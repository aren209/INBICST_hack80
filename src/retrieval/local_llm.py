"""Local 4-bit Linear-Hybrid Qwen/Qwen3.8-27B LLM Client Factory (vLLM INT4 & HuggingFace BitsAndBytes NF4)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import threading
from types import SimpleNamespace
from typing import Any, Iterator
import uuid

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

# Candidate model identifiers for local 4-bit Qwen/Qwen3.8-27B
LOCAL_QWEN_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.8-27B-INT4",
    "RedHatAI/Qwen3.8-27B-INT4",
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_XML_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_XML_FUNC_RE = re.compile(r"<function\s*=\s*([^\s>]+)\s*>(.*?)</function>", re.DOTALL | re.IGNORECASE)
_XML_PARAM_RE = re.compile(r"<parameter\s*=\s*([^\s>]+)\s*>(.*?)</parameter>", re.DOTALL | re.IGNORECASE)

_BNB_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_BNB_LOCK = threading.RLock()


def strip_thinking_tags(text: str | None) -> str:
    """Strip `<think>...</think>` reasoning blocks emitted by Qwen3.8 models when thinking occurs."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in cleaned and "</think>" not in cleaned:
        # Defensive guard if max_tokens truncated inside an unclosed <think> block
        before, _, _ = cleaned.partition("<think>")
        if before.strip():
            cleaned = before
    return cleaned.strip()


def parse_qwen3_xml_tool_calls(raw_text: str) -> tuple[str, list[Any]]:
    """
    Parse Qwen3.8 XML tool calls (`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`)
    or JSON `<tool_call>{"name": ..., "arguments": ...}</tool_call>` into OpenAI-compatible tool_call objects.
    Returns `(remaining_content_text, tool_calls_list)`.
    """
    if "<tool_call>" not in raw_text:
        return strip_thinking_tags(raw_text), []

    tool_calls: list[Any] = []
    for idx, match in enumerate(_XML_TOOL_CALL_RE.finditer(raw_text), start=1):
        inner = match.group(1).strip()
        fn_match = _XML_FUNC_RE.search(inner)
        if fn_match:
            fn_name = fn_match.group(1).strip()
            fn_body = fn_match.group(2)
            args_dict: dict[str, Any] = {}
            for p_match in _XML_PARAM_RE.finditer(fn_body):
                p_key = p_match.group(1).strip()
                p_val_raw = p_match.group(2).strip()
                try:
                    p_val = json.loads(p_val_raw)
                except Exception:
                    p_val = p_val_raw
                args_dict[p_key] = p_val
            tool_calls.append(
                SimpleNamespace(
                    id=f"call_{uuid.uuid4().hex[:10]}_{idx}",
                    type="function",
                    function=SimpleNamespace(
                        name=fn_name,
                        arguments=json.dumps(args_dict, ensure_ascii=False),
                    ),
                )
            )
            continue

        # Fallback: JSON inside <tool_call>{...}</tool_call>
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict) and parsed.get("name"):
                raw_args = parsed.get("arguments", {})
                args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                tool_calls.append(
                    SimpleNamespace(
                        id=f"call_{uuid.uuid4().hex[:10]}_{idx}",
                        type="function",
                        function=SimpleNamespace(
                            name=str(parsed["name"]),
                            arguments=args_str,
                        ),
                    )
                )
        except Exception:
            logger.debug("Could not parse <tool_call> block: %s", inner[:200])

    cleaned_content = _XML_TOOL_CALL_RE.sub("", raw_text)
    cleaned_content = strip_thinking_tags(cleaned_content)
    return cleaned_content, tool_calls


def _load_bnb_4bit_qwen(model_id: str) -> tuple[Any, Any]:
    """Load `Qwen/Qwen3.8-27B` in 4-bit quantization via HuggingFace `transformers` + `BitsAndBytesConfig`."""
    if model_id in _BNB_MODEL_CACHE:
        return _BNB_MODEL_CACHE[model_id]

    import torch
    from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading in-process 4-bit Qwen3.8 model '%s' via BitsAndBytes NF4...", model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tokenizer = AutoProcessor.from_pretrained(model_id)

    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
    except Exception:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

    _BNB_MODEL_CACHE[model_id] = (model, tokenizer)
    return model, tokenizer


@dataclass
class _TransformersCompletionsEndpoint:
    """OpenAI-compatible `.chat.completions` adapter for in-process 4-bit HuggingFace Qwen3.8-27B."""

    default_model_id: str
    enable_thinking: bool = False

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        import torch

        target_repo = self.default_model_id
        with _BNB_LOCK:
            hf_model, tok = _load_bnb_4bit_qwen(target_repo)
            chat_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": self.enable_thinking,
            }
            if tools and tool_choice != "none":
                chat_kwargs["tools"] = tools

            prompt_text = tok.apply_chat_template(messages, **chat_kwargs)
            inputs = tok(prompt_text, return_tensors="pt").to(hf_model.device)
            prompt_len = int(inputs["input_ids"].shape[-1])

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens or 2048,
                "do_sample": bool(temperature and temperature > 1e-4),
            }
            if gen_kwargs["do_sample"]:
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = 0.9

            with torch.inference_mode():
                out_ids = hf_model.generate(**inputs, **gen_kwargs)

            gen_ids = out_ids[0][prompt_len:]
            raw_output = tok.decode(gen_ids, skip_special_tokens=True)

        content, parsed_tools = parse_qwen3_xml_tool_calls(raw_output)
        if not stream:
            msg = SimpleNamespace(
                role="assistant",
                content=content,
                tool_calls=parsed_tools if parsed_tools else None,
            )
            choice = SimpleNamespace(
                index=0,
                message=msg,
                finish_reason="tool_calls" if parsed_tools else "stop",
            )
            return SimpleNamespace(model=model or target_repo, choices=[choice])

        return self._stream_single_response(model or target_repo, content, parsed_tools)

    @staticmethod
    def _stream_single_response(
        model_name: str,
        content: str,
        parsed_tools: list[Any],
    ) -> Iterator[Any]:
        if content:
            delta = SimpleNamespace(content=content, tool_calls=None)
            yield SimpleNamespace(model=model_name, choices=[SimpleNamespace(delta=delta, finish_reason=None)])
        if parsed_tools:
            tc_deltas = [
                SimpleNamespace(
                    index=i,
                    id=tc.id,
                    type="function",
                    function=SimpleNamespace(name=tc.function.name, arguments=tc.function.arguments),
                )
                for i, tc in enumerate(parsed_tools)
            ]
            delta_tc = SimpleNamespace(content=None, tool_calls=tc_deltas)
            yield SimpleNamespace(model=model_name, choices=[SimpleNamespace(delta=delta_tc, finish_reason="tool_calls")])
        else:
            yield SimpleNamespace(
                model=model_name,
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop")],
            )


@dataclass
class Transformers4BitChatClient:
    """OpenAI-compatible client wrapper around local 4-bit HuggingFace `Qwen/Qwen3.8-27B`."""

    model_id: str = field(default_factory=lambda: settings.LOCAL_LLM_MODEL)
    enable_thinking: bool = field(default_factory=lambda: settings.LLM_ENABLE_THINKING)

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=_TransformersCompletionsEndpoint(
                default_model_id=self.model_id,
                enable_thinking=self.enable_thinking,
            )
        )


def create_llm_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float,
    max_retries: int = 2,
    llm_provider: str = "local",
) -> Any:
    """
    Create the configured LLM client:
    - If `api_key` is empty -> returns `None` (triggers offline deterministic fallback in unit tests).
    - If `llm_provider == "local"` and `settings.LLM_BACKEND == "local_bnb_4bit"` -> returns in-process
      `Transformers4BitChatClient` (`BitsAndBytesConfig(load_in_4bit=True)`).
    - Otherwise (`local_vllm` default on `http://localhost:8001/v1` serving 4-bit `Qwen/Qwen3.8-27B`,
      or explicit external OpenAI-compatible endpoint) -> returns `OpenAI(...)`.
    """
    if not api_key:
        return None
    if llm_provider == "local" and settings.LLM_BACKEND.strip().lower() == "local_bnb_4bit":
        return Transformers4BitChatClient(
            model_id=settings.LOCAL_LLM_MODEL,
            enable_thinking=settings.LLM_ENABLE_THINKING,
        )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
