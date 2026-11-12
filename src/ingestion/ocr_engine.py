"""Two-tier Hybrid OCR Engine (Polza AI Qwen3-VL + Local RapidOCR v3 PP-OCRv5 ESLAV fallback) with SHA-256 disk cache."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import statistics
import threading
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

from src.config import (
    OCR_CACHE_DIR,
    VLM_MAX_TOKENS,
    VLM_SYSTEM_PROMPT,
    VLM_TEMPERATURE,
    VLM_TIMEOUT_SEC,
    settings,
)

logger = logging.getLogger(__name__)

# Anti-loop regexes for blank fill-in lines in forms and signature sheets
_REPEAT_UNDERSCORES_RE = re.compile(r"_{4,}")
_REPEAT_DOTS_RE = re.compile(r"\.{6,}")
_REPEAT_DASHES_RE = re.compile(r"[─━—–]{5,}")
_MULTI_BLANK_MARKER_RE = re.compile(r"(?:\[___\]\s*){2,}")
_CODE_FENCE_START_RE = re.compile(r"^\s*```(?:markdown|md|text|txt)?\s*\n?", re.IGNORECASE)
_CODE_FENCE_END_RE = re.compile(r"\n?\s*```\s*$")


def sanitize_ocr_text(text: str) -> str:
    """
    Clean OCR text and enforce anti-loop blank line replacement (`[___]`):
    - Strips accidental Markdown code fences wrapped by VLM
    - Replaces long sequences of `_`, `.`, or `─` on signature/blank lines with `[___]`
    - Normalizes private-use bullet characters and non-breaking spaces
    """
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = _CODE_FENCE_START_RE.sub("", cleaned)
    cleaned = _CODE_FENCE_END_RE.sub("", cleaned)

    # Normalize line endings and special spaces/bullets
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    for pua_bullet in ("\uf02d", "\uf0b7", "\uf0a7", "\uf076", "\uf0d8", "\uf0fc"):
        cleaned = cleaned.replace(pua_bullet, "-")

    # Replace long fill-in underscores/dots/dashes with [___] (except inside Markdown table separator rows |---|)
    out_lines: list[str] = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        is_md_table_sep = (
            stripped.startswith("|")
            and stripped.endswith("|")
            and all(re.match(r"^[-:\s]*$", cell) for cell in stripped.strip("|").split("|"))
        )
        if not is_md_table_sep:
            line = _REPEAT_UNDERSCORES_RE.sub("[___]", line)
            line = _REPEAT_DOTS_RE.sub("[___]", line)
            line = _REPEAT_DASHES_RE.sub("[___]", line)
            line = _MULTI_BLANK_MARKER_RE.sub("[___] ", line)
        out_lines.append(re.sub(r"[ \t]{2,}", " ", line).strip())

    result = "\n".join(out_lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


class HybridOCREngine:
    """
    Two-tier OCR engine with SHA-256 PNG page caching in `data/ocr_cache/`:
    1. Primary (`hybrid_vlm`): Polza AI OpenAI-compatible Vision API (`qwen/qwen3-vl-235b-a22b-instruct`,
       `temperature=0.0`, `max_tokens=2500`) with strict anti-loop system prompt.
    2. Automatic Fallback (`local_rapidocr`): Local CPU `RapidOCR v3` (`PP-OCRv5`, `LangRec.ESLAV`,
       `ModelType.MOBILE`, `Global.use_cls=False`, `Det.box_thresh=0.3`) triggered when:
       - `OCR_BACKEND == "local_rapidocr"`
       - VLM returns `finish_reason == "length"` (blank form underscore loop protection)
       - VLM API times out, errors, or has no API key configured
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir: Path = cache_dir or OCR_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rapidocr_engine: Any = None
        self._rapidocr_lock = threading.Lock()
        self._openai_client: OpenAI | None = None

    def _get_openai_client(self) -> OpenAI | None:
        """Lazily initialize OpenAI-compatible client for Polza AI VLM."""
        if not settings.POLZA_API_KEY:
            return None
        if self._openai_client is None:
            self._openai_client = OpenAI(
                api_key=settings.POLZA_API_KEY,
                base_url=settings.POLZA_BASE_URL,
                timeout=VLM_TIMEOUT_SEC,
            )
        return self._openai_client

    def _get_rapidocr_engine(self) -> Any:
        """
        Lazily initialize RapidOCR v3 with verified Cyrillic/Russian parameters:
        - Rec.ocr_version = OCRVersion.PPOCRV5
        - Rec.lang_type = LangRec.ESLAV
        - Rec.model_type = ModelType.MOBILE
        - Global.use_cls = False (CRITICAL: prevents 180-degree rotation of wide Russian lines)
        - Det.box_thresh = 0.3 (preserves short headers and clause numbers)
        """
        if self._rapidocr_engine is None:
            from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

            params: dict[str, Any] = {
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.lang_type": LangRec.ESLAV,
                "Rec.model_type": ModelType.MOBILE,
                "Global.use_cls": False,
                "Det.box_thresh": 0.3,
            }
            self._rapidocr_engine = RapidOCR(params=params)
        return self._rapidocr_engine

    @staticmethod
    def compute_image_sha256(png_bytes: bytes) -> str:
        """Compute SHA-256 hex digest of rendered page PNG bytes."""
        return hashlib.sha256(png_bytes).hexdigest()

    def _read_cache(self, sha256_hex: str, requested_backend: str) -> dict[str, Any] | None:
        """Read cached OCR result from `data/ocr_cache/<sha256>.json` if compatible with requested_backend."""
        json_path = self.cache_dir / f"{sha256_hex}.json"
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                cached_text = str(data.get("text") or "").strip()
                backend_used = str(data.get("backend_used") or "")
                vlm_attempted = bool(data.get("vlm_attempted", False))
                if cached_text:
                    # If hybrid_vlm is requested, accept cache if VLM succeeded or VLM was attempted and fell back
                    if requested_backend == "hybrid_vlm":
                        if backend_used == "hybrid_vlm" or vlm_attempted:
                            return data
                    else:
                        return data
            except Exception as exc:
                logger.warning("Failed to read OCR cache %s: %s", json_path, exc)

        txt_path = self.cache_dir / f"{sha256_hex}.txt"
        if txt_path.exists() and requested_backend == "local_rapidocr":
            try:
                cached_text = txt_path.read_text(encoding="utf-8").strip()
                if cached_text:
                    return {
                        "sha256": sha256_hex,
                        "text": cached_text,
                        "backend_used": "cache_txt",
                        "vlm_attempted": False,
                        "finish_reason": "cached",
                    }
            except Exception as exc:
                logger.warning("Failed to read OCR txt cache %s: %s", txt_path, exc)

        return None

    def _write_cache(
        self,
        sha256_hex: str,
        text: str,
        backend_used: str,
        vlm_attempted: bool,
        finish_reason: str = "",
        fallback_triggered: bool = False,
        file_name: str = "",
        page_num: int = 1,
    ) -> None:
        """Atomically persist OCR result to `data/ocr_cache/<sha256>.json`."""
        if not text or not text.strip():
            return
        payload = {
            "sha256": sha256_hex,
            "text": text,
            "backend_used": backend_used,
            "vlm_attempted": vlm_attempted,
            "finish_reason": finish_reason,
            "fallback_triggered": fallback_triggered,
            "file_name": file_name,
            "page_num": page_num,
        }
        target_path = self.cache_dir / f"{sha256_hex}.json"
        tmp_path = self.cache_dir / f"{sha256_hex}.json.tmp"
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(target_path)
        except Exception as exc:
            logger.warning("Failed to write OCR cache %s: %s", target_path, exc)

    def _call_polza_vlm(
        self,
        png_bytes: bytes,
        file_name: str = "",
        page_num: int = 1,
    ) -> tuple[str | None, str]:
        """
        Call Polza AI Vision model (`qwen/qwen3-vl-235b-a22b-instruct`).
        Returns `(recognized_text_or_None, finish_reason)`.
        If `finish_reason == "length"` or an API error occurs, returns `(None, reason)` to trigger RapidOCR fallback.
        """
        client = self._get_openai_client()
        if client is None:
            return None, "no_api_key"

        b64_img = base64.b64encode(png_bytes).decode("ascii")
        user_prompt = (
            "Распознай весь текст на этой странице документа МФТИ. "
            "Сохрани все номера разделов и пунктов (например, 1.1., 2.3., 6.1.), реквизиты, "
            "штампы ЭДО («Зарегистрировано в ЭДО МФТИ от ... № ...») и таблицы в формате Markdown. "
            "Помни: никогда не повторяй символы `_`, `.` или пробелы на пустых графах для подписи или заполнения — "
            "заменяй любую пустую линию бланка на один короткий маркер `[___]`."
        )

        try:
            response = client.chat.completions.create(
                model=settings.POLZA_VLM_MODEL,
                temperature=VLM_TEMPERATURE,
                max_tokens=VLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": VLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                            },
                        ],
                    },
                ],
            )
            if not response.choices:
                logger.warning("Polza VLM returned empty choices on %s p.%d", file_name, page_num)
                return None, "empty_choices"

            choice = response.choices[0]
            finish_reason = str(choice.finish_reason or "stop")
            raw_content = str(choice.message.content or "").strip()

            if finish_reason == "length":
                logger.warning(
                    "Polza VLM hit finish_reason='length' on %s p.%d; switching page to local RapidOCR fallback!",
                    file_name,
                    page_num,
                )
                return None, "length"

            # Additional guard: detect runaway repetition if model ignored finish_reason
            if "__________" in raw_content and raw_content.count("_") > 400:
                logger.warning(
                    "Polza VLM produced excessive underscores on %s p.%d; sanitizing/checking",
                    file_name,
                    page_num,
                )

            cleaned = sanitize_ocr_text(raw_content)
            if not cleaned:
                return None, "empty_text"

            return cleaned, finish_reason
        except Exception as exc:
            logger.warning(
                "Polza VLM API error on %s p.%d (%s); switching to local RapidOCR fallback",
                file_name,
                page_num,
                exc,
            )
            return None, f"error:{type(exc).__name__}"

    def _call_local_rapidocr(self, png_bytes: bytes, file_name: str = "", page_num: int = 1) -> str:
        """
        Run local offline CPU fallback using RapidOCR v3 (PP-OCRv5 ESLAV MOBILE, use_cls=False, box_thresh=0.3)
        and reconstruct reading-order lines by grouping horizontal bounding boxes.
        """
        try:
            with self._rapidocr_lock:
                engine = self._get_rapidocr_engine()
                pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                result = engine(pil_img)
            txts = getattr(result, "txts", None)
            boxes = getattr(result, "boxes", None)

            if not txts:
                return ""

            if boxes is None or len(boxes) != len(txts):
                return sanitize_ocr_text("\n".join(str(t) for t in txts if t))

            # Group text boxes into horizontal reading lines
            items: list[dict[str, Any]] = []
            heights: list[float] = []
            for box, txt in zip(boxes, txts):
                clean_t = str(txt or "").strip()
                if not clean_t:
                    continue
                xs = [float(pt[0]) for pt in box]
                ys = [float(pt[1]) for pt in box]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                h = max(1.0, y1 - y0)
                heights.append(h)
                items.append(
                    {
                        "text": clean_t,
                        "x0": x0,
                        "x1": x1,
                        "y0": y0,
                        "y1": y1,
                        "yc": (y0 + y1) / 2.0,
                        "h": h,
                    }
                )

            if not items:
                return ""

            median_h = statistics.median(heights) if heights else 20.0
            line_tol = max(8.0, median_h * 0.55)

            # Sort primarily by vertical center
            items.sort(key=lambda it: (it["yc"], it["x0"]))

            lines: list[list[dict[str, Any]]] = []
            for it in items:
                if not lines:
                    lines.append([it])
                    continue
                prev_line = lines[-1]
                prev_yc = sum(x["yc"] for x in prev_line) / len(prev_line)
                if abs(it["yc"] - prev_yc) <= line_tol:
                    prev_line.append(it)
                else:
                    lines.append([it])

            reconstructed_lines: list[str] = []
            for line_items in lines:
                line_items.sort(key=lambda it: it["x0"])
                parts: list[str] = []
                for idx, it in enumerate(line_items):
                    if idx > 0:
                        gap = it["x0"] - line_items[idx - 1]["x1"]
                        # Keep space between words on the same line
                        if gap > median_h * 4.0 and len(line_items) >= 3:
                            parts.append(" | ")
                        else:
                            parts.append(" ")
                    parts.append(it["text"])
                reconstructed_lines.append("".join(parts).strip())

            return sanitize_ocr_text("\n".join(reconstructed_lines))
        except Exception as exc:
            logger.error("Local RapidOCR failed on %s p.%d: %s", file_name, page_num, exc)
            return ""

    def recognize_page(
        self,
        png_bytes: bytes,
        ocr_backend: str | None = None,
        page_num: int = 1,
        file_name: str = "",
    ) -> dict[str, Any]:
        """
        Recognize a rendered PDF page PNG image with SHA-256 disk caching and automatic fallback.
        Returns a dict:
        {
            "text": str,
            "backend_used": "hybrid_vlm" | "local_rapidocr",
            "from_cache": bool,
            "sha256": str,
            "fallback_triggered": bool,
            "finish_reason": str,
        }
        """
        backend = (ocr_backend or settings.OCR_BACKEND or "hybrid_vlm").strip().lower()
        if backend not in {"hybrid_vlm", "local_rapidocr"}:
            backend = "hybrid_vlm"

        sha256_hex = self.compute_image_sha256(png_bytes)

        cached = self._read_cache(sha256_hex, requested_backend=backend)
        if cached is not None:
            return {
                "text": sanitize_ocr_text(str(cached.get("text") or "")),
                "backend_used": str(cached.get("backend_used") or backend),
                "from_cache": True,
                "sha256": sha256_hex,
                "fallback_triggered": bool(cached.get("fallback_triggered", False)),
                "finish_reason": str(cached.get("finish_reason") or "cached"),
            }

        if backend == "local_rapidocr":
            text = self._call_local_rapidocr(png_bytes, file_name=file_name, page_num=page_num)
            self._write_cache(
                sha256_hex=sha256_hex,
                text=text,
                backend_used="local_rapidocr",
                vlm_attempted=False,
                finish_reason="local",
                fallback_triggered=False,
                file_name=file_name,
                page_num=page_num,
            )
            return {
                "text": text,
                "backend_used": "local_rapidocr",
                "from_cache": False,
                "sha256": sha256_hex,
                "fallback_triggered": False,
                "finish_reason": "local",
            }

        # Primary mode: hybrid_vlm -> fallback to local_rapidocr on finish_reason=="length" or API error
        vlm_text, finish_reason = self._call_polza_vlm(png_bytes, file_name=file_name, page_num=page_num)
        if vlm_text is not None and finish_reason != "length":
            self._write_cache(
                sha256_hex=sha256_hex,
                text=vlm_text,
                backend_used="hybrid_vlm",
                vlm_attempted=True,
                finish_reason=finish_reason,
                fallback_triggered=False,
                file_name=file_name,
                page_num=page_num,
            )
            return {
                "text": vlm_text,
                "backend_used": "hybrid_vlm",
                "from_cache": False,
                "sha256": sha256_hex,
                "fallback_triggered": False,
                "finish_reason": finish_reason,
            }

        # Fallback to local RapidOCR v3
        fallback_text = self._call_local_rapidocr(png_bytes, file_name=file_name, page_num=page_num)
        # Only mark vlm_attempted=True in cache if VLM failed due to content ("length"),
        # not transient network/API key absence, or cache it so repeated runs are fast.
        vlm_permanent_fallback = finish_reason == "length"
        self._write_cache(
            sha256_hex=sha256_hex,
            text=fallback_text,
            backend_used="local_rapidocr",
            vlm_attempted=vlm_permanent_fallback,
            finish_reason=finish_reason,
            fallback_triggered=True,
            file_name=file_name,
            page_num=page_num,
        )
        return {
            "text": fallback_text,
            "backend_used": "local_rapidocr",
            "from_cache": False,
            "sha256": sha256_hex,
            "fallback_triggered": True,
            "finish_reason": finish_reason,
        }

    def ocr_page_bytes(
        self,
        png_bytes: bytes,
        ocr_backend: str | None = None,
        page_num: int = 1,
        file_name: str = "",
    ) -> str:
        """Convenience wrapper returning recognized text string for a page PNG."""
        res = self.recognize_page(
            png_bytes=png_bytes,
            ocr_backend=ocr_backend,
            page_num=page_num,
            file_name=file_name,
        )
        return str(res.get("text") or "")


_ENGINE_SINGLETON: HybridOCREngine | None = None


def get_ocr_engine() -> HybridOCREngine:
    """Return singleton HybridOCREngine instance."""
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        _ENGINE_SINGLETON = HybridOCREngine()
    return _ENGINE_SINGLETON
