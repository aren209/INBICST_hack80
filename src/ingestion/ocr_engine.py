"""Two-tier Local Hybrid OCR Engine (PaddlePaddle/PaddleOCR-VL-1.6 + Local RapidOCR v3 PP-OCRv5 ESLAV fallback) with SHA-256 disk cache."""

from __future__ import annotations

import base64
import hashlib
from html.parser import HTMLParser
import io
import json
import logging
import re
import statistics
import threading
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageFilter

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
_REPEAT_PHRASE_LOOP_RE = re.compile(r"\b(\S+(?:\s+\S+){0,2})(?:\s+\1){4,}", re.IGNORECASE)
_CODE_FENCE_START_RE = re.compile(r"^\s*```(?:markdown|md|text|txt|html)?\s*\n?", re.IGNORECASE)
_CODE_FENCE_END_RE = re.compile(r"\n?\s*```\s*$")
_HTML_TABLE_BLOCK_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_OTSL_CELL_SPLIT_RE = re.compile(r"(<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)")

# Process-wide singleton cache for local PaddleOCR-VL-1.6 model & processor
_PADDLEOCR_VL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_PADDLEOCR_VL_LOCK = threading.RLock()


class _SimpleHTMLTableParser(HTMLParser):
    """Lightweight HTML `<table>` -> 2D grid parser supporting `colspan` and `rowspan`."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[tuple[str, int, int, bool]]] = []
        self._in_row = False
        self._in_cell = False
        self._is_header = False
        self._colspan = 1
        self._rowspan = 1
        self._cur_row: list[tuple[str, int, int, bool]] = []
        self._cell_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "tr":
            self._in_row = True
            self._cur_row = []
        elif t in ("td", "th") and self._in_row:
            self._in_cell = True
            self._is_header = t == "th"
            attr_map = {k.lower(): (v or "") for k, v in attrs}
            try:
                self._colspan = max(1, int(attr_map.get("colspan") or 1))
            except ValueError:
                self._colspan = 1
            try:
                self._rowspan = max(1, int(attr_map.get("rowspan") or 1))
            except ValueError:
                self._rowspan = 1
            self._cell_buf = []
        elif t == "br" and self._in_cell:
            self._cell_buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._in_cell:
            cell_txt = re.sub(r"\s+", " ", "".join(self._cell_buf)).strip()
            self._cur_row.append((cell_txt, self._colspan, self._rowspan, self._is_header))
            self._in_cell = False
        elif t == "tr" and self._in_row:
            if self._cur_row:
                self.rows.append(self._cur_row)
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buf.append(data)


def _html_table_to_markdown(html_table: str) -> str:
    """Convert an HTML `<table>...</table>` emitted by PaddleOCR-VL into pipe-separated Markdown."""
    parser = _SimpleHTMLTableParser()
    try:
        parser.feed(html_table)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html_table)

    if not parser.rows:
        return ""

    # Expand rowspan / colspan into a rectangular grid
    grid: list[dict[int, str]] = [{} for _ in range(len(parser.rows))]
    max_col = 0
    for r_idx, raw_cells in enumerate(parser.rows):
        c_idx = 0
        for txt, colspan, rowspan, _is_hdr in raw_cells:
            while c_idx in grid[r_idx]:
                c_idx += 1
            clean_cell = txt.replace("|", "/").strip()
            for dr in range(rowspan):
                target_r = r_idx + dr
                while target_r >= len(grid):
                    grid.append({})
                for dc in range(colspan):
                    target_c = c_idx + dc
                    grid[target_r][target_c] = clean_cell if (dr == 0 and dc == 0) else (clean_cell if dr > 0 and dc == 0 else "")
                    if target_c + 1 > max_col:
                        max_col = target_c + 1
            c_idx += colspan

    if max_col == 0:
        return ""

    matrix: list[list[str]] = []
    for row_map in grid:
        row_vals = [row_map.get(c, "") for c in range(max_col)]
        if any(v.strip() for v in row_vals):
            matrix.append(row_vals)

    if not matrix:
        return ""

    lines: list[str] = []
    lines.append("| " + " | ".join(matrix[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_col) + " |")
    for row_vals in matrix[1:]:
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


def _otsl_to_markdown(otsl_text: str) -> str:
    """Convert PaddleOCR-VL OTSL table output (`<fcel>...<ecel>...<nl>`) into a clean Markdown pipe table."""
    raw = (
        otsl_text.replace("</s>", "")
        .replace("<s>", "")
        .replace("<unk>", "")
        .replace("<|end_of_sentence|>", "")
        .replace("<|begin_of_sentence|>", "")
        .strip()
    )
    if "<fcel>" not in raw and "<ecel>" not in raw:
        return raw

    raw_rows = [r.strip() for r in raw.split("<nl>") if r.strip()]
    parsed_rows: list[list[tuple[str, str]]] = []
    max_cols = 0

    for r_str in raw_rows:
        parts = _OTSL_CELL_SPLIT_RE.split(r_str)
        row_cells: list[tuple[str, str]] = []
        cur_tag: str | None = None
        for p in parts:
            if p in {"<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>"}:
                cur_tag = p
            elif cur_tag is not None:
                val = re.sub(r"\s+", " ", p).replace("|", "/").strip()
                row_cells.append((cur_tag, val))
                cur_tag = None
        if row_cells:
            parsed_rows.append(row_cells)
            if len(row_cells) > max_cols:
                max_cols = len(row_cells)

    if not parsed_rows or max_cols == 0:
        return ""

    matrix: list[list[str]] = []
    for r_idx, row_cells in enumerate(parsed_rows):
        row_vals: list[str] = []
        for c_idx in range(max_cols):
            if c_idx < len(row_cells):
                tag, val = row_cells[c_idx]
                if tag == "<ucel>" and r_idx > 0:
                    val = matrix[r_idx - 1][c_idx]
                elif tag in {"<ecel>", "<lcel>", "<xcel>"} and not val:
                    val = ""
                row_vals.append(val)
            else:
                row_vals.append("")
        if any(v.strip() for v in row_vals):
            matrix.append(row_vals)

    if not matrix:
        return ""

    lines: list[str] = []
    lines.append("| " + " | ".join(matrix[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row_vals in matrix[1:]:
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


def _image_has_grid_table(pil_img: Image.Image) -> bool:
    """Fast morphology heuristic (~4ms) detecting ruled grid tables via MinFilter(3) dark line runs."""
    try:
        gray = pil_img.convert("L").filter(ImageFilter.MinFilter(3)).resize((600, 800))
        px = gray.load()
        if px is None:
            return False
        w, h = gray.size
        horiz_lines = 0
        for y in range(20, h - 20, 2):
            dark_run = 0
            max_run = 0
            for x in range(20, w - 20):
                if px[x, y] < 185:
                    dark_run += 1
                    if dark_run > max_run:
                        max_run = dark_run
                else:
                    dark_run = 0
            if max_run >= int(w * 0.42):
                horiz_lines += 1

        vert_lines = 0
        for x in range(20, w - 20, 2):
            dark_run = 0
            max_run = 0
            for y in range(20, h - 20):
                if px[x, y] < 185:
                    dark_run += 1
                    if dark_run > max_run:
                        max_run = dark_run
                else:
                    dark_run = 0
            if max_run >= int(h * 0.14):
                vert_lines += 1

        return horiz_lines >= 4 and vert_lines >= 3
    except Exception:
        return False


def _convert_html_tables_in_text(text: str) -> str:
    """Convert any inline HTML `<table>...</table>` or OTSL blocks in VLM output into Markdown tables."""
    out = text
    if "<fcel>" in out or "<ecel>" in out:
        out = _otsl_to_markdown(out)
    if "<table" not in out.lower():
        return out
    converted = _HTML_TABLE_BLOCK_RE.sub(lambda m: "\n" + _html_table_to_markdown(m.group(0)) + "\n", out)
    # Strip any leftover basic inline formatting tags outside tables (e.g. <b>, <i>, <sup>, <sub>)
    converted = re.sub(r"</?(?:b|i|u|strong|em|sup|sub|span|div|p|thead|tbody)\b[^>]*>", "", converted, flags=re.IGNORECASE)
    return converted


def sanitize_ocr_text(text: str) -> str:
    """
    Clean OCR text and enforce anti-loop blank line replacement (`[___]`):
    - Strips accidental Markdown code fences wrapped by VLM
    - Converts HTML `<table>...</table>` blocks (from PaddleOCR-VL) into clean Markdown pipe tables
    - Replaces long sequences of `_`, `.`, or `─` on signature/blank lines with `[___]`
    - Normalizes private-use bullet characters and non-breaking spaces
    """
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = _CODE_FENCE_START_RE.sub("", cleaned)
    cleaned = _CODE_FENCE_END_RE.sub("", cleaned)
    cleaned = _convert_html_tables_in_text(cleaned)

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
            line = _REPEAT_PHRASE_LOOP_RE.sub(r"\1", line)
        out_lines.append(re.sub(r"[ \t]{2,}", " ", line).strip())

    result = "\n".join(out_lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def _resolve_ocr_vl_device(preferred: str = "auto") -> str:
    pref = (preferred or "auto").strip().lower()
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_paddleocr_vl_components(model_name: str, device: str) -> tuple[Any, Any]:
    """Lazily load and cache local `PaddlePaddle/PaddleOCR-VL-1.6` model and processor."""
    key = (model_name, device)
    if key in _PADDLEOCR_VL_CACHE:
        return _PADDLEOCR_VL_CACHE[key]

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    logger.info("Loading local OCR VLM '%s' on device '%s' (dtype=%s)...", model_name, device, dtype)
    processor = AutoProcessor.from_pretrained(model_name)
    if device.startswith("cuda"):
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
        ).eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(device).eval()

    _PADDLEOCR_VL_CACHE[key] = (model, processor)
    return model, processor


class HybridOCREngine:
    """
    Two-tier Local Hybrid OCR engine with SHA-256 PNG page caching in `data/ocr_cache/`:
    1. Primary (`paddleocr_vl` / `hybrid_vlm`): Local `PaddlePaddle/PaddleOCR-VL-1.6` (0.9B SOTA Document VLM)
       running directly on GPU/CPU via HuggingFace `transformers`, converting text, stamps, formulas, and tables
       into clean Markdown with anti-loop sanitization.
    2. Automatic Fallback (`local_rapidocr`): Local CPU `RapidOCR v3` (`PP-OCRv5`, `LangRec.ESLAV`,
       `ModelType.MOBILE`, `Global.use_cls=False`, `Det.box_thresh=0.3`) triggered when:
       - `OCR_BACKEND == "local_rapidocr"`
       - VLM hits max token limit (`finish_reason == "length"`, blank form loop protection)
       - VLM encounters runtime/OOM error
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        vl_model_name: str | None = None,
        vl_device: str | None = None,
    ) -> None:
        self.cache_dir: Path = cache_dir or OCR_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vl_model_name: str = vl_model_name or settings.OCR_VL_MODEL
        self.vl_device_pref: str = vl_device or settings.OCR_VL_DEVICE
        self._rapidocr_engine: Any = None
        self._rapidocr_lock = threading.Lock()
        self._openai_client: OpenAI | None = None

    @property
    def resolved_vl_device(self) -> str:
        return _resolve_ocr_vl_device(self.vl_device_pref)

    def _get_openai_client(self) -> OpenAI | None:
        """Optional legacy OpenAI-compatible VLM client (only used if `OCR_BACKEND='polza_vlm'` and key is set)."""
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
                    if requested_backend in {"paddleocr_vl", "hybrid_vlm", "polza_vlm"}:
                        if backend_used in {"paddleocr_vl", "hybrid_vlm", "polza_vlm"} or vlm_attempted:
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
            "model_used": self.vl_model_name if backend_used == "paddleocr_vl" else backend_used,
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

    def _call_paddleocr_vl(
        self,
        png_bytes: bytes,
        file_name: str = "",
        page_num: int = 1,
    ) -> tuple[str | None, str]:
        """
        Run local `PaddlePaddle/PaddleOCR-VL-1.6` Vision-Language OCR on a rendered page PNG (`use_cache=True`).
        When a ruled grid table is present on the page, also runs `Table Recognition:` (OTSL -> Markdown)
        so multi-column row alignment is preserved.
        Returns `(recognized_text_or_None, finish_reason)`.
        """
        try:
            import torch

            pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            has_grid = _image_has_grid_table(pil_img)

            def _run_vl_prompt(model: Any, processor: Any, prompt_str: str, keep_special: bool = False) -> tuple[str, int]:
                min_pixels = getattr(getattr(processor, "image_processor", None), "min_pixels", 28 * 28 * 4)
                max_pixels = int(getattr(settings, "OCR_VL_MAX_PIXELS", 1280 * 28 * 28) or (1280 * 28 * 28))
                msgs = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": prompt_str},
                        ],
                    }
                ]
                text_prompt = processor.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=[text_prompt],
                    images=[pil_img],
                    return_tensors="pt",
                    size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
                ).to(model.device)
                prompt_len = int(inputs["input_ids"].shape[-1])
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=VLM_MAX_TOKENS,
                        do_sample=False,
                        use_cache=True,
                    )
                gen_ids = outputs[0][prompt_len:]
                decoded = processor.decode(gen_ids, skip_special_tokens=not keep_special).strip()
                return decoded, int(gen_ids.shape[-1])

            with _PADDLEOCR_VL_LOCK:
                model, processor = _get_paddleocr_vl_components(self.vl_model_name, self.resolved_vl_device)
                raw_content, gen_len = _run_vl_prompt(model, processor, "OCR:", keep_special=False)
                md_table_block = ""
                if has_grid and "|" not in raw_content:
                    raw_otsl, _ = _run_vl_prompt(model, processor, "Table Recognition:", keep_special=True)
                    if "<fcel>" in raw_otsl or "<ecel>" in raw_otsl:
                        md_table_block = _otsl_to_markdown(raw_otsl)

            if gen_len >= VLM_MAX_TOKENS - 4:
                lines = [ln.strip() for ln in raw_content.splitlines() if ln.strip()]
                unique_ratio = (len(set(lines)) / len(lines)) if len(lines) >= 10 else 1.0
                if raw_content.count("_") > 300 or raw_content.count(".") > 500 or unique_ratio < 0.35:
                    logger.warning(
                        "PaddleOCR-VL-1.6 hit max_new_tokens=%d with repetition on %s p.%d; switching to RapidOCR fallback",
                        VLM_MAX_TOKENS,
                        file_name,
                        page_num,
                    )
                    return None, "length"

            if md_table_block:
                prose_clauses = [
                    ln for ln in raw_content.splitlines()
                    if re.match(r"^\s*(?:\d+\.\d+\.?|Приложение\s*№)", ln)
                ]
                combined = ("\n".join(prose_clauses) + "\n\n" + md_table_block).strip() if prose_clauses else md_table_block
            else:
                combined = raw_content
            cleaned = sanitize_ocr_text(combined)
            if not cleaned:
                return None, "empty_text"
            return cleaned, "stop"
        except Exception as exc:
            logger.warning(
                "Local PaddleOCR-VL-1.6 error on %s p.%d (%s); switching to local RapidOCR fallback",
                file_name,
                page_num,
                exc,
            )
            return None, f"error:{type(exc).__name__}"

    def _call_polza_vlm(
        self,
        png_bytes: bytes,
        file_name: str = "",
        page_num: int = 1,
    ) -> tuple[str | None, str]:
        """Legacy remote OpenAI-compatible Vision API call (used only when `OCR_BACKEND='polza_vlm'`)."""
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
                return None, "empty_choices"

            choice = response.choices[0]
            finish_reason = str(choice.finish_reason or "stop")
            raw_content = str(choice.message.content or "").strip()

            if finish_reason == "length":
                return None, "length"

            cleaned = sanitize_ocr_text(raw_content)
            if not cleaned:
                return None, "empty_text"

            return cleaned, finish_reason
        except Exception as exc:
            logger.warning("Remote VLM API error on %s p.%d (%s); switching to RapidOCR", file_name, page_num, exc)
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
            "backend_used": "paddleocr_vl" | "hybrid_vlm" | "local_rapidocr",
            "from_cache": bool,
            "sha256": str,
            "fallback_triggered": bool,
            "finish_reason": str,
        }
        """
        raw_backend = (ocr_backend or settings.OCR_BACKEND or "paddleocr_vl").strip().lower()
        if raw_backend not in {"paddleocr_vl", "hybrid_vlm", "polza_vlm", "local_rapidocr"}:
            raw_backend = "paddleocr_vl"

        # Route legacy "hybrid_vlm" alias to local "paddleocr_vl" unless remote Polza VLM is explicitly configured
        if raw_backend == "hybrid_vlm" and not settings.POLZA_API_KEY:
            backend = "paddleocr_vl"
        else:
            backend = raw_backend

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

        # Primary local VLM mode: PaddlePaddle/PaddleOCR-VL-1.6 (or explicit polza_vlm) -> RapidOCR fallback
        if backend == "polza_vlm":
            vlm_text, finish_reason = self._call_polza_vlm(png_bytes, file_name=file_name, page_num=page_num)
            used_label = "hybrid_vlm"
        else:
            vlm_text, finish_reason = self._call_paddleocr_vl(png_bytes, file_name=file_name, page_num=page_num)
            used_label = "paddleocr_vl"

        if vlm_text is not None and finish_reason != "length":
            self._write_cache(
                sha256_hex=sha256_hex,
                text=vlm_text,
                backend_used=used_label,
                vlm_attempted=True,
                finish_reason=finish_reason,
                fallback_triggered=False,
                file_name=file_name,
                page_num=page_num,
            )
            return {
                "text": vlm_text,
                "backend_used": used_label,
                "from_cache": False,
                "sha256": sha256_hex,
                "fallback_triggered": False,
                "finish_reason": finish_reason,
            }

        # Automatic fallback to local RapidOCR v3 (PP-OCRv5 ESLAV)
        fallback_text = self._call_local_rapidocr(png_bytes, file_name=file_name, page_num=page_num)
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
