"""Unified Document Ingestion Parsers for PDF, DOCX, and XLSX (Module 1 -> ParsedBlock stream)."""

from __future__ import annotations

import concurrent.futures
import io
import logging
import re
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import docx
import openpyxl
import pdfplumber
import pypdfium2 as pdfium
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from PIL import Image

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fitz

from src.config import (
    DETACHED_CLAUSE_NUM_RE,
    MAX_OCR_IMAGE_SIDE,
    OCR_MIN_IMAGE_COVERAGE,
    OCR_RENDER_DPI,
    OCR_SMALL_NATIVE_TEXT_CHARS,
)
from src.ingestion.ocr_engine import HybridOCREngine, get_ocr_engine, sanitize_ocr_text
from src.schemas import BlockType, ParsedBlock, ZoneType

logger = logging.getLogger(__name__)

# WordprocessingML XML namespace
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_W_NS_MAP = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

# Private Use Area bullets from Symbol/Wingdings fonts in PDF/DOCX
_PUA_BULLETS = ("\uf02d", "\uf0b7", "\uf0a7", "\uf076", "\uf0d8", "\uf0fc", "\uf06e", "\uf0a8")

# Requisite 1x2 table pattern ("от 28.03.2023 | № 794-1")
_REQUISITE_CELL_RE = re.compile(
    r"(?:от\s+\d{2}\.\d{2}\.\d{4}|(?:№|N)\s*\d{1,5}(?:-\d{1,3})?)",
    re.IGNORECASE,
)

# Official registration / ЭДО stamp line pattern
_REGISTRATION_STAMP_LINE_RE = re.compile(
    r"^\s*(?:Зарегистрировано\s+(?:в\s+(?:ЭДО|СЭД|реестре|журнале)|от\b)|"
    r"Регистрационный\s*(?:№|N|номер)\b|Рег\.\s*(?:№|N)\s*\d)",
    re.IGNORECASE,
)

# Marker for detecting registration / ЭДО stamps inside embedded raster stamp images on digital PDFs
_STAMP_OCR_MARKER_RE = re.compile(
    r"(?:зарегистрирован[оа]?|регистрационн[а-яё]+\s*(?:№|n|номер)|рег\.\s*(?:№|n)\s*\d|в\s+эдо\b)",
    re.IGNORECASE,
)

# Order / disposition header requisite marker for Word section headers
_DOCX_ORDER_HEADER_RE = re.compile(
    r"(?:ПРИКАЗ|РАСПОРЯЖЕНИЕ)\s+(?:от\s+\d{1,2}[.\s]|(?:№|N)\s*\d)",
    re.IGNORECASE,
)

# Standalone single-level or multi-level clause number on its own line in PDFs
_STANDALONE_CLAUSE_LINE_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})+|\d{1,2}\.)\s*$")


def _normalize_inline_whitespace(text: str) -> str:
    """Normalize non-breaking spaces, PUA bullets, and collapse multiple horizontal spaces."""
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    for b_ch in _PUA_BULLETS:
        s = s.replace(b_ch, "-")
    # Replace leading mathematical minus '−' (U+2212) at start of line with standard hyphen '-'
    s = re.sub(r"(?m)^\s*−\s*", "- ", s)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    return "\n".join(lines).strip()


def _format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render table headers and rows into a clean pipe-separated Markdown table string."""
    if not headers and not rows:
        return ""
    col_count = max(
        len(headers),
        max((len(r) for r in rows), default=0),
    )
    if col_count == 0:
        return ""

    norm_headers = [
        (headers[i].replace("\n", " ").strip() if i < len(headers) else "")
        for i in range(col_count)
    ]
    lines: list[str] = []
    if any(norm_headers):
        lines.append("| " + " | ".join(norm_headers) + " |")
        lines.append("| " + " | ".join(["---"] * col_count) + " |")

    for row in rows:
        norm_row = [
            (str(row[i] if row[i] is not None else "").replace("\n", " ").strip() if i < len(row) else "")
            for i in range(col_count)
        ]
        if any(norm_row):
            lines.append("| " + " | ".join(norm_row) + " |")

    return "\n".join(lines)


def _forward_fill_cfo_hierarchy(
    headers: list[str],
    rows: list[list[str]],
) -> tuple[list[list[str]], list[str]]:
    """
    Forward-fill hierarchical tree columns (e.g. `Финансодержатель -> ЦФО 1 -> ЦФО 2 -> ЦФО 3 -> ЦФО 4`
    or `Направление -> Поднаправление`) down `rows` until a parent branch changes.
    Also removes pure group-header rows where a parent cell was only set on a blank row and is immediately
    inherited by the next child row, OR keeps all rows aligned between DOCX and XLSX.
    """
    if not rows or not headers:
        return rows, []

    header_low = [h.lower().replace("ё", "е").strip() for h in headers]
    is_cfo_table = any("цфо" in h or "финансодержатель" in h for h in header_low)
    is_dir_table = len(header_low) >= 2 and "направление" in header_low[0] and "поднаправление" in header_low[1]

    if not (is_cfo_table or is_dir_table):
        return rows, []

    tree_depth = min(len(headers), 5) if is_cfo_table else 2
    running: list[str] = [""] * tree_depth
    filled_rows: list[list[str]] = []
    top_hierarchy: list[str] = []

    for raw_row in rows:
        row = [
            "" if (c is None or str(c).strip() == "None") else _normalize_inline_whitespace(str(c)).replace("\n", " ")
            for c in raw_row
        ]
        if len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))
        elif len(row) > len(headers):
            row = row[: len(headers)]

        if not any(row):
            continue

        for col_i in range(tree_depth):
            if row[col_i]:
                if row[col_i] != running[col_i]:
                    running[col_i] = row[col_i]
                    # Reset deeper levels if they are not explicitly set on this row
                    for deeper_i in range(col_i + 1, tree_depth):
                        if not row[deeper_i]:
                            running[deeper_i] = ""
            elif running[col_i]:
                # Forward-fill parent level only if at least one deeper level has a value on this row
                # (or for column 0 Финансодержатель, always forward-fill when any deeper column is non-empty)
                if any(row[d] for d in range(col_i + 1, len(row))):
                    row[col_i] = running[col_i]

        if not top_hierarchy and any(running):
            top_hierarchy = [v for v in running if v]

        filled_rows.append(row)

    return filled_rows, top_hierarchy


# ---------------------------------------------------------------------------
# 1. PDF Parser (PyMuPDF / fitz + pypdfium2 + pdfplumber + HybridOCREngine)
# ---------------------------------------------------------------------------
def _compute_pdf_page_image_coverage(fitz_page: Any) -> tuple[int, float]:
    """
    Compute `(image_count, max_image_coverage)` on a PyMuPDF page.
    Checks both `get_images(full=True)` + `get_image_rects` and dict image blocks.
    """
    rect = fitz_page.rect
    page_area = float(rect.width * rect.height)
    if page_area <= 0:
        return 0, 0.0

    images = fitz_page.get_images(full=True)
    img_count = len(images)
    max_cov = 0.0

    for img_info in images:
        xref = img_info[0]
        try:
            for bbox in fitz_page.get_image_rects(xref):
                clipped = bbox & rect
                if not clipped.is_empty:
                    cov = float(clipped.width * clipped.height) / page_area
                    if cov > max_cov:
                        max_cov = min(1.0, cov)
        except Exception as exc:
            logger.debug("Ignoring image rect lookup error for xref %s: %s", xref, exc)

    if max_cov == 0.0:
        try:
            page_dict = fitz_page.get_text("dict")
            for blk in page_dict.get("blocks", []):
                if blk.get("type") == 1 and "bbox" in blk:
                    bx0, by0, bx1, by1 = blk["bbox"]
                    w = max(0.0, min(bx1, rect.x1) - max(bx0, rect.x0))
                    h = max(0.0, min(by1, rect.y1) - max(by0, rect.y0))
                    cov = (w * h) / page_area
                    img_count = max(img_count, 1)
                    if cov > max_cov:
                        max_cov = min(1.0, cov)
        except Exception as exc:
            logger.debug("Ignoring dict block coverage fallback error: %s", exc)

    return img_count, max_cov


def _should_trigger_ocr(native_text_len: int, image_count: int, max_image_coverage: float) -> bool:
    """
    Determine whether a PDF page requires OCR according to MIPT V3 rules:
    - Native text < 600 chars (`OCR_SMALL_NATIVE_TEXT_CHARS`) with raster images present, OR
    - `max_image_coverage >= 0.55` (`OCR_MIN_IMAGE_COVERAGE`) with native text < 600 chars, OR
    - Completely empty native text (`native_text_len == 0`).
    """
    if native_text_len == 0:
        return True
    if native_text_len < OCR_SMALL_NATIVE_TEXT_CHARS and image_count > 0:
        return True
    if max_image_coverage >= OCR_MIN_IMAGE_COVERAGE and native_text_len < OCR_SMALL_NATIVE_TEXT_CHARS:
        return True
    return False


def _render_pdf_page_to_png(
    pdfium_doc: Any,
    fitz_page: Any,
    page_idx: int,
    dpi: int = OCR_RENDER_DPI,
    max_side: int = MAX_OCR_IMAGE_SIDE,
) -> bytes:
    """
    Render a PDF page at 300 DPI via `pypdfium2` (with `fitz` fallback),
    auto-rotate landscape scan pages if needed, and clamp the longest image side
    to `MAX_OCR_IMAGE_SIDE = 3509` px.
    """
    scale = float(dpi) / 72.0
    pil_img: Image.Image | None = None

    if pdfium_doc is not None:
        try:
            pdfium_page = pdfium_doc[page_idx]
            bitmap = pdfium_page.render(scale=scale)
            pil_img = bitmap.to_pil().convert("RGB")
        except Exception as exc:
            logger.debug("pypdfium2 render fallback to fitz on page %d: %s", page_idx + 1, exc)

    if pil_img is None:
        pix = fitz_page.get_pixmap(dpi=dpi, alpha=False)
        pil_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    # Auto-rotate landscape scanned pages by 90 degrees when width > height
    w, h = pil_img.size
    if w > int(h * 1.15):
        pil_img = pil_img.rotate(90, expand=True)
        w, h = pil_img.size

    # Clamp longest side to MAX_OCR_IMAGE_SIDE (3509 px) to protect VLM & RAM on giant pages
    longest = max(w, h)
    if longest > max_side:
        ratio = float(max_side) / float(longest)
        new_w = max(1, int(round(w * ratio)))
        new_h = max(1, int(round(h * ratio)))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _merge_native_stamp_with_ocr(native_text: str, ocr_text: str) -> str:
    """
    Preserve non-duplicated native text (e.g. overlaid ЭДО registration stamps or approval headers)
    alongside OCR text without duplicating lines already recognized by OCR.
    """
    clean_native = _normalize_inline_whitespace(native_text)
    clean_ocr = (ocr_text or "").strip()
    if not clean_native:
        return clean_ocr
    if not clean_ocr:
        return clean_native

    ocr_norm_alnum = re.sub(r"[^а-яa-z0-9]+", "", clean_ocr.lower().replace("ё", "е"))
    missing_top_lines: list[str] = []
    missing_bottom_lines: list[str] = []

    native_lines = [ln.strip() for ln in clean_native.split("\n") if ln.strip()]
    is_top_approval = any(
        kw in clean_native.upper() for kw in ("СОГЛАСОВАНО", "УТВЕРЖДАЮ", "ПРЕДСЕДАТЕЛЬ ПРОФКОМА")
    )

    for ln in native_lines:
        ln_alnum = re.sub(r"[^а-яa-z0-9]+", "", ln.lower().replace("ё", "е"))
        if len(ln_alnum) < 3:
            continue
        if ln_alnum in ocr_norm_alnum:
            continue
        if is_top_approval:
            missing_top_lines.append(ln)
        else:
            missing_bottom_lines.append(ln)

    parts: list[str] = []
    if missing_top_lines:
        parts.append("\n".join(missing_top_lines))
    parts.append(clean_ocr)
    if missing_bottom_lines:
        parts.append("\n".join(missing_bottom_lines))

    return "\n\n".join(p for p in parts if p).strip()


def _glue_detached_clause_numbers_in_lines(lines: list[str]) -> list[str]:
    """
    Glue standalone detached clause numbers (e.g. '1.1.' or '2.3.' on its own line in
    `Регламент поиска и подбора персонала.pdf`) to the following text line.
    Also unwraps single-word-per-line PDF justification artifacts.
    """
    if not lines:
        return []

    # Step 1: Collapse single-word-per-line justification runs (where >= 4 consecutive lines are single words)
    unwrapped: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        # Check if this starts a run of >= 4 single-word non-clause lines
        if " " not in ln and not _STANDALONE_CLAUSE_LINE_RE.match(ln) and not DETACHED_CLAUSE_NUM_RE.match(ln):
            run_end = i
            while (
                run_end < n
                and lines[run_end].strip()
                and " " not in lines[run_end].strip()
                and not _STANDALONE_CLAUSE_LINE_RE.match(lines[run_end].strip())
                and not DETACHED_CLAUSE_NUM_RE.match(lines[run_end].strip())
                and not lines[run_end].strip().isupper()
            ):
                run_end += 1
            if run_end - i >= 4:
                merged_words = " ".join(lines[k].strip() for k in range(i, run_end))
                if unwrapped and not _STANDALONE_CLAUSE_LINE_RE.match(unwrapped[-1]):
                    unwrapped[-1] = f"{unwrapped[-1]} {merged_words}".strip()
                else:
                    unwrapped.append(merged_words)
                i = run_end
                continue
        unwrapped.append(ln)
        i += 1

    # Step 2: Glue standalone clause number lines ('1.1.', '2.3.') to the next line
    glued: list[str] = []
    idx = 0
    while idx < len(unwrapped):
        curr = unwrapped[idx].strip()
        if (
            (DETACHED_CLAUSE_NUM_RE.match(curr) or _STANDALONE_CLAUSE_LINE_RE.match(curr))
            and idx + 1 < len(unwrapped)
            and unwrapped[idx + 1].strip()
            and not (
                DETACHED_CLAUSE_NUM_RE.match(unwrapped[idx + 1].strip())
                or _STANDALONE_CLAUSE_LINE_RE.match(unwrapped[idx + 1].strip())
            )
        ):
            num_tok = curr if curr.endswith(".") else f"{curr}."
            glued.append(f"{num_tok} {unwrapped[idx + 1].strip()}")
            idx += 2
            continue
        glued.append(curr)
        idx += 1

    return glued


def _text_to_parsed_blocks(
    page_text: str,
    page_num: int,
    is_ocr: bool,
    prev_blocks: list[ParsedBlock] | None = None,
) -> list[ParsedBlock]:
    """
    Convert a page's cleaned text (from OCR or digital layer) into a list of `ParsedBlock`s,
    splitting out embedded Markdown tables (`| col1 | col2 |`), merging cross-page table
    continuation rows into the preceding page's table, and gluing detached clause numbers.
    """
    norm_text = _normalize_inline_whitespace(page_text)
    if not norm_text:
        return []

    raw_lines = [ln.strip() for ln in norm_text.split("\n") if ln.strip()]
    # Strip trailing standalone page footer number (e.g. '3', '4', '6' at bottom of page)
    if len(raw_lines) > 1 and re.match(r"^\d{1,3}$", raw_lines[-1]):
        raw_lines.pop()

    glued_lines = _glue_detached_clause_numbers_in_lines(raw_lines)
    if not glued_lines:
        return []

    blocks: list[ParsedBlock] = []
    para_buf: list[str] = []
    md_table_buf: list[str] = []

    def _flush_para() -> None:
        if not para_buf:
            return
        txt = "\n".join(para_buf).strip()
        if txt and not re.match(r"^\d{1,3}$", txt):
            blocks.append(
                ParsedBlock(
                    text=txt,
                    block_type=BlockType.PARAGRAPH.value,
                    zone=ZoneType.NORMATIVE.value,
                    page_num=page_num,
                    page_numbers=[page_num],
                    is_ocr=is_ocr,
                )
            )
        para_buf.clear()

    def _flush_md_table() -> None:
        if not md_table_buf:
            return
        parsed_rows: list[list[str]] = []
        for ln in md_table_buf:
            inner = ln.strip().strip("|")
            cells = [c.strip() for c in inner.split("|")]
            if all(re.match(r"^[-:\s]*$", c) for c in cells if c):
                continue
            if any(cells):
                parsed_rows.append(cells)

        # Check if this table fragment at the start of the page is a continuation of the previous page's table
        if (
            not blocks
            and prev_blocks
            and prev_blocks[-1].block_type == BlockType.TABLE.value
            and prev_blocks[-1].table_headers
            and parsed_rows
            and len(parsed_rows[0]) == len(prev_blocks[-1].table_headers)
            and parsed_rows[0] != prev_blocks[-1].table_headers
        ):
            prev_tbl = prev_blocks[-1]
            for r_idx, cont_row in enumerate(parsed_rows):
                if len(cont_row) != len(prev_tbl.table_headers):
                    continue
                if r_idx == 0 and not cont_row[0].strip() and prev_tbl.table_rows:
                    last_row = prev_tbl.table_rows[-1]
                    for c_i, val in enumerate(cont_row):
                        if val:
                            last_row[c_i] = f"{last_row[c_i]} {val}".strip() if last_row[c_i] else val
                else:
                    prev_tbl.table_rows.append(cont_row)
            prev_tbl.text = _format_markdown_table(prev_tbl.table_headers, prev_tbl.table_rows)
            if page_num not in prev_tbl.page_numbers:
                prev_tbl.page_numbers.append(page_num)
            md_table_buf.clear()
            return

        if len(parsed_rows) >= 2:
            headers = parsed_rows[0]
            data_rows = parsed_rows[1:]
            filled_rows, top_hier = _forward_fill_cfo_hierarchy(headers, data_rows)
            md_str = _format_markdown_table(headers, filled_rows)
            blocks.append(
                ParsedBlock(
                    text=md_str,
                    block_type=BlockType.TABLE.value,
                    zone=ZoneType.TABLE.value,
                    page_num=page_num,
                    page_numbers=[page_num],
                    is_ocr=is_ocr,
                    table_headers=headers,
                    table_rows=filled_rows,
                    parent_hierarchy=top_hier,
                )
            )
        else:
            para_buf.extend(md_table_buf)
            _flush_para()
        md_table_buf.clear()

    for ln in glued_lines:
        if ln.startswith("|") and ln.endswith("|") and ln.count("|") >= 3:
            _flush_para()
            md_table_buf.append(ln)
        elif _REGISTRATION_STAMP_LINE_RE.match(ln):
            if md_table_buf:
                _flush_md_table()
            _flush_para()
            blocks.append(
                ParsedBlock(
                    text=ln,
                    block_type=BlockType.STAMP.value,
                    zone=ZoneType.NORMATIVE.value,
                    page_num=page_num,
                    page_numbers=[page_num],
                    is_ocr=is_ocr,
                )
            )
        else:
            if md_table_buf:
                _flush_md_table()
            para_buf.append(ln)

    _flush_md_table()
    _flush_para()
    return blocks


def _extract_valid_pdfplumber_tables(
    plumber_page: Any,
    fitz_page: Any,
) -> list[dict[str, Any]]:
    """
    Extract genuine multi-column tables from a digital PDF page via `pdfplumber`,
    filtering out false-positive indented paragraph blocks (such as in `Регламент о командировании.pdf`).
    Returns a list of dicts: `{"bbox": (x0, top, x1, bottom), "headers": list[str], "rows": list[list[str]]}`.
    """
    if plumber_page is None:
        return []

    try:
        found_tables = plumber_page.find_tables()
    except Exception as exc:
        logger.debug("pdfplumber find_tables error: %s", exc)
        return []

    if not found_tables:
        return []

    # Check if PyMuPDF also sees tables on this page
    fitz_has_tables = False
    try:
        ft = fitz_page.find_tables()
        fitz_has_tables = bool(ft and ft.tables)
    except Exception as exc:
        logger.debug("PyMuPDF find_tables fallback check failed: %s", exc)

    valid_tables: list[dict[str, Any]] = []
    for tbl_obj in found_tables:
        try:
            raw_grid = tbl_obj.extract()
        except Exception:
            continue
        if not raw_grid or len(raw_grid) < 2:
            continue

        cleaned_grid: list[list[str]] = []
        rows_with_multi_cols = 0
        max_non_empty_cols = 0

        for raw_row in raw_grid:
            if not raw_row:
                continue
            row_cells = [
                _normalize_inline_whitespace(str(c or "")).replace("\n", " ")
                for c in raw_row
            ]
            non_empty = sum(1 for c in row_cells if c)
            if non_empty == 0:
                continue
            if non_empty >= 2:
                rows_with_multi_cols += 1
            if non_empty > max_non_empty_cols:
                max_non_empty_cols = non_empty
            cleaned_grid.append(row_cells)

        if len(cleaned_grid) < 2 or max_non_empty_cols < 2:
            continue
        if not fitz_has_tables and rows_with_multi_cols < 2:
            continue

        valid_tables.append(
            {
                "bbox": tuple(float(v) for v in tbl_obj.bbox),
                "headers": cleaned_grid[0],
                "rows": cleaned_grid[1:],
            }
        )

    return valid_tables


def _bbox_overlaps_any_table(
    blk_bbox: tuple[float, float, float, float],
    table_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    """Return True if a PyMuPDF text block's center or >50% area falls inside any extracted table bbox."""
    bx0, by0, bx1, by1 = blk_bbox
    cx = (bx0 + bx1) / 2.0
    cy = (by0 + by1) / 2.0
    b_area = max(1.0, (bx1 - bx0) * (by1 - by0))

    for tx0, ty0, tx1, ty1 in table_bboxes:
        if (tx0 - 4.0) <= cx <= (tx1 + 4.0) and (ty0 - 4.0) <= cy <= (ty1 + 4.0):
            return True
        ix0 = max(bx0, tx0)
        iy0 = max(by0, ty0)
        ix1 = min(bx1, tx1)
        iy1 = min(by1, ty1)
        if ix1 > ix0 and iy1 > iy0:
            inter_area = (ix1 - ix0) * (iy1 - iy0)
            if (inter_area / b_area) >= 0.45:
                return True
    return False


def _extract_digital_page_raster_stamps(
    fitz_page: Any,
    engine: HybridOCREngine,
    page_num: int,
    file_name: str,
    native_text: str,
) -> list[tuple[float, ParsedBlock]]:
    """
    Inspect small/medium embedded raster banner images on a digital PDF page (e.g. overlaid ЭДО registration stamps)
    using cached local RapidOCR, returning `(y0, ParsedBlock(block_type='stamp'))` for genuine registration stamps
    not already present in the native text layer.
    """
    rect = fitz_page.rect
    page_area = float(rect.width * rect.height)
    if page_area <= 0:
        return []

    images = fitz_page.get_images(full=True)
    if not images:
        return []

    native_alnum = re.sub(r"[^а-яa-z0-9]+", "", (native_text or "").lower().replace("ё", "е"))
    candidate_rects: list[Any] = []
    seen_bbox_keys: set[tuple[int, int, int, int]] = set()

    for img_info in images:
        xref = img_info[0]
        try:
            for bbox in fitz_page.get_image_rects(xref):
                clipped = bbox & rect
                if clipped.is_empty:
                    continue
                cov = float(clipped.width * clipped.height) / page_area
                if (
                    clipped.width < 80
                    or clipped.height < 18
                    or clipped.width < 1.5 * clipped.height
                    or cov >= OCR_MIN_IMAGE_COVERAGE
                ):
                    continue
                bkey = (int(round(clipped.x0)), int(round(clipped.y0)), int(round(clipped.x1)), int(round(clipped.y1)))
                if bkey not in seen_bbox_keys:
                    seen_bbox_keys.add(bkey)
                    candidate_rects.append(clipped)
        except Exception as exc:
            logger.debug("Skipping stamp image rect lookup on %s p.%d: %s", file_name, page_num, exc)

    if not candidate_rects or len(candidate_rects) > 6:
        return []

    candidate_rects.sort(key=lambda r: (float(r.y0), float(r.x0)))
    stamp_blocks: list[tuple[float, ParsedBlock]] = []

    for clipped in candidate_rects[:4]:
        try:
            pix = fitz_page.get_pixmap(clip=clipped, dpi=300, alpha=False)
            png_bytes = pix.tobytes("png")
            ocr_res = engine.recognize_page(
                png_bytes=png_bytes,
                ocr_backend="local_rapidocr",
                page_num=page_num,
                file_name=f"{file_name}#stamp",
            )
            stamp_txt = _normalize_inline_whitespace(str(ocr_res.get("text") or ""))
            if not stamp_txt or not _STAMP_OCR_MARKER_RE.search(stamp_txt):
                continue
            stamp_flat = re.sub(r"\s+", " ", stamp_txt).strip()
            stamp_alnum = re.sub(r"[^а-яa-z0-9]+", "", stamp_flat.lower().replace("ё", "е"))
            if len(stamp_alnum) < 6 or (stamp_alnum in native_alnum):
                continue
            native_alnum += stamp_alnum
            stamp_blocks.append(
                (
                    float(clipped.y0),
                    ParsedBlock(
                        text=stamp_flat,
                        block_type=BlockType.STAMP.value,
                        zone=ZoneType.NORMATIVE.value,
                        page_num=page_num,
                        page_numbers=[page_num],
                        is_ocr=True,
                    ),
                )
            )
        except Exception as exc:
            logger.debug("Skipping raster stamp OCR on %s p.%d: %s", file_name, page_num, exc)

    return stamp_blocks


def parse_pdf(
    file_path: str | Path,
    ocr_backend: str | None = None,
    ocr_engine: HybridOCREngine | None = None,
) -> tuple[list[ParsedBlock], dict[str, Any]]:
    """
    Parse a digital or scanned PDF into `ParsedBlock`s with per-page fault isolation:
    - Triggers OCR when native text < 600 chars with raster images OR image_coverage >= 0.55 with text < 600
    - Renders OCR pages at 300 DPI via `pypdfium2`, auto-rotates landscape scans, clamps longest side <= 3509 px
    - Preserves non-duplicated overlaid native ЭДО/approval stamps alongside OCR text
    - Glues detached standalone clause numbers ('1.1.', '2.3.') to the following line
    - Extracts genuine digital tables via `pdfplumber` without duplicating native table text
    """
    path = Path(file_path)
    engine = ocr_engine or get_ocr_engine()
    blocks: list[ParsedBlock] = []
    warnings_list: list[str] = []
    errors_list: list[str] = []
    ocr_pages_count = 0
    pages_total = 0

    try:
        fitz_doc = fitz.open(str(path))
    except Exception as exc:
        msg = f"Failed to open PDF {path.name}: {exc}"
        logger.error(msg)
        return [], {
            "file_format": "pdf",
            "pages_total": 0,
            "ocr_pages": 0,
            "warnings": [],
            "errors": [msg],
        }

    pdfium_doc: Any = None
    try:
        pdfium_doc = pdfium.PdfDocument(str(path))
    except Exception as exc:
        logger.debug("pypdfium2 open skipped for %s: %s", path.name, exc)

    plumber_doc: Any = None
    try:
        plumber_doc = pdfplumber.open(str(path))
    except Exception as exc:
        logger.debug("pdfplumber open skipped for %s: %s", path.name, exc)

    try:
        pages_total = len(fitz_doc)

        # Pre-check and render OCR pages so multi-page scanned PDFs can run VLM OCR concurrently
        ocr_render_map: dict[int, tuple[bytes, str]] = {}
        for page_idx in range(pages_total):
            try:
                fitz_page = fitz_doc[page_idx]
                raw_native_text = fitz_page.get_text("text") or ""
                norm_native_text = _normalize_inline_whitespace(raw_native_text)
                native_len = len(norm_native_text)
                img_count, max_cov = _compute_pdf_page_image_coverage(fitz_page)
                if _should_trigger_ocr(native_len, img_count, max_cov):
                    png_bytes = _render_pdf_page_to_png(
                        pdfium_doc=pdfium_doc,
                        fitz_page=fitz_page,
                        page_idx=page_idx,
                        dpi=OCR_RENDER_DPI,
                        max_side=MAX_OCR_IMAGE_SIDE,
                    )
                    ocr_render_map[page_idx] = (png_bytes, norm_native_text)
            except Exception as prep_exc:
                err_msg = f"Ошибка подготовки стр. {page_idx + 1} в {path.name}: {prep_exc}"
                logger.warning(err_msg)
                warnings_list.append(err_msg)
                errors_list.append(err_msg)

        ocr_results_map: dict[int, dict[str, Any]] = {}
        if len(ocr_render_map) == 1:
            p_idx, (png_b, _) = next(iter(ocr_render_map.items()))
            ocr_results_map[p_idx] = engine.recognize_page(
                png_bytes=png_b,
                ocr_backend=ocr_backend,
                page_num=p_idx + 1,
                file_name=path.name,
            )
        elif len(ocr_render_map) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                future_to_idx = {
                    pool.submit(
                        engine.recognize_page,
                        png_b,
                        ocr_backend,
                        p_idx + 1,
                        path.name,
                    ): p_idx
                    for p_idx, (png_b, _) in ocr_render_map.items()
                }
                for fut, p_idx in future_to_idx.items():
                    try:
                        ocr_results_map[p_idx] = fut.result()
                    except Exception as ocr_exc:
                        err_msg = f"Ошибка OCR стр. {p_idx + 1} в {path.name}: {ocr_exc}"
                        logger.warning(err_msg)
                        warnings_list.append(err_msg)
                        errors_list.append(err_msg)

        for page_idx in range(pages_total):
            page_num = page_idx + 1
            try:
                if page_idx in ocr_render_map:
                    ocr_pages_count += 1
                    _, norm_native_text = ocr_render_map[page_idx]
                    ocr_res = ocr_results_map.get(page_idx, {})
                    ocr_text = str(ocr_res.get("text") or "")
                    if ocr_res.get("fallback_triggered"):
                        warnings_list.append(
                            f"Стр. {page_num}: VLM переключен на локальный фоллбэк RapidOCR (причина: {ocr_res.get('finish_reason')})"
                        )
                    combined_text = _merge_native_stamp_with_ocr(norm_native_text, ocr_text)
                    page_blocks = _text_to_parsed_blocks(
                        combined_text, page_num=page_num, is_ocr=True, prev_blocks=blocks
                    )
                    blocks.extend(page_blocks)
                    continue

                fitz_page = fitz_doc[page_idx]
                raw_native_text = fitz_page.get_text("text") or ""
                norm_native_text = _normalize_inline_whitespace(raw_native_text)

                raster_stamps: list[tuple[float, ParsedBlock]] = []
                if page_idx == 0 or page_idx == pages_total - 1:
                    raster_stamps = _extract_digital_page_raster_stamps(
                        fitz_page=fitz_page,
                        engine=engine,
                        page_num=page_num,
                        file_name=path.name,
                        native_text=norm_native_text,
                    )

                # Digital page processing: check for genuine multi-column tables via pdfplumber
                plumber_page = (
                    plumber_doc.pages[page_idx]
                    if (plumber_doc is not None and page_idx < len(plumber_doc.pages))
                    else None
                )
                digital_tables = _extract_valid_pdfplumber_tables(plumber_page, fitz_page)

                if not digital_tables:
                    top_stamps = [sb for sy0, sb in raster_stamps if sy0 < float(fitz_page.rect.height) * 0.5]
                    bot_stamps = [sb for sy0, sb in raster_stamps if sy0 >= float(fitz_page.rect.height) * 0.5]
                    blocks.extend(top_stamps)
                    page_blocks = _text_to_parsed_blocks(
                        norm_native_text, page_num=page_num, is_ocr=False, prev_blocks=blocks
                    )
                    blocks.extend(page_blocks)
                    blocks.extend(bot_stamps)
                    continue

                # Interleave non-overlapping PyMuPDF text blocks and extracted digital tables by vertical y0 order
                table_bboxes = [t["bbox"] for t in digital_tables]
                ordered_items: list[tuple[float, str, Any]] = []

                for sy0, sb in raster_stamps:
                    ordered_items.append((sy0, "stamp", sb))

                for tbl in digital_tables:
                    ordered_items.append((tbl["bbox"][1], "table", tbl))

                raw_blocks = fitz_page.get_text("blocks")
                for rb in raw_blocks:
                    if len(rb) >= 5:
                        bx0, by0, bx1, by1, btext = float(rb[0]), float(rb[1]), float(rb[2]), float(rb[3]), str(rb[4])
                        btype = int(rb[6]) if len(rb) >= 7 else 0
                        if btype != 0:
                            continue
                        if _bbox_overlaps_any_table((bx0, by0, bx1, by1), table_bboxes):
                            continue
                        clean_bt = _normalize_inline_whitespace(btext)
                        if clean_bt:
                            ordered_items.append((by0, "text", clean_bt))

                ordered_items.sort(key=lambda item: item[0])
                text_acc: list[str] = []

                def _flush_text_acc() -> None:
                    if not text_acc:
                        return
                    joined = "\n".join(text_acc)
                    blocks.extend(_text_to_parsed_blocks(joined, page_num=page_num, is_ocr=False))
                    text_acc.clear()

                for _, kind, payload in ordered_items:
                    if kind == "text":
                        text_acc.append(payload)
                    elif kind == "stamp":
                        _flush_text_acc()
                        blocks.append(payload)
                    else:
                        _flush_text_acc()
                        headers = payload["headers"]
                        rows = payload["rows"]
                        filled_rows, top_hier = _forward_fill_cfo_hierarchy(headers, rows)
                        md_str = _format_markdown_table(headers, filled_rows)
                        blocks.append(
                            ParsedBlock(
                                text=md_str,
                                block_type=BlockType.TABLE.value,
                                zone=ZoneType.TABLE.value,
                                page_num=page_num,
                                page_numbers=[page_num],
                                is_ocr=False,
                                table_headers=headers,
                                table_rows=filled_rows,
                                parent_hierarchy=top_hier,
                            )
                        )

                _flush_text_acc()

            except Exception as page_exc:
                err_msg = f"Ошибка разбора стр. {page_num} в {path.name}: {page_exc}"
                logger.warning(err_msg)
                warnings_list.append(err_msg)
                errors_list.append(err_msg)
    finally:
        try:
            fitz_doc.close()
        except Exception as exc:
            logger.debug("Error closing fitz_doc: %s", exc)
        if pdfium_doc is not None:
            try:
                pdfium_doc.close()
            except Exception as exc:
                logger.debug("Error closing pdfium_doc: %s", exc)
        if plumber_doc is not None:
            try:
                plumber_doc.close()
            except Exception as exc:
                logger.debug("Error closing plumber_doc: %s", exc)

    return blocks, {
        "file_format": "pdf",
        "pages_total": pages_total,
        "ocr_pages": ocr_pages_count,
        "warnings": warnings_list,
        "errors": errors_list,
    }


# ---------------------------------------------------------------------------
# 2. DOCX Parser (python-docx + Word XML w:numPr & word/numbering.xml + vMerge)
# ---------------------------------------------------------------------------
class WordNumberingResolver:
    """
    Resolves automatic Word XML list numbering (`w:numPr` + `word/numbering.xml` + `word/styles.xml`)
    into explicit clause prefixes (`1.`, `1.1.`, `а)`, `-`) so uploaded `.docx` documents
    using automatic Word numbering never lose their clause numbers.
    """

    def __init__(self, docx_path: Path) -> None:
        self.abstract_nums: dict[str, dict[int, dict[str, Any]]] = {}
        self.num_to_abstract: dict[str, str] = {}
        self.num_overrides: dict[tuple[str, int], int] = {}
        self.style_num_pr: dict[str, tuple[str | None, int | None]] = {}
        self.style_based_on: dict[str, str] = {}
        self.counters: dict[tuple[str, int], int] = {}
        self._load_from_zip(docx_path)

    def _load_from_zip(self, docx_path: Path) -> None:
        try:
            with zipfile.ZipFile(docx_path, "r") as zf:
                namelist = set(zf.namelist())
                if "word/styles.xml" in namelist:
                    self._parse_styles_xml(zf.read("word/styles.xml"))
                if "word/numbering.xml" in namelist:
                    self._parse_numbering_xml(zf.read("word/numbering.xml"))
        except Exception as exc:
            logger.debug("WordNumberingResolver XML parse skipped for %s: %s", docx_path.name, exc)

    def _parse_styles_xml(self, xml_bytes: bytes) -> None:
        root = ET.fromstring(xml_bytes)
        for style_el in root.findall("w:style", _W_NS_MAP):
            style_id = style_el.attrib.get(f"{_W_NS}styleId")
            if not style_id:
                continue
            based_el = style_el.find("w:basedOn", _W_NS_MAP)
            if based_el is not None:
                b_val = based_el.attrib.get(f"{_W_NS}val")
                if b_val:
                    self.style_based_on[style_id] = b_val
            num_pr = style_el.find(".//w:pPr/w:numPr", _W_NS_MAP)
            if num_pr is not None:
                num_id_el = num_pr.find("w:numId", _W_NS_MAP)
                ilvl_el = num_pr.find("w:ilvl", _W_NS_MAP)
                num_id = num_id_el.attrib.get(f"{_W_NS}val") if num_id_el is not None else None
                ilvl = int(ilvl_el.attrib.get(f"{_W_NS}val", "0")) if ilvl_el is not None else None
                self.style_num_pr[style_id] = (num_id, ilvl)

    def _parse_numbering_xml(self, xml_bytes: bytes) -> None:
        root = ET.fromstring(xml_bytes)
        for abs_el in root.findall("w:abstractNum", _W_NS_MAP):
            abs_id = abs_el.attrib.get(f"{_W_NS}abstractNumId")
            if abs_id is None:
                continue
            levels: dict[int, dict[str, Any]] = {}
            for lvl_el in abs_el.findall("w:lvl", _W_NS_MAP):
                ilvl_str = lvl_el.attrib.get(f"{_W_NS}ilvl", "0")
                if not ilvl_str.isdigit():
                    continue
                ilvl = int(ilvl_str)
                start_el = lvl_el.find("w:start", _W_NS_MAP)
                fmt_el = lvl_el.find("w:numFmt", _W_NS_MAP)
                txt_el = lvl_el.find("w:lvlText", _W_NS_MAP)

                start_val = int(start_el.attrib.get(f"{_W_NS}val", "1")) if start_el is not None else 1
                num_fmt = fmt_el.attrib.get(f"{_W_NS}val", "decimal") if fmt_el is not None else "decimal"
                lvl_text = txt_el.attrib.get(f"{_W_NS}val", f"%{ilvl + 1}.") if txt_el is not None else f"%{ilvl + 1}."
                levels[ilvl] = {
                    "start": start_val,
                    "numFmt": num_fmt,
                    "lvlText": lvl_text,
                }
            self.abstract_nums[abs_id] = levels

        for num_el in root.findall("w:num", _W_NS_MAP):
            num_id = num_el.attrib.get(f"{_W_NS}numId")
            abs_ref = num_el.find("w:abstractNumId", _W_NS_MAP)
            if num_id and abs_ref is not None:
                abs_id = abs_ref.attrib.get(f"{_W_NS}val")
                if abs_id is not None:
                    self.num_to_abstract[num_id] = abs_id
            if num_id:
                for ov_el in num_el.findall("w:lvlOverride", _W_NS_MAP):
                    ilvl_str = ov_el.attrib.get(f"{_W_NS}ilvl", "0")
                    start_ov = ov_el.find("w:startOverride", _W_NS_MAP)
                    if ilvl_str.isdigit() and start_ov is not None:
                        s_val = start_ov.attrib.get(f"{_W_NS}val")
                        if s_val and s_val.isdigit():
                            self.num_overrides[(num_id, int(ilvl_str))] = int(s_val)

    def _resolve_style_num_pr(self, style_id: str | None) -> tuple[str | None, int | None]:
        visited: set[str] = set()
        curr = style_id
        while curr and curr not in visited:
            visited.add(curr)
            if curr in self.style_num_pr:
                return self.style_num_pr[curr]
            curr = self.style_based_on.get(curr)
        return None, None

    @staticmethod
    def _format_counter(val: int, num_fmt: str) -> str:
        if num_fmt == "decimalZero":
            return f"{val:02d}"
        if num_fmt in {"lowerLetter", "russianLower"}:
            alphabet = "абвгдежзийклмнопрстуфхцчшщэюя"
            return alphabet[(val - 1) % len(alphabet)]
        if num_fmt in {"upperLetter", "russianUpper"}:
            alphabet = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ"
            return alphabet[(val - 1) % len(alphabet)]
        if num_fmt in {"lowerRoman", "upperRoman"}:
            romans = [
                (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
            ]
            res = ""
            n = max(1, val)
            for r_val, r_sym in romans:
                while n >= r_val:
                    res += r_sym
                    n -= r_val
            return res.lower() if num_fmt == "lowerRoman" else res
        return str(val)

    def resolve_paragraph_prefix(self, p_element: CT_P) -> str | None:
        """Return resolved numbering prefix (e.g. '1.', '1.1.', '-') for a `<w:p>` element, or None."""
        if not self.num_to_abstract:
            return None

        p_pr = p_element.find("w:pPr", _W_NS_MAP)
        if p_pr is None:
            return None

        num_id: str | None = None
        ilvl: int | None = None

        num_pr = p_pr.find("w:numPr", _W_NS_MAP)
        if num_pr is not None:
            num_id_el = num_pr.find("w:numId", _W_NS_MAP)
            ilvl_el = num_pr.find("w:ilvl", _W_NS_MAP)
            if num_id_el is not None:
                num_id = num_id_el.attrib.get(f"{_W_NS}val")
            if ilvl_el is not None:
                raw_ilvl = ilvl_el.attrib.get(f"{_W_NS}val", "0")
                if raw_ilvl.isdigit():
                    ilvl = int(raw_ilvl)

        if num_id is None:
            p_style = p_pr.find("w:pStyle", _W_NS_MAP)
            style_id = p_style.attrib.get(f"{_W_NS}val") if p_style is not None else None
            s_num_id, s_ilvl = self._resolve_style_num_pr(style_id)
            num_id = s_num_id
            if ilvl is None:
                ilvl = s_ilvl

        if not num_id or num_id == "0":
            return None

        abs_id = self.num_to_abstract.get(num_id)
        if not abs_id or abs_id not in self.abstract_nums:
            return None

        lvl = ilvl if ilvl is not None else 0
        levels_map = self.abstract_nums[abs_id]
        lvl_info = levels_map.get(lvl, {"start": 1, "numFmt": "decimal", "lvlText": f"%{lvl + 1}."})
        num_fmt = str(lvl_info.get("numFmt") or "decimal")

        if num_fmt == "none":
            return None
        if num_fmt == "bullet":
            return "-"

        # Increment counter for (abs_id, lvl) and reset deeper levels
        key = (abs_id, lvl)
        if key not in self.counters:
            start_val = self.num_overrides.get((num_id, lvl), int(lvl_info.get("start", 1)))
            self.counters[key] = start_val
        else:
            self.counters[key] += 1

        for deeper_lvl in range(lvl + 1, 10):
            self.counters.pop((abs_id, deeper_lvl), None)

        lvl_text = str(lvl_info.get("lvlText") or f"%{lvl + 1}.")
        rendered = lvl_text
        for k in range(lvl + 1):
            k_info = levels_map.get(k, {"start": 1, "numFmt": "decimal"})
            k_val = self.counters.get((abs_id, k), int(k_info.get("start", 1)))
            k_str = self._format_counter(k_val, str(k_info.get("numFmt") or "decimal"))
            rendered = rendered.replace(f"%{k + 1}", k_str)

        rendered = rendered.strip()
        if not rendered:
            return None
        if re.match(r"^\d+(?:\.\d+)*$", rendered):
            rendered += "."
        return rendered


def _detect_paragraph_bold_and_heading(para: DocxParagraph) -> tuple[bool, str, int | None]:
    """Extract `(is_bold, style_name, heading_level)` from a `python-docx` Paragraph."""
    style_name = str(para.style.name if (para.style and para.style.name) else "Normal")
    heading_level: int | None = None

    m_head = re.search(r"(?:Heading|Заголовок)\s*(\d+)", style_name, re.IGNORECASE)
    if m_head:
        heading_level = int(m_head.group(1))

    non_empty_runs = [r for r in para.runs if r.text and r.text.strip()]
    is_bold = False
    if non_empty_runs:
        bold_chars = sum(len(r.text.strip()) for r in non_empty_runs if r.bold)
        total_chars = sum(len(r.text.strip()) for r in non_empty_runs)
        if total_chars > 0 and (bold_chars / total_chars) >= 0.65:
            is_bold = True
    if not is_bold and para.style and para.style.font and para.style.font.bold:
        is_bold = True
    if heading_level is not None:
        is_bold = True

    return is_bold, style_name, heading_level


def _extract_docx_table_grid(table: DocxTable) -> list[list[str]]:
    """
    Extract a 2D grid of cell strings from a `python-docx` Table:
    - Deduplicates horizontally merged cells within the same row by `id(cell._tc)` (setting continuation slots to `""`)
    - Preserves vertical `vMerge` across rows (repeating the merged cell's text down its column)
    - Trims trailing completely empty columns and completely empty rows
    """
    raw_rows: list[list[str]] = []
    max_cols = 0

    for row in table.rows:
        seen_tc_in_row: set[int] = set()
        row_vals: list[str] = []
        for cell in row.cells:
            tc_id = id(cell._tc)
            if tc_id in seen_tc_in_row:
                row_vals.append("")
            else:
                seen_tc_in_row.add(tc_id)
                cell_txt = _normalize_inline_whitespace(cell.text).replace("\n", " ")
                row_vals.append(cell_txt)
        if any(row_vals):
            max_cols = max(max_cols, len(row_vals))
            raw_rows.append(row_vals)

    if not raw_rows or max_cols == 0:
        return []

    # Normalize row lengths and trim trailing empty columns
    for r in raw_rows:
        if len(r) < max_cols:
            r.extend([""] * (max_cols - len(r)))

    non_empty_col_indices = [
        c_idx for c_idx in range(max_cols) if any(r[c_idx] for r in raw_rows)
    ]
    if not non_empty_col_indices:
        return []

    last_col = non_empty_col_indices[-1] + 1
    trimmed_rows = [r[:last_col] for r in raw_rows if any(r[:last_col])]
    return trimmed_rows


def _split_pre_header_banner_rows(grid: list[list[str]]) -> tuple[list[str], list[str], list[list[str]]]:
    """
    If the top row(s) of a table are single-cell banner titles (e.g. 'Финансовая структура в целях бюджетирования...'
    or 'Ед. изм.: руб.' before the real multi-column header row), extract them as `banner_paragraphs`
    so the actual column header row becomes `table_headers`.
    Returns `(banner_paragraphs, headers, data_rows)`.
    """
    if not grid:
        return [], [], []

    banners: list[str] = []
    idx = 0
    while idx + 1 < len(grid):
        row = grid[idx]
        unique_non_empty = list(dict.fromkeys(c for c in row if c))
        next_row = grid[idx + 1]
        next_unique = list(dict.fromkeys(c for c in next_row if c))

        if len(unique_non_empty) == 1 and len(next_unique) >= 2:
            banners.append(unique_non_empty[0])
            idx += 1
            continue
        break

    remaining = grid[idx:]
    if not remaining:
        return banners, [], []
    if len(remaining) == 1:
        return banners, remaining[0], []
    return banners, remaining[0], remaining[1:]


def parse_docx(file_path: str | Path) -> tuple[list[ParsedBlock], dict[str, Any]]:
    """
    Parse a `.docx` document into `ParsedBlock`s:
    - Iterates `doc.element.body.iterchildren()` to preserve exact paragraph (`CT_P`) and table (`CT_Tbl`) order
    - Resolves both explicit text numbering and automatic Word XML numbering (`w:numPr` + `word/numbering.xml`)
    - Deduplicates horizontally merged cells by `id(cell._tc)` while preserving vertical `vMerge`
    - Unwraps 1-row layout/stamp/signature tables and forwards-fills hierarchical ЦФО tables
    """
    path = Path(file_path)
    blocks: list[ParsedBlock] = []
    warnings_list: list[str] = []
    errors_list: list[str] = []

    try:
        doc = docx.Document(str(path))
    except Exception as exc:
        msg = f"Failed to open DOCX {path.name}: {exc}"
        logger.error(msg)
        return [], {
            "file_format": "docx",
            "pages_total": 0,
            "ocr_pages": 0,
            "warnings": [],
            "errors": [msg],
        }

    num_resolver = WordNumberingResolver(path)
    page_num = 1

    for child in doc.element.body.iterchildren():
        try:
            # Track explicit Word page breaks (`w:br w:type="page"`) if present
            if child.findall(".//w:br[@w:type='page']", _W_NS_MAP):
                page_num += 1

            if isinstance(child, CT_P):
                para = DocxParagraph(child, doc)
                text = _normalize_inline_whitespace(para.text)
                xml_prefix = num_resolver.resolve_paragraph_prefix(child)

                if not text:
                    continue

                if xml_prefix:
                    # Prepend resolved Word XML numbering if not already written explicitly in the text
                    if not text.lstrip().startswith(xml_prefix):
                        text = f"{xml_prefix} {text.lstrip()}"

                is_bold, style_name, heading_level = _detect_paragraph_bold_and_heading(para)
                btype = BlockType.HEADING.value if heading_level is not None else BlockType.PARAGRAPH.value

                blocks.append(
                    ParsedBlock(
                        text=text,
                        block_type=btype,
                        zone=ZoneType.NORMATIVE.value,
                        page_num=page_num,
                        page_numbers=[page_num],
                        is_bold=is_bold,
                        style_name=style_name,
                        heading_level=heading_level,
                        is_ocr=False,
                        xml_num_prefix=xml_prefix,
                    )
                )

            elif isinstance(child, CT_Tbl):
                table = DocxTable(child, doc)
                grid = _extract_docx_table_grid(table)
                if not grid:
                    continue

                # Handle 1-row layout tables (requisite frames, single-cell appendix stamps, 2-cell sign-offs)
                if len(grid) == 1:
                    row0 = grid[0]
                    non_empty_cells = [c for c in row0 if c]
                    if not non_empty_cells:
                        continue
                    joined_pipe = " | ".join(non_empty_cells)

                    # Requisite header table ("от 28.03.2023 | № 794-1"): emit as 1-row table for requisite extractor
                    if len(non_empty_cells) <= 2 and _REQUISITE_CELL_RE.search(joined_pipe):
                        blocks.append(
                            ParsedBlock(
                                text=joined_pipe,
                                block_type=BlockType.TABLE.value,
                                zone=ZoneType.NORMATIVE.value,
                                page_num=page_num,
                                page_numbers=[page_num],
                                table_headers=row0,
                                table_rows=[],
                            )
                        )
                        continue

                    # Single-cell stamp (e.g. "Приложение 1 к Регламенту..." in 3065-1) or 2-cell signature ("Ректор | Д.В. Орлов")
                    if len(non_empty_cells) <= 2:
                        para_text = " ".join(non_empty_cells)
                        blocks.append(
                            ParsedBlock(
                                text=para_text,
                                block_type=BlockType.PARAGRAPH.value,
                                zone=ZoneType.NORMATIVE.value,
                                page_num=page_num,
                                page_numbers=[page_num],
                            )
                        )
                        continue

                banners, headers, data_rows = _split_pre_header_banner_rows(grid)
                for banner_txt in banners:
                    blocks.append(
                        ParsedBlock(
                            text=banner_txt,
                            block_type=BlockType.HEADING.value,
                            zone=ZoneType.NORMATIVE.value,
                            page_num=page_num,
                            page_numbers=[page_num],
                            is_bold=True,
                        )
                    )

                if not headers and not data_rows:
                    continue

                filled_rows, top_hier = _forward_fill_cfo_hierarchy(headers, data_rows)
                md_str = _format_markdown_table(headers, filled_rows)
                tbl_title = banners[-1] if banners else ""

                blocks.append(
                    ParsedBlock(
                        text=md_str,
                        block_type=BlockType.TABLE.value,
                        zone=ZoneType.TABLE.value,
                        page_num=page_num,
                        page_numbers=[page_num],
                        is_ocr=False,
                        table_headers=headers,
                        table_rows=filled_rows,
                        parent_hierarchy=top_hier,
                        metadata={"table_title": tbl_title} if tbl_title else {},
                    )
                )

        except Exception as child_exc:
            w_msg = f"Ошибка элемента DOCX в {path.name}: {child_exc}"
            logger.warning(w_msg)
            warnings_list.append(w_msg)

    # Check Word section headers/footers ONLY for genuine order requisite headers or registration/ЭДО stamps
    try:
        body_alnum = re.sub(
            r"[^а-яa-z0-9]+",
            "",
            "\n".join(b.text for b in blocks if b.text).lower().replace("ё", "е"),
        )
        hdr_blocks: list[ParsedBlock] = []
        ftr_blocks: list[ParsedBlock] = []
        seen_hf_elements: set[int] = set()

        for sec in doc.sections:
            for is_footer, hf_obj in (
                (False, getattr(sec, "first_page_header", None)),
                (False, getattr(sec, "header", None)),
                (True, getattr(sec, "first_page_footer", None)),
                (True, getattr(sec, "footer", None)),
            ):
                if hf_obj is None or getattr(hf_obj, "_element", None) is None:
                    continue
                el_id = id(hf_obj._element)
                if el_id in seen_hf_elements:
                    continue
                seen_hf_elements.add(el_id)

                hf_lines: list[str] = []
                for p in getattr(hf_obj, "paragraphs", []):
                    pt = _normalize_inline_whitespace(p.text)
                    if pt:
                        hf_lines.append(pt)
                for tbl in getattr(hf_obj, "tables", []):
                    grid = _extract_docx_table_grid(tbl)
                    for row in grid:
                        rt = " | ".join(c for c in row if c)
                        if rt:
                            hf_lines.append(rt)

                if not hf_lines:
                    continue
                combined_hf = "\n".join(hf_lines).strip()
                has_reg_stamp = any(_REGISTRATION_STAMP_LINE_RE.match(ln) for ln in hf_lines) or bool(
                    _STAMP_OCR_MARKER_RE.search(combined_hf)
                )
                has_ord_hdr = (not is_footer) and bool(
                    _DOCX_ORDER_HEADER_RE.search(combined_hf) or _REQUISITE_CELL_RE.search(combined_hf)
                )
                if not (has_reg_stamp or has_ord_hdr):
                    continue
                hf_alnum = re.sub(r"[^а-яa-z0-9]+", "", combined_hf.lower().replace("ё", "е"))
                if len(hf_alnum) < 6 or hf_alnum in body_alnum:
                    continue
                body_alnum += hf_alnum
                sb = ParsedBlock(
                    text=combined_hf,
                    block_type=BlockType.STAMP.value,
                    zone=ZoneType.NORMATIVE.value,
                    page_num=1 if not is_footer else max(1, page_num),
                    page_numbers=[1 if not is_footer else max(1, page_num)],
                    is_ocr=False,
                )
                if is_footer:
                    ftr_blocks.append(sb)
                else:
                    hdr_blocks.append(sb)

        if hdr_blocks or ftr_blocks:
            blocks = hdr_blocks + blocks + ftr_blocks
    except Exception as hf_exc:
        logger.debug("Skipping DOCX header/footer stamp check for %s: %s", path.name, hf_exc)

    return blocks, {
        "file_format": "docx",
        "pages_total": max(1, page_num),
        "ocr_pages": 0,
        "warnings": warnings_list,
        "errors": errors_list,
    }


# ---------------------------------------------------------------------------
# 3. XLSX Parser (openpyxl dual-mode: data_only=True & data_only=False)
# ---------------------------------------------------------------------------
def _format_excel_scalar(val: Any, number_format: str = "") -> str:
    """
    Format a scalar Excel cell value without ever emitting `'None'`.
    Converts percentage-formatted decimals (`0.2` with `nf='0%'`, `0.082` with `nf='0.0%'`)
    into explicit human-readable percentages (`'20% (0.2)'`, `'8.2% (0.082)'`).
    """
    if val is None:
        return ""
    if isinstance(val, bool):
        return "Да" if val else "Нет"
    if isinstance(val, (int, float)):
        nf = str(number_format or "")
        if "%" in nf:
            pct = round(float(val) * 100.0, 4)
            pct_str = f"{pct:g}%"
            raw_str = f"{val:g}"
            if float(val) != 0.0:
                return f"{pct_str} ({raw_str})"
            return pct_str
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return f"{val:g}" if isinstance(val, float) else str(val)

    s = _normalize_inline_whitespace(str(val)).replace("\n", " ")
    if s == "None":
        return ""
    return sanitize_ocr_text(s)


def _format_excel_cell(val_cached: Any, val_formula: Any, number_format: str = "") -> str:
    """
    Combine dual-mode `openpyxl` values (`data_only=True` cached value and `data_only=False` formula):
    - NEVER outputs `'None'` when `val_cached is None`!
    - Emits `[формула: =...]` when a cell contains an Excel formula (`=...`).
    """
    is_formula = isinstance(val_formula, str) and val_formula.strip().startswith("=")
    if is_formula:
        f_clean = val_formula.strip()
        cached_str = _format_excel_scalar(val_cached, number_format=number_format)
        if cached_str:
            return f"{cached_str} [формула: {f_clean}]"
        return f"[формула: {f_clean}]"

    # Non-formula cell: prefer val_cached if present, else val_formula
    chosen = val_cached if val_cached is not None else val_formula
    return _format_excel_scalar(chosen, number_format=number_format)


def _parse_xlsx_smeta_form_sheet(
    sheet_title: str,
    page_num: int,
    grid_cells: list[list[str]],
) -> list[ParsedBlock]:
    """
    Format a budget/smeta Excel worksheet (such as `смет_поквартально` and `смета_помесячно` in
    `Приложение 1_новая смета 2019 (с незащищенными формулами).xlsx`) into structured `zone='form'` blocks:
    - Preserves sheet title (`Приложение № 1а` / `Приложение № 1б`), legend rates (`20% (0.2)`, `8.2% (0.082)`),
      KVR (`КВР`, forward-filled from parent category row to sub-items!), KOSGU (`КОСГУ`), and
      deduplicated representative formulas (`[формула: =...]`) without any `'None'`.
    """
    blocks: list[ParsedBlock] = []
    if not grid_cells:
        return blocks

    # Locate the main smeta table header row ("Наименование показателя" | "КВР" | "КОСГУ...")
    header_row_idx: int | None = None
    kvr_col_idx: int = 3
    kosgu_col_idx: int = 4

    for r_idx, row in enumerate(grid_cells):
        row_join = " ".join(row).lower()
        if "наименование показателя" in row_join and ("квр" in row_join or "косгу" in row_join):
            header_row_idx = r_idx
            for c_idx, cell_txt in enumerate(row):
                cl = cell_txt.lower()
                if "квр" in cl:
                    kvr_col_idx = c_idx
                elif "косгу" in cl or "вид затрат" in cl:
                    kosgu_col_idx = c_idx
            break

    form_lines: list[str] = [f"Лист: {sheet_title}"]

    # 1. Format preamble & legend rows (above header_row_idx)
    preamble_end = header_row_idx if header_row_idx is not None else len(grid_cells)
    for r_idx in range(preamble_end):
        row = grid_cells[r_idx]
        non_empty = [c for c in row if c and c != "None"]
        if not non_empty:
            continue
        form_lines.append(" | ".join(non_empty))

    if header_row_idx is None:
        fallback_headers = [f"Колонка {i + 1}" for i in range(max((len(r) for r in grid_cells), default=1))]
        return [
            ParsedBlock(
                text="\n".join(form_lines),
                block_type=BlockType.FORM.value,
                zone=ZoneType.FORM.value,
                page_num=page_num,
                page_numbers=[page_num],
                sheet_name=sheet_title,
                metadata={
                    "sheet_name": sheet_title,
                    "table_title": f"Лист: {sheet_title}",
                    "structured_headers": fallback_headers,
                    "structured_rows": [r for r in grid_cells if any(r)],
                },
            )
        ]

    # 2. Format smeta table rows with KVR inheritance and deduplicated period formulas,
    # while simultaneously building rich structured columns/rows for StructuredTableStore!
    hdr_row = grid_cells[header_row_idx]
    hdr_non_empty = [c for c in hdr_row if c and c != "None"]
    form_lines.append("Шапка сметы: " + " | ".join(hdr_non_empty[:10]))

    # Identify period & total columns after KOSGU (e.g. 1 квартал..4 квартал, Сумма OR январь..декабрь, Сумма)
    period_cols: list[tuple[int, str]] = []
    notes_start_col = len(hdr_row)
    for c_i in range(kosgu_col_idx + 1, len(hdr_row)):
        h_lbl = hdr_row[c_i].strip()
        if h_lbl and h_lbl != "None" and "[формула:" not in h_lbl:
            period_cols.append((c_i, h_lbl))
            if "сумма" in h_lbl.lower() or "итого" in h_lbl.lower():
                notes_start_col = c_i + 1
                break
    if not period_cols:
        notes_start_col = kosgu_col_idx + 1

    structured_headers: list[str] = [
        "Наименование показателя",
        "Ставка",
        "КВР",
        "КОСГУ / Вид затрат",
        *[lbl for _, lbl in period_cols],
        "Формулы",
        "Примечание",
    ]
    structured_rows: list[list[str]] = []

    # Capture legend rate/formula definitions above header_row_idx (e.g. 20% (0.2), 8.2% (0.082))
    app_banner = ""
    for r_idx in range(preamble_end):
        prow = grid_cells[r_idx]
        non_empty = [c.strip() for c in prow if c and c.strip() and c.strip() != "None"]
        if not non_empty:
            continue
        if any("приложение" in c.lower() for c in non_empty) and not app_banner:
            app_banner = non_empty[0]
        has_legend_rate_or_formula = any(
            ("%" in c and any(ch.isdigit() for ch in c)) or "[формула:" in c
            for c in non_empty
        )
        if has_legend_rate_or_formula:
            leg_rate = next((c for c in non_empty if "%" in c and any(ch.isdigit() for ch in c)), "")
            leg_form = next((c for c in non_empty if "[формула:" in c), "")
            leg_notes = [c for c in non_empty if c not in {leg_rate, leg_form}]
            structured_rows.append(
                [
                    "Условные обозначения бланка сметы",
                    leg_rate,
                    "",
                    "",
                    *["" for _ in period_cols],
                    leg_form,
                    " | ".join(leg_notes),
                ]
            )

    active_parent_kvr = ""

    for r_idx in range(header_row_idx + 1, len(grid_cells)):
        row = grid_cells[r_idx]
        if not any(row):
            continue

        indicator = row[0] if len(row) > 0 else ""
        # Rate / percentage column (usually col B / index 1)
        rate_parts: list[str] = []
        for c_i in range(1, min(kvr_col_idx, len(row))):
            val_c = row[c_i].strip()
            if val_c and val_c != ")":
                rate_parts.append(val_c)
            elif val_c == ")" and indicator.endswith("("):
                indicator += ")"
        if indicator.count("(") > indicator.count(")"):
            indicator = indicator.rstrip() + ")"

        raw_kvr = row[kvr_col_idx].strip() if kvr_col_idx < len(row) else ""
        raw_kosgu = row[kosgu_col_idx].strip() if kosgu_col_idx < len(row) else ""

        # Forward-fill parent KVR code (e.g. 111, 112, 113, 119, 241, 243, 244, 851, 852, 853) to child KOSGU rows
        if raw_kvr and raw_kvr not in {"Х", "X", "-"}:
            active_parent_kvr = raw_kvr
            effective_kvr = raw_kvr
        elif raw_kvr in {"Х", "X"}:
            if raw_kosgu in {"Х", "X", ""}:
                active_parent_kvr = ""
            effective_kvr = raw_kvr
        else:
            effective_kvr = active_parent_kvr if (raw_kosgu and raw_kosgu not in {"Х", "X"}) else ""

        # Collect formulas and side comments from columns after KOSGU
        formulas_seen: list[str] = []
        comments_seen: list[str] = []
        for c_i in range(kosgu_col_idx + 1, len(row)):
            cell_v = row[c_i].strip()
            if not cell_v or cell_v in {"Х", "X", "None"}:
                continue
            if "[формула:" in cell_v:
                # Avoid repeating 4 quarterly or 12 monthly identical formula patterns on the same line:
                # keep the first period formula and the final total column formula (e.g. SUM)
                if len(formulas_seen) == 0:
                    formulas_seen.append(cell_v)
                elif "SUM(" in cell_v and not any("SUM(" in f for f in formulas_seen):
                    formulas_seen.append(cell_v)
                elif len(formulas_seen) < 2 and cell_v not in formulas_seen:
                    # Check if it's just the same formula shifted to column G/H/I
                    norm_pat = re.sub(r"\b[F-R](\$?\d+)", r"COL\1", cell_v)
                    existing_pats = [re.sub(r"\b[F-R](\$?\d+)", r"COL\1", f) for f in formulas_seen]
                    if norm_pat not in existing_pats:
                        formulas_seen.append(cell_v)
            else:
                comments_seen.append(cell_v)

        row_pieces: list[str] = []
        if indicator:
            row_pieces.append(indicator)
        if rate_parts:
            row_pieces.append(f"Ставка: {', '.join(rate_parts)}")
        if effective_kvr:
            row_pieces.append(f"КВР: {effective_kvr}")
        if raw_kosgu:
            row_pieces.append(f"КОСГУ: {raw_kosgu}")
        row_pieces.extend(formulas_seen)
        row_pieces.extend(comments_seen)

        if row_pieces:
            form_lines.append(" | ".join(row_pieces))

        period_vals = [
            (row[col_idx].strip() if col_idx < len(row) and row[col_idx].strip() != "None" else "")
            for col_idx, _ in period_cols
        ]
        side_notes = [
            row[c_i].strip()
            for c_i in range(notes_start_col, len(row))
            if row[c_i].strip() and row[c_i].strip() != "None" and "[формула:" not in row[c_i]
        ]
        s_row = [
            indicator,
            ", ".join(rate_parts),
            effective_kvr,
            raw_kosgu,
            *period_vals,
            " ; ".join(formulas_seen),
            " | ".join(side_notes or comments_seen),
        ]
        if any(s_row):
            structured_rows.append(s_row)

    smeta_title = f"{app_banner} — Смета доходов и расходов ({sheet_title})" if app_banner else f"Смета ({sheet_title})"
    blocks.append(
        ParsedBlock(
            text="\n".join(form_lines),
            block_type=BlockType.FORM.value,
            zone=ZoneType.FORM.value,
            page_num=page_num,
            page_numbers=[page_num],
            sheet_name=sheet_title,
            metadata={
                "sheet_name": sheet_title,
                "table_title": smeta_title,
                "structured_headers": structured_headers,
                "structured_rows": structured_rows,
            },
        )
    )
    return blocks


def parse_xlsx(file_path: str | Path) -> tuple[list[ParsedBlock], dict[str, Any]]:
    """
    Parse an `.xlsx` workbook in two passes (`data_only=True` & `data_only=False`):
    - Never outputs `'None'` when `cached_value is None`; emits `[формула: =...]`
    - Preserves KVR/KOSGU codes, percentage rates (`20% (0.2)`, `8.2% (0.082)`), and legend comments
    - Expands `merged_cells` and forward-fills multi-column `A..E` ЦФО hierarchy (`Финансодержатель -> ЦФО 1..4`)
    """
    path = Path(file_path)
    blocks: list[ParsedBlock] = []
    warnings_list: list[str] = []
    errors_list: list[str] = []

    try:
        wb_val = openpyxl.load_workbook(str(path), data_only=True)
        wb_form = openpyxl.load_workbook(str(path), data_only=False)
    except Exception as exc:
        msg = f"Failed to open XLSX {path.name}: {exc}"
        logger.error(msg)
        return [], {
            "file_format": "xlsx",
            "pages_total": 0,
            "ocr_pages": 0,
            "warnings": [],
            "errors": [msg],
        }

    sheet_names = list(wb_form.sheetnames)
    try:
        for sheet_idx, sheet_name in enumerate(sheet_names, start=1):
            try:
                ws_v = wb_val[sheet_name]
                ws_f = wb_form[sheet_name]

                if getattr(ws_f, "sheet_state", "visible") != "visible":
                    continue

                max_r = ws_f.max_row or 0
                max_c = min(ws_f.max_column or 0, 60)
                if max_r == 0 or max_c == 0:
                    continue

                # Build merged cell lookup: vertical merges propagate down the column;
                # horizontal merges in the same row keep value in the leftmost cell and "" in right cells
                merged_lookup: dict[tuple[int, int], tuple[Any, Any, str]] = {}
                for m_range in ws_f.merged_cells.ranges:
                    tl_v = ws_v.cell(m_range.min_row, m_range.min_col).value
                    tl_f = ws_f.cell(m_range.min_row, m_range.min_col).value
                    tl_nf = str(ws_f.cell(m_range.min_row, m_range.min_col).number_format or "")
                    for r in range(m_range.min_row, m_range.max_row + 1):
                        for c in range(m_range.min_col, m_range.max_col + 1):
                            if c == m_range.min_col:
                                merged_lookup[(r, c)] = (tl_v, tl_f, tl_nf)
                            else:
                                merged_lookup[(r, c)] = (None, None, "")

                raw_grid: list[list[str]] = []
                formula_count = 0
                highest_non_empty_col = 0

                for r in range(1, max_r + 1):
                    row_vals: list[str] = []
                    for c in range(1, max_c + 1):
                        if (r, c) in merged_lookup:
                            v_cached, v_form, nf = merged_lookup[(r, c)]
                        else:
                            cell_v = ws_v.cell(r, c)
                            cell_f = ws_f.cell(r, c)
                            v_cached = cell_v.value
                            v_form = cell_f.value
                            nf = str(cell_f.number_format or "")

                        if isinstance(v_form, str) and v_form.strip().startswith("="):
                            formula_count += 1

                        formatted = _format_excel_cell(v_cached, v_form, number_format=nf)
                        if formatted:
                            highest_non_empty_col = max(highest_non_empty_col, c)
                        row_vals.append(formatted)

                    if any(row_vals):
                        raw_grid.append(row_vals)

                if not raw_grid or highest_non_empty_col == 0:
                    continue

                trimmed_grid = [r[:highest_non_empty_col] for r in raw_grid if any(r[:highest_non_empty_col])]
                if not trimmed_grid:
                    continue

                # Determine whether this sheet is a formula-driven smeta/form or a relational data table
                is_smeta_form = formula_count > 0 or "смет" in sheet_name.lower() or "форма" in sheet_name.lower()
                if is_smeta_form:
                    blocks.extend(
                        _parse_xlsx_smeta_form_sheet(
                            sheet_title=sheet_name,
                            page_num=sheet_idx,
                            grid_cells=trimmed_grid,
                        )
                    )
                    continue

                # Data Table Sheet (e.g. `2026` in `Приложение к приказу об утверждении фин. структуры 2026г..xlsx`)
                banners, headers, data_rows = _split_pre_header_banner_rows(trimmed_grid)
                for banner_txt in banners:
                    blocks.append(
                        ParsedBlock(
                            text=banner_txt,
                            block_type=BlockType.HEADING.value,
                            zone=ZoneType.NORMATIVE.value,
                            page_num=sheet_idx,
                            page_numbers=[sheet_idx],
                            is_bold=True,
                        )
                    )

                filled_rows, top_hier = _forward_fill_cfo_hierarchy(headers, data_rows)
                md_str = _format_markdown_table(headers, filled_rows)
                tbl_title = banners[-1] if banners else f"Лист: {sheet_name}"

                # Keep sheet_name in metadata so table slices hash identically to matching DOCX tables (e.g. 3614-1)
                blocks.append(
                    ParsedBlock(
                        text=md_str,
                        block_type=BlockType.TABLE.value,
                        zone=ZoneType.TABLE.value,
                        page_num=sheet_idx,
                        page_numbers=[sheet_idx],
                        is_ocr=False,
                        table_headers=headers,
                        table_rows=filled_rows,
                        parent_hierarchy=top_hier,
                        sheet_name=None,
                        metadata={"sheet_name": sheet_name, "table_title": tbl_title},
                    )
                )

            except Exception as sheet_exc:
                w_msg = f"Ошибка разбора листа '{sheet_name}' в {path.name}: {sheet_exc}"
                logger.warning(w_msg)
                warnings_list.append(w_msg)
                errors_list.append(w_msg)
    finally:
        try:
            wb_val.close()
        except Exception as exc:
            logger.debug("Error closing wb_val: %s", exc)
        try:
            wb_form.close()
        except Exception as exc:
            logger.debug("Error closing wb_form: %s", exc)

    return blocks, {
        "file_format": "xlsx",
        "pages_total": max(1, len(sheet_names)),
        "ocr_pages": 0,
        "warnings": warnings_list,
        "errors": errors_list,
    }


# ---------------------------------------------------------------------------
# Unified Entrypoint & DocumentParser Class
# ---------------------------------------------------------------------------
def parse_document(
    file_path: str | Path,
    ocr_backend: str | None = None,
    doc_id: str = "",
) -> tuple[list[ParsedBlock], dict[str, Any]]:
    """
    Unified fault-tolerant entrypoint for Module 1 Document Ingestion:
    Dispatches `.pdf`, `.docx`, and `.xlsx` files to their specialized parsers and returns
    `(list[ParsedBlock], parser_meta_dict)`. If `doc_id` is supplied, pre-assigns deterministic
    `table_id` (`<doc_id>::tbl_01`...) onto eligible structured table blocks.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        msg = f"Файл не найден: {path}"
        logger.error(msg)
        return [], {
            "file_format": path.suffix.lstrip(".").lower(),
            "pages_total": 0,
            "ocr_pages": 0,
            "warnings": [msg],
            "errors": [msg],
        }

    ext = path.suffix.lower()
    if ext == ".pdf":
        blocks, meta = parse_pdf(path, ocr_backend=ocr_backend)
    elif ext == ".docx":
        blocks, meta = parse_docx(path)
    elif ext in {".xlsx", ".xlsm"}:
        blocks, meta = parse_xlsx(path)
    else:
        msg = f"Неподдерживаемый формат файла: {ext} ({path.name})"
        logger.warning(msg)
        return [], {
            "file_format": ext.lstrip("."),
            "pages_total": 0,
            "ocr_pages": 0,
            "warnings": [msg],
            "errors": [msg],
        }

    if doc_id and blocks:
        from src.indexing.table_store import extract_logical_tables_and_assign_ids
        from src.schemas import DocumentMeta

        specs = extract_logical_tables_and_assign_ids(
            blocks,
            DocumentMeta(doc_id=doc_id, file_name=path.name, file_format=ext.lstrip(".")),
        )
        meta["tables_count"] = len(specs)

    return blocks, meta


parse_file = parse_document


class DocumentParser:
    """Object-oriented wrapper around `parse_document` for pipeline compatibility."""

    def __init__(self, ocr_backend: str | None = None) -> None:
        self.ocr_backend = ocr_backend

    def parse(
        self,
        file_path: str | Path,
        ocr_backend: str | None = None,
    ) -> tuple[list[ParsedBlock], dict[str, Any]]:
        return parse_document(file_path, ocr_backend=ocr_backend or self.ocr_backend)
