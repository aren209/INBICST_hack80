"""Unified Document Ingestion Parsers for PDF, DOCX, DOC, XLSX, and XLS (Module 1 -> ParsedBlock stream)."""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import email
from html.parser import HTMLParser
import io
import logging
import re
import struct
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import docx
import olefile
import openpyxl
import pdfplumber
import pypdfium2 as pdfium
import xlrd
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


_EXCEL_CELL_FORMULA_TAG_RE = re.compile(r"\[формула:\s*(=[^\]]+)\]", re.IGNORECASE)
_EXCEL_HEADER_FORMULA_TAG_RE = re.compile(r"\s*\[формула:\s*=[^\]]+\]", re.IGNORECASE)


def has_smeta_kvr_header(grid_cells: list[list[str]]) -> bool:
    """
    Return True iff at least one row (joined lowercase, `ё -> е`) contains
    `"наименование показателя"` AND at least one of `("квр", "косгу", "вид затрат")`.
    """
    if not grid_cells:
        return False
    for row in grid_cells:
        row_join = " ".join(str(c) for c in row if c).lower().replace("ё", "е")
        if "наименование показателя" in row_join and any(
            kw in row_join for kw in ("квр", "косгу", "вид затрат")
        ):
            return True
    return False


def _strip_excel_header_formula_tag(header: str) -> str:
    """Strip `[формула: ...]` tag from an Excel header label while keeping readable header text."""
    if not header:
        return ""
    return _EXCEL_HEADER_FORMULA_TAG_RE.sub("", str(header)).strip()


def _extract_excel_cell_formulas(cell_val: str) -> list[str]:
    """Extract Excel `=...` formulas from `[формула: =...]` tags or raw leading `=` cell strings."""
    if not cell_val:
        return []
    tagged = [m.group(1).strip() for m in _EXCEL_CELL_FORMULA_TAG_RE.finditer(cell_val) if m.group(1).strip()]
    if tagged:
        return tagged
    stripped = cell_val.strip()
    if stripped.startswith("=") and len(stripped) > 1:
        return [stripped]
    return []


def _extract_structured_formulas_from_rows(
    headers: list[str],
    rows: list[list[str]],
) -> list[dict[str, Any]]:
    """
    Extract `{"row_index": r_idx, "column": col_name, "formula": f_str, "cell": cell_val}`
    for every cell in `rows` containing `[формула: =...]` or a leading `=`.
    """
    structured_formulas: list[dict[str, Any]] = []
    for r_idx, row in enumerate(rows):
        for c_idx, cell_raw in enumerate(row):
            cell_val = str(cell_raw) if cell_raw is not None else ""
            cell_formulas = _extract_excel_cell_formulas(cell_val)
            if not cell_formulas:
                continue
            col_name = (
                headers[c_idx].strip()
                if c_idx < len(headers) and headers[c_idx].strip()
                else f"Колонка {c_idx + 1}"
            )
            for f_str in cell_formulas:
                structured_formulas.append(
                    {
                        "row_index": r_idx,
                        "column": col_name,
                        "formula": f_str,
                        "cell": cell_val,
                    }
                )
    return structured_formulas


def _build_excel_data_table_blocks(
    sheet_name: str,
    sheet_idx: int,
    trimmed_grid: list[list[str]],
    *,
    include_page_or_sheet: bool = False,
) -> list[ParsedBlock]:
    """Build heading + structured `TABLE` ParsedBlocks for a non-smeta Excel data sheet."""
    out_blocks: list[ParsedBlock] = []
    banners, raw_headers, data_rows = _split_pre_header_banner_rows(trimmed_grid)
    headers = [_strip_excel_header_formula_tag(h) for h in raw_headers]

    for banner_txt in banners:
        heading_meta = (
            {"sheet_name": sheet_name, "page_or_sheet": sheet_name}
            if include_page_or_sheet
            else {}
        )
        out_blocks.append(
            ParsedBlock(
                text=banner_txt,
                block_type=BlockType.HEADING.value,
                zone=ZoneType.NORMATIVE.value,
                page_num=sheet_idx,
                page_numbers=[sheet_idx],
                is_bold=True,
                metadata=heading_meta,
            )
        )

    if not headers and not data_rows:
        return out_blocks

    filled_rows, top_hier = _forward_fill_cfo_hierarchy(headers, data_rows)
    md_str = _format_markdown_table(headers, filled_rows)
    tbl_title = banners[-1] if banners else f"Лист: {sheet_name}"
    structured_formulas = _extract_structured_formulas_from_rows(headers, filled_rows)

    table_meta: dict[str, Any] = {
        "sheet_name": sheet_name,
        "table_title": tbl_title,
        "structured_formulas": structured_formulas,
        "has_formulas": bool(structured_formulas),
        "formula_count": len(structured_formulas),
    }
    if include_page_or_sheet:
        table_meta["page_or_sheet"] = sheet_name

    # Keep sheet_name in metadata so table slices hash identically to matching DOCX tables (e.g. 3614-1)
    out_blocks.append(
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
            metadata=table_meta,
        )
    )
    return out_blocks


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
        row_join = " ".join(str(c) for c in row if c).lower().replace("ё", "е")
        if "наименование показателя" in row_join and any(
            kw in row_join for kw in ("квр", "косгу", "вид затрат")
        ):
            header_row_idx = r_idx
            for c_idx, cell_txt in enumerate(row):
                cl = str(cell_txt).lower().replace("ё", "е")
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

                # Determine whether this sheet is a KVR/KOSGU smeta form or a relational data table
                is_smeta_form = has_smeta_kvr_header(trimmed_grid)
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
                blocks.extend(
                    _build_excel_data_table_blocks(
                        sheet_name=sheet_name,
                        sheet_idx=sheet_idx,
                        trimmed_grid=trimmed_grid,
                        include_page_or_sheet=False,
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
# 4. XLS Parser (xlrd + python-calamine + HTML/XML disguised .xls fallback)
# ---------------------------------------------------------------------------
def _grid_to_excel_sheet_blocks(
    sheet_name: str,
    sheet_idx: int,
    trimmed_grid: list[list[str]],
    formula_count: int = 0,
) -> list[ParsedBlock]:
    """Convert a cleaned 2D worksheet cell grid into ParsedBlock records identically to `parse_xlsx`."""
    if not trimmed_grid:
        return []

    is_smeta_form = has_smeta_kvr_header(trimmed_grid)
    if is_smeta_form:
        form_blocks = _parse_xlsx_smeta_form_sheet(
            sheet_title=sheet_name,
            page_num=sheet_idx,
            grid_cells=trimmed_grid,
        )
        for fb in form_blocks:
            fb.metadata.setdefault("page_or_sheet", sheet_name)
        return form_blocks

    return _build_excel_data_table_blocks(
        sheet_name=sheet_name,
        sheet_idx=sheet_idx,
        trimmed_grid=trimmed_grid,
        include_page_or_sheet=True,
    )


def _format_xlrd_cell(wb: xlrd.Book, cell: Any) -> str:
    """Format an `xlrd.sheet.Cell` (dates, numbers, percentages, booleans, strings) cleanly."""
    ctype = cell.ctype
    val = cell.value
    if ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK, xlrd.XL_CELL_ERROR) or val is None:
        return ""
    if ctype == xlrd.XL_CELL_BOOLEAN:
        return "Да" if bool(val) else "Нет"
    if ctype == xlrd.XL_CELL_DATE:
        try:
            y, m, d, hh, mm, ss = xlrd.xldate_as_tuple(float(val), wb.datemode)
            if y > 0:
                if hh or mm or ss:
                    return f"{d:02d}.{m:02d}.{y:04d} {hh:02d}:{mm:02d}"
                return f"{d:02d}.{m:02d}.{y:04d}"
            return f"{hh:02d}:{mm:02d}:{ss:02d}" if ss else f"{hh:02d}:{mm:02d}"
        except Exception:
            return _format_excel_scalar(val)
    if ctype == xlrd.XL_CELL_NUMBER:
        fmt_str = ""
        if getattr(wb, "formatting_info", False) and cell.xf_index is not None:
            try:
                if 0 <= cell.xf_index < len(wb.xf_list):
                    xf = wb.xf_list[cell.xf_index]
                    fmt_obj = wb.format_map.get(xf.format_key)
                    if fmt_obj is not None and fmt_obj.format_str:
                        fmt_str = str(fmt_obj.format_str)
            except Exception:
                fmt_str = ""
        return _format_excel_scalar(val, number_format=fmt_str)
    return _format_excel_scalar(val)


def _format_calamine_value(val: Any) -> str:
    """Format a Python value returned by `python_calamine`."""
    if val is None:
        return ""
    if isinstance(val, _dt.datetime):
        if val.hour or val.minute or val.second:
            return val.strftime("%d.%m.%Y %H:%M")
        return val.strftime("%d.%m.%Y")
    if isinstance(val, _dt.date):
        return val.strftime("%d.%m.%Y")
    if isinstance(val, _dt.time):
        return val.strftime("%H:%M")
    return _format_excel_scalar(val)


def parse_xls(file_path: str | Path) -> tuple[list[ParsedBlock], dict[str, Any]]:
    """
    Parse legacy `.xls` workbooks (BIFF8/BIFF5 via `xlrd` + `python_calamine`, plus disguised
    ZIP `.xlsx` or HTML/XML exports) into `ParsedBlock` table/form records with `sheet_name`,
    merged cell expansion, date/number formatting, and `StructuredTableStore` compatibility.
    """
    path = Path(file_path)
    blocks: list[ParsedBlock] = []
    warnings_list: list[str] = []
    errors_list: list[str] = []

    try:
        head_bytes = path.read_bytes()[:512]
    except Exception as exc:
        msg = f"Failed to read XLS {path.name}: {exc}"
        logger.error(msg)
        return [], {
            "file_format": "xls",
            "pages_total": 0,
            "ocr_pages": 0,
            "warnings": [],
            "errors": [msg],
        }

    stripped_head = head_bytes.lstrip()
    if head_bytes.startswith(b"PK\x03\x04"):
        sub_blocks, sub_meta = parse_xlsx(path)
        sub_meta["file_format"] = "xls"
        return sub_blocks, sub_meta

    if stripped_head[:10].lower().startswith((b"<html", b"<!doctype", b"<table", b"<?xml")):
        raw_bytes = path.read_bytes()
        html_blocks = _parse_html_or_xml_to_blocks(raw_bytes, default_sheet_mode=True)
        if html_blocks:
            sheet_cnt = max((b.page_num for b in html_blocks), default=1)
            return html_blocks, {
                "file_format": "xls",
                "pages_total": max(1, sheet_cnt),
                "ocr_pages": 0,
                "warnings": warnings_list,
                "errors": errors_list,
            }

    # Primary BIFF8 / BIFF5 path via xlrd
    wb: xlrd.Book | None = None
    try:
        try:
            wb = xlrd.open_workbook(str(path), formatting_info=True)
        except Exception:
            wb = xlrd.open_workbook(str(path), formatting_info=False)
    except Exception as xlrd_exc:
        logger.info("xlrd could not open %s (%s); trying python_calamine fallback.", path.name, xlrd_exc)
        wb = None

    if wb is not None:
        sheet_names = list(wb.sheet_names())
        visible_idx = 0
        for sh in wb.sheets():
            if getattr(sh, "visibility", 0) != 0:
                continue
            visible_idx += 1
            sheet_name = sh.name
            try:
                nrows = sh.nrows or 0
                ncols = min(sh.ncols or 0, 60)
                if nrows == 0 or ncols == 0:
                    continue

                merged_lookup: dict[tuple[int, int], str] = {}
                for rlo, rhi, clo, chi in getattr(sh, "merged_cells", ()):
                    if rlo >= nrows or clo >= ncols:
                        continue
                    tl_str = _format_xlrd_cell(wb, sh.cell(rlo, clo))
                    for r in range(rlo, min(rhi, nrows)):
                        for c in range(clo, min(chi, ncols)):
                            merged_lookup[(r, c)] = tl_str if c == clo else ""

                raw_grid: list[list[str]] = []
                highest_non_empty_col = 0
                for r in range(nrows):
                    row_vals: list[str] = []
                    for c in range(ncols):
                        if (r, c) in merged_lookup:
                            formatted = merged_lookup[(r, c)]
                        else:
                            formatted = _format_xlrd_cell(wb, sh.cell(r, c))
                        if formatted:
                            highest_non_empty_col = max(highest_non_empty_col, c + 1)
                        row_vals.append(formatted)
                    if any(row_vals):
                        raw_grid.append(row_vals)

                if not raw_grid or highest_non_empty_col == 0:
                    continue
                trimmed_grid = [r[:highest_non_empty_col] for r in raw_grid if any(r[:highest_non_empty_col])]
                blocks.extend(
                    _grid_to_excel_sheet_blocks(
                        sheet_name=sheet_name,
                        sheet_idx=visible_idx,
                        trimmed_grid=trimmed_grid,
                    )
                )
            except Exception as sheet_exc:
                w_msg = f"Ошибка разбора листа '{sheet_name}' в {path.name}: {sheet_exc}"
                logger.warning(w_msg)
                warnings_list.append(w_msg)
                errors_list.append(w_msg)

        try:
            wb.release_resources()
        except Exception:
            pass

        return blocks, {
            "file_format": "xls",
            "pages_total": max(1, visible_idx or len(sheet_names)),
            "ocr_pages": 0,
            "warnings": warnings_list,
            "errors": errors_list,
        }

    # Secondary fallback via python_calamine
    try:
        from python_calamine import CalamineWorkbook

        cwb = CalamineWorkbook.from_path(str(path))
        c_sheets = list(cwb.sheet_names)
        for sheet_idx, sheet_name in enumerate(c_sheets, start=1):
            sh_obj = cwb.get_sheet_by_name(sheet_name)
            py_rows = sh_obj.to_python(skip_empty_area=False)
            raw_grid = []
            highest_non_empty_col = 0
            for raw_r in py_rows:
                row_vals = []
                for c_idx, val in enumerate(raw_r[:60]):
                    formatted = _format_calamine_value(val)
                    if formatted:
                        highest_non_empty_col = max(highest_non_empty_col, c_idx + 1)
                    row_vals.append(formatted)
                if any(row_vals):
                    raw_grid.append(row_vals)
            if not raw_grid or highest_non_empty_col == 0:
                continue
            trimmed_grid = [r[:highest_non_empty_col] for r in raw_grid if any(r[:highest_non_empty_col])]
            blocks.extend(
                _grid_to_excel_sheet_blocks(
                    sheet_name=sheet_name,
                    sheet_idx=sheet_idx,
                    trimmed_grid=trimmed_grid,
                )
            )
        return blocks, {
            "file_format": "xls",
            "pages_total": max(1, len(c_sheets)),
            "ocr_pages": 0,
            "warnings": warnings_list,
            "errors": errors_list,
        }
    except Exception as cal_exc:
        msg = f"Failed to open XLS {path.name}: {cal_exc}"
        logger.error(msg)
        return [], {
            "file_format": "xls",
            "pages_total": 0,
            "ocr_pages": 0,
            "warnings": warnings_list,
            "errors": [msg],
        }


# ---------------------------------------------------------------------------
# 5. Pure-Python Multi-Format DOC Parser (OLE2 Word 97-2003, RTF, HTML/MHTML/XML, ZIP)
# ---------------------------------------------------------------------------
_DOC_CHARS_PER_PAGE: int = 3000

_RTF_SKIP_DESTINATIONS: frozenset[str] = frozenset(
    {
        "fonttbl",
        "colortbl",
        "stylesheet",
        "info",
        "pict",
        "object",
        "objdata",
        "header",
        "footer",
        "headerl",
        "headerr",
        "headerf",
        "footerl",
        "footerr",
        "footerf",
        "fldinst",
        "xmlnstbl",
        "listtable",
        "listoverridetable",
        "rsidtbl",
        "generator",
        "themedata",
        "colorschememapping",
        "latentstyles",
        "datastore",
        "pgdsctbl",
        "bkmkstart",
        "bkmkend",
        "datafield",
    }
)

_RTF_CODEPAGE_MAP: dict[int, str] = {
    1251: "cp1251",
    1252: "cp1252",
    1250: "cp1250",
    1257: "cp1257",
    866: "cp866",
    20866: "koi8-r",
    10007: "mac_cyrillic",
    65001: "utf-8",
}

_RTF_SYMBOL_WORDS: dict[str, str] = {
    "emdash": "—",
    "endash": "–",
    "emspace": " ",
    "enspace": " ",
    "qmspace": " ",
    "lquote": "‘",
    "rquote": "’",
    "ldblquote": "«",
    "rdblquote": "»",
    "bullet": "•",
    "tab": "\t",
}


def _emit_doc_table_grid(grid: list[list[str]], page_num: int) -> list[ParsedBlock]:
    """Convert a reconstructed 2D Word table grid into `ParsedBlock` records identically to `parse_docx`."""
    cleaned_grid: list[list[str]] = []
    max_cols = max((len(r) for r in grid), default=0)
    if max_cols == 0:
        return []
    for r in grid:
        norm_r = [
            _normalize_inline_whitespace(r[i] if i < len(r) else "").replace("\n", " ")
            for i in range(max_cols)
        ]
        if any(norm_r):
            cleaned_grid.append(norm_r)
    if not cleaned_grid:
        return []

    if len(cleaned_grid) == 1:
        row0 = cleaned_grid[0]
        non_empty = [c for c in row0 if c]
        if not non_empty:
            return []
        joined_pipe = " | ".join(non_empty)
        if _REQUISITE_CELL_RE.search(joined_pipe):
            return [
                ParsedBlock(
                    text=joined_pipe,
                    block_type=BlockType.TABLE.value,
                    zone=ZoneType.NORMATIVE.value,
                    page_num=page_num,
                    page_numbers=[page_num],
                    table_headers=row0,
                    table_rows=[],
                )
            ]
        if len(non_empty) <= 2:
            return [
                ParsedBlock(
                    text=" ".join(non_empty),
                    block_type=BlockType.PARAGRAPH.value,
                    zone=ZoneType.NORMATIVE.value,
                    page_num=page_num,
                    page_numbers=[page_num],
                )
            ]

    banners, headers, data_rows = _split_pre_header_banner_rows(cleaned_grid)
    out: list[ParsedBlock] = []
    for banner_txt in banners:
        out.append(
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
        return out

    filled_rows, top_hier = _forward_fill_cfo_hierarchy(headers, data_rows)
    md_str = _format_markdown_table(headers, filled_rows)
    tbl_title = banners[-1] if banners else ""
    out.append(
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
    return out


def _parse_rtf_doc(raw_bytes: bytes) -> tuple[list[ParsedBlock], int]:
    """
    Pure-Python RTF parser supporting Russian `\\ansicpg1251`, `\\'xx` hex escapes,
    `\\uN` signed 16-bit Unicode escapes (`\\ucN` skip count), `\\par`/`\\line`/`\\page`,
    and `\\cell`/`\\row` table reconstruction.
    """
    blocks: list[ParsedBlock] = []
    encoding = "cp1251"
    uc_skip = 1
    skip_dest = False
    in_table = False

    # Stack stores (encoding, uc_skip, skip_dest)
    stack: list[tuple[str, int, bool]] = []
    byte_buf = bytearray()
    cur_chars: list[str] = []
    row_cells: list[str] = []
    table_grid: list[list[str]] = []

    explicit_page = 1
    running_chars = 0
    has_explicit_pages = False

    def _effective_page() -> int:
        if has_explicit_pages:
            return explicit_page
        return max(1, 1 + running_chars // _DOC_CHARS_PER_PAGE)

    def _flush_bytes() -> None:
        if byte_buf:
            cur_chars.append(byte_buf.decode(encoding, errors="replace"))
            byte_buf.clear()

    def _pop_current_text() -> str:
        _flush_bytes()
        raw_s = "".join(cur_chars)
        cur_chars.clear()
        if not raw_s:
            return ""
        try:
            raw_s = raw_s.encode("utf-16", "surrogatepass").decode("utf-16", errors="replace")
        except Exception:
            pass
        return _normalize_inline_whitespace(raw_s)

    def _flush_table_grid() -> None:
        if table_grid:
            blocks.extend(_emit_doc_table_grid(table_grid, _effective_page()))
            table_grid.clear()

    def _flush_paragraph() -> None:
        nonlocal running_chars
        txt = _pop_current_text()
        if not txt:
            return
        if not in_table and table_grid and not row_cells:
            _flush_table_grid()
        p_num = _effective_page()
        blocks.extend(_text_to_parsed_blocks(txt, page_num=p_num, is_ocr=False, prev_blocks=blocks))
        running_chars += len(txt) + 1

    n = len(raw_bytes)
    i = 0
    skip_ansi_chars = 0

    while i < n:
        b = raw_bytes[i]

        if b == 0x7B:  # '{'
            _flush_bytes()
            stack.append((encoding, uc_skip, skip_dest))
            i += 1
            # Check for ignorable destination `{\*`
            if i + 1 < n and raw_bytes[i] == 0x5C and raw_bytes[i + 1] == 0x2A:
                i += 2
                # Peek destination control word name: keep `fldrslt`, skip others
                m_dest = re.match(rb"\s*\\([a-zA-Z]+)", raw_bytes[i : i + 40])
                dest_name = m_dest.group(1).decode("ascii", errors="ignore").lower() if m_dest else ""
                if dest_name != "fldrslt":
                    skip_dest = True
            continue

        if b == 0x7D:  # '}'
            _flush_bytes()
            if stack:
                encoding, uc_skip, skip_dest = stack.pop()
            i += 1
            continue

        if b in (0x0D, 0x0A):  # CR/LF in RTF source are ignored whitespace
            i += 1
            continue

        if b == 0x5C:  # '\'
            i += 1
            if i >= n:
                break
            nxt = raw_bytes[i]

            # Hex byte escape \'xx
            if nxt == 0x27 and i + 2 < n:
                hex_part = raw_bytes[i + 1 : i + 3]
                try:
                    val = int(hex_part.decode("ascii"), 16)
                    i += 3
                    if skip_ansi_chars > 0:
                        skip_ansi_chars -= 1
                    elif not skip_dest:
                        byte_buf.append(val)
                    continue
                except ValueError:
                    pass

            # Single-character control symbol (\\, \{, \}, \~, \-, \_, \n, \r, \*)
            if not (0x41 <= nxt <= 0x5A or 0x61 <= nxt <= 0x7A):
                _flush_bytes()
                i += 1
                if skip_dest:
                    continue
                if skip_ansi_chars > 0 and nxt in (0x5C, 0x7B, 0x7D, 0x7E, 0x5F, 0x2D):
                    skip_ansi_chars -= 1
                    continue
                if nxt in (0x5C, 0x7B, 0x7D):
                    cur_chars.append(chr(nxt))
                elif nxt == 0x7E:  # \~ non-breaking space
                    cur_chars.append(" ")
                elif nxt == 0x5F:  # \_ non-breaking hyphen
                    cur_chars.append("-")
                elif nxt in (0x0A, 0x0D):  # \<LF> or \<CR> acts as \par
                    if row_cells or in_table:
                        cur_chars.append("\n")
                    else:
                        _flush_paragraph()
                continue

            # Alphabetic control word \word[[-]num][space]
            w_start = i
            while i < n and (0x41 <= raw_bytes[i] <= 0x5A or 0x61 <= raw_bytes[i] <= 0x7A):
                i += 1
            word = raw_bytes[w_start:i].decode("ascii").lower()

            num_sign = 1
            has_num = False
            num_val = 0
            if i < n and raw_bytes[i] in (0x2D, 0x2B):
                if raw_bytes[i] == 0x2D:
                    num_sign = -1
                i += 1
            d_start = i
            while i < n and 0x30 <= raw_bytes[i] <= 0x39:
                i += 1
            if i > d_start:
                has_num = True
                num_val = num_sign * int(raw_bytes[d_start:i].decode("ascii"))

            # Optional single space delimiter after control word
            if i < n and raw_bytes[i] == 0x20:
                i += 1

            if word in _RTF_SKIP_DESTINATIONS:
                _flush_bytes()
                skip_dest = True
                continue

            if word == "ansicpg" and has_num:
                _flush_bytes()
                encoding = _RTF_CODEPAGE_MAP.get(num_val, encoding)
                continue

            if word == "mac":
                _flush_bytes()
                encoding = "mac_cyrillic"
                continue

            if word == "pc" or word == "pca":
                _flush_bytes()
                encoding = "cp866"
                continue

            if skip_dest:
                continue

            if word == "uc" and has_num:
                _flush_bytes()
                uc_skip = max(0, num_val)
                continue

            if word == "u" and has_num:
                _flush_bytes()
                codepoint = num_val if num_val >= 0 else (num_val + 65536)
                if 0 <= codepoint <= 0x10FFFF:
                    cur_chars.append(chr(codepoint))
                skip_ansi_chars = uc_skip
                continue

            _flush_bytes()
            if word in _RTF_SYMBOL_WORDS:
                cur_chars.append(_RTF_SYMBOL_WORDS[word])
            elif word == "line":
                cur_chars.append("\n")
            elif word in ("intbl", "trowd"):
                in_table = True
            elif word == "pard":
                in_table = False
            elif word in ("cell", "nestcell"):
                in_table = True
                cell_txt = _pop_current_text()
                row_cells.append(cell_txt)
                running_chars += len(cell_txt) + 2
            elif word in ("row", "nestrow"):
                trailing = _pop_current_text()
                if trailing and not row_cells:
                    row_cells.append(trailing)
                if row_cells:
                    table_grid.append(list(row_cells))
                    row_cells.clear()
                in_table = False
            elif word in ("page", "pagebb"):
                if row_cells:
                    table_grid.append(list(row_cells))
                    row_cells.clear()
                _flush_table_grid()
                _flush_paragraph()
                has_explicit_pages = True
                explicit_page += 1
            elif word in ("par", "sect"):
                if in_table:
                    cur_chars.append("\n")
                else:
                    _flush_table_grid()
                    _flush_paragraph()
            continue

        # Literal character byte
        i += 1
        if skip_ansi_chars > 0:
            skip_ansi_chars -= 1
            continue
        if not skip_dest:
            byte_buf.append(b)

    if row_cells:
        trailing = _pop_current_text()
        if trailing:
            row_cells.append(trailing)
        table_grid.append(list(row_cells))
        row_cells.clear()
    _flush_table_grid()
    _flush_paragraph()

    pages_total = max((b.page_num for b in blocks), default=1)
    return blocks, max(1, pages_total)


class _SimpleDocHTMLParser(HTMLParser):
    """Lightweight HTML/MHTML extractor for paragraphs, headings, lists, and `<table>` grids."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[tuple[str, Any]] = []  # ("heading", (lvl, txt)) | ("para", txt) | ("table", grid) | ("page", None)
        self._skip_depth = 0
        self._in_table = 0
        self._cur_grid: list[list[str]] = []
        self._cur_row: list[str] | None = None
        self._cur_cell: list[str] | None = None
        self._cur_cell_colspan: int = 1
        self._cur_para: list[str] = []
        self._cur_heading_lvl: int | None = None

    def _flush_para(self) -> None:
        if not self._cur_para:
            return
        txt = _normalize_inline_whitespace("".join(self._cur_para))
        self._cur_para.clear()
        if txt:
            if self._cur_heading_lvl is not None:
                self.items.append(("heading", (self._cur_heading_lvl, txt)))
            else:
                self.items.append(("para", txt))
        self._cur_heading_lvl = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in ("script", "style", "head", "xml"):
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        attr_map = {k.lower(): (v or "") for k, v in attrs}
        style_val = attr_map.get("style", "").lower()
        if "page-break-before:always" in style_val.replace(" ", "") or "page-break-after:always" in style_val.replace(" ", ""):
            if self._in_table == 0:
                self._flush_para()
                self.items.append(("page", None))

        if t == "table":
            if self._in_table == 0:
                self._flush_para()
                self._cur_grid = []
            self._in_table += 1
        elif t == "tr" and self._in_table == 1:
            self._cur_row = []
        elif t in ("td", "th") and self._in_table == 1 and self._cur_row is not None:
            self._cur_cell = []
            try:
                self._cur_cell_colspan = max(1, min(20, int(attr_map.get("colspan") or "1")))
            except ValueError:
                self._cur_cell_colspan = 1
        elif t == "br":
            if self._cur_cell is not None:
                self._cur_cell.append("\n")
            else:
                self._cur_para.append("\n")
        elif self._in_table == 0 and t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush_para()
            self._cur_heading_lvl = int(t[1])
        elif self._in_table == 0 and t in ("p", "div", "li", "blockquote", "section", "article"):
            self._flush_para()
            if t == "li":
                self._cur_para.append("- ")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "head", "xml"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return

        if t in ("td", "th") and self._in_table == 1 and self._cur_row is not None and self._cur_cell is not None:
            cell_txt = _normalize_inline_whitespace("".join(self._cur_cell)).replace("\n", " ")
            self._cur_row.append(cell_txt)
            for _ in range(self._cur_cell_colspan - 1):
                self._cur_row.append("")
            self._cur_cell = None
            self._cur_cell_colspan = 1
        elif t == "tr" and self._in_table == 1 and self._cur_row is not None:
            if any(self._cur_row):
                self._cur_grid.append(self._cur_row)
            self._cur_row = None
        elif t == "table" and self._in_table > 0:
            self._in_table -= 1
            if self._in_table == 0 and self._cur_grid:
                self.items.append(("table", list(self._cur_grid)))
                self._cur_grid = []
        elif self._in_table == 0 and t in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "li", "blockquote"):
            self._flush_para()

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._cur_cell is not None:
            self._cur_cell.append(data)
        elif self._in_table == 0:
            self._cur_para.append(data)


def _decode_markup_bytes(raw_bytes: bytes) -> str:
    """Decode HTML/XML bytes detecting BOM, `<meta charset>`, XML encoding, UTF-8, or CP1251."""
    if raw_bytes.startswith(b"\xff\xfe"):
        return raw_bytes.decode("utf-16-le", errors="replace")
    if raw_bytes.startswith(b"\xfe\xff"):
        return raw_bytes.decode("utf-16-be", errors="replace")
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return raw_bytes[3:].decode("utf-8", errors="replace")

    head_ascii = raw_bytes[:1024].decode("ascii", errors="ignore")
    m_enc = re.search(r'charset\s*=\s*["\']?([a-zA-Z0-9_-]+)', head_ascii, re.IGNORECASE)
    if not m_enc:
        m_enc = re.search(r'encoding\s*=\s*["\']([a-zA-Z0-9_-]+)["\']', head_ascii, re.IGNORECASE)
    if m_enc:
        enc_cand = m_enc.group(1).strip().lower()
        if enc_cand in ("windows-1251", "cp1251"):
            return raw_bytes.decode("cp1251", errors="replace")
        try:
            return raw_bytes.decode(enc_cand, errors="replace")
        except LookupError:
            pass

    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("cp1251", errors="replace")


def _parse_wordml_or_spreadsheetml_xml(xml_text: str, default_sheet_mode: bool = False) -> list[ParsedBlock]:
    """Parse Word 2003 XML (`w:wordDocument`), Flat OPC Word XML, or SpreadsheetML 2003 XML."""
    root = ET.fromstring(xml_text)

    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    # SpreadsheetML 2003 (`<Workbook><Worksheet ss:Name="..."><Table><Row><Cell><Data>`)
    worksheets = [el for el in root.iter() if _local(el.tag) == "Worksheet"]
    if worksheets:
        blocks: list[ParsedBlock] = []
        for s_idx, ws in enumerate(worksheets, start=1):
            sh_name = f"Лист {s_idx}"
            for k, v in ws.attrib.items():
                if _local(k) == "Name" and v:
                    sh_name = v
                    break
            grid: list[list[str]] = []
            for row_el in ws.iter():
                if _local(row_el.tag) != "Row":
                    continue
                row_vals: list[str] = []
                col_cursor = 0
                for cell_el in row_el:
                    if _local(cell_el.tag) != "Cell":
                        continue
                    for k, v in cell_el.attrib.items():
                        if _local(k) == "Index" and str(v).isdigit():
                            target_idx = max(0, int(v) - 1)
                            while col_cursor < target_idx:
                                row_vals.append("")
                                col_cursor += 1
                    cell_text = _normalize_inline_whitespace("".join(cell_el.itertext())).replace("\n", " ")
                    row_vals.append(cell_text)
                    col_cursor += 1
                    for k, v in cell_el.attrib.items():
                        if _local(k) == "MergeAcross" and str(v).isdigit():
                            for _ in range(int(v)):
                                row_vals.append("")
                                col_cursor += 1
                if any(row_vals):
                    grid.append(row_vals)
            if grid:
                blocks.extend(_grid_to_excel_sheet_blocks(sh_name, s_idx, grid))
        return blocks

    # WordprocessingML 2003 / Flat OPC (`w:body` -> `w:p` | `w:tbl` | `w:sectPr`)
    bodies = [el for el in root.iter() if _local(el.tag) == "body"]
    if not bodies:
        return []

    blocks = []
    running_chars = 0
    explicit_page = 1

    def _walk_container(container: ET.Element) -> None:
        nonlocal running_chars, explicit_page
        for child in container:
            cname = _local(child.tag)
            if cname == "p":
                texts: list[str] = []
                has_page_br = False
                for sub in child.iter():
                    sname = _local(sub.tag)
                    if sname == "t" and sub.text:
                        texts.append(sub.text)
                    elif sname == "tab":
                        texts.append("\t")
                    elif sname == "br":
                        br_type = next((v for k, v in sub.attrib.items() if _local(k) == "type"), "")
                        if br_type == "page":
                            has_page_br = True
                        else:
                            texts.append("\n")
                p_txt = _normalize_inline_whitespace("".join(texts))
                p_num = max(explicit_page, 1 + running_chars // _DOC_CHARS_PER_PAGE)
                if p_txt:
                    blocks.extend(_text_to_parsed_blocks(p_txt, page_num=p_num, is_ocr=False, prev_blocks=blocks))
                    running_chars += len(p_txt) + 1
                if has_page_br:
                    explicit_page += 1
            elif cname == "tbl":
                grid: list[list[str]] = []
                for tr in child:
                    if _local(tr.tag) != "tr":
                        continue
                    row_cells: list[str] = []
                    for tc in tr:
                        if _local(tc.tag) != "tc":
                            continue
                        cell_paras: list[str] = []
                        for p in tc.iter():
                            if _local(p.tag) == "p":
                                pt = _normalize_inline_whitespace(
                                    "".join(sub.text or "" for sub in p.iter() if _local(sub.tag) == "t")
                                )
                                if pt:
                                    cell_paras.append(pt)
                        row_cells.append(" ".join(cell_paras))
                    if any(row_cells):
                        grid.append(row_cells)
                if grid:
                    p_num = max(explicit_page, 1 + running_chars // _DOC_CHARS_PER_PAGE)
                    if default_sheet_mode:
                        blocks.extend(_grid_to_excel_sheet_blocks(f"Таблица {len(blocks) + 1}", len(blocks) + 1, grid))
                    else:
                        blocks.extend(_emit_doc_table_grid(grid, p_num))
                    running_chars += sum(len(c) for r in grid for c in r)
            elif cname in ("sub-section", "sdt", "sdtContent"):
                _walk_container(child)

    for body_el in bodies:
        _walk_container(body_el)
    return blocks


def _parse_html_or_xml_to_blocks(raw_bytes: bytes, default_sheet_mode: bool = False) -> list[ParsedBlock]:
    """Parse MHTML, Word/Excel XML, or HTML documents saved with `.doc` or `.xls` extension."""
    stripped = raw_bytes.lstrip()
    if stripped[:60].lower().startswith(b"mime-version:") or b"multipart/related" in raw_bytes[:1024].lower():
        try:
            msg = email.message_from_bytes(raw_bytes)
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes) and payload:
                        raw_bytes = payload
                        break
        except Exception as mhtml_exc:
            logger.debug("MHTML unwrap skipped: %s", mhtml_exc)

    decoded = _decode_markup_bytes(raw_bytes)
    stripped_txt = decoded.lstrip()
    if stripped_txt.startswith("<?xml") or stripped_txt.startswith("<w:") or stripped_txt.startswith("<Workbook"):
        try:
            xml_blocks = _parse_wordml_or_spreadsheetml_xml(decoded, default_sheet_mode=default_sheet_mode)
            if xml_blocks:
                return xml_blocks
        except Exception as xml_exc:
            logger.debug("XML parse fallback to HTMLParser: %s", xml_exc)

    parser = _SimpleDocHTMLParser()
    parser.feed(decoded)
    parser._flush_para()

    blocks: list[ParsedBlock] = []
    explicit_page = 1
    running_chars = 0
    sheet_idx = 0

    for kind, payload in parser.items:
        if kind == "page":
            explicit_page += 1
            continue
        p_num = max(explicit_page, 1 + running_chars // _DOC_CHARS_PER_PAGE)
        if kind == "heading":
            lvl, txt = payload
            blocks.append(
                ParsedBlock(
                    text=txt,
                    block_type=BlockType.HEADING.value,
                    zone=ZoneType.NORMATIVE.value,
                    page_num=p_num,
                    page_numbers=[p_num],
                    is_bold=True,
                    heading_level=lvl,
                )
            )
            running_chars += len(txt) + 1
        elif kind == "para":
            blocks.extend(_text_to_parsed_blocks(payload, page_num=p_num, is_ocr=False, prev_blocks=blocks))
            running_chars += len(payload) + 1
        elif kind == "table":
            if default_sheet_mode:
                sheet_idx += 1
                blocks.extend(_grid_to_excel_sheet_blocks(f"Лист {sheet_idx}", sheet_idx, payload))
            else:
                blocks.extend(_emit_doc_table_grid(payload, p_num))
            running_chars += sum(len(c) for r in payload for c in r)

    return blocks


def _decode_word_compressed_piece(raw_piece: bytes) -> str:
    """
    Decode an 8-bit compressed Clx piece (`fCompressed=1`).
    In MS-DOC, compressed pieces are CP1252 for Western/ASCII runs; if high bytes `0xC0..0xFF`
    or `0xA8/0xB8/0xB9` appear (used by Russian third-party `.doc` generators), decode as `cp1251`.
    """
    has_cyrillic_high = any((0xC0 <= b <= 0xFF) or b in (0xA8, 0xB8, 0xB9) for b in raw_piece)
    enc = "cp1251" if has_cyrillic_high else "cp1252"
    return raw_piece.decode(enc, errors="replace")


def _strip_word_field_codes(text: str) -> str:
    """
    Strip binary Word field instructions `\\x13 instr \\x14 visible \\x15` keeping only `visible`
    (or removing `\\x13 ... \\x15` when no `\\x14` display value exists, e.g. TOC/BOOKMARK/SEQ codes).
    Handles nested fields cleanly.
    """
    if "\x13" not in text and "\x15" not in text:
        return text

    out_stack: list[tuple[list[str], list[str], bool]] = []
    root_chars: list[str] = []

    for ch in text:
        if ch == "\x13":  # Field begin
            out_stack.append(([], [], False))
        elif ch == "\x14":  # Field separator (switch from instruction to visible result)
            if out_stack:
                instr_buf, vis_buf, _ = out_stack[-1]
                out_stack[-1] = (instr_buf, vis_buf, True)
        elif ch == "\x15":  # Field end
            if out_stack:
                _instr_buf, vis_buf, seen_sep = out_stack.pop()
                kept = "".join(vis_buf) if seen_sep else ""
                if out_stack:
                    _, parent_vis, parent_sep = out_stack[-1]
                    if parent_sep:
                        parent_vis.append(kept)
                else:
                    root_chars.append(kept)
        else:
            if out_stack:
                instr_buf, vis_buf, seen_sep = out_stack[-1]
                if seen_sep:
                    vis_buf.append(ch)
                else:
                    instr_buf.append(ch)
            else:
                root_chars.append(ch)

    # Flush any unclosed field gracefully
    for _instr_buf, vis_buf, seen_sep in out_stack:
        if seen_sep:
            root_chars.extend(vis_buf)
    return "".join(root_chars)


def _clean_word_stream_chars(text: str) -> str:
    """Clean binary Word control markers while preserving `\\r`, `\\x07` (table cell/row), and `\\x0c` (page break)."""
    s = _strip_word_field_codes(text)
    s = s.replace("\r\n", "\r").replace("\n", "\r")
    s = s.replace("\x0b", "\n")  # Vertical tab = soft line break inside paragraph
    s = s.replace("\x1e", "-")   # Non-breaking hyphen
    s = s.replace("\x1f", "")    # Optional soft hyphen
    s = s.replace("\xa0", " ")
    # Remove inline picture/drawn-object/footnote/annotation anchors
    s = re.sub(r"[\x00-\x06\x08\x0e-\x12\x16-\x1d]", "", s)
    return s


def _word_binary_text_to_blocks(raw_text: str) -> tuple[list[ParsedBlock], int]:
    """
    Reconstruct structured `ParsedBlock` paragraphs, headings, page numbers, and `\\x07`-delimited
    Word tables from the decoded `WordDocument` character stream.
    """
    cleaned = _clean_word_stream_chars(raw_text)
    blocks: list[ParsedBlock] = []
    explicit_page = 1
    has_explicit_page = "\x0c" in cleaned
    running_chars = 0

    def _cur_page() -> int:
        if has_explicit_page:
            return explicit_page
        return max(1, 1 + running_chars // _DOC_CHARS_PER_PAGE)

    def _emit_prose_chunk(prose: str) -> None:
        nonlocal explicit_page, running_chars
        if not prose:
            return
        # Handle explicit \x0c page/section breaks inside prose
        page_parts = prose.split("\x0c")
        for p_idx, part in enumerate(page_parts):
            if p_idx > 0:
                explicit_page += 1
            lines = [
                _normalize_inline_whitespace(p)
                for p in part.split("\r")
            ]
            non_empty_lines = [ln for ln in lines if ln]
            if non_empty_lines:
                joined = "\n".join(non_empty_lines)
                p_num = _cur_page()
                blocks.extend(_text_to_parsed_blocks(joined, page_num=p_num, is_ocr=False, prev_blocks=blocks))
                running_chars += len(joined) + 1

    if "\x07" not in cleaned:
        _emit_prose_chunk(cleaned)
        pages_total = max((b.page_num for b in blocks), default=1)
        return blocks, max(1, pages_total)

    segments = cleaned.split("\x07")
    head_segments = segments[:-1]  # Each segment in head_segments was terminated by \x07
    tail_prose = segments[-1]      # Trailing text after the very last \x07

    table_rows: list[list[str]] = []
    current_row: list[str] = []
    expected_cols: int = 0

    def _flush_table() -> None:
        nonlocal expected_cols
        if current_row:
            if any(current_row):
                table_rows.append(list(current_row))
            current_row.clear()
        if table_rows:
            blocks.extend(_emit_doc_table_grid(table_rows, _cur_page()))
            table_rows.clear()
        expected_cols = 0

    for seg in head_segments:
        # Word 6/95 or multi-table transitions: if we are between rows (`not current_row`),
        # check whether `seg` contains preceding prose paragraphs separated by `\r` or `\x0c`.
        if not current_row and ("\r" in seg or "\x0c" in seg):
            # Note: `\r\x07` (seg == "\r") is a classic Word 6/95 row-end marker!
            if seg == "\r" and table_rows:
                continue
            r_parts = seg.split("\r")
            preceding_prose = "\r".join(r_parts[:-1])
            first_cell_raw = r_parts[-1]
            if preceding_prose.strip() or "\x0c" in preceding_prose:
                _flush_table()
                _emit_prose_chunk(preceding_prose)
            seg = first_cell_raw

        # Normalize cell text (convert internal `\r` inside a multi-paragraph cell to space/newline)
        cell_clean = _normalize_inline_whitespace(seg.replace("\x0c", "").replace("\r", "\n")).replace("\n", " ")

        if cell_clean == "":
            # An empty segment between two `\x07` marks is EITHER the row-terminating TTP `\x07`
            # OR an empty cell inside the current row!
            if not current_row:
                # Consecutive row terminator or leading empty cell
                if expected_cols > 1:
                    current_row.append("")
                continue
            if expected_cols > 0 and len(current_row) < expected_cols:
                current_row.append("")
            else:
                table_rows.append(list(current_row))
                expected_cols = max(expected_cols, len(current_row))
                current_row.clear()
        else:
            current_row.append(cell_clean)
            running_chars += len(cell_clean) + 2

    _flush_table()
    if tail_prose:
        _emit_prose_chunk(tail_prose)

    pages_total = max((b.page_num for b in blocks), default=1)
    return blocks, max(1, pages_total)


def _extract_ole2_pieces_from_fib(word_stream: bytes, ole: olefile.OleFileIO) -> tuple[str, str] | None:
    """
    Parse MS-DOC FIB (`FibBase` + `FibRgW97` + `FibRgLw97` + `FibRgFcLcb`) and `Clx` piece table
    from `1Table` / `0Table` to reconstruct `(main_story_text, header_story_text)` in exact CP order.
    """
    if len(word_stream) < 68:
        return None

    flags = struct.unpack_from("<H", word_stream, 0x0A)[0]
    tbl_stream_name = "1Table" if (flags & 0x0200) else "0Table"
    if not ole.exists(tbl_stream_name):
        alt_name = "0Table" if tbl_stream_name == "1Table" else "1Table"
        if ole.exists(alt_name):
            tbl_stream_name = alt_name
        else:
            return None

    # Walk variable-length FIB arrays dynamically (`csw` -> `cslw` -> `cbRgFcLcb`)
    csw = struct.unpack_from("<H", word_stream, 32)[0]
    pos_cslw = 34 + csw * 2
    if pos_cslw + 2 > len(word_stream):
        return None
    cslw = struct.unpack_from("<H", word_stream, pos_cslw)[0]
    pos_rglw = pos_cslw + 2
    if pos_rglw + cslw * 4 + 2 > len(word_stream) or cslw < 6:
        return None

    ccp_text = struct.unpack_from("<i", word_stream, pos_rglw + 3 * 4)[0]
    ccp_ftn = struct.unpack_from("<i", word_stream, pos_rglw + 4 * 4)[0]
    ccp_hdd = struct.unpack_from("<i", word_stream, pos_rglw + 5 * 4)[0]

    pos_cb_fclcb = pos_rglw + cslw * 4
    cb_fclcb = struct.unpack_from("<H", word_stream, pos_cb_fclcb)[0]
    pos_fclcb = pos_cb_fclcb + 2
    if cb_fclcb <= 33 or pos_fclcb + 34 * 8 > len(word_stream):
        return None

    fc_clx, lcb_clx = struct.unpack_from("<II", word_stream, pos_fclcb + 33 * 8)
    if lcb_clx == 0:
        return None

    tbl_data = ole.openstream(tbl_stream_name).read()
    if fc_clx + lcb_clx > len(tbl_data):
        return None
    clx = tbl_data[fc_clx : fc_clx + lcb_clx]

    pos = 0
    while pos < len(clx) and clx[pos] == 0x01:  # Skip Prc property records
        if pos + 3 > len(clx):
            return None
        cb_grpprl = struct.unpack_from("<H", clx, pos + 1)[0]
        pos += 3 + cb_grpprl

    if pos + 5 > len(clx) or clx[pos] != 0x02:  # Expect Pcdt (0x02)
        return None

    lcb_plc = struct.unpack_from("<I", clx, pos + 1)[0]
    plc_data = clx[pos + 5 : pos + 5 + lcb_plc]
    if len(plc_data) < 16 or (len(plc_data) - 4) % 12 != 0:
        return None

    n_pieces = (len(plc_data) - 4) // 12
    a_cp = [struct.unpack_from("<I", plc_data, i * 4)[0] for i in range(n_pieces + 1)]
    pcd_base = (n_pieces + 1) * 4

    pieces_str: list[str] = []
    for i in range(n_pieces):
        cp_start, cp_end = a_cp[i], a_cp[i + 1]
        if cp_end <= cp_start:
            continue
        char_count = cp_end - cp_start
        fc_raw = struct.unpack_from("<I", plc_data, pcd_base + i * 8 + 2)[0]
        is_compressed = bool(fc_raw & 0x40000000)
        if is_compressed:
            byte_offset = (fc_raw & ~0x40000000) // 2
            raw_piece = word_stream[byte_offset : byte_offset + char_count]
            pieces_str.append(_decode_word_compressed_piece(raw_piece))
        else:
            byte_offset = fc_raw
            raw_piece = word_stream[byte_offset : byte_offset + char_count * 2]
            pieces_str.append(raw_piece.decode("utf-16-le", errors="replace"))

    full_chars = "".join(pieces_str)
    if not full_chars:
        return None

    if ccp_text > 0:
        main_text = full_chars[:ccp_text]
        hdd_start = max(0, ccp_text + max(0, ccp_ftn))
        hdd_end = hdd_start + max(0, ccp_hdd)
        hdd_text = full_chars[hdd_start:hdd_end] if ccp_hdd > 0 and hdd_start < len(full_chars) else ""
    else:
        main_text = full_chars
        hdd_text = ""

    return main_text, hdd_text


def _extract_ole2_fallback_text(word_stream: bytes) -> str:
    """
    Clean stream-scanning fallback when `Clx` is absent (e.g. Word 6.0 non-complex fast-save or
    stripped FIB): slices `fcMin..fcMac` if valid, otherwise extracts printable UTF-16LE / CP1251 runs.
    """
    if len(word_stream) >= 0x20:
        flags = struct.unpack_from("<H", word_stream, 0x0A)[0]
        fc_min, fc_mac = struct.unpack_from("<II", word_stream, 0x18)
        is_complex = bool(flags & 0x0004)
        if not is_complex and 0 < fc_min < fc_mac <= len(word_stream):
            span = word_stream[fc_min:fc_mac]
            # Check if UTF-16LE (Word 97 non-complex) vs 8-bit CP1251 (Word 6/95)
            zero_odd = sum(1 for idx in range(1, min(len(span), 400), 2) if span[idx] in (0x00, 0x04))
            if len(span) >= 4 and zero_odd >= min(len(span), 400) // 4:
                return span.decode("utf-16-le", errors="replace")
            return _decode_word_compressed_piece(span)

    payload = word_stream[512:] if len(word_stream) > 512 else word_stream
    # Scan for UTF-16LE runs (ASCII + Cyrillic U+0400..U+045F + punctuation) and CP1251 runs
    chunks: list[str] = []
    for m in re.finditer(rb"(?:[\x07\x09\x0a\x0d\x20-\x7e]\x00|[\x00-\x5f]\x04){6,}", payload):
        s = m.group(0).decode("utf-16-le", errors="ignore").strip()
        if s:
            chunks.append(s)
    if not chunks:
        for m in re.finditer(rb"[\x07\x09\x0a\x0d\x20-\x7e\xa8\xb8\xb9\xc0-\xff]{8,}", payload):
            s = m.group(0).decode("cp1251", errors="ignore").strip()
            if s:
                chunks.append(s)
    return "\r".join(chunks)


def _parse_ole2_word_doc(raw_bytes: bytes, file_name: str) -> tuple[list[ParsedBlock], int, list[str]]:
    """Parse genuine OLE2 Compound Binary Word 97–2003 (`.doc`) file via `olefile`."""
    warnings_list: list[str] = []
    with olefile.OleFileIO(io.BytesIO(raw_bytes)) as ole:
        if not ole.exists("WordDocument"):
            raise ValueError(f"OLE2 container '{file_name}' has no 'WordDocument' stream.")
        word_stream = ole.openstream("WordDocument").read()
        fib_res = _extract_ole2_pieces_from_fib(word_stream, ole)
        if fib_res is not None:
            main_text, hdd_text = fib_res
        else:
            w_msg = f"Нестандартная таблица кусков Clx в {file_name}; применен потоковый разбор WordDocument."
            warnings_list.append(w_msg)
            main_text = _extract_ole2_fallback_text(word_stream)
            hdd_text = ""

    blocks, pages_total = _word_binary_text_to_blocks(main_text)

    # Check header/footer story for order requisite frames or registration/ЭДО stamps (like parse_docx)
    if hdd_text.strip():
        hdd_blocks, _ = _word_binary_text_to_blocks(hdd_text)
        body_alnum = re.sub(
            r"[^а-яa-z0-9]+",
            "",
            "\n".join(b.text for b in blocks if b.text).lower().replace("ё", "е"),
        )
        prefix_stamps: list[ParsedBlock] = []
        suffix_stamps: list[ParsedBlock] = []
        for hb in hdd_blocks:
            htxt = (hb.text or "").strip()
            if not htxt:
                continue
            has_reg_stamp = bool(_REGISTRATION_STAMP_LINE_RE.search(htxt) or _STAMP_OCR_MARKER_RE.search(htxt))
            has_ord_hdr = bool(_DOCX_ORDER_HEADER_RE.search(htxt) or _REQUISITE_CELL_RE.search(htxt))
            if not (has_reg_stamp or has_ord_hdr):
                continue
            h_alnum = re.sub(r"[^а-яa-z0-9]+", "", htxt.lower().replace("ё", "е"))
            if len(h_alnum) < 6 or h_alnum in body_alnum:
                continue
            body_alnum += h_alnum
            stamp_blk = ParsedBlock(
                text=htxt,
                block_type=BlockType.STAMP.value,
                zone=ZoneType.NORMATIVE.value,
                page_num=pages_total if has_reg_stamp else 1,
                page_numbers=[pages_total if has_reg_stamp else 1],
            )
            if has_reg_stamp:
                suffix_stamps.append(stamp_blk)
            else:
                prefix_stamps.append(stamp_blk)
        if prefix_stamps or suffix_stamps:
            blocks = prefix_stamps + blocks + suffix_stamps

    return blocks, max(1, pages_total), warnings_list


def parse_doc(file_path: str | Path) -> tuple[list[ParsedBlock], dict[str, Any]]:
    """
    Pure-Python multi-format `.doc` parser inspecting header magic bytes first:
    a) ZIP (`PK\\x03\\x04`) -> delegates to `parse_docx`
    b) RTF (`{\\rtf`) -> pure-Python Russian RTF + table parser
    c) HTML / MHTML / WordML XML (`<html`, `<!DOCTYPE`, `<?xml`, `MIME-Version:`) -> HTML/XML + table extractor
    d) Genuine OLE2 Compound Binary Word 97–2003 (`\\xd0\\xcf\\x11\\xe0`) via `olefile` (FIB + Clx piece table + `\\x07` tables)
    """
    path = Path(file_path)
    warnings_list: list[str] = []
    errors_list: list[str] = []

    try:
        raw_bytes = path.read_bytes()
    except Exception as exc:
        msg = f"Failed to read DOC {path.name}: {exc}"
        logger.error(msg)
        return [], {
            "file_format": "doc",
            "pages_total": 0,
            "ocr_pages": 0,
            "warnings": [],
            "errors": [msg],
        }

    if not raw_bytes:
        msg = f"Empty DOC file: {path.name}"
        return [], {
            "file_format": "doc",
            "pages_total": 0,
            "ocr_pages": 0,
            "warnings": [msg],
            "errors": [msg],
        }

    stripped_head = raw_bytes[:512].lstrip()

    # (a) Disguised OOXML .docx (ZIP header PK\x03\x04)
    if raw_bytes.startswith(b"PK\x03\x04"):
        blocks, sub_meta = parse_docx(path)
        sub_meta["file_format"] = "doc"
        return blocks, sub_meta

    # (b) Rich Text Format ({\rtf)
    if stripped_head.startswith(b"{\\rtf"):
        try:
            blocks, pages_total = _parse_rtf_doc(raw_bytes)
            return blocks, {
                "file_format": "doc",
                "pages_total": pages_total,
                "ocr_pages": 0,
                "warnings": warnings_list,
                "errors": errors_list,
            }
        except Exception as rtf_exc:
            msg = f"Failed to parse RTF DOC {path.name}: {rtf_exc}"
            logger.error(msg)
            return [], {
                "file_format": "doc",
                "pages_total": 0,
                "ocr_pages": 0,
                "warnings": warnings_list,
                "errors": [msg],
            }

    # (c) HTML / MHTML / WordprocessingML XML
    low_head = stripped_head[:64].lower()
    if low_head.startswith(
        (b"<html", b"<!doctype", b"<?xml", b"mime-version:", b"<w:worddocument", b"<body", b"<table", b"<div", b"<p")
    ):
        try:
            blocks = _parse_html_or_xml_to_blocks(raw_bytes, default_sheet_mode=False)
            pages_total = max((b.page_num for b in blocks), default=1)
            return blocks, {
                "file_format": "doc",
                "pages_total": max(1, pages_total),
                "ocr_pages": 0,
                "warnings": warnings_list,
                "errors": errors_list,
            }
        except Exception as html_exc:
            msg = f"Failed to parse HTML/XML DOC {path.name}: {html_exc}"
            logger.error(msg)
            return [], {
                "file_format": "doc",
                "pages_total": 0,
                "ocr_pages": 0,
                "warnings": warnings_list,
                "errors": [msg],
            }

    # (d) Genuine OLE2 Compound Binary Word 97–2003 (\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1)
    if raw_bytes.startswith(b"\xd0\xcf\x11\xe0"):
        try:
            blocks, pages_total, ole_warnings = _parse_ole2_word_doc(raw_bytes, path.name)
            warnings_list.extend(ole_warnings)
            return blocks, {
                "file_format": "doc",
                "pages_total": pages_total,
                "ocr_pages": 0,
                "warnings": warnings_list,
                "errors": errors_list,
            }
        except Exception as ole_exc:
            msg = f"Failed to parse OLE2 DOC {path.name}: {ole_exc}"
            logger.error(msg)
            return [], {
                "file_format": "doc",
                "pages_total": 0,
                "ocr_pages": 0,
                "warnings": warnings_list,
                "errors": [msg],
            }

    # Plain-text fallback if a plain UTF-8 / CP1251 file was saved with `.doc` extension
    decoded_txt = _decode_markup_bytes(raw_bytes)
    norm_txt = _normalize_inline_whitespace(decoded_txt)
    if norm_txt:
        blocks, pages_total = _word_binary_text_to_blocks(norm_txt.replace("\n", "\r"))
        return blocks, {
            "file_format": "doc",
            "pages_total": pages_total,
            "ocr_pages": 0,
            "warnings": warnings_list,
            "errors": errors_list,
        }

    msg = f"Не удалось распознать структуру файла DOC: {path.name}"
    return [], {
        "file_format": "doc",
        "pages_total": 0,
        "ocr_pages": 0,
        "warnings": [msg],
        "errors": [msg],
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
    Dispatches `.pdf`, `.docx`, `.doc`, `.xlsx`, and `.xls` files to their specialized parsers and returns
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
    elif ext == ".doc":
        blocks, meta = parse_doc(path)
    elif ext in {".xlsx", ".xlsm"}:
        blocks, meta = parse_xlsx(path)
    elif ext == ".xls":
        blocks, meta = parse_xls(path)
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
