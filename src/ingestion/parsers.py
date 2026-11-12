from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl
import pdfplumber
import pymupdf
from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from src.config import OCR_SMALL_NATIVE_TEXT_CHARS, TABLE_SLICE_ROWS
from src.ingestion.ocr_engine import OcrEngine
from src.ingestion.registry import DocumentMetadata, DocumentRegistry


@dataclass
class ParsedBlock:
    block_type: str  # "paragraph" | "table" | "form"
    text: str
    page_no: int = 1
    zone: str = "body"  # "body" | "approval" | "table" | "form"
    source: str = "native"  # "native" | "ocr"
    is_bold: bool = False
    style_name: str = ""
    table_header: list[str] = field(default_factory=list)
    table_rows: list[list[str]] = field(default_factory=list)


def clean_pdf_text(text: str) -> str:
    """Clean PDF extraction artifacts, hyphenations, and glued footnote numbers."""
    # Fix glued footnotes in Регламент о командировании.pdf:
    # "от 26.12.2005 № 8123" -> "от 26.12.2005 № 812"
    # "от 02.10.2002 № 7292" -> "от 02.10.2002 № 729"
    text = re.sub(r"(26\.12\.2005\s*№\s*812)3\b", r"\1", text)
    text = re.sub(r"(02\.10\.2002\s*№\s*729)2\b", r"\1", text)
    text = re.sub(r"(13\.10\.2008\s*№\s*749)1\b", r"\1", text)

    # Normalize non-breaking spaces and soft hyphens
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    # Join words broken by hyphen across lines (excluding numbered lists)
    text = re.sub(r"([а-яА-ЯёЁ]{2,})-\s*\n\s*([а-яё]{2,})", r"\1\2", text)
    # Remove standalone page number lines
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if re.fullmatch(r"(?:Стр\.?\s*)?\d{1,3}(?:\s+из\s+\d{1,3})?", s, re.IGNORECASE):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines)


class DocxParser:
    """Order-preserving DOCX parser using body.iterchildren() and merged-cell deduplication."""

    APPROVAL_TRIGGER = re.compile(r"^\s*ЛИСТ\s+СОГЛАСОВАНИЯ\b", re.IGNORECASE)
    NORMATIVE_RESUME_TRIGGER = re.compile(
        r"^(?:Приложение|УТВЕРЖДЕН|РЕГЛАМЕНТ|ПОЛОЖЕНИЕ|ИНСТРУКЦИЯ|ПОРЯДОК|КАРТОЧКА\s+ПРОЕКТА|БЛАНК|ФОРМА)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _extract_table_rows(table: DocxTable) -> list[list[str]]:
        rows_out: list[list[str]] = []
        for row in table.rows:
            seen_tc: set[int] = set()
            row_cells: list[str] = []
            for cell in row.cells:
                tc_id = id(cell._tc)
                if tc_id in seen_tc:
                    continue
                seen_tc.add(tc_id)
                cell_text = " ".join(
                    p.text.strip() for p in cell.paragraphs if p.text.strip()
                )
                cell_text = re.sub(r"\s+", " ", cell_text).strip()
                row_cells.append(cell_text)
            if any(row_cells):
                rows_out.append(row_cells)
        return rows_out

    @staticmethod
    def _format_markdown_table(header: list[str], rows: list[list[str]]) -> str:
        lines: list[str] = []
        if header:
            lines.append(" | ".join(c for c in header if c))
            lines.append("-" * min(80, max(20, len(lines[0]))))
        for r in rows:
            non_empty = [c for c in r if c]
            if non_empty:
                lines.append(" | ".join(r))
        return "\n".join(lines)

    def parse(self, file_path: Path, meta: DocumentMetadata) -> list[ParsedBlock]:
        # If this DOCX is a standalone duplicate of appendices already inside Приказ 3699-1,
        # emit a compact summary block pointing to master_doc_id while still parsing key headers
        doc = DocxDocument(str(file_path))
        blocks: list[ParsedBlock] = []
        in_approval_zone = False

        if meta.is_duplicate_appendix_of:
            blocks.append(
                ParsedBlock(
                    block_type="paragraph",
                    text=(
                        f"Документ «{meta.doc_title}» представляет собой отдельную выгрузку "
                        f"Приложений № 1–3 к Приказу № 3699-1 от 18.08.2025 "
                        f"(основной документ: {meta.is_duplicate_appendix_of})."
                    ),
                    page_no=1,
                    zone="body",
                )
            )
            return blocks

        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                para = DocxParagraph(child, doc)
                raw = para.text.replace("\u00a0", " ").strip()
                if not raw:
                    continue
                if self.APPROVAL_TRIGGER.search(raw):
                    in_approval_zone = True
                elif in_approval_zone and self.NORMATIVE_RESUME_TRIGGER.search(raw):
                    in_approval_zone = False

                is_bold = any(run.bold for run in para.runs if run.text.strip())
                style_name = para.style.name if para.style else ""
                blocks.append(
                    ParsedBlock(
                        block_type="paragraph",
                        text=raw,
                        page_no=1,
                        zone="approval" if in_approval_zone else "body",
                        is_bold=is_bold,
                        style_name=style_name,
                    )
                )

            elif isinstance(child, CT_Tbl):
                tbl = DocxTable(child, doc)
                rows = self._extract_table_rows(tbl)
                if not rows:
                    continue

                flat_sample = " ".join(" ".join(r) for r in rows[:3])
                is_approval_tbl = (
                    bool(self.APPROVAL_TRIGGER.search(flat_sample))
                    or in_approval_zone
                    or ("согласующий" in flat_sample.lower() and "результат" in flat_sample.lower())
                )
                if is_approval_tbl:
                    blocks.append(
                        ParsedBlock(
                            block_type="table",
                            text=self._format_markdown_table(rows[0], rows[1:]),
                            page_no=1,
                            zone="approval",
                            table_header=rows[0],
                            table_rows=rows[1:],
                        )
                    )
                    # Approval sheet table ends the approval block; resume normal parsing for appendices
                    in_approval_zone = False
                    continue

                # If this is the huge 525-row CFO table inside Приказ 3614-1.docx,
                # emit a bridge note because the full forward-filled table is parsed from XLSX
                if meta.skip_embedded_cfo_table and len(rows) > 100:
                    blocks.append(
                        ParsedBlock(
                            block_type="paragraph",
                            text=(
                                "Таблица «Финансовая структура МФТИ на 2026 год» (перечень ЦФО 1, 2 и 3 уровней, "
                                f"всего {len(rows)} строк) приведена в Приложении к Приказу № 3614-1 от 11.08.2025 "
                                "(см. связанные табличные блоки Приложения к приказу № 3614-1)."
                            ),
                            page_no=1,
                            zone="body",
                        )
                    )
                    continue

                # 1-row layout boxes in Word (e.g. "от 04.04.2023 | № 889-1" or "Ректор | Д.В. Орлов")
                # should be merged as paragraphs rather than standalone table chunks
                if len(rows) == 1:
                    blocks.append(
                        ParsedBlock(
                            block_type="paragraph",
                            text=" ".join(c for c in rows[0] if c),
                            page_no=1,
                            zone="body",
                        )
                    )
                    continue

                # Slice large normative tables (e.g. 98-row KFO table in Распоряжение 700)
                header = rows[0]
                body_rows = rows[1:] if len(rows) > 1 else rows
                if len(body_rows) <= TABLE_SLICE_ROWS:
                    blocks.append(
                        ParsedBlock(
                            block_type="table",
                            text=self._format_markdown_table(header, body_rows),
                            page_no=1,
                            zone="table",
                            table_header=header,
                            table_rows=body_rows,
                        )
                    )
                else:
                    for start in range(0, len(body_rows), TABLE_SLICE_ROWS):
                        slice_rows = body_rows[start : start + TABLE_SLICE_ROWS]
                        blocks.append(
                            ParsedBlock(
                                block_type="table",
                                text=self._format_markdown_table(header, slice_rows),
                                page_no=1,
                                zone="table",
                                table_header=header,
                                table_rows=slice_rows,
                            )
                        )

        return blocks


class PdfParser:
    """Hybrid PDF parser: native text + pdfplumber tables for digital PDFs, OcrEngine for scans."""

    def __init__(self, ocr_engine: OcrEngine) -> None:
        self.ocr_engine = ocr_engine

    def parse(self, file_path: Path, meta: DocumentMetadata) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        pdf_doc = pymupdf.open(str(file_path))

        with pdfplumber.open(str(file_path)) as plumber_doc:
            for idx, page in enumerate(pdf_doc):
                page_no = idx + 1
                native_text = page.get_text("text") or ""
                printable_chars = len(re.sub(r"\s+", "", native_text))

                # Страницы-сканы иногда имеют тонкий текстовый слой (колонтитул/хвост
                # страницы). Если текста мало, а на странице есть картинка — OCR
                # обязателен, иначе теряются шапка приказа и начало разделов.
                has_images = bool(page.get_images(full=True))
                native_text_is_incomplete = (
                    has_images and printable_chars < OCR_SMALL_NATIVE_TEXT_CHARS
                )

                if (
                    printable_chars >= 60
                    and page_no not in meta.rotated_pages
                    and not native_text_is_incomplete
                ):
                    cleaned = clean_pdf_text(native_text)
                    for line in cleaned.splitlines():
                        s = line.strip()
                        if s:
                            blocks.append(
                                ParsedBlock(
                                    block_type="paragraph",
                                    text=s,
                                    page_no=page_no,
                                    zone="body",
                                )
                            )

                    # Also extract structured tables if present on digital page
                    if idx < len(plumber_doc.pages):
                        p_tables = plumber_doc.pages[idx].extract_tables() or []
                        for tbl in p_tables:
                            clean_rows = [
                                [
                                    re.sub(r"\s+", " ", (cell or "")).strip()
                                    for cell in row
                                ]
                                for row in tbl
                                if row and any(cell and cell.strip() for cell in row)
                            ]
                            if len(clean_rows) >= 2:
                                header = clean_rows[0]
                                for start in range(1, len(clean_rows), TABLE_SLICE_ROWS):
                                    sub = clean_rows[start : start + TABLE_SLICE_ROWS]
                                    md_lines = [" | ".join(header)] + [
                                        " | ".join(r) for r in sub
                                    ]
                                    blocks.append(
                                        ParsedBlock(
                                            block_type="table",
                                            text="\n".join(md_lines),
                                            page_no=page_no,
                                            zone="table",
                                            table_header=header,
                                            table_rows=sub,
                                        )
                                    )
                else:
                    # Scanned page or rotated landscape table page -> OCR
                    rotate_cw = 90 if page_no in meta.rotated_pages else 0
                    ocr_text = self.ocr_engine.ocr_page(
                        page=page,
                        file_sha256=meta.file_sha256,
                        file_name=meta.file_name,
                        page_no=page_no,
                        rotate_cw_degrees=rotate_cw,
                    )
                    cleaned_ocr = clean_pdf_text(ocr_text)
                    for line in cleaned_ocr.splitlines():
                        s = line.strip()
                        if s:
                            blocks.append(
                                ParsedBlock(
                                    block_type="paragraph",
                                    text=s,
                                    page_no=page_no,
                                    zone="body",
                                    source="ocr",
                                )
                            )

        pdf_doc.close()
        return blocks


class XlsxParser:
    """XLSX parser with hierarchical merged-cell forward-fill and formula extraction."""

    def parse(self, file_path: Path, meta: DocumentMetadata) -> list[ParsedBlock]:
        wb_val = openpyxl.load_workbook(str(file_path), data_only=True)
        wb_frm = openpyxl.load_workbook(str(file_path), data_only=False)
        blocks: list[ParsedBlock] = []

        for sheet_name in wb_val.sheetnames:
            ws_val = wb_val[sheet_name]
            ws_frm = wb_frm[sheet_name]

            # Build 2D grid of values and formulas, resolving merged cell ranges via top-left value
            max_r = ws_val.max_row or 0
            max_c = min(ws_val.max_column or 0, 35)
            if max_r == 0 or max_c == 0:
                continue

            val_grid: list[list[str]] = [
                ["" for _ in range(max_c)] for _ in range(max_r)
            ]
            frm_grid: list[list[str]] = [
                ["" for _ in range(max_c)] for _ in range(max_r)
            ]

            for r in range(1, max_r + 1):
                for c in range(1, max_c + 1):
                    v = ws_val.cell(r, c).value
                    f = ws_frm.cell(r, c).value
                    val_grid[r - 1][c - 1] = (
                        str(v).strip() if v is not None else ""
                    )
                    frm_grid[r - 1][c - 1] = (
                        str(f).strip() if f is not None else ""
                    )

            # Propagate merged cells across their span
            for m_range in ws_val.merged_cells.ranges:
                if m_range.min_row > max_r or m_range.min_col > max_c:
                    continue
                top_val = val_grid[m_range.min_row - 1][m_range.min_col - 1]
                if not top_val:
                    continue
                for r in range(m_range.min_row, min(m_range.max_row, max_r) + 1):
                    for c in range(m_range.min_col, min(m_range.max_col, max_c) + 1):
                        if not val_grid[r - 1][c - 1]:
                            val_grid[r - 1][c - 1] = top_val

            # Case 1: Large hierarchical reference table (Финансовая структура 2026)
            if "фин" in meta.file_name.lower() or max_r > 200:
                blocks.extend(self._parse_cfo_hierarchy(val_grid, sheet_name))
            else:
                # Case 2: Cost estimate templates (Бланк сметы РЦНИ / новая смета 2019)
                blocks.extend(
                    self._parse_estimate_form(val_grid, frm_grid, sheet_name, meta)
                )

        wb_val.close()
        wb_frm.close()
        return blocks

    def _parse_cfo_hierarchy(
        self, grid: list[list[str]], sheet_name: str
    ) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        header = [
            "Уровень 1 (Код и ЦФО)",
            "Уровень 2 (Код и ЦФО)",
            "Уровень 3 (Код и ЦФО)",
            "Тип подразделения",
            "Руководитель ЦФО / Ответственный",
        ]

        curr_l1 = ""
        curr_l2 = ""
        structured_rows: list[list[str]] = []

        for row in grid[3:]:
            if not any(row):
                continue
            c0 = row[0] if len(row) > 0 else ""
            c1 = row[1] if len(row) > 1 else ""
            c2 = row[2] if len(row) > 2 else ""
            c3 = row[3] if len(row) > 3 else ""
            c4 = row[4] if len(row) > 4 else ""
            c5 = row[5] if len(row) > 5 else ""
            c6 = row[6] if len(row) > 6 else ""
            c7 = row[7] if len(row) > 7 else ""
            c8 = row[8] if len(row) > 8 else ""

            if c0 or c1:
                curr_l1 = f"{c0} {c1}".strip()
            if c2 or c3:
                curr_l2 = f"{c2} {c3}".strip()
            l3 = f"{c4} {c5}".strip()
            unit_type = c6
            head_info = " ".join(x for x in (c7, c8) if x).strip()

            row_summary = [
                curr_l1 or "-",
                curr_l2 or "-",
                l3 or "-",
                unit_type or "-",
                head_info or "-",
            ]
            if any(x != "-" for x in row_summary):
                structured_rows.append(row_summary)

        for start in range(0, len(structured_rows), TABLE_SLICE_ROWS):
            sub = structured_rows[start : start + TABLE_SLICE_ROWS]
            l2_names = sorted({r[1] for r in sub if r[1] != "-"})
            group_label = (
                f"Лист «{sheet_name}» — Подразделения: {', '.join(l2_names[:3])}"
                if l2_names
                else f"Лист «{sheet_name}» (строки {start + 1}–{start + len(sub)})"
            )
            lines = [group_label, " | ".join(header)]
            for r in sub:
                lines.append(" | ".join(r))
            blocks.append(
                ParsedBlock(
                    block_type="table",
                    text="\n".join(lines),
                    page_no=1,
                    zone="table",
                    table_header=header,
                    table_rows=sub,
                )
            )
        return blocks

    def _parse_estimate_form(
        self,
        val_grid: list[list[str]],
        frm_grid: list[list[str]],
        sheet_name: str,
        meta: DocumentMetadata,
    ) -> list[ParsedBlock]:
        lines: list[str] = [
            f"Форма документа: {meta.doc_title} (Лист Excel: «{sheet_name}»)"
        ]
        formulas_found: list[str] = []

        for r_idx, row in enumerate(val_grid):
            frm_row = frm_grid[r_idx]
            non_empty_vals: list[str] = []
            for c_idx, val in enumerate(row):
                frm = frm_row[c_idx]
                if frm.startswith("="):
                    formulas_found.append(f"R{r_idx + 1}C{c_idx + 1}: {frm}")
                    if val and val != frm:
                        non_empty_vals.append(f"{val} [формула: {frm}]")
                    else:
                        non_empty_vals.append(f"[формула: {frm}]")
                elif val:
                    non_empty_vals.append(val)

            if non_empty_vals:
                dedup_row: list[str] = []
                for item in non_empty_vals:
                    if not dedup_row or dedup_row[-1] != item:
                        dedup_row.append(item)
                lines.append(" | ".join(dedup_row[:8]))

        if formulas_found:
            lines.append(
                "Ключевые расчётные формулы и коэффициенты в бланке сметы: "
                + "; ".join(formulas_found[:20])
            )

        # Split into 900-char form blocks so nothing gets truncated
        blocks: list[ParsedBlock] = []
        chunk_lines: list[str] = [lines[0]]
        curr_len = len(lines[0])
        for line in lines[1:]:
            if curr_len + len(line) > 950 and len(chunk_lines) > 2:
                blocks.append(
                    ParsedBlock(
                        block_type="form",
                        text="\n".join(chunk_lines),
                        page_no=1,
                        zone="form",
                    )
                )
                chunk_lines = [lines[0], line]
                curr_len = len(lines[0]) + len(line)
            else:
                chunk_lines.append(line)
                curr_len += len(line)

        if len(chunk_lines) > 1:
            blocks.append(
                ParsedBlock(
                    block_type="form",
                    text="\n".join(chunk_lines),
                    page_no=1,
                    zone="form",
                )
            )
        return blocks


class DocumentIngestionRouter:
    """Routes any document in данные/ to the appropriate format parser and resolves metadata."""

    def __init__(
        self,
        registry: DocumentRegistry | None = None,
        ocr_engine: OcrEngine | None = None,
    ) -> None:
        self.registry = registry or DocumentRegistry()
        self.ocr_engine = ocr_engine or OcrEngine()
        self.docx_parser = DocxParser()
        self.pdf_parser = PdfParser(self.ocr_engine)
        self.xlsx_parser = XlsxParser()

    def parse_file(self, file_path: Path) -> tuple[DocumentMetadata, list[ParsedBlock]]:
        ext = file_path.suffix.lower()
        prelim_meta = self.registry.resolve(file_path, sample_text="")

        if ext == ".docx":
            blocks = self.docx_parser.parse(file_path, prelim_meta)
        elif ext == ".pdf":
            blocks = self.pdf_parser.parse(file_path, prelim_meta)
        elif ext in (".xlsx", ".xls"):
            blocks = self.xlsx_parser.parse(file_path, prelim_meta)
        else:
            raise ValueError(f"Unsupported file extension: {ext} ({file_path})")

        sample_text = "\n".join(b.text for b in blocks[:15])
        final_meta = self.registry.resolve(file_path, sample_text=sample_text)
        return final_meta, blocks
