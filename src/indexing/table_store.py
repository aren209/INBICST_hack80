"""SQLite Structured Table Store & OpenAI Tool Execution for Adaptive 3-Tier Table Architecture."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from src.config import (
    APPROVAL_TRIGGER_RE,
    LARGE_TABLE_MIN_CHARS,
    LARGE_TABLE_MIN_ROWS,
    NORMATIVE_RESET_RE,
    TABLES_DB_PATH,
    UNIT_BOUNDARY_RE,
)
from src.schemas import (
    BlockType,
    ChunkRecord,
    DocumentMeta,
    ParsedBlock,
    TableColumnSchema,
    TablePassport,
    TableQueryRequest,
    TableQueryResponse,
    ZoneType,
)

logger = logging.getLogger(__name__)

# Regexes for formula extraction, header requisite detection, and numeric parsing
_FORMULA_TAG_RE = re.compile(r"\[формула:\s*(=[^\]]+)\]", re.IGNORECASE)
_REQUISITE_HEADER_PROBE_RE = re.compile(
    r"(?:от\s+\d{2}\.\d{2}\.\d{4}.*(?:№|N)\s*\d|(?:№|N)\s*\d.*от\s+\d{2}\.\d{2}\.\d{4})",
    re.IGNORECASE,
)
_COMP_OP_RE = re.compile(r"^\s*(>=|<=|!=|<>|==|>|<|=)\s*(.+)$")
_DATE_LIKE_RE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}(?:\s*г\.?)?$")


def _norm_ru(text: str) -> str:
    """Normalize Russian string for forgiving comparisons: lowercase, ё -> е, collapsed spaces."""
    if not text:
        return ""
    s = str(text).lower().replace("ё", "е").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()


def _clean_cell_str(val: Any) -> str:
    """Sanitize a cell value, never emitting literal 'None'."""
    if val is None:
        return ""
    s = str(val).replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s == "None":
        return ""
    return s


def should_store_structured_table(
    file_format: str,
    row_count: int,
    char_count: int = 0,
) -> bool:
    """
    Adaptive 3-Tier Table Eligibility Rule:
    - Returns True for ALL Excel tables/sheets (`.xlsx` / `.xlsm`) regardless of size (preserves formulas & types).
    - Returns True for large Word/PDF tables (`row_count >= LARGE_TABLE_MIN_ROWS` OR `char_count >= LARGE_TABLE_MIN_CHARS`).
    - Returns False for compact Word/PDF tables (`< 25` rows AND `< 5000` chars), which stay inline via `ParentSectionRecord`.
    """
    fmt = (file_format or "").strip().lower().lstrip(".")
    if fmt in {"xlsx", "xlsm", "xls"}:
        return True
    if int(row_count) >= LARGE_TABLE_MIN_ROWS or int(char_count) >= LARGE_TABLE_MIN_CHARS:
        return True
    return False


def _extract_formulas_from_cell(cell_text: str) -> list[str]:
    """Extract all `=...` formula strings from `[формула: =...]` tags or raw `=...` cells."""
    if not cell_text:
        return []
    found = [m.group(1).strip() for m in _FORMULA_TAG_RE.finditer(cell_text) if m.group(1).strip()]
    if found:
        return found
    stripped = cell_text.strip()
    if stripped.startswith("=") and len(stripped) > 1:
        return [stripped]
    return []


def _parse_numeric_value(cell_text: str) -> float | None:
    """
    Extract a clean float from numbers, currency strings ('1 250 000,50 руб.'),
    percentages ('20% (0.2)', '8.2% (0.082)', '16,4%'), or cached formula values ('1500 [формула: =SUM(...)]').
    Returns None when the cell is non-numeric, a pure formula without cached value, or a date/code range.
    """
    if not cell_text:
        return None

    # Strip `[формула: ...]` suffix first so cell references like F14 are never mistaken for numbers
    without_formula = _FORMULA_TAG_RE.sub("", cell_text).strip()
    if not without_formula or without_formula.startswith("="):
        return None

    if _DATE_LIKE_RE.match(without_formula):
        return None

    # Handle formatted Excel percentages like "20% (0.2)" or "8.2% (0.082)" or "16,4%"
    m_pct = re.match(
        r"^([+-]?\d+(?:[.,]\d+)?)\s*%(?:\s*\([+-]?\d+(?:[.,]\d+)?\))?$",
        without_formula,
    )
    if m_pct:
        try:
            return float(m_pct.group(1).replace(",", "."))
        except ValueError:
            return None

    # Strip trailing currency / unit markers
    cleaned = re.sub(
        r"(?:\s*(?:тыс\.?\s*руб\.?|млн\.?\s*руб\.?|руб\.?|р\.|ед\.?|шт\.?|чел\.?|дн\.?|час\.?))\.?\s*$",
        "",
        without_formula,
        flags=re.IGNORECASE,
    ).strip()

    # Reject hyphenated order numbers / ranges like "3065-1" or "2026-2028" or KVR/KOSGU like "900 (510)"
    if re.match(r"^\d+\s*[-–—]\s*\d+$", cleaned):
        return None
    if re.match(r"^\d+\s*\(\d+\)$", cleaned):
        return None

    # Remove space thousand separators inside numeric strings
    compact = cleaned.replace(" ", "")
    if re.match(r"^[+-]?\d+(?:[.,]\d+)?$", compact):
        try:
            return float(compact.replace(",", "."))
        except ValueError:
            return None

    return None


def _deduplicate_headers(raw_headers: list[str], col_count: int) -> list[str]:
    """Ensure every column has a non-empty, unique name preserving original Russian labels."""
    seen: dict[str, int] = {}
    headers: list[str] = []
    for idx in range(col_count):
        raw = _clean_cell_str(raw_headers[idx]) if idx < len(raw_headers) else ""
        base = raw if raw else f"Колонка {idx + 1}"
        norm_base = _norm_ru(base)
        cnt = seen.get(norm_base, 0) + 1
        seen[norm_base] = cnt
        col_name = base if cnt == 1 else f"{base} ({cnt})"
        headers.append(col_name)
    return headers


def compute_table_hash(headers: list[str], rows: list[list[str]]) -> str:
    """
    Compute a normalized SHA-256 hash of a table's headers + data rows (`ё->е`, lowercase, whitespace-normalized)
    so identical tables across `.docx` and `.xlsx` (e.g. Order 3614-1 Table 4 and Fin Structure 2026.xlsx)
    produce the exact same `table_hash` regardless of `file_name` or `sheet_name`.
    """
    col_count = max(len(headers), max((len(r) for r in rows), default=0))
    if col_count == 0:
        return ""

    norm_lines: list[str] = []
    hdr_norm = " | ".join(_norm_ru(headers[i] if i < len(headers) else "") for i in range(col_count))
    norm_lines.append(hdr_norm)

    for r in rows:
        r_cells = [_norm_ru(r[i] if i < len(r) else "") for i in range(col_count)]
        if any(r_cells):
            norm_lines.append(" | ".join(r_cells))

    joined = "\n".join(norm_lines)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _is_1x2_requisite_block(block: ParsedBlock) -> bool:
    """Return True if `block` is a tiny 1-2 row Word requisite header frame ('от 28.03.2023 | № 794-1')."""
    total_rows = (1 if block.table_headers else 0) + len(block.table_rows)
    if total_rows > 2:
        return False
    pieces: list[str] = []
    if block.table_headers:
        pieces.append(" | ".join(_clean_cell_str(c) for c in block.table_headers if _clean_cell_str(c)))
    for r in block.table_rows:
        pieces.append(" | ".join(_clean_cell_str(c) for c in r if _clean_cell_str(c)))
    if block.text:
        pieces.append(block.text.strip())
    combined = " | ".join(p for p in pieces if p)
    if not combined or len(combined) > 220:
        return False
    return bool(_REQUISITE_HEADER_PROBE_RE.search(combined))


def _extract_grid_from_parsed_block(
    block: ParsedBlock,
    file_format: str = "",
) -> tuple[list[str], list[list[str]], str, str] | None:
    """
    Extract `(headers, rows, sheet_name, table_title)` from a `ParsedBlock` if it represents
    a table or structured Excel form sheet; otherwise return `None`.
    """
    meta = block.metadata or {}
    sheet_name = _clean_cell_str(block.sheet_name or meta.get("sheet_name") or "")
    table_title = _clean_cell_str(meta.get("table_title") or "")

    # 1. Explicit structured_headers / structured_rows prepared by XLSX smeta form parser
    if meta.get("structured_rows"):
        s_headers = [_clean_cell_str(h) for h in (meta.get("structured_headers") or [])]
        s_rows = [
            [_clean_cell_str(c) for c in r]
            for r in (meta.get("structured_rows") or [])
            if any(_clean_cell_str(c) for c in r)
        ]
        if s_rows:
            return s_headers, s_rows, sheet_name, table_title

    # 2. Standard table_headers / table_rows on ParsedBlock
    if block.table_rows:
        headers = [_clean_cell_str(h) for h in (block.table_headers or [])]
        rows = [
            [_clean_cell_str(c) for c in r]
            for r in block.table_rows
            if any(_clean_cell_str(c) for c in r)
        ]
        if not headers and rows:
            headers = rows[0]
            rows = rows[1:]
        if rows:
            return headers, rows, sheet_name, table_title

    # 3. Markdown pipe table or XLSX form text fallback
    is_tbl_or_xlsx_form = (
        block.block_type == BlockType.TABLE.value
        or block.zone == ZoneType.TABLE.value
        or (
            (file_format.lower().lstrip(".") in {"xlsx", "xlsm", "xls"} or bool(sheet_name))
            and (block.block_type == BlockType.FORM.value or block.zone == ZoneType.FORM.value)
        )
    )
    if not is_tbl_or_xlsx_form or not block.text:
        return None

    lines = [ln.strip() for ln in block.text.split("\n") if ln.strip()]
    pipe_lines = [ln for ln in lines if "|" in ln]
    if not pipe_lines:
        return None

    # Check if formatted as `Шапка сметы: ...` form lines
    smeta_hdr: list[str] = []
    smeta_rows: list[list[str]] = []
    has_smeta_marker = any(ln.startswith("Шапка сметы:") or "КВР:" in ln or "КОСГУ:" in ln for ln in lines)
    if has_smeta_marker:
        smeta_hdr = ["Наименование показателя", "Ставка", "КВР", "КОСГУ / Вид затрат", "Формулы", "Примечание"]
        for ln in lines:
            if ln.startswith("Лист:") or ln.startswith("Шапка сметы:"):
                continue
            parts = [_clean_cell_str(p) for p in ln.split("|") if _clean_cell_str(p)]
            if not parts:
                continue
            ind, rate, kvr, kosgu = "", "", "", ""
            formulas_acc: list[str] = []
            notes_acc: list[str] = []
            for idx_p, p in enumerate(parts):
                if p.startswith("Ставка:"):
                    rate = p.split(":", 1)[1].strip()
                elif p.startswith("КВР:"):
                    kvr = p.split(":", 1)[1].strip()
                elif p.startswith("КОСГУ:"):
                    kosgu = p.split(":", 1)[1].strip()
                elif "[формула:" in p or p.startswith("="):
                    formulas_acc.append(p)
                elif re.match(r"^[+-]?\d+(?:[.,]\d+)?%.*", p) and not rate and idx_p > 0:
                    rate = p
                elif not ind and idx_p == 0:
                    ind = p
                else:
                    notes_acc.append(p)
            if any((ind, rate, kvr, kosgu, formulas_acc, notes_acc)):
                smeta_rows.append(
                    [ind, rate, kvr, kosgu, " ; ".join(formulas_acc), " | ".join(notes_acc)]
                )
        if smeta_rows:
            return smeta_hdr, smeta_rows, sheet_name, table_title

    # Standard pipe-separated Markdown table fallback
    md_rows: list[list[str]] = []
    for ln in pipe_lines:
        inner = ln.strip().strip("|")
        cells = [_clean_cell_str(c) for c in inner.split("|")]
        if cells and all(re.match(r"^[-:\s]*$", c) for c in cells if c):
            continue
        if any(cells):
            md_rows.append(cells)

    if len(md_rows) >= 2:
        return md_rows[0], md_rows[1:], sheet_name, table_title
    return None


def extract_logical_tables_and_assign_ids(
    blocks: list[ParsedBlock],
    doc_meta: DocumentMeta,
) -> list[dict[str, Any]]:
    """
    Scan `blocks` for a document, group contiguous same-sheet/continuation table blocks into unified
    logical tables, check `should_store_structured_table(file_format, row_count, char_count)`,
    assign deterministic `table_id = '<doc_id>::tbl_NN'` IN-PLACE onto eligible `ParsedBlock`s
    (`block.table_id` and `block.metadata['table_id']`), and return the list of eligible logical table specs.
    """
    doc_id = doc_meta.doc_id or "doc"
    file_format = (
        doc_meta.file_format
        or (doc_meta.file_name.rsplit(".", 1)[-1].lower() if "." in (doc_meta.file_name or "") else "")
    ).lower().lstrip(".")

    in_approval = False
    last_heading_text = ""
    current_unit_id = "u01_main"
    unit_idx = 1

    raw_candidates: list[dict[str, Any]] = []

    for blk in blocks:
        first_line = (blk.text or "").split("\n", 1)[0].strip()
        if APPROVAL_TRIGGER_RE.match(first_line):
            in_approval = True
        elif in_approval and (NORMATIVE_RESET_RE.match(first_line) or UNIT_BOUNDARY_RE.match(first_line)):
            in_approval = False

        if in_approval or blk.zone == ZoneType.APPROVAL.value:
            continue

        if UNIT_BOUNDARY_RE.match(first_line) and len(first_line) <= 160:
            if not (current_unit_id == "u01_main" and not raw_candidates and not first_line.lower().startswith("приложени")):
                unit_idx += 1
                current_unit_id = f"u{unit_idx:02d}"
            last_heading_text = first_line

        if blk.block_type == BlockType.HEADING.value and first_line:
            last_heading_text = first_line
            continue

        if _is_1x2_requisite_block(blk):
            continue

        extracted = _extract_grid_from_parsed_block(blk, file_format=file_format)
        if extracted is None:
            if blk.block_type == BlockType.PARAGRAPH.value and first_line and len(first_line) <= 160:
                if any(
                    kw in first_line.lower()
                    for kw in ("таблица", "приложение", "структура", "смета", "перечень", "расшифровка", "форма")
                ):
                    last_heading_text = first_line
            continue

        headers, rows, sheet_name, explicit_title = extracted
        inferred_title = explicit_title or last_heading_text or (f"Лист {sheet_name}" if sheet_name else "")

        # Merge contiguous table blocks belonging to the exact same Excel sheet & matching column count/headers
        if (
            raw_candidates
            and sheet_name
            and raw_candidates[-1]["sheet_name"] == sheet_name
            and len(headers) == len(raw_candidates[-1]["headers"])
            and [_norm_ru(h) for h in headers] == [_norm_ru(h) for h in raw_candidates[-1]["headers"]]
        ):
            prev = raw_candidates[-1]
            prev["rows"].extend(rows)
            prev["blocks"].append(blk)
            prev["char_count"] += len(blk.text or "")
            continue

        char_cnt = max(
            len(blk.text or ""),
            sum(sum(len(c) + 3 for c in r) for r in rows) + sum(len(h) + 3 for h in headers),
        )
        raw_candidates.append(
            {
                "headers": headers,
                "rows": rows,
                "sheet_name": sheet_name,
                "table_title": inferred_title,
                "unit_id": current_unit_id,
                "char_count": char_cnt,
                "blocks": [blk],
            }
        )

    eligible_specs: list[dict[str, Any]] = []
    tbl_seq = 0
    for cand in raw_candidates:
        row_cnt = len(cand["rows"])
        char_cnt = int(cand["char_count"])
        if row_cnt == 0:
            continue
        if not should_store_structured_table(file_format, row_count=row_cnt, char_count=char_cnt):
            # Explicitly clear any stale table_id on compact inline Word/PDF tables
            for b in cand["blocks"]:
                if not b.table_id:
                    b.table_id = ""
            continue

        tbl_seq += 1
        existing_tid = next(
            (b.table_id for b in cand["blocks"] if b.table_id and b.table_id.startswith(f"{doc_id}::")),
            "",
        )
        tid = existing_tid or f"{doc_id}::tbl_{tbl_seq:02d}"
        for b in cand["blocks"]:
            b.table_id = tid
            if b.metadata is None:
                b.metadata = {}
            b.metadata["table_id"] = tid

        cand["table_id"] = tid
        cand["file_format"] = file_format
        eligible_specs.append(cand)

    return eligible_specs


def _format_preview_markdown(columns: list[str], rows: list[dict[str, Any]], max_rows: int = 30) -> str:
    """Format a list of row dicts into a compact pipe-separated Markdown preview table."""
    if not columns or not rows:
        return ""
    shown = rows[:max_rows]
    lines: list[str] = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for r in shown:
        vals = [_clean_cell_str(r.get(col, "")).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n*... показано {max_rows} из {len(rows)} строк*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Tool Definitions
# ---------------------------------------------------------------------------
OPENAI_TABLE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_structured_table",
            "description": (
                "Детерминированный SQL/аналитический запрос к большой нормативной таблице или Excel-смете МФТИ "
                "в базе StructuredTableStore. Используйте этот инструмент для точной фильтрации строк, "
                "подсчета количества (COUNT / DISTINCT), суммирования (SUM / AVG / MIN / MAX), группировки "
                "(GROUP BY) по Финансодержателям/ЦФО/КВР/КОСГУ и извлечения формул Excel ([формула: =...])."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {
                        "type": "string",
                        "description": "Точный идентификатор таблицы из TablePassport (например, 'order_3614_1::tbl_01').",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["filter_rows", "aggregate", "get_formulas", "get_rows", "get_schema"],
                        "description": (
                            "Тип операции: 'filter_rows' — поиск/фильтрация строк; "
                            "'aggregate' — точный подсчет/сумма/группировка; "
                            "'get_formulas' — извлечение формул расчета [формула: =...]; "
                            "'get_rows' — чтение среза строк по offset/limit; "
                            "'get_schema' — паспорт и колонки таблицы."
                        ),
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Словарь фильтров по колонкам {имя_колонки: значение}. Поддерживаются частичные названия "
                            "колонок, подстроки без учета регистра и ё/е, списки значений ['111', '112'] и числовые "
                            "сравнения ('> 0', '>= 20', '= 111')."
                        ),
                        "additionalProperties": True,
                    },
                    "search_text": {
                        "type": "string",
                        "description": "Полнотекстовый поиск подстроки/ключевых слов по всей строке таблицы.",
                    },
                    "select_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список колонок для вывода (оставьте пустым для вывода всех колонок).",
                    },
                    "agg_func": {
                        "type": "string",
                        "enum": ["count", "distinct", "sum", "avg", "min", "max"],
                        "description": "Функция агрегации (для operation='aggregate'). По умолчанию 'count'.",
                    },
                    "agg_column": {
                        "type": "string",
                        "description": "Колонка для агрегации (обязательна для distinct/sum/avg/min/max).",
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Колонка группировки для подсчета/суммирования в разрезе групп (например, 'Финансодержатель' или 'ЦФО 1 уровня').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное число возвращаемых строк (по умолчанию 25, максимум 200).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Смещение для постраничного просмотра (по умолчанию 0).",
                    },
                },
                "required": ["table_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": (
                "Получить подробный паспорт таблицы (TablePassport): список всех колонок, их типы, "
                "примеры значений, наличие формул и первые строки предпросмотра."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {
                        "type": "string",
                        "description": "Идентификатор таблицы из TablePassport.",
                    },
                },
                "required": ["table_id"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# StructuredTableStore (SQLite Backend)
# ---------------------------------------------------------------------------
class StructuredTableStore:
    """
    Deterministic SQLite store (`data/tables_store.db`) for Excel workbooks and large Word/PDF tables.
    Supports schema inference, cross-file `table_hash` deduplication (`also_in_files`),
    forgiving Russian column/text filtering, numeric comparisons, aggregations, and formula inspection.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path: Path = Path(db_path) if db_path is not None else TABLES_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS tables_meta (
                        table_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        file_name TEXT NOT NULL DEFAULT '',
                        doc_number TEXT NOT NULL DEFAULT '',
                        doc_date TEXT NOT NULL DEFAULT '',
                        file_format TEXT NOT NULL DEFAULT '',
                        unit_id TEXT NOT NULL DEFAULT '',
                        section_id TEXT NOT NULL DEFAULT '',
                        section_title TEXT NOT NULL DEFAULT '',
                        sheet_name TEXT NOT NULL DEFAULT '',
                        table_title TEXT NOT NULL DEFAULT '',
                        row_count INTEGER NOT NULL DEFAULT 0,
                        col_count INTEGER NOT NULL DEFAULT 0,
                        char_count INTEGER NOT NULL DEFAULT 0,
                        columns_json TEXT NOT NULL DEFAULT '[]',
                        has_formulas INTEGER NOT NULL DEFAULT 0,
                        formula_count INTEGER NOT NULL DEFAULT 0,
                        delivery_mode TEXT NOT NULL DEFAULT 'tool_query',
                        table_hash TEXT NOT NULL DEFAULT '',
                        also_in_files_json TEXT NOT NULL DEFAULT '[]',
                        preview_rows_json TEXT NOT NULL DEFAULT '[]'
                    );

                    CREATE INDEX IF NOT EXISTS idx_tables_meta_doc_id ON tables_meta(doc_id);
                    CREATE INDEX IF NOT EXISTS idx_tables_meta_hash ON tables_meta(table_hash);

                    CREATE TABLE IF NOT EXISTS table_rows (
                        table_id TEXT NOT NULL,
                        doc_id TEXT NOT NULL,
                        row_index INTEGER NOT NULL,
                        norm_text TEXT NOT NULL DEFAULT '',
                        cells_json TEXT NOT NULL DEFAULT '{}',
                        numeric_json TEXT NOT NULL DEFAULT '{}',
                        formulas_json TEXT NOT NULL DEFAULT '{}',
                        PRIMARY KEY (table_id, row_index)
                    );

                    CREATE INDEX IF NOT EXISTS idx_table_rows_table_id ON table_rows(table_id);
                    CREATE INDEX IF NOT EXISTS idx_table_rows_doc_id ON table_rows(doc_id);
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Schema Inference & Single/Batch Table Upsert
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_columns_schema(
        columns: list[str],
        row_dicts: list[dict[str, str]],
        numeric_dicts: list[dict[str, float]],
        formula_dicts: list[dict[str, str]],
    ) -> list[TableColumnSchema]:
        schemas: list[TableColumnSchema] = []
        for idx, col_name in enumerate(columns):
            norm_col = _norm_ru(col_name)
            is_hier = any(
                kw in norm_col
                for kw in ("финансодержатель", "цфо", "направление", "поднаправление", "квр", "уровня")
            )
            non_empty_vals: list[str] = []
            num_cnt = 0
            form_cnt = 0

            for r_idx, r_cells in enumerate(row_dicts):
                v = r_cells.get(col_name, "")
                if v:
                    non_empty_vals.append(v)
                    if col_name in numeric_dicts[r_idx]:
                        num_cnt += 1
                    if col_name in formula_dicts[r_idx]:
                        form_cnt += 1

            non_empty_total = len(non_empty_vals)
            if non_empty_total == 0:
                inferred_type = "text"
            elif form_cnt > 0 and num_cnt == 0:
                inferred_type = "formula"
            elif num_cnt >= max(1, int(non_empty_total * 0.65)):
                inferred_type = "numeric"
            elif num_cnt > 0 or form_cnt > 0:
                inferred_type = "mixed"
            else:
                inferred_type = "text"

            # Collect up to 5 distinct representative sample values (truncated cleanly)
            samples: list[str] = []
            for v in non_empty_vals:
                short_v = v if len(v) <= 80 else (v[:77] + "...")
                if short_v not in samples:
                    samples.append(short_v)
                if len(samples) >= 5:
                    break

            schemas.append(
                TableColumnSchema(
                    name=col_name,
                    col_index=idx,
                    inferred_type=inferred_type,
                    non_empty_count=non_empty_total,
                    sample_values=samples,
                    is_hierarchy=is_hier,
                    has_formulas=(form_cnt > 0),
                )
            )
        return schemas

    @staticmethod
    def _reconcile_hash_dedup_in_tx(conn: sqlite3.Connection, table_hashes: set[str]) -> None:
        """Update `also_in_files_json` across all tables sharing any hash in `table_hashes`."""
        clean_hashes = {h for h in table_hashes if h}
        for th in clean_hashes:
            cur = conn.execute(
                "SELECT table_id, doc_id, file_name FROM tables_meta WHERE table_hash = ? ORDER BY table_id",
                (th,),
            )
            entries = cur.fetchall()
            if not entries:
                continue
            all_labels: list[str] = []
            for ent in entries:
                lbl = (ent["file_name"] or ent["doc_id"] or "").strip()
                if lbl and lbl not in all_labels:
                    all_labels.append(lbl)

            for ent in entries:
                own_lbl = (ent["file_name"] or ent["doc_id"] or "").strip()
                others = [lbl for lbl in all_labels if lbl != own_lbl]
                conn.execute(
                    "UPDATE tables_meta SET also_in_files_json = ? WHERE table_id = ?",
                    (json.dumps(others, ensure_ascii=False), ent["table_id"]),
                )

    def upsert_table(
        self,
        table_id: str,
        doc_id: str,
        headers: list[str],
        rows: list[list[str]],
        *,
        file_name: str = "",
        doc_number: str = "",
        doc_date: str = "",
        file_format: str = "",
        unit_id: str = "",
        section_id: str = "",
        section_title: str = "",
        sheet_name: str = "",
        table_title: str = "",
        char_count: int = 0,
        delivery_mode: str = "tool_query",
        reconcile_dedup: bool = True,
    ) -> TablePassport:
        """Upsert a single structured table and its rows into SQLite and return its hydrated `TablePassport`."""
        col_count = max(len(headers), max((len(r) for r in rows), default=0))
        columns = _deduplicate_headers(headers, col_count)

        row_dicts: list[dict[str, str]] = []
        numeric_dicts: list[dict[str, float]] = []
        formula_dicts: list[dict[str, str]] = []
        norm_texts: list[str] = []
        clean_grid_rows: list[list[str]] = []
        total_formulas = 0
        computed_chars = sum(len(c) + 3 for c in columns)

        for raw_row in rows:
            cells_list = [
                _clean_cell_str(raw_row[i]) if i < len(raw_row) else ""
                for i in range(col_count)
            ]
            if not any(cells_list):
                continue

            clean_grid_rows.append(cells_list)
            c_map: dict[str, str] = {}
            n_map: dict[str, float] = {}
            f_map: dict[str, str] = {}

            for idx, col_name in enumerate(columns):
                val = cells_list[idx]
                c_map[col_name] = val
                computed_chars += len(val) + 3
                if val:
                    num_v = _parse_numeric_value(val)
                    if num_v is not None:
                        n_map[col_name] = num_v
                    f_list = _extract_formulas_from_cell(val)
                    if f_list:
                        f_map[col_name] = " ; ".join(f_list)
                        total_formulas += len(f_list)

            row_dicts.append(c_map)
            numeric_dicts.append(n_map)
            formula_dicts.append(f_map)
            norm_texts.append(_norm_ru(" | ".join(v for v in cells_list if v)))

        actual_chars = max(int(char_count), computed_chars)
        col_schemas = self._infer_columns_schema(columns, row_dicts, numeric_dicts, formula_dicts)
        t_hash = compute_table_hash(headers, clean_grid_rows)
        preview_rows = row_dicts[:5]

        eff_title = (
            table_title.strip()
            or (f"{file_name} — лист «{sheet_name}»" if sheet_name and file_name else "")
            or (f"Лист «{sheet_name}»" if sheet_name else "")
            or section_title.strip()
            or file_name
            or table_id
        )
        eff_format = (
            file_format.strip().lower().lstrip(".")
            or (file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "")
        )

        with self._lock:
            conn = self._connect()
            try:
                old_row = conn.execute(
                    "SELECT table_hash FROM tables_meta WHERE table_id = ?",
                    (table_id,),
                ).fetchone()
                old_hash = str(old_row["table_hash"]) if old_row else ""

                conn.execute("DELETE FROM table_rows WHERE table_id = ?", (table_id,))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tables_meta (
                        table_id, doc_id, file_name, doc_number, doc_date, file_format,
                        unit_id, section_id, section_title, sheet_name, table_title,
                        row_count, col_count, char_count, columns_json,
                        has_formulas, formula_count, delivery_mode,
                        table_hash, also_in_files_json, preview_rows_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        table_id,
                        doc_id,
                        file_name,
                        doc_number,
                        doc_date,
                        eff_format,
                        unit_id,
                        section_id,
                        section_title,
                        sheet_name,
                        eff_title,
                        len(row_dicts),
                        col_count,
                        actual_chars,
                        json.dumps([c.model_dump() for c in col_schemas], ensure_ascii=False),
                        1 if total_formulas > 0 else 0,
                        total_formulas,
                        delivery_mode,
                        t_hash,
                        "[]",
                        json.dumps(preview_rows, ensure_ascii=False),
                    ),
                )

                if row_dicts:
                    conn.executemany(
                        """
                        INSERT INTO table_rows (
                            table_id, doc_id, row_index, norm_text, cells_json, numeric_json, formulas_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                table_id,
                                doc_id,
                                r_idx,
                                norm_texts[r_idx],
                                json.dumps(row_dicts[r_idx], ensure_ascii=False),
                                json.dumps(numeric_dicts[r_idx], ensure_ascii=False),
                                json.dumps(formula_dicts[r_idx], ensure_ascii=False),
                            )
                            for r_idx in range(len(row_dicts))
                        ],
                    )

                if reconcile_dedup:
                    self._reconcile_hash_dedup_in_tx(conn, {old_hash, t_hash})

                conn.commit()
            finally:
                conn.close()

        passport = self.get_table(table_id)
        if passport is None:
            raise RuntimeError(f"Failed to load TablePassport after upserting {table_id}")
        return passport

    def ingest_document_tables(
        self,
        doc_meta: DocumentMeta,
        blocks: list[ParsedBlock],
        chunks: list[ChunkRecord] | None = None,
    ) -> list[TablePassport]:
        """
        Idempotently replace all structured tables for `doc_meta.doc_id` from `blocks`:
        - Promotes ALL `.xlsx` tables/sheets and large `.docx`/`.pdf` tables (`>= 25` rows or `>= 5000` chars)
        - Assigns deterministic `table_id` (`<doc_id>::tbl_01`...) onto `blocks` (and matching `chunks` if passed)
        - Reconciles cross-file `table_hash` deduplication (`also_in_files`)
        """
        doc_id = doc_meta.doc_id
        self.delete_document_tables(doc_id)

        specs = extract_logical_tables_and_assign_ids(blocks, doc_meta)
        if not specs:
            doc_meta.tables_count = 0
            return []

        # Map table_id -> chunk hierarchy info if chunks are provided
        chunk_ctx_by_tid: dict[str, tuple[str, str, str]] = {}
        if chunks:
            for ch in chunks:
                ch_tid = ch.table_id or str((ch.metadata or {}).get("table_id") or "")
                if ch_tid and ch_tid not in chunk_ctx_by_tid:
                    chunk_ctx_by_tid[ch_tid] = (ch.unit_id, ch.section_id, ch.section_title)

        passports: list[TablePassport] = []
        hashes_to_reconcile: set[str] = set()

        for spec in specs:
            tid = spec["table_id"]
            u_id, sec_id, sec_title = chunk_ctx_by_tid.get(tid, (spec["unit_id"], "", ""))
            passport = self.upsert_table(
                table_id=tid,
                doc_id=doc_id,
                headers=spec["headers"],
                rows=spec["rows"],
                file_name=doc_meta.file_name,
                doc_number=doc_meta.doc_number,
                doc_date=doc_meta.doc_date,
                file_format=spec["file_format"],
                unit_id=u_id,
                section_id=sec_id,
                section_title=sec_title,
                sheet_name=spec["sheet_name"],
                table_title=spec["table_title"],
                char_count=spec["char_count"],
                delivery_mode="tool_query",
                reconcile_dedup=False,
            )
            hashes_to_reconcile.add(passport.table_hash)
            passports.append(passport)

        if hashes_to_reconcile:
            with self._lock:
                conn = self._connect()
                try:
                    self._reconcile_hash_dedup_in_tx(conn, hashes_to_reconcile)
                    conn.commit()
                finally:
                    conn.close()

        refreshed = self.list_tables(doc_id=doc_id)
        doc_meta.tables_count = len(refreshed)
        return refreshed

    def upsert_document_tables(
        self,
        blocks_or_meta: DocumentMeta | list[ParsedBlock],
        meta_or_blocks: list[ParsedBlock] | DocumentMeta | None = None,
        chunks: list[ChunkRecord] | None = None,
    ) -> list[TablePassport]:
        """Flexible-argument alias supporting both `(doc_meta, blocks)` and `(blocks, doc_meta)`."""
        if isinstance(blocks_or_meta, DocumentMeta):
            doc_meta = blocks_or_meta
            blocks = list(meta_or_blocks) if isinstance(meta_or_blocks, list) else []
        else:
            blocks = list(blocks_or_meta or [])
            if not isinstance(meta_or_blocks, DocumentMeta):
                raise TypeError("DocumentMeta is required in upsert_document_tables")
            doc_meta = meta_or_blocks
        return self.ingest_document_tables(doc_meta=doc_meta, blocks=blocks, chunks=chunks)

    index_document_tables = upsert_document_tables

    # ------------------------------------------------------------------
    # Deletion & Metadata Lookup
    # ------------------------------------------------------------------
    def delete_document_tables(self, doc_id: str) -> int:
        """Delete all tables and rows belonging to `doc_id` and reconcile `also_in_files` on peers."""
        if not doc_id:
            return 0
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT table_id, table_hash FROM tables_meta WHERE doc_id = ?",
                    (doc_id,),
                ).fetchall()
                if not rows:
                    return 0
                affected_hashes = {str(r["table_hash"]) for r in rows if r["table_hash"]}
                deleted_count = len(rows)

                conn.execute("DELETE FROM table_rows WHERE doc_id = ?", (doc_id,))
                conn.execute("DELETE FROM tables_meta WHERE doc_id = ?", (doc_id,))
                if affected_hashes:
                    self._reconcile_hash_dedup_in_tx(conn, affected_hashes)
                conn.commit()
                return deleted_count
            finally:
                conn.close()

    def clear_all(self) -> None:
        """Remove all structured tables and rows from SQLite."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM table_rows;")
                conn.execute("DELETE FROM tables_meta;")
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _row_to_passport(row: sqlite3.Row) -> TablePassport:
        cols_raw = json.loads(row["columns_json"] or "[]")
        also_raw = json.loads(row["also_in_files_json"] or "[]")
        prev_raw = json.loads(row["preview_rows_json"] or "[]")
        return TablePassport(
            table_id=row["table_id"],
            doc_id=row["doc_id"],
            file_name=row["file_name"],
            doc_number=row["doc_number"],
            doc_date=row["doc_date"],
            file_format=row["file_format"],
            unit_id=row["unit_id"],
            section_id=row["section_id"],
            section_title=row["section_title"],
            sheet_name=row["sheet_name"],
            table_title=row["table_title"],
            row_count=int(row["row_count"]),
            col_count=int(row["col_count"]),
            char_count=int(row["char_count"]),
            columns=[TableColumnSchema.model_validate(c) for c in cols_raw],
            has_formulas=bool(row["has_formulas"]),
            formula_count=int(row["formula_count"]),
            delivery_mode=row["delivery_mode"] or "tool_query",
            table_hash=row["table_hash"] or "",
            also_in_files=also_raw,
            preview_rows=prev_raw,
        )

    def get_table(self, table_id: str) -> TablePassport | None:
        """Fetch a single `TablePassport` by `table_id`, or return `None` if not found."""
        if not table_id:
            return None
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM tables_meta WHERE table_id = ?",
                    (table_id,),
                ).fetchone()
                return self._row_to_passport(row) if row is not None else None
            finally:
                conn.close()

    get_passport = get_table

    def list_tables(self, doc_id: str | None = None) -> list[TablePassport]:
        """List all `TablePassport`s in SQLite, optionally filtered by `doc_id`."""
        with self._lock:
            conn = self._connect()
            try:
                if doc_id:
                    rows = conn.execute(
                        "SELECT * FROM tables_meta WHERE doc_id = ? ORDER BY table_id",
                        (doc_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM tables_meta ORDER BY doc_id, table_id"
                    ).fetchall()
                return [self._row_to_passport(r) for r in rows]
            finally:
                conn.close()

    get_tables = list_tables

    def count_tables(self, doc_id: str | None = None) -> int:
        """Count structured tables in SQLite."""
        with self._lock:
            conn = self._connect()
            try:
                if doc_id:
                    r = conn.execute(
                        "SELECT COUNT(*) AS cnt FROM tables_meta WHERE doc_id = ?",
                        (doc_id,),
                    ).fetchone()
                else:
                    r = conn.execute("SELECT COUNT(*) AS cnt FROM tables_meta").fetchone()
                return int(r["cnt"]) if r else 0
            finally:
                conn.close()

    def find_tables_for_hit(
        self,
        doc_id: str,
        table_id: str = "",
        section_id: str = "",
        sheet_name: str = "",
        unit_id: str = "",
    ) -> list[TablePassport]:
        """
        Resolve `TablePassport`(s) matching a search hit:
        1) Direct `table_id` lookup if non-empty.
        2) Fallback match by `(doc_id, sheet_name)` or `(doc_id, section_id)` or single-table `.xlsx` `doc_id`.
        """
        if table_id:
            direct = self.get_table(table_id)
            if direct is not None:
                return [direct]

        if not doc_id:
            return []

        doc_tables = self.list_tables(doc_id=doc_id)
        if not doc_tables:
            return []

        if sheet_name:
            norm_sh = _norm_ru(sheet_name)
            by_sheet = [
                t for t in doc_tables
                if _norm_ru(t.sheet_name) == norm_sh or norm_sh in _norm_ru(t.table_title)
            ]
            if by_sheet:
                return by_sheet

        if section_id:
            base_sec = re.sub(r"_p\d+$", "", section_id)
            by_sec = [
                t for t in doc_tables
                if t.section_id and re.sub(r"_p\d+$", "", t.section_id) == base_sec
            ]
            if by_sec:
                return by_sec

        if unit_id:
            by_unit = [t for t in doc_tables if t.unit_id == unit_id]
            if by_unit:
                return by_unit

        if len(doc_tables) == 1 and doc_tables[0].file_format in {"xlsx", "xls", "xlsm"}:
            return doc_tables

        return []

    # ------------------------------------------------------------------
    # Column Resolution & Row Filtering Engine
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_column_name(requested: str, available_columns: list[str]) -> str | None:
        """
        Forgiving Russian column resolver:
        1) Exact normalized match (`ё->е`, lowercase)
        2) Prefix match (`'ЦФО 1'` -> `'ЦФО 1 уровня'`, `'КОСГУ'` -> `'КОСГУ / Вид затрат'`)
        3) Substring match (preferring shortest matching column name)
        """
        req_norm = _norm_ru(requested)
        if not req_norm or not available_columns:
            return None

        norm_map = [(col, _norm_ru(col)) for col in available_columns]

        for col, n_col in norm_map:
            if n_col == req_norm:
                return col

        prefix_matches = [
            col for col, n_col in norm_map
            if n_col.startswith(req_norm) or req_norm.startswith(n_col)
        ]
        if prefix_matches:
            prefix_matches.sort(key=len)
            return prefix_matches[0]

        sub_matches = [
            col for col, n_col in norm_map
            if req_norm in n_col or (len(n_col) >= 3 and n_col in req_norm)
        ]
        if sub_matches:
            sub_matches.sort(key=len)
            return sub_matches[0]

        return None

    @classmethod
    def _match_single_condition(
        cls,
        cell_str: str,
        cell_num: float | None,
        cond_val: Any,
    ) -> bool:
        """Evaluate a single filter condition (substring, list, number, comparison operator, or dict) against a cell."""
        if cond_val is None:
            return True

        if isinstance(cond_val, (list, tuple, set)):
            items = list(cond_val)
            if not items:
                return True
            return any(cls._match_single_condition(cell_str, cell_num, item) for item in items)

        if isinstance(cond_val, dict):
            # Support structured operators like {"gte": 20, "lt": 100} or {"op": ">=", "value": 20} or {"contains": "ФРКТ"}
            if "op" in cond_val and "value" in cond_val:
                return cls._match_single_condition(cell_str, cell_num, f"{cond_val['op']} {cond_val['value']}")
            op_aliases = {
                "gt": ">",
                "gte": ">=",
                "lt": "<",
                "lte": "<=",
                "eq": "=",
                "ne": "!=",
            }
            for k, v in cond_val.items():
                kl = str(k).lower()
                if kl in op_aliases:
                    if not cls._match_single_condition(cell_str, cell_num, f"{op_aliases[kl]} {v}"):
                        return False
                elif kl in {"contains", "like", "substring"}:
                    if _norm_ru(str(v)) not in _norm_ru(cell_str):
                        return False
            return True

        if isinstance(cond_val, (int, float)) and not isinstance(cond_val, bool):
            target = float(cond_val)
            if cell_num is not None:
                if abs(cell_num - target) < 1e-9:
                    return True
                if "%" in cell_str and abs((cell_num / 100.0) - target) < 1e-9:
                    return True
            return _norm_ru(f"{cond_val:g}") in _norm_ru(cell_str)

        s_cond = str(cond_val).strip()
        if not s_cond:
            return True

        m_op = _COMP_OP_RE.match(s_cond)
        if m_op:
            op, rhs = m_op.group(1), m_op.group(2).strip()
            rhs_num = _parse_numeric_value(rhs)
            if rhs_num is not None and cell_num is not None:
                candidates = [cell_num]
                if "%" in cell_str and "%" not in rhs and abs(rhs_num) < 1.0 < abs(cell_num):
                    candidates.append(cell_num / 100.0)
                if op == ">":
                    return any(c > rhs_num for c in candidates)
                if op == ">=":
                    return any(c >= rhs_num - 1e-9 for c in candidates)
                if op == "<":
                    return any(c < rhs_num for c in candidates)
                if op == "<=":
                    return any(c <= rhs_num + 1e-9 for c in candidates)
                if op in {"=", "=="}:
                    return any(abs(c - rhs_num) < 1e-9 for c in candidates) or (_norm_ru(rhs) == _norm_ru(cell_str))
                if op in {"!=", "<>"}:
                    return all(abs(c - rhs_num) >= 1e-9 for c in candidates)
            else:
                if op in {"=", "=="}:
                    return _norm_ru(rhs) == _norm_ru(cell_str) or (
                        rhs_num is None and _norm_ru(rhs) in _norm_ru(cell_str)
                    )
                if op in {"!=", "<>"}:
                    return _norm_ru(rhs) != _norm_ru(cell_str)
                return False

        # Default: forgiving normalized substring match (`ё -> е`, case-insensitive)
        norm_needle = _norm_ru(s_cond)
        norm_hay = _norm_ru(cell_str)
        if norm_needle in norm_hay:
            return True
        return False

    def _load_stored_rows(self, table_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                db_rows = conn.execute(
                    "SELECT row_index, norm_text, cells_json, numeric_json, formulas_json "
                    "FROM table_rows WHERE table_id = ? ORDER BY row_index ASC",
                    (table_id,),
                ).fetchall()
            finally:
                conn.close()

        parsed: list[dict[str, Any]] = []
        for r in db_rows:
            parsed.append(
                {
                    "row_index": int(r["row_index"]),
                    "norm_text": str(r["norm_text"] or ""),
                    "cells": json.loads(r["cells_json"] or "{}"),
                    "numerics": json.loads(r["numeric_json"] or "{}"),
                    "formulas": json.loads(r["formulas_json"] or "{}"),
                }
            )
        return parsed

    def _filter_stored_rows(
        self,
        stored_rows: list[dict[str, Any]],
        available_columns: list[str],
        filters: dict[str, Any] | None,
        search_text: str = "",
    ) -> list[dict[str, Any]]:
        """Filter stored rows deterministically using `search_text` and column `filters`."""
        norm_search = _norm_ru(search_text)
        search_tokens = [tok for tok in norm_search.split(" ") if tok] if norm_search else []

        resolved_filters: list[tuple[str | None, Any]] = []
        for raw_col, cond_val in (filters or {}).items():
            if cond_val is None or (isinstance(cond_val, str) and not cond_val.strip()):
                continue
            key_norm = _norm_ru(raw_col)
            if key_norm in {"*", "_all", "any", "text", "search"}:
                resolved_filters.append(("*", cond_val))
            else:
                col_match = self._resolve_column_name(raw_col, available_columns)
                resolved_filters.append((col_match, cond_val))

        # If any filter targeted a completely non-existent column, no rows can match
        if any(col_name is None for col_name, _ in resolved_filters):
            return []

        matched: list[dict[str, Any]] = []
        for item in stored_rows:
            r_norm = item["norm_text"]
            if search_tokens:
                if norm_search not in r_norm and not all(tok in r_norm for tok in search_tokens):
                    continue

            cells: dict[str, str] = item["cells"]
            numerics: dict[str, float] = item["numerics"]
            ok = True
            for col_name, cond_val in resolved_filters:
                if col_name == "*":
                    if _norm_ru(str(cond_val)) not in r_norm:
                        ok = False
                        break
                else:
                    assert col_name is not None
                    c_str = cells.get(col_name, "")
                    c_num = numerics.get(col_name)
                    if not self._match_single_condition(c_str, c_num, cond_val):
                        ok = False
                        break
            if ok:
                matched.append(item)

        return matched

    # ------------------------------------------------------------------
    # Deterministic Query Engine (`query_table`)
    # ------------------------------------------------------------------
    def query_table(self, req: TableQueryRequest | dict[str, Any]) -> TableQueryResponse:
        """
        Execute a deterministic structured query (`filter_rows`, `aggregate`, `get_formulas`,
        `get_rows`, or `get_schema`) against SQLite `tables_store.db`.
        """
        t0 = time.perf_counter()
        q = req if isinstance(req, TableQueryRequest) else TableQueryRequest.model_validate(req)
        op = (q.operation or "filter_rows").strip().lower()

        passport = self.get_table(q.table_id)
        if passport is None:
            return TableQueryResponse(
                table_id=q.table_id,
                table_title="",
                operation=op,
                total_table_rows=0,
                matched_count=0,
                returned_count=0,
                markdown_preview=f"Таблица '{q.table_id}' не найдена в базе таблиц.",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        all_columns = [c.name for c in passport.columns]

        # 1. Schema / Passport inspection
        if op == "get_schema":
            preview_slice = passport.preview_rows[q.offset : q.offset + q.limit]
            schema_lines = [
                f"**Таблица:** {passport.table_title} (`{passport.table_id}`)",
                f"- Строк: {passport.row_count}, Колонок: {passport.col_count}, Формул: {passport.formula_count}",
                "- Колонки:",
            ]
            for c in passport.columns:
                samp = ", ".join(c.sample_values[:3])
                schema_lines.append(
                    f"  * `{c.name}` ({c.inferred_type}, заполнено: {c.non_empty_count}): {samp}"
                )
            md_prev = "\n".join(schema_lines)
            if preview_slice:
                md_prev += "\n\n" + _format_preview_markdown(all_columns, preview_slice, max_rows=5)

            return TableQueryResponse(
                table_id=passport.table_id,
                table_title=passport.table_title,
                operation=op,
                total_table_rows=passport.row_count,
                matched_count=passport.row_count,
                returned_count=len(preview_slice),
                columns=all_columns,
                rows=preview_slice,
                aggregation_result=passport.model_dump(),
                truncated=(q.offset + len(preview_slice) < passport.row_count),
                markdown_preview=md_prev,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        stored_rows = self._load_stored_rows(passport.table_id)
        total_rows = len(stored_rows)

        # Resolve optional select_columns projection
        selected_cols: list[str] = []
        if q.select_columns:
            for req_c in q.select_columns:
                resolved_c = self._resolve_column_name(req_c, all_columns)
                if resolved_c and resolved_c not in selected_cols:
                    selected_cols.append(resolved_c)
        output_cols = selected_cols if selected_cols else all_columns

        # 2. Formula Inspection (`get_formulas`)
        if op == "get_formulas":
            formula_rows = [r for r in stored_rows if r["formulas"]]
            matched_f_rows = self._filter_stored_rows(
                formula_rows,
                available_columns=all_columns,
                filters=q.filters,
                search_text=q.search_text,
            )

            all_formula_entries: list[dict[str, Any]] = []
            label_col = all_columns[0] if all_columns else ""
            for r_item in matched_f_rows:
                cells = r_item["cells"]
                row_label = cells.get(label_col, "") or next((v for v in cells.values() if v), "")
                compact_ctx = {
                    k: v
                    for k, v in cells.items()
                    if v and ("[формула:" not in v or k == label_col)
                }
                for col_name, f_str in r_item["formulas"].items():
                    all_formula_entries.append(
                        {
                            "row_index": r_item["row_index"],
                            "row_label": row_label,
                            "column": col_name,
                            "formula": f_str,
                            "cell_text": cells.get(col_name, ""),
                            "context": compact_ctx,
                        }
                    )

            paged_rows_raw = matched_f_rows[q.offset : q.offset + q.limit]
            projected_rows = [
                {col: r["cells"].get(col, "") for col in output_cols}
                for r in paged_rows_raw
            ]
            paged_formulas = all_formula_entries[q.offset : q.offset + max(q.limit * 2, 50)]

            return TableQueryResponse(
                table_id=passport.table_id,
                table_title=passport.table_title,
                operation=op,
                total_table_rows=total_rows,
                matched_count=len(matched_f_rows),
                returned_count=len(projected_rows),
                columns=output_cols,
                rows=projected_rows,
                formulas=paged_formulas,
                markdown_preview=_format_preview_markdown(output_cols, projected_rows),
                truncated=(q.offset + len(projected_rows) < len(matched_f_rows)),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # Filter rows for `filter_rows`, `get_rows`, and `aggregate`
        matched_rows = self._filter_stored_rows(
            stored_rows,
            available_columns=all_columns,
            filters=q.filters,
            search_text=q.search_text,
        )

        # 3. Aggregation (`aggregate`)
        if op == "aggregate":
            agg_fn = (q.agg_func or "count").strip().lower()
            agg_col = (
                self._resolve_column_name(q.agg_column, all_columns)
                if q.agg_column and q.agg_column.strip() != "*"
                else None
            )
            grp_col = (
                self._resolve_column_name(q.group_by, all_columns)
                if q.group_by and q.group_by.strip()
                else None
            )

            def _compute_metric(bucket: list[dict[str, Any]]) -> Any:
                if agg_fn == "count":
                    if agg_col is None:
                        return len(bucket)
                    return sum(1 for item in bucket if item["cells"].get(agg_col, ""))
                if agg_fn in {"distinct", "unique"}:
                    target_c = agg_col or (all_columns[0] if all_columns else "")
                    vals: list[str] = []
                    for item in bucket:
                        v = item["cells"].get(target_c, "")
                        if v and v not in vals:
                            vals.append(v)
                    return vals
                if agg_fn == "count_distinct":
                    target_c = agg_col or (all_columns[0] if all_columns else "")
                    return len({item["cells"].get(target_c, "") for item in bucket if item["cells"].get(target_c, "")})

                # Numeric aggregations: sum, avg/mean, min, max
                target_c = agg_col or (all_columns[-1] if all_columns else "")
                nums: list[float] = [
                    float(item["numerics"][target_c])
                    for item in bucket
                    if target_c in item["numerics"]
                ]
                if not nums:
                    return 0.0 if agg_fn == "sum" else None
                if agg_fn == "sum":
                    return round(sum(nums), 6)
                if agg_fn in {"avg", "mean"}:
                    return round(sum(nums) / len(nums), 6)
                if agg_fn == "min":
                    return min(nums)
                if agg_fn == "max":
                    return max(nums)
                return len(bucket)

            if grp_col is not None:
                groups: dict[str, list[dict[str, Any]]] = {}
                for item in matched_rows:
                    g_key = item["cells"].get(grp_col, "").strip() or "(не указано)"
                    groups.setdefault(g_key, []).append(item)

                agg_dict: dict[str, Any] = {}
                summary_rows: list[dict[str, Any]] = []
                val_col_label = f"{agg_fn}({agg_col or '*'})"

                for g_key, g_bucket in groups.items():
                    m_val = _compute_metric(g_bucket)
                    display_val = len(m_val) if isinstance(m_val, list) else m_val
                    agg_dict[g_key] = m_val
                    row_entry: dict[str, Any] = {
                        grp_col: g_key,
                        val_col_label: display_val,
                        "rows_count": len(g_bucket),
                    }
                    if isinstance(m_val, list):
                        row_entry["distinct_values"] = ", ".join(m_val[:10])
                    summary_rows.append(row_entry)

                summary_rows.sort(
                    key=lambda x: (
                        float(x[val_col_label]) if isinstance(x[val_col_label], (int, float)) else 0.0,
                        int(x["rows_count"]),
                    ),
                    reverse=True,
                )
                paged_summary = summary_rows[q.offset : q.offset + q.limit]
                sum_cols = list(paged_summary[0].keys()) if paged_summary else [grp_col, val_col_label, "rows_count"]

                return TableQueryResponse(
                    table_id=passport.table_id,
                    table_title=passport.table_title,
                    operation=op,
                    total_table_rows=total_rows,
                    matched_count=len(matched_rows),
                    returned_count=len(paged_summary),
                    columns=sum_cols,
                    rows=paged_summary,
                    aggregation_result=agg_dict,
                    markdown_preview=_format_preview_markdown(sum_cols, paged_summary),
                    truncated=(q.offset + len(paged_summary) < len(summary_rows)),
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )

            # Ungrouped aggregation
            scalar_or_list = _compute_metric(matched_rows)
            if isinstance(scalar_or_list, list):
                target_c = agg_col or (all_columns[0] if all_columns else "Значение")
                freq: dict[str, int] = {}
                for item in matched_rows:
                    v = item["cells"].get(target_c, "")
                    if v:
                        freq[v] = freq.get(v, 0) + 1
                dist_rows = [{target_c: v, "count": freq.get(v, 0)} for v in scalar_or_list]
                paged_dist = dist_rows[q.offset : q.offset + q.limit]
                dist_cols = [target_c, "count"]
                return TableQueryResponse(
                    table_id=passport.table_id,
                    table_title=passport.table_title,
                    operation=op,
                    total_table_rows=total_rows,
                    matched_count=len(matched_rows),
                    returned_count=len(paged_dist),
                    columns=dist_cols,
                    rows=paged_dist,
                    aggregation_result=scalar_or_list,
                    markdown_preview=_format_preview_markdown(dist_cols, paged_dist),
                    truncated=(q.offset + len(paged_dist) < len(dist_rows)),
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )

            agg_summary_cols = ["agg_func", "column", "result", "matched_rows"]
            agg_summary_rows = [
                {
                    "agg_func": agg_fn,
                    "column": agg_col or "*",
                    "result": scalar_or_list,
                    "matched_rows": len(matched_rows),
                }
            ]
            return TableQueryResponse(
                table_id=passport.table_id,
                table_title=passport.table_title,
                operation=op,
                total_table_rows=total_rows,
                matched_count=len(matched_rows),
                returned_count=1,
                columns=agg_summary_cols,
                rows=agg_summary_rows,
                aggregation_result=scalar_or_list,
                markdown_preview=_format_preview_markdown(agg_summary_cols, agg_summary_rows),
                truncated=False,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # 4. Default: `filter_rows` / `get_rows`
        paged = matched_rows[q.offset : q.offset + q.limit]
        out_rows = [
            {col: item["cells"].get(col, "") for col in output_cols}
            for item in paged
        ]

        return TableQueryResponse(
            table_id=passport.table_id,
            table_title=passport.table_title,
            operation=op,
            total_table_rows=total_rows,
            matched_count=len(matched_rows),
            returned_count=len(out_rows),
            columns=output_cols,
            rows=out_rows,
            markdown_preview=_format_preview_markdown(output_cols, out_rows),
            truncated=(q.offset + len(out_rows) < len(matched_rows)),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    # ------------------------------------------------------------------
    # OpenAI Tool Call Dispatcher
    # ------------------------------------------------------------------
    def execute_tool_call(
        self,
        name: str,
        arguments: dict[str, Any] | str | None,
    ) -> tuple[dict[str, Any], TableQueryResponse | None]:
        """
        Execute an OpenAI function-calling tool (`query_structured_table` or `get_table_schema`)
        and return `(json_serializable_dict_for_llm, TableQueryResponse | None)`.
        """
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return (
                    {"error": f"Некорректный JSON аргументов инструмента {name}: {exc}"},
                    None,
                )
        elif isinstance(arguments, dict):
            args = dict(arguments)
        else:
            args = {}

        table_id = str(args.get("table_id") or "").strip()
        if not table_id:
            return ({"error": "Не указан обязательный параметр table_id."}, None)

        if name == "get_table_schema":
            resp = self.query_table(
                TableQueryRequest(
                    table_id=table_id,
                    operation="get_schema",
                    limit=5,
                )
            )
            passport = self.get_table(table_id)
            if passport is None:
                return (
                    {
                        "error": f"Таблица '{table_id}' не найдена.",
                        "available_tables": [t.table_id for t in self.list_tables()[:20]],
                    },
                    resp,
                )
            payload = {
                "table_id": passport.table_id,
                "table_title": passport.table_title,
                "file_name": passport.file_name,
                "sheet_name": passport.sheet_name,
                "row_count": passport.row_count,
                "col_count": passport.col_count,
                "has_formulas": passport.has_formulas,
                "formula_count": passport.formula_count,
                "also_in_files": passport.also_in_files,
                "columns": [c.model_dump() for c in passport.columns],
                "preview_rows": passport.preview_rows[:5],
            }
            return payload, resp

        if name == "query_structured_table":
            raw_limit = int(args.get("limit", 25) or 25)
            clamped_limit = max(1, min(200, raw_limit))
            raw_offset = max(0, int(args.get("offset", 0) or 0))
            req = TableQueryRequest(
                table_id=table_id,
                operation=str(args.get("operation") or "filter_rows"),
                filters=dict(args.get("filters") or {}),
                search_text=str(args.get("search_text") or ""),
                select_columns=list(args.get("select_columns") or []),
                agg_func=str(args.get("agg_func") or "count"),
                agg_column=str(args.get("agg_column") or ""),
                group_by=str(args.get("group_by") or ""),
                limit=clamped_limit,
                offset=raw_offset,
            )
            resp = self.query_table(req)
            result_dict: dict[str, Any] = {
                "table_id": resp.table_id,
                "table_title": resp.table_title,
                "operation": resp.operation,
                "total_table_rows": resp.total_table_rows,
                "matched_count": resp.matched_count,
                "returned_count": resp.returned_count,
                "truncated": resp.truncated,
                "columns": resp.columns,
            }
            if resp.aggregation_result is not None:
                result_dict["aggregation_result"] = resp.aggregation_result
            if resp.formulas:
                result_dict["formulas"] = resp.formulas[:40]
            if resp.rows:
                result_dict["rows"] = resp.rows
            if resp.markdown_preview:
                result_dict["markdown_preview"] = resp.markdown_preview
            return result_dict, resp

        return ({"error": f"Неизвестный инструмент: {name}"}, None)


# ---------------------------------------------------------------------------
# Singleton Factory & Module-Level Tool Helper
# ---------------------------------------------------------------------------
_STORE_SINGLETONS: dict[str, StructuredTableStore] = {}
_SINGLETON_LOCK = threading.Lock()


def get_table_store(db_path: Path | str | None = None) -> StructuredTableStore:
    """Return a thread-safe cached `StructuredTableStore` instance keyed by resolved SQLite path."""
    target = (Path(db_path) if db_path is not None else TABLES_DB_PATH).resolve()
    key = str(target)
    with _SINGLETON_LOCK:
        inst = _STORE_SINGLETONS.get(key)
        if inst is None:
            inst = StructuredTableStore(db_path=target)
            _STORE_SINGLETONS[key] = inst
        return inst


def execute_tool_call(
    name: str,
    arguments: dict[str, Any] | str | None,
    store: StructuredTableStore | None = None,
) -> tuple[dict[str, Any], TableQueryResponse | None]:
    """Module-level convenience wrapper around `StructuredTableStore.execute_tool_call`."""
    active_store = store or get_table_store()
    return active_store.execute_tool_call(name=name, arguments=arguments)
