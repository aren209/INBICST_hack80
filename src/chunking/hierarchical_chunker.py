"""V3 7-Stage Hierarchical Chunker (Parent-Child + 12% Bilateral Word-Boundary Overlap) for MIPT Normative RAG."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.chunking.glossary import (
    HEADER_TABLE_REQUISITE_RE,
    GlossaryManager,
    get_glossary_manager,
)
from src.config import (
    APPROVAL_TRIGGER_RE,
    CHUNK_OVERLAP_RATIO,
    CONTINUATION_HEADER_PREVIEW_CHARS,
    DATE_PREFIX_RE,
    DETACHED_CLAUSE_NUM_RE,
    FOOTNOTE_DEF_LINE_RE,
    FOOTNOTE_ORDER_RE,
    HYPHEN_WRAP_RE,
    IMPERATIVE_VERB_STOPWORDS,
    MAX_CLAUSE_CHARS,
    MAX_PARENT_SECTION_CHARS,
    MAX_SECTION_HEADER_CHARS,
    MAX_TABLE_FORM_CHARS,
    MIN_CLAUSE_CHARS,
    MIN_TABLE_FORM_CHARS,
    MULTI_LEVEL_FIRST_MAX,
    MULTI_LEVEL_PART_MAX,
    NON_ANSWER_ZONES,
    NORMATIVE_RESET_RE,
    PAGE_FOOTER_RE,
    PARA_OVERLAP_MAX_CHARS,
    PARA_OVERLAP_MIN_CHARS,
    SECTION_TITLE_STARTERS,
    SINGLE_LEVEL_CLAUSE_MAX,
    STATUS_RU_MAP,
    SUBITEM_RE,
    TABLE_OVERLAP_MAX_CHARS,
    TABLE_OVERLAP_MIN_CHARS,
    TABLE_SLICE_ROWS,
    UNIT_BOUNDARY_RE,
)
from src.schemas import (
    BlockType,
    ChunkRecord,
    DocumentMeta,
    ParentSectionRecord,
    ParsedBlock,
    ZoneType,
    compute_content_hash,
    make_point_uuid,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Additional Compiled Regexes for V3 7-Stage FSM
# ---------------------------------------------------------------------------

# Extended appendix / unit trigger at line start
APPENDIX_LINE_RE = re.compile(
    r"^\s*(?:"
    r"Приложени[ея]\s*(?:№|N)?\s*(?:\d+[а-яa-z]?|[а-яa-z]|к\s+приказу|к\s+распоряжению|к\s+регламенту|к\s+положению)\b|"
    r"УТВЕРЖДЕН[ОЫА]?\s*(?:приказом|распоряжением|$)|"
    r"(?:ВНУТРЕННИЙ\s+|ТЕХНИЧЕСКИЙ\s+)?РЕГЛАМЕНТ\b(?!\s*\))|"
    r"ПОЛОЖЕНИЕ\s+(?:о|об)\b|"
    r"ИНСТРУКЦИЯ\s+(?:по|о|об)\b|"
    r"ПОРЯДОК\s+(?:присвоения|открытия|формирования|проведения|по\s+определению)\b|"
    r"КАРТОЧКА\s+ПРОЕКТА\b"
    r")",
    re.IGNORECASE,
)

# Explicit "Раздел N" or Roman numeral section header
EXPLICIT_SECTION_RE = re.compile(
    r"^\s*(?:Раздел|Глава|РАЗДЕЛ|ГЛАВА)\s+(\d+|[IVXLC]+)\.?\s*(.*)$"
)

ROMAN_SECTION_RE = re.compile(
    r"^\s*([IVX]{1,5})\.\s+([А-ЯЁ][^\n;]{3,160})$"
)

# Clause candidate at start of line: captures optional "п."/"пункт", number, trailing dot, and rest
CLAUSE_LINE_RE = re.compile(
    r"^\s*(?:(п\.|пункт)\s*)?(\d+(?:\.\d+)*)(\.)?(?:\s+|$)(.*)",
    re.IGNORECASE | re.DOTALL,
)

# Order directive trigger switching u01_main into order-clause mode
ORDER_DIRECTIVE_TRIGGER_RE = re.compile(
    r"\b(?:ПРИКАЗЫВАЮ|РАСПОРЯЖАЮСЬ|ОБЯЗЫВАЮ)\s*:",
    re.IGNORECASE,
)

# Dative / directive starters that mark an order clause rather than a section title
ORDER_CLAUSE_PHRASE_RE = re.compile(
    r"(?:\bвозложить\s+на\b|\bоставляю\s+за\s+собой\b|\bдовести\s+настоящ|\bв\s+срок\s+до\b|"
    r"\bпризнать\s+утративш|\bотменить\s+действие\b|\bвнести\s+изменени|\bутвердить\s+прилагаем|"
    r"\bоткрыть\s+финансов|\bответственн(?:ый|ые)\s*:)",
    re.IGNORECASE,
)

DATIVE_ACTOR_STARTERS: tuple[str, ...] = (
    "заведующ",
    "начальник",
    "директор",
    "проректор",
    "главному бухгалтеру",
    "финансово-экономическому управлению",
    "центру трансфера",
    "административному отделу",
    "управлению",
    "отделу",
    "руководител",
    "фэу ",
    "пфо ",
    "уит ",
    "кмо ",
    "в целях ",
    "в связи с ",
    "в качестве ",
    "для оптимизации",
    "для выполнения",
    "при планировании",
    "при внесении",
    "сумма планируемого",
    "от получившегося",
    "после этого",
    "затем оставшаяся",
    "резерв отпускных",
    "расходы по договорам",
)

# Form indicators
FORM_MARKER_RE = re.compile(
    r"(?:\[формула:\s*=|\[___\]|_{5,}|ФЛС\s*___|с\s+ДД\.ММ\.ГГГГ\s+по\s+ДД\.ММ\.ГГГГ|"
    r"\bБланк\s+сметы\b|\bСМЕТА\s+ДОХОДОВ\s+И\s+РАСХОДОВ\b|\bФорма\s+заявки\b)",
    re.IGNORECASE,
)

# Footnote digit glued to ALL-CAPS heading word (e.g. "ПРОЕКТА1" -> "ПРОЕКТА")
CAPS_FOOTNOTE_RE = re.compile(r"\b([А-ЯЁ]{4,})\d(?=\s|$|[,.:;)])")

# Unicode superscript digits
UNICODE_SUPERSCRIPT_RE = re.compile(r"(№\s*\d+(?:-\d+)?)[\u00b9\u00b2\u00b3\u2070-\u2079]+")

_SUPERSCRIPT_TO_ASCII: dict[str, str] = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",
}
_UNICODE_FN_DEF_LINE_RE = re.compile(r"^\s*([\u00b9\u00b2\u00b3\u2070-\u2079])\s*\S")


def extract_page_footnote_digits(text: str) -> set[str]:
    """
    Scan lines of a page/block for explicit footnote definitions:
    - Bare single digit without dot/paren followed by uppercase start (`3 Не облагается НДФЛ...`)
    - Explicit Unicode superscript marker at line start (`³ Не облагается...`)
    """
    if not text:
        return set()
    digits: set[str] = set()
    norm = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    for line in norm.split("\n"):
        m_sup = _UNICODE_FN_DEF_LINE_RE.match(line)
        if m_sup:
            mapped = _SUPERSCRIPT_TO_ASCII.get(m_sup.group(1))
            if mapped:
                digits.add(mapped)
            continue
        m_def = FOOTNOTE_DEF_LINE_RE.match(line)
        if m_def:
            digits.add(m_def.group(1))
    return digits


# ---------------------------------------------------------------------------
# Stage 1 & 2 Helper Functions (Exported for direct unit testing)
# ---------------------------------------------------------------------------
def clean_block_text(text: str, page_footnotes: set[str] | None = None) -> str:
    """
    Stage 1 text cleaner:
    - De-hyphenates PDF word wraps across lines
    - Strips page footers ('Стр. 3 из 14', standalone page numbers)
    - Strips footnote superscripts glued to order numbers ONLY when either:
      (a) it is an explicit Unicode superscript character (`¹²³⁴⁵⁶⁷⁸⁹⁰`), OR
      (b) a matching footnote definition line exists on the same page (`page_footnotes`),
      preserving real 4-digit act numbers (`№ 1138`, `№ 1363`, `№ 1853`, `№ 3065`, `№ 3699`)!
    """
    if not text:
        return ""

    # Normalize non-breaking spaces and carriage returns
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")

    # 1. De-hyphenate wrapped Russian words across line breaks
    cleaned = HYPHEN_WRAP_RE.sub(r"\1\2", cleaned)

    # 2. Strip footnote superscripts glued to order numbers & uppercase headings
    cleaned = UNICODE_SUPERSCRIPT_RE.sub(r"\1", cleaned)
    if page_footnotes is not None:
        cleaned = FOOTNOTE_ORDER_RE.sub(
            lambda m: m.group(1) if m.group(2) in page_footnotes else m.group(0),
            cleaned,
        )
    elif "\n" in cleaned:
        fn_digits = extract_page_footnote_digits(cleaned)
        cleaned = FOOTNOTE_ORDER_RE.sub(
            lambda m: m.group(1) if m.group(2) in fn_digits else m.group(0),
            cleaned,
        )
    else:
        # Standalone 1-line helper invocation without page context (unit-test compatibility)
        cleaned = FOOTNOTE_ORDER_RE.sub(r"\1", cleaned)
    cleaned = CAPS_FOOTNOTE_RE.sub(r"\1", cleaned)

    # 3. Strip page footers line by line
    raw_lines = cleaned.split("\n")
    kept_lines: list[str] = []
    total_lines = len(raw_lines)
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        # Explicit "Стр. X из Y" or "- X -"
        if re.match(
            r"^\s*(?:Стр\.?\s*\d+\s*(?:из|/)\s*\d+|Страница\s+\d+\s+из\s+\d+|-\s*\d+\s*-)\s*$",
            stripped,
            re.IGNORECASE,
        ):
            continue
        # Standalone bottom page number (1-3 digits without trailing dot) at start/end of multi-line block or footer
        if re.match(r"^\d{1,3}$", stripped) and (total_lines == 1 or idx == 0 or idx == total_lines - 1):
            continue
        kept_lines.append(re.sub(r"[ \t]+", " ", stripped))

    result = "\n".join(kept_lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def is_header_requisite_table(block: ParsedBlock) -> bool:
    """Detect 1x2 / 2x2 Word requisite frame tables (e.g., 'от 28.03.2023 | № 794-1') for unwrapping into paragraph."""
    total_rows = (1 if block.table_headers else 0) + len(block.table_rows)
    if total_rows > 2:
        return False

    parts: list[str] = []
    if block.table_headers:
        parts.append(" | ".join(str(c).strip() for c in block.table_headers if str(c).strip()))
    for r in block.table_rows:
        parts.append(" | ".join(str(c).strip() for c in r if str(c).strip()))
    if block.text:
        parts.append(block.text.strip())

    combined = " | ".join(p for p in parts if p).strip()
    if not combined or len(combined) > 220:
        return False

    return bool(HEADER_TABLE_REQUISITE_RE.search(combined))


def is_valid_clause_number(
    num_str: str,
    rest_text: str = "",
    has_trailing_dot: bool = True,
    has_explicit_prefix: bool = False,
) -> bool:
    """
    Validate candidate clause number against V3 Date vs Clause disambiguation rules:
    - Reject leading zeros in any segment ('01.09' is a date, never clause '1.9')
    - Single-level 'N.': 1 <= parts[0] <= 300 (SINGLE_LEVEL_CLAUSE_MAX), must have trailing dot or 'п.'
    - Multi-level 'X.Y' / 'X.Y.Z': 1 <= parts[0] <= 120 (MULTI_LEVEL_FIRST_MAX),
      all(1 <= p <= 200) (MULTI_LEVEL_PART_MAX), max 4 levels,
      no 4-digit year in 3rd segment, and not followed by a year in rest_text.
    """
    if not num_str:
        return False

    raw_parts = num_str.strip(".").split(".")
    if not raw_parts or len(raw_parts) > 4:
        return False

    # Reject any segment with a leading zero like "01", "09" (dates DD.MM)
    for rp in raw_parts:
        if not rp.isdigit():
            return False
        if len(rp) > 1 and rp.startswith("0"):
            return False

    parts = [int(rp) for rp in raw_parts]
    rest_clean = rest_text.strip()

    if len(parts) == 1:
        if not (has_trailing_dot or has_explicit_prefix):
            return False
        if not (1 <= parts[0] <= SINGLE_LEVEL_CLAUSE_MAX):
            return False
        # Reject if followed immediately by date/currency/measure units
        if re.match(
            r"^(?:г\.|года|год\b|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|"
            r"руб|тыс|млн|%|ч\.|дн\.)",
            rest_clean,
            re.IGNORECASE,
        ):
            return False
        return True

    # Multi-level (X.Y, X.Y.Z, X.Y.Z.W)
    if len(parts) >= 3:
        # 3rd segment is a 4-digit year -> Date DD.MM.YYYY!
        if len(raw_parts[2]) == 4 or parts[2] >= 1900:
            return False

    if not (1 <= parts[0] <= MULTI_LEVEL_FIRST_MAX):
        return False
    if not all(1 <= p <= MULTI_LEVEL_PART_MAX for p in parts):
        return False

    # Reject if immediately followed by currency or percentage marker (e.g. "75.50 руб.", "15.5 %")
    if re.match(r"^(?:руб|тыс\.?\s*руб|млн\.?\s*руб|%)", rest_clean, re.IGNORECASE):
        return False

    # Extra guard for X.Y where X <= 31 and Y <= 12: reject if rest starts with a year or date marker
    if len(parts) == 2 and parts[0] <= 31 and parts[1] <= 12:
        if re.match(r"^(?:19\d{2}|20\d{2}|\d{2}\s*г\.?)\b", rest_clean):
            return False
        if not has_trailing_dot and not has_explicit_prefix and re.match(r"^\d{2,4}\b", rest_clean):
            return False

    return True


def parse_clause_number(line: str) -> tuple[str, str] | None:
    """
    Parse clause number and remaining text from the start of a line.
    Returns (clause_no, rest_text) or None if the line is a date or ordinary text.
    """
    if not line:
        return None
    stripped = line.strip()
    if not stripped:
        return None

    # 1. Explicit date prefix check (DD.MM.YYYY or DD.MM.YY)
    if DATE_PREFIX_RE.match(stripped):
        return None

    # 2. Match clause candidate
    m = CLAUSE_LINE_RE.match(stripped)
    if not m:
        return None

    explicit_kw = bool(m.group(1))
    num_str = m.group(2)
    has_dot = bool(m.group(3))
    rest = (m.group(4) or "").strip()

    if not is_valid_clause_number(
        num_str,
        rest_text=rest,
        has_trailing_dot=has_dot,
        has_explicit_prefix=explicit_kw,
    ):
        return None

    return num_str, rest


_parse_clause_number = parse_clause_number


def is_section_header(
    text: str,
    is_bold: bool = False,
    style_name: str = "",
    heading_level: int | None = None,
    in_order_body: bool = False,
) -> bool:
    """
    Stage 2 Level 3 disambiguator:
    Distinguishes real Parent section headers ('2. Порядок открытия ФЛС')
    from imperative order clauses ('2. Утвердить прилагаемую форму...', '3. Контроль за исполнением приказа возложить...').
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped or "\n" in stripped:
        return False
    if len(stripped) > MAX_SECTION_HEADER_CHARS:
        return False
    if stripped.endswith(";"):
        return False

    # 1. Explicit "Раздел N" / "Глава N" or Roman numeral "I. Общие положения"
    if EXPLICIT_SECTION_RE.match(stripped) or ROMAN_SECTION_RE.match(stripped):
        return True

    # 2. Inside "ПРИКАЗЫВАЮ:" order body (before appendices/reglaments), numbered items are order clauses
    if in_order_body:
        return False

    # 3. Check numbered line "N. <Title>"
    parsed = parse_clause_number(stripped)
    if parsed is None:
        return False

    num_str, body = parsed
    # Multi-level numbers ("1.1.", "2.3.1") are child clauses, never top-level parent section headers
    if "." in num_str:
        return False

    sec_num = int(num_str)
    if sec_num > 60 or not body:
        return False

    norm_body = body.lower().replace("ё", "е").strip()
    words = re.findall(r"[а-яa-z0-9\-]+", norm_body)
    if not words:
        return False
    first_word = words[0]

    # Check imperative verb stop-list and order-clause phrase patterns
    if ORDER_CLAUSE_PHRASE_RE.search(norm_body):
        return False

    if any(norm_body.startswith(prefix) for prefix in DATIVE_ACTOR_STARTERS):
        return False

    if first_word in IMPERATIVE_VERB_STOPWORDS:
        if first_word == "контроль":
            # "Контроль и ответственность" is a section title; "Контроль за исполнением..." is an order clause
            if any(
                kw in norm_body
                for kw in ("за исполнением", "исполнения", "возложить", "оставляю", "приказа", "распоряжения")
            ):
                return False
        else:
            return False

    # Reject if body has multiple sentences separated by ". "
    if re.search(r"[а-яё]{3,}\.\s+[А-ЯЁ]", body):
        return False

    # Positive signals:
    # A. Starts with SECTION_TITLE_STARTERS (or close stem)
    for starter in SECTION_TITLE_STARTERS:
        norm_starter = starter.lower().replace("ё", "е")
        if norm_body.startswith(norm_starter):
            return True
        # Also match plural/inflected starter stems (e.g., "обзор", "мероприятия", "источники", "оценка", "кадровый")
    extra_starters = (
        "обзор",
        "мероприятия",
        "количественные показатели",
        "кадровый состав",
        "оценка",
        "источники",
        "связанные проекты",
        "приложения",
        "основные понятия",
        "обязанности",
        "права",
        "режим",
        "порядок и условия",
        "правила и порядок",
    )
    if any(norm_body.startswith(st) for st in extra_starters):
        return True

    # B. Explicit Heading style or ALL-CAPS short header
    style_low = (style_name or "").lower()
    if heading_level is not None or "heading" in style_low or "заголовок" in style_low:
        if len(body) <= 140:
            return True

    if body.isupper() and 4 <= len(body) <= 140:
        return True

    # C. Bold short noun-phrase header without trailing period
    if is_bold and len(body) <= 110 and not body.endswith("."):
        return True

    return False


_is_section_header = is_section_header


# ---------------------------------------------------------------------------
# Stage 6 Overlap Helpers (Strictly Word-Boundary Aligned)
# ---------------------------------------------------------------------------
def _extract_tail_overlap(
    text: str,
    min_chars: int = PARA_OVERLAP_MIN_CHARS,
    max_chars: int = PARA_OVERLAP_MAX_CHARS,
    target_chars: int | None = None,
) -> str:
    """
    Extract a word-boundary-aligned suffix from `text` clamped to [min_chars, max_chars].
    NEVER cuts a word, number, date, or code in half.
    """
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return ""
    if len(s) <= max_chars:
        return s

    if target_chars is None:
        target_chars = int(round(len(s) * CHUNK_OVERLAP_RATIO))
    target = max(min_chars, min(max_chars, target_chars))

    # Collect all valid word-start indices `pos` where pos == 0 or (s[pos-1].isspace() and not s[pos].isspace())
    word_starts: list[int] = [0]
    for idx in range(1, len(s)):
        if s[idx - 1].isspace() and not s[idx].isspace():
            word_starts.append(idx)

    # Candidates within [min_chars, max_chars]
    in_bounds: list[tuple[int, str]] = []
    under_max: list[tuple[int, str]] = []
    over_max: list[tuple[int, str]] = []

    for pos in word_starts:
        cand = s[pos:].strip()
        if not cand:
            continue
        length = len(cand)
        if min_chars <= length <= max_chars:
            in_bounds.append((abs(length - target), cand))
        elif length < min_chars:
            under_max.append((min_chars - length, cand))
        else:
            over_max.append((length - max_chars, cand))

    if in_bounds:
        in_bounds.sort(key=lambda item: item[0])
        return in_bounds[0][1]
    if under_max:
        under_max.sort(key=lambda item: item[0])
        return under_max[0][1]
    if over_max:
        over_max.sort(key=lambda item: item[0])
        return over_max[0][1]
    return ""


def _extract_head_overlap(
    text: str,
    min_chars: int = PARA_OVERLAP_MIN_CHARS,
    max_chars: int = PARA_OVERLAP_MAX_CHARS,
    target_chars: int | None = None,
) -> str:
    """
    Extract a word-boundary-aligned prefix from `text` clamped to [min_chars, max_chars].
    NEVER cuts a word, number, date, or code in half.
    """
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return ""
    if len(s) <= max_chars:
        return s

    if target_chars is None:
        target_chars = int(round(len(s) * CHUNK_OVERLAP_RATIO))
    target = max(min_chars, min(max_chars, target_chars))

    # Collect all valid word-end indices `pos` where pos == len(s) or (not s[pos-1].isspace() and s[pos].isspace())
    word_ends: list[int] = []
    for idx in range(1, len(s)):
        if not s[idx - 1].isspace() and s[idx].isspace():
            word_ends.append(idx)
    word_ends.append(len(s))

    in_bounds: list[tuple[int, str]] = []
    under_max: list[tuple[int, str]] = []
    over_max: list[tuple[int, str]] = []

    for pos in word_ends:
        cand = s[:pos].strip()
        if not cand:
            continue
        length = len(cand)
        if min_chars <= length <= max_chars:
            in_bounds.append((abs(length - target), cand))
        elif length < min_chars:
            under_max.append((min_chars - length, cand))
        else:
            over_max.append((length - max_chars, cand))

    if in_bounds:
        in_bounds.sort(key=lambda item: item[0])
        return in_bounds[0][1]
    if under_max:
        under_max.sort(key=lambda item: item[0])
        return under_max[0][1]
    if over_max:
        over_max.sort(key=lambda item: item[0])
        return over_max[0][1]
    return ""


# ---------------------------------------------------------------------------
# Stage 4 Helpers: Long Clause Splitting & 3-Pass Short Chunk Merging
# ---------------------------------------------------------------------------
def _make_continuation_preview(text: str, max_preview: int = CONTINUATION_HEADER_PREVIEW_CHARS) -> str:
    """Extract the first <= 120 chars of a clause on a word boundary for '(Продолжение п. X: ...)'."""
    flat = re.sub(r"\s+", " ", text.strip())
    if len(flat) <= max_preview:
        return flat.rstrip(" .,;:")
    cut = flat[:max_preview]
    if not flat[max_preview].isspace() and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" .,;:")


def _split_into_atomic_segments(text: str, max_body_len: int) -> list[str]:
    """Split text into atomic segments (lines -> sentences -> words) each <= max_body_len."""
    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    segments: list[str] = []

    for line in raw_lines:
        if len(line) <= max_body_len:
            segments.append(line)
            continue
        # Split long line by sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9«\"(\-])|(?<=;)\s+", line)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= max_body_len:
                segments.append(sent)
            else:
                # Fallback: split by comma/clause or word boundary, never mid-word
                words = sent.split(" ")
                curr_w: list[str] = []
                curr_l = 0
                for w in words:
                    add_l = len(w) + (1 if curr_w else 0)
                    if curr_w and curr_l + add_l > max_body_len:
                        segments.append(" ".join(curr_w))
                        curr_w = [w]
                        curr_l = len(w)
                    else:
                        curr_w.append(w)
                        curr_l += add_l
                if curr_w:
                    segments.append(" ".join(curr_w))
    return segments


def _split_long_chunk(
    chunk: ChunkRecord,
    max_chars: int | None = None,
    min_chars: int = MIN_CLAUSE_CHARS,
) -> list[ChunkRecord]:
    """
    Stage 4 Long Clause Splitter (`flush_clause`):
    - Splits chunks > MAX_CLAUSE_CHARS (880) or > MAX_TABLE_FORM_CHARS (560 for table/form)
      strictly along line/sub-item/sentence boundaries.
    - Prepends `(Продолжение п. X: <first 120 chars>...)` to parts 2+.
    - Numbers parts `п. X (ч. 1)`, `п. X (ч. 2)`.
    - Re-attaches any final remainder < MIN_CLAUSE_CHARS (160) to the penultimate part.
    """
    if chunk.zone in NON_ANSWER_ZONES:
        return [chunk]

    is_table_or_form = chunk.zone in {ZoneType.TABLE.value, ZoneType.FORM.value} or chunk.block_type in {
        BlockType.TABLE.value,
        BlockType.FORM.value,
    }
    limit = max_chars or (MAX_TABLE_FORM_CHARS if is_table_or_form else MAX_CLAUSE_CHARS)

    text = chunk.clean_text.strip()
    if len(text) <= limit:
        return [chunk]

    base_clause = (chunk.clause_no or "").strip()
    clean_clause_label = re.sub(r"^(?:п\.|пункт)\s*", "", base_clause, flags=re.IGNORECASE).strip() or "б/н"
    preview = _make_continuation_preview(text, CONTINUATION_HEADER_PREVIEW_CHARS)
    cont_header = f"(Продолжение п. {clean_clause_label}: {preview}...)"
    max_part2_body = max(limit - len(cont_header) - 2, min_chars + 40)

    segments = _split_into_atomic_segments(text, max_part2_body)
    if len(segments) <= 1:
        return [chunk]

    parts_bodies: list[str] = []
    curr_segs: list[str] = []
    curr_len = 0

    for seg in segments:
        cap = limit if not parts_bodies else max_part2_body
        sep_len = 1 if curr_segs else 0
        if curr_segs and (curr_len + sep_len + len(seg) > cap):
            parts_bodies.append("\n".join(curr_segs))
            curr_segs = [seg]
            curr_len = len(seg)
        else:
            curr_segs.append(seg)
            curr_len += sep_len + len(seg)

    if curr_segs:
        parts_bodies.append("\n".join(curr_segs))

    if len(parts_bodies) <= 1:
        return [chunk]

    # If the final remainder is shorter than min_chars (160), re-attach it to the penultimate part
    if len(parts_bodies) >= 2 and len(parts_bodies[-1]) < min_chars:
        tail = parts_bodies.pop()
        parts_bodies[-1] = f"{parts_bodies[-1]}\n{tail}"

    if len(parts_bodies) <= 1:
        return [chunk]

    split_records: list[ChunkRecord] = []
    base_pages = sorted(set(chunk.page_numbers or [1]))
    orig_cpm = dict((chunk.metadata or {}).get("clause_page_map") or {})

    for idx, body_part in enumerate(parts_bodies, start=1):
        part_text = body_part if idx == 1 else f"{cont_header}\n{body_part}"
        part_clause_no = f"п. {clean_clause_label} (ч. {idx})"
        part_meta = dict(chunk.metadata)
        part_meta["base_clause_no"] = clean_clause_label
        part_meta["part_index"] = idx
        part_meta["parts_total"] = len(parts_bodies)
        part_cpm = dict(orig_cpm)
        if clean_clause_label and clean_clause_label != "б/н":
            part_cpm.setdefault(clean_clause_label, list(base_pages))
        part_cpm[part_clause_no] = list(base_pages)
        part_meta["clause_page_map"] = part_cpm

        rec = chunk.model_copy(
            update={
                "chunk_id": f"{chunk.chunk_id}_p{idx}",
                "point_uuid": "",
                "clause_no": part_clause_no,
                "clean_text": part_text,
                "raw_text": part_text,
                "char_count": len(part_text),
                "content_hash": compute_content_hash(part_text),
                "metadata": part_meta,
            }
        )
        rec.point_uuid = make_point_uuid(rec.chunk_id)
        split_records.append(rec)

    return split_records


flush_clause = _split_long_chunk


def _combine_clause_nos(first_no: str, second_no: str) -> str:
    """Combine two clause numbers cleanly (e.g. '1.2' + '1.3' -> '1.2, 1.3') without duplicates."""
    items: list[str] = []
    for raw in (first_no, second_no):
        for piece in (raw or "").split(","):
            p = piece.strip()
            if p and p not in items:
                items.append(p)
    return ", ".join(items)


def _major_clause_prefix(clause_no: str) -> str:
    """Return the major section prefix of a multi-level clause (e.g., '1.2' -> '1', '2.3.1' -> '2')."""
    if not clause_no:
        return ""
    first_token = clause_no.split(",")[0].strip()
    first_token = re.sub(r"^(?:п\.|пункт)\s*", "", first_token, flags=re.IGNORECASE).strip()
    if "." in first_token:
        return first_token.split(".")[0].strip()
    return ""


def _can_merge_pair(
    a: ChunkRecord,
    b: ChunkRecord,
    require_same_section: bool = True,
    max_chars: int = MAX_CLAUSE_CHARS,
) -> bool:
    """Check whether two adjacent chunks can be merged safely in Stage 4."""
    if a.doc_id != b.doc_id or a.unit_id != b.unit_id:
        return False
    if a.zone in NON_ANSWER_ZONES or b.zone in NON_ANSWER_ZONES:
        return False
    if a.zone != b.zone:
        return False

    if a.table_id and b.table_id and a.table_id != b.table_id:
        return False

    # Never merge two chunks that have the exact same non-empty clause_no (duplicate clause protection: e.g. two "2.6")
    if a.clause_no and b.clause_no and a.clause_no.strip() == b.clause_no.strip():
        return False

    if require_same_section:
        if a.section_id != b.section_id:
            return False
        # If both have multi-level clause numbers (e.g. 1.2 and 2.3), require same major prefix
        maj_a = _major_clause_prefix(a.clause_no)
        maj_b = _major_clause_prefix(b.clause_no)
        if maj_a and maj_b and maj_a != maj_b:
            return False

    limit = MAX_TABLE_FORM_CHARS if a.zone in {ZoneType.TABLE.value, ZoneType.FORM.value} else max_chars
    if len(a.clean_text) + 1 + len(b.clean_text) > limit:
        return False

    return True


def _merge_two_chunks(first: ChunkRecord, second: ChunkRecord, keep_section_of: ChunkRecord) -> ChunkRecord:
    """Merge `first` and `second` into a single ChunkRecord preserving combined clause_no and exact per-clause page_numbers."""
    merged_text = f"{first.clean_text.strip()}\n{second.clean_text.strip()}".strip()
    merged_clause = _combine_clause_nos(first.clause_no, second.clause_no)
    merged_pages = sorted(set((first.page_numbers or []) + (second.page_numbers or [])))
    merged_roles = list(dict.fromkeys((first.roles or []) + (second.roles or [])))
    merged_topics = list(dict.fromkeys((first.topics or []) + (second.topics or [])))
    merged_terms = list(dict.fromkeys((first.terms or []) + (second.terms or [])))
    merged_tid = first.table_id or second.table_id
    merged_meta = dict(keep_section_of.metadata or {})
    if merged_tid:
        merged_meta["table_id"] = merged_tid

    # Preserve exact individual clause -> page_numbers mapping across merges
    combined_cpm: dict[str, list[int]] = {}
    for src_ch in (first, second):
        src_pages = sorted(set(src_ch.page_numbers or []))
        src_cpm = (src_ch.metadata or {}).get("clause_page_map")
        if isinstance(src_cpm, dict) and src_cpm:
            for cl_k, p_list in src_cpm.items():
                cl_str = str(cl_k).strip()
                if cl_str and p_list:
                    combined_cpm[cl_str] = sorted(set(combined_cpm.get(cl_str, [])) | {int(p) for p in p_list})
        elif src_ch.clause_no and src_pages:
            cl_str = src_ch.clause_no.strip()
            if cl_str:
                combined_cpm[cl_str] = sorted(set(combined_cpm.get(cl_str, [])) | set(src_pages))
    if merged_clause and merged_pages:
        combined_cpm[merged_clause] = list(merged_pages)
    if combined_cpm:
        merged_meta["clause_page_map"] = combined_cpm

    rec = keep_section_of.model_copy(
        update={
            "clause_no": merged_clause,
            "table_id": merged_tid,
            "clean_text": merged_text,
            "raw_text": merged_text,
            "char_count": len(merged_text),
            "content_hash": compute_content_hash(merged_text),
            "page_numbers": merged_pages,
            "roles": merged_roles,
            "topics": merged_topics,
            "terms": merged_terms,
            "metadata": merged_meta,
        }
    )
    return rec


def _merge_short_chunks(
    chunks: list[ChunkRecord],
    min_chars: int = MIN_CLAUSE_CHARS,
    max_chars: int = MAX_CLAUSE_CHARS,
) -> list[ChunkRecord]:
    """
    Stage 4 3-Pass Short Chunk Merger (threshold `MIN_CLAUSE_CHARS = 160`):
    - Pass 1 (Forward within same section_id & zone): merges short chunk into the next chunk ('1.2, 1.3').
    - Pass 2 (Backward at end of section): merges trailing short chunk into previous chunk of same section ('1.4, 1.5').
    - Pass 3 (Orphan header/stamp cleanup within same unit_id): attaches remaining ultra-short orphan blocks
      to an adjacent chunk inside the same unit_id.
    """
    if len(chunks) <= 1:
        return list(chunks)

    # Identify any duplicate clause_no within the same (doc_id, unit_id) (e.g. duplicate "2.6" or "3.2")
    # so they are never merged away into a combined clause_no like "2.6, 2.7".
    clause_counts: dict[tuple[str, str, str], int] = {}
    for ch in chunks:
        if ch.clause_no and ch.zone not in NON_ANSWER_ZONES:
            key = (ch.doc_id, ch.unit_id, ch.clause_no.strip())
            clause_counts[key] = clause_counts.get(key, 0) + 1
    dup_clause_keys = {k for k, cnt in clause_counts.items() if cnt > 1}

    def _is_dup_clause(c: ChunkRecord) -> bool:
        return bool(c.clause_no and (c.doc_id, c.unit_id, c.clause_no.strip()) in dup_clause_keys)

    # Pass 1: Forward merge within same (unit_id, section_id, zone)
    pass1: list[ChunkRecord] = []
    i = 0
    n = len(chunks)
    while i < n:
        curr = chunks[i]
        if (
            curr.zone not in NON_ANSWER_ZONES
            and not _is_dup_clause(curr)
            and len(curr.clean_text.strip()) < min_chars
            and i + 1 < n
            and not _is_dup_clause(chunks[i + 1])
            and _can_merge_pair(curr, chunks[i + 1], require_same_section=True, max_chars=max_chars)
        ):
            merged = _merge_two_chunks(curr, chunks[i + 1], keep_section_of=chunks[i + 1])
            chunks[i + 1] = merged
            i += 1
            continue
        pass1.append(curr)
        i += 1

    # Pass 2: Backward merge for short chunks remaining at the end of a section (within same section_id & zone)
    pass2: list[ChunkRecord] = []
    for curr in pass1:
        if (
            curr.zone not in NON_ANSWER_ZONES
            and not _is_dup_clause(curr)
            and len(curr.clean_text.strip()) < min_chars
            and pass2
            and not _is_dup_clause(pass2[-1])
            and _can_merge_pair(pass2[-1], curr, require_same_section=True, max_chars=max_chars)
        ):
            pass2[-1] = _merge_two_chunks(pass2[-1], curr, keep_section_of=pass2[-1])
        else:
            pass2.append(curr)

    # Pass 3: Orphan stamp / header / inter-section short block cleanup within same unit_id
    pass3: list[ChunkRecord] = []
    idx = 0
    m = len(pass2)
    while idx < m:
        curr = pass2[idx]
        is_short_orphan = (
            curr.zone not in NON_ANSWER_ZONES
            and not _is_dup_clause(curr)
            and len(curr.clean_text.strip()) < min_chars
            and (not curr.clause_no or curr.block_type in {BlockType.STAMP.value, BlockType.HEADING.value} or len(curr.clean_text.strip()) < 90)
        )
        if is_short_orphan:
            # Prefer merging forward into next chunk in same unit_id, else backward into previous chunk in same unit_id
            if (
                idx + 1 < m
                and not _is_dup_clause(pass2[idx + 1])
                and _can_merge_pair(curr, pass2[idx + 1], require_same_section=False, max_chars=max_chars)
            ):
                pass2[idx + 1] = _merge_two_chunks(curr, pass2[idx + 1], keep_section_of=pass2[idx + 1])
                idx += 1
                continue
            if (
                pass3
                and not _is_dup_clause(pass3[-1])
                and _can_merge_pair(pass3[-1], curr, require_same_section=False, max_chars=max_chars)
            ):
                pass3[-1] = _merge_two_chunks(pass3[-1], curr, keep_section_of=pass3[-1])
                idx += 1
                continue
        pass3.append(curr)
        idx += 1

    return pass3


# ---------------------------------------------------------------------------
# Main V3 7-Stage Hierarchical Chunker
# ---------------------------------------------------------------------------
class HierarchicalChunker:
    """
    7-Stage V3 Structural-Hierarchical Chunker (Parent-Child + 12% Bilateral Overlap):
    - Stage 1: Zoning ('normative', 'table', 'form', 'approval', 'meta') & noise cleanup
    - Stage 2: 5-Level Hierarchy (file -> unit -> section -> clause -> sub-item)
    - Stage 3: Special slicing for tables (8 rows + repeated header + hierarchy), forms (380-560 chars), u00_meta
    - Stage 4: Min/Max size control (>880 split with continuation header; <160 3-pass merge)
    - Stage 5: Clean ParentSectionRecord assembly (up to 12,000 chars, zero overlap duplication)
    - Stage 6: 12% Bilateral word-boundary overlap isolated strictly within unit_id
    - Stage 7: Enriched bracketed embed_text & normalized content_hash
    """

    def __init__(
        self,
        glossary: GlossaryManager | None = None,
        merge_short: bool = True,
    ) -> None:
        self.glossary: GlossaryManager = glossary or get_glossary_manager()
        self.merge_short: bool = merge_short

    # Expose static helpers on instance/class as well
    _extract_tail_overlap = staticmethod(_extract_tail_overlap)
    _extract_head_overlap = staticmethod(_extract_head_overlap)
    _split_long_chunk = staticmethod(_split_long_chunk)
    _merge_short_chunks = staticmethod(_merge_short_chunks)
    is_section_header = staticmethod(is_section_header)
    parse_clause_number = staticmethod(parse_clause_number)

    def chunk_document(
        self,
        blocks: list[ParsedBlock] | DocumentMeta,
        doc_meta: DocumentMeta | list[ParsedBlock] | None = None,
        file_name: str = "",
        doc_id: str = "",
    ) -> tuple[list[ChunkRecord], list[ParentSectionRecord]]:
        """Execute all 7 stages of the V3 chunking pipeline and return (chunks, parent_sections)."""
        # Support flexible argument order: (blocks, doc_meta) or (doc_meta, blocks)
        if isinstance(blocks, DocumentMeta):
            actual_meta: DocumentMeta | None = blocks
            actual_blocks: list[ParsedBlock] = list(doc_meta) if isinstance(doc_meta, list) else []
        else:
            actual_blocks = list(blocks or [])
            actual_meta = doc_meta if isinstance(doc_meta, DocumentMeta) else None

        # Normalize dicts into ParsedBlock if needed
        typed_blocks: list[ParsedBlock] = [
            b if isinstance(b, ParsedBlock) else ParsedBlock.model_validate(b)
            for b in actual_blocks
        ]

        # Build or enrich DocumentMeta
        if actual_meta is None:
            actual_meta = self.glossary.extract_document_meta(
                typed_blocks,
                file_name=file_name,
            )
            if doc_id:
                actual_meta.doc_id = doc_id
        else:
            if doc_id and not actual_meta.doc_id:
                actual_meta.doc_id = doc_id
            if file_name and not actual_meta.file_name:
                actual_meta.file_name = file_name
            if not actual_meta.doc_id:
                extracted = self.glossary.extract_document_meta(
                    typed_blocks,
                    file_name=actual_meta.file_name or file_name,
                )
                actual_meta.doc_id = extracted.doc_id

        # Auto-learn domain abbreviations from full normative text before chunking
        norm_text_for_learning = "\n".join(
            b.text for b in typed_blocks if b.text and b.zone != ZoneType.APPROVAL.value
        )
        self.glossary.learn_from_text(norm_text_for_learning)

        # Assign deterministic table_id ('<doc_id>::tbl_01'...) onto eligible structured table/XLSX blocks
        from src.indexing.table_store import extract_logical_tables_and_assign_ids

        eligible_table_specs = extract_logical_tables_and_assign_ids(typed_blocks, actual_meta)
        actual_meta.tables_count = len(eligible_table_specs)

        # Stage 1: Zoning & Noise Cleanup
        cleaned_blocks = self._stage1_zone_and_clean(typed_blocks, actual_meta)

        # Stage 2 & Stage 3: 5-Level Hierarchy + Tables, Forms, and u00_meta
        raw_chunks = self._stage2_and_3_hierarchy_and_special_zones(cleaned_blocks, actual_meta)

        # Stage 4: Bilateral Min/Max Size Control
        sized_chunks = self._stage4_size_control(raw_chunks, actual_meta)

        # Stage 5: Clean ParentSectionRecord Assembly (STRICTLY BEFORE Stage 6 overlap!)
        parent_sections = self._stage5_build_parent_sections(sized_chunks, actual_meta)

        # Stage 6: 12% Bilateral Word-Boundary Overlap (isolated per unit_id)
        overlapped_chunks = self._stage6_apply_bilateral_overlap(sized_chunks)

        # Stage 7: Build raw_text, bracketed embed_text, and content_hash
        final_chunks = self._stage7_build_embed_text(overlapped_chunks, actual_meta)

        actual_meta.parent_count = len(parent_sections)
        actual_meta.chunk_count = len(final_chunks)

        return final_chunks, parent_sections

    # Alias methods for caller convenience
    chunk_blocks = chunk_document
    chunk = chunk_document
    process = chunk_document

    # -------------------------------------------------------------------
    # STAGE 1: Zoning & Noise Cleanup
    # -------------------------------------------------------------------
    def _stage1_zone_and_clean(
        self,
        blocks: list[ParsedBlock],
        doc_meta: DocumentMeta,
    ) -> list[ParsedBlock]:
        """
        Stage 1:
        - Unwraps 1x2 requisite header frame tables into paragraphs & updates doc_meta
        - Cleans hyphen wraps, page footers, and footnote superscripts
        - Glues detached clause numbers ('1.1.') to the following paragraph block
        - Splits multi-line paragraph blocks on structural boundaries (approval/reset/unit/section/clause)
        - Classifies zone ('normative', 'table', 'form', 'approval') with reset after ЛИСТ СОГЛАСОВАНИЯ
        """
        unwrapped: list[ParsedBlock] = []

        # Pre-scan non-table blocks per page for explicit footnote definition lines
        # so real 4-digit act numbers (e.g. № 1138, № 1363, № 1853) are never stripped on pages without footnotes.
        footnotes_by_page: dict[int, set[str]] = {}
        for raw_b in blocks:
            if raw_b.block_type == BlockType.TABLE.value or raw_b.table_rows:
                continue
            fn_set = extract_page_footnote_digits(raw_b.text or "")
            if not fn_set:
                continue
            pages_for_b = list(raw_b.page_numbers or [])
            if raw_b.page_num and raw_b.page_num not in pages_for_b:
                pages_for_b.append(raw_b.page_num)
            if not pages_for_b:
                pages_for_b = [1]
            for pg_n in pages_for_b:
                footnotes_by_page.setdefault(pg_n, set()).update(fn_set)

        for b in blocks:
            blk = b.model_copy(deep=True)
            blk_pg = blk.page_num or (blk.page_numbers[0] if blk.page_numbers else 1)
            page_fns: set[str] = set(footnotes_by_page.get(blk_pg, set()))
            for extra_pg in blk.page_numbers or []:
                page_fns.update(footnotes_by_page.get(extra_pg, set()))

            # 1. Unwrap 1x2 requisite header frame tables
            if is_header_requisite_table(blk):
                parts: list[str] = []
                if blk.table_headers:
                    parts.append(" | ".join(str(c).strip() for c in blk.table_headers if str(c).strip()))
                for row in blk.table_rows:
                    parts.append(" | ".join(str(c).strip() for c in row if str(c).strip()))
                if blk.text and blk.text.strip() not in parts:
                    parts.append(blk.text.strip())
                combined = " | ".join(p for p in parts if p)

                m_req = HEADER_TABLE_REQUISITE_RE.search(combined)
                if m_req and (blk.page_num or 1) == 1:
                    raw_d = m_req.group(1) or m_req.group(4) or ""
                    raw_n = (m_req.group(2) or m_req.group(3) or "").strip()
                    if raw_n and re.match(r"^\d", raw_n) and "_" not in raw_n and not doc_meta.doc_number:
                        doc_meta.doc_number = raw_n
                    if raw_d and doc_meta.doc_number and not doc_meta.doc_date:
                        m_d = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})$", raw_d.strip())
                        if m_d:
                            doc_meta.doc_date = f"{m_d.group(3)}-{m_d.group(2)}-{m_d.group(1)}"

                blk.block_type = BlockType.PARAGRAPH.value
                blk.zone = ZoneType.NORMATIVE.value
                blk.text = re.sub(r"\s*\|\s*", " ", combined).strip()
                blk.table_headers = []
                blk.table_rows = []

            # Prepend xml_num_prefix from Word numbering XML if not already present in text
            if blk.xml_num_prefix and blk.text:
                pref = blk.xml_num_prefix.strip()
                if pref and not blk.text.lstrip().startswith(pref):
                    blk.text = f"{pref} {blk.text.lstrip()}"

            # Clean block text with page-scoped footnote awareness
            blk.text = clean_block_text(blk.text, page_footnotes=page_fns)

            # Skip empty non-table blocks
            if not blk.text and not blk.table_rows:
                continue

            unwrapped.append(blk)

        # 2. Glue standalone detached clause numbers (e.g. "1.1." on its own block) to the next paragraph block
        glued: list[ParsedBlock] = []
        idx = 0
        while idx < len(unwrapped):
            curr = unwrapped[idx]
            if (
                curr.block_type in {BlockType.PARAGRAPH.value, BlockType.HEADING.value}
                and not curr.table_rows
                and DETACHED_CLAUSE_NUM_RE.match(curr.text.strip())
                and idx + 1 < len(unwrapped)
                and unwrapped[idx + 1].block_type == BlockType.PARAGRAPH.value
                and not unwrapped[idx + 1].table_rows
            ):
                num_tok = curr.text.strip()
                if not num_tok.endswith("."):
                    num_tok += "."
                nxt = unwrapped[idx + 1].model_copy(deep=True)
                nxt.text = f"{num_tok} {nxt.text.lstrip()}"
                nxt.page_numbers = sorted(set((curr.page_numbers or []) + (nxt.page_numbers or [])))
                glued.append(nxt)
                idx += 2
                continue
            glued.append(curr)
            idx += 1

        # 3. Expand multi-line paragraph blocks when structural triggers (ЛИСТ СОГЛАСОВАНИЯ, Приложение,
        # Markdown tables, or new clauses) appear on interior lines (common in OCR pages).
        expanded: list[ParsedBlock] = []
        for blk in glued:
            if blk.block_type in {BlockType.TABLE.value, BlockType.FORM.value} or blk.table_rows:
                expanded.append(blk)
                continue
            if "\n" not in blk.text:
                expanded.append(blk)
                continue
            expanded.extend(self._split_multiline_paragraph_block(blk))

        # 4. Run Zoning FSM (handling ЛИСТ СОГЛАСОВАНИЯ -> zone="approval" and normative reset on Приложение/УТВЕРЖДЕН)
        zoned: list[ParsedBlock] = []
        in_approval = False

        for blk in expanded:
            first_line = blk.text.split("\n", 1)[0].strip() if blk.text else ""

            if APPROVAL_TRIGGER_RE.match(first_line):
                in_approval = True
            elif in_approval and (
                NORMATIVE_RESET_RE.match(first_line)
                or UNIT_BOUNDARY_RE.match(first_line)
                or APPENDIX_LINE_RE.match(first_line)
            ):
                in_approval = False

            if in_approval:
                blk.zone = ZoneType.APPROVAL.value
                zoned.append(blk)
                continue

            # Normative / Table / Form classification
            if blk.zone == ZoneType.FORM.value or blk.block_type == BlockType.FORM.value:
                blk.zone = ZoneType.FORM.value
                blk.block_type = BlockType.FORM.value
            elif blk.zone == ZoneType.TABLE.value or blk.block_type == BlockType.TABLE.value or blk.table_rows:
                # Check if table is actually an Excel smeta form with formulas
                table_text_probe = blk.text + " " + " ".join(" ".join(r) for r in blk.table_rows[:10])
                if "[формула:" in table_text_probe:
                    blk.zone = ZoneType.FORM.value
                    blk.block_type = BlockType.FORM.value
                else:
                    blk.zone = ZoneType.TABLE.value
                    blk.block_type = BlockType.TABLE.value
            elif FORM_MARKER_RE.search(blk.text):
                blk.zone = ZoneType.FORM.value
                blk.block_type = BlockType.FORM.value
            else:
                blk.zone = ZoneType.NORMATIVE.value
                if blk.block_type not in {BlockType.HEADING.value, BlockType.STAMP.value}:
                    blk.block_type = BlockType.PARAGRAPH.value

            zoned.append(blk)

        return zoned

    def _split_multiline_paragraph_block(self, block: ParsedBlock) -> list[ParsedBlock]:
        """
        Split a multi-line paragraph block (e.g. from OCR) into sub-blocks when interior lines
        start a Markdown table, ЛИСТ СОГЛАСОВАНИЯ, unit boundary, section header, or new clause.
        Also glues detached single-line clause numbers ('1.1.') to the next line.
        """
        raw_lines = [ln.strip() for ln in block.text.split("\n") if ln.strip()]
        if len(raw_lines) <= 1:
            return [block]

        # Glue detached clause number lines inside the block first
        lines: list[str] = []
        i = 0
        while i < len(raw_lines):
            ln = raw_lines[i]
            if DETACHED_CLAUSE_NUM_RE.match(ln) and i + 1 < len(raw_lines):
                tok = ln if ln.endswith(".") else f"{ln}."
                lines.append(f"{tok} {raw_lines[i + 1]}")
                i += 2
            else:
                lines.append(ln)
                i += 1

        out_blocks: list[ParsedBlock] = []
        curr_para_lines: list[str] = []
        curr_md_table_lines: list[str] = []

        def _flush_para() -> None:
            if not curr_para_lines:
                return
            sub = block.model_copy(
                update={
                    "text": "\n".join(curr_para_lines),
                    "table_headers": [],
                    "table_rows": [],
                }
            )
            out_blocks.append(sub)
            curr_para_lines.clear()

        def _flush_md_table() -> None:
            if not curr_md_table_lines:
                return
            headers, rows = self._parse_markdown_table_lines(curr_md_table_lines)
            if rows:
                sub = block.model_copy(
                    update={
                        "text": "\n".join(curr_md_table_lines),
                        "block_type": BlockType.TABLE.value,
                        "zone": ZoneType.TABLE.value,
                        "table_headers": headers,
                        "table_rows": rows,
                    }
                )
                out_blocks.append(sub)
            else:
                curr_para_lines.extend(curr_md_table_lines)
                _flush_para()
            curr_md_table_lines.clear()

        for ln in lines:
            # Check Markdown table row (`| col1 | col2 |`)
            if ln.startswith("|") and ln.endswith("|") and ln.count("|") >= 3:
                _flush_para()
                curr_md_table_lines.append(ln)
                continue
            elif curr_md_table_lines:
                _flush_md_table()

            # Check if `ln` starts a new structural unit, approval zone, section, or clause
            starts_new_block = (
                bool(APPROVAL_TRIGGER_RE.match(ln))
                or bool(NORMATIVE_RESET_RE.match(ln))
                or bool(UNIT_BOUNDARY_RE.match(ln))
                or bool(APPENDIX_LINE_RE.match(ln))
                or parse_clause_number(ln) is not None
                or bool(EXPLICIT_SECTION_RE.match(ln))
            )
            if starts_new_block and curr_para_lines:
                _flush_para()
            curr_para_lines.append(ln)

        _flush_md_table()
        _flush_para()
        return out_blocks

    @staticmethod
    def _parse_markdown_table_lines(md_lines: list[str]) -> tuple[list[str], list[list[str]]]:
        """Parse Markdown table lines into (table_headers, table_rows)."""
        parsed_rows: list[list[str]] = []
        for ln in md_lines:
            inner = ln.strip().strip("|")
            cells = [c.strip() for c in inner.split("|")]
            # Skip separator row `|---|---|`
            if all(re.match(r"^[-:]+$", c.replace(" ", "")) for c in cells if c):
                continue
            parsed_rows.append(cells)
        if not parsed_rows:
            return [], []
        if len(parsed_rows) == 1:
            return [], parsed_rows
        return parsed_rows[0], parsed_rows[1:]

    # -------------------------------------------------------------------
    # STAGE 2 & STAGE 3: 5-Level Hierarchy + Tables, Forms, u00_meta
    # -------------------------------------------------------------------
    def _stage2_and_3_hierarchy_and_special_zones(
        self,
        blocks: list[ParsedBlock],
        doc_meta: DocumentMeta,
    ) -> list[ChunkRecord]:
        """
        Stage 2 & Stage 3 FSM:
        Builds 5-level hierarchy (file -> unit -> section -> clause -> sub-item)
        and handles special slicing for zone='table', zone='form', zone='approval', and u00_meta.
        """
        chunks: list[ChunkRecord] = []

        # Stage 3: Emit synthetic u00_meta card first if document has version/quality/edition alerts
        u00_meta_chunk = self.glossary.build_u00_meta_chunk(doc_meta)
        if u00_meta_chunk is not None:
            chunks.append(u00_meta_chunk)

        doc_id = doc_meta.doc_id

        # Hierarchy state
        unit_idx = 1
        unit_id = "u01_main"
        unit_title = "Основной текст"
        unit_has_content = False
        unit_chunk_counter = 0

        sec_counter = 0
        section_id = f"{doc_id}::{unit_id}::s00"
        section_num = "0"
        section_title = "Основной текст"

        in_order_body = False
        just_saw_approval = False

        # Active clause accumulation buffer (Level 4 & Level 5 sub-items)
        curr_clause_no: str = ""
        curr_lines: list[str] = []
        curr_pages: list[int] = []
        curr_clause_pages: list[int] = []
        curr_zone: str = ZoneType.NORMATIVE.value
        curr_block_type: str = BlockType.PARAGRAPH.value

        # Active form buffer for packing adjacent form paragraphs into 380-560 chars
        form_buffer_lines: list[str] = []
        form_buffer_tids: list[str] = []
        form_buffer_pages: list[int] = []
        form_title_header: str = ""

        # Approval buffer
        approval_lines: list[str] = []
        approval_pages: list[int] = []

        def _next_chunk_id(target_unit_id: str) -> str:
            nonlocal unit_chunk_counter
            unit_chunk_counter += 1
            return f"{doc_id}::{target_unit_id}::c{unit_chunk_counter:03d}"

        def _flush_clause_buffer() -> None:
            nonlocal curr_clause_no, curr_lines, curr_pages, curr_clause_pages, unit_has_content
            if not curr_lines:
                return
            text_val = "\n".join(curr_lines).strip()
            if text_val:
                cid = _next_chunk_id(unit_id)
                eff_clause_pages = sorted(set(curr_clause_pages)) if curr_clause_pages else (
                    sorted(set(curr_pages)) if curr_pages else [1]
                )
                eff_chunk_pages = eff_clause_pages if curr_clause_no else (
                    sorted(set(curr_pages)) if curr_pages else [1]
                )
                rec_meta: dict[str, Any] = {}
                if curr_clause_no:
                    rec_meta["clause_page_map"] = {curr_clause_no: list(eff_clause_pages)}
                rec = ChunkRecord(
                    chunk_id=cid,
                    doc_id=doc_id,
                    unit_id=unit_id,
                    unit_title=unit_title,
                    section_id=section_id,
                    section_num=section_num,
                    section_title=section_title,
                    clause_no=curr_clause_no,
                    zone=curr_zone,
                    block_type=curr_block_type,
                    clean_text=text_val,
                    raw_text=text_val,
                    page_numbers=eff_chunk_pages,
                    metadata=rec_meta,
                )
                chunks.append(rec)
                unit_has_content = True
            curr_clause_no = ""
            curr_lines = []
            curr_pages = []
            curr_clause_pages = []

        def _flush_form_buffer() -> None:
            nonlocal form_buffer_lines, form_buffer_tids, form_buffer_pages, unit_has_content
            if not form_buffer_lines:
                return
            packed_chunks = self._pack_form_lines(
                lines=form_buffer_lines,
                form_title=form_title_header or unit_title,
                doc_id=doc_id,
                unit_id=unit_id,
                unit_title=unit_title,
                section_id=section_id,
                section_num=section_num,
                section_title=section_title,
                clause_no=curr_clause_no or "Форма",
                pages=sorted(set(form_buffer_pages)) if form_buffer_pages else [1],
                next_id_fn=lambda: _next_chunk_id(unit_id),
                line_table_ids=form_buffer_tids,
            )
            if packed_chunks:
                chunks.extend(packed_chunks)
                unit_has_content = True
            form_buffer_lines = []
            form_buffer_tids = []
            form_buffer_pages = []

        def _flush_approval_buffer() -> None:
            nonlocal approval_lines, approval_pages
            if not approval_lines:
                return
            app_text = "\n".join(approval_lines).strip()
            if app_text:
                cid = _next_chunk_id(unit_id)
                chunks.append(
                    ChunkRecord(
                        chunk_id=cid,
                        doc_id=doc_id,
                        unit_id=unit_id,
                        unit_title=unit_title,
                        section_id=f"{doc_id}::{unit_id}::s_approval",
                        section_num="0",
                        section_title="Лист согласования",
                        clause_no="",
                        zone=ZoneType.APPROVAL.value,
                        block_type=BlockType.PARAGRAPH.value,
                        clean_text=app_text,
                        raw_text=app_text,
                        page_numbers=sorted(set(approval_pages)) if approval_pages else [1],
                    )
                )
            approval_lines = []
            approval_pages = []

        def _start_new_unit(boundary_title: str) -> None:
            nonlocal unit_idx, unit_id, unit_title, unit_has_content, unit_chunk_counter
            nonlocal sec_counter, section_id, section_num, section_title, in_order_body, just_saw_approval
            _flush_clause_buffer()
            _flush_form_buffer()
            _flush_approval_buffer()

            unit_idx += 1
            unit_id = f"u{unit_idx:02d}"
            unit_title = boundary_title.strip() or f"Часть {unit_idx}"
            unit_has_content = False
            unit_chunk_counter = 0

            sec_counter = 0
            section_id = f"{doc_id}::{unit_id}::s00"
            section_num = "0"
            section_title = unit_title
            in_order_body = False
            just_saw_approval = False

        for blk in blocks:
            blk_text = (blk.text or "").strip()
            blk_pages = blk.page_numbers or ([blk.page_num] if blk.page_num else [1])

            # 1. Approval zone blocks
            if blk.zone == ZoneType.APPROVAL.value:
                _flush_clause_buffer()
                _flush_form_buffer()
                approval_lines.append(blk_text)
                approval_pages.extend(blk_pages)
                just_saw_approval = True
                continue
            elif approval_lines:
                _flush_approval_buffer()

            # 2. Check Level 2 Autonomous Unit Boundary (`u01_main` -> `u02` -> `u03`...)
            first_line = blk_text.split("\n", 1)[0].strip() if blk_text else ""
            is_unit_trigger = bool(
                UNIT_BOUNDARY_RE.match(first_line) or APPENDIX_LINE_RE.match(first_line)
            )
            # Exclude inline references like "Приложения 2 к настоящему Регламенту" or "10. ПРИЛОЖЕНИЯ (перечислить):"
            if is_unit_trigger and re.match(r"^\s*\d+\.\s*ПРИЛОЖЕНИЯ", first_line, re.IGNORECASE):
                is_unit_trigger = False
            if is_unit_trigger and len(first_line) > 160:
                is_unit_trigger = False

            if is_unit_trigger:
                is_explicit_appendix = bool(re.match(r"^\s*Приложени[ея]\b", first_line, re.IGNORECASE))
                if unit_id == "u01_main" and not unit_has_content and not just_saw_approval and not is_explicit_appendix and not curr_lines:
                    # Top of standalone Reglament/Position: enrich u01_main title
                    if unit_title == "Основной текст":
                        unit_title = first_line
                        section_title = first_line
                    elif first_line not in unit_title and len(unit_title) < 90:
                        unit_title = f"{unit_title} — {first_line}"
                        section_title = unit_title
                    curr_lines.append(blk_text)
                    curr_pages.extend(blk_pages)
                    continue
                elif unit_id != "u01_main" and not unit_has_content and not curr_lines and not is_explicit_appendix:
                    # Multi-line stamp at the top of u02+ (e.g. "Приложение к приказу..." -> "УТВЕРЖДЕН" -> "Регламент")
                    if first_line not in unit_title and len(unit_title) < 120:
                        unit_title = f"{unit_title} — {first_line}"
                        section_title = unit_title
                    curr_lines.append(blk_text)
                    curr_pages.extend(blk_pages)
                    continue
                elif (
                    unit_id != "u01_main"
                    and not unit_has_content
                    and len(curr_lines) <= 3
                    and all(
                        bool(UNIT_BOUNDARY_RE.match(ln) or APPENDIX_LINE_RE.match(ln) or ln.lower().startswith("приказом"))
                        for ln in curr_lines
                    )
                    and not (is_explicit_appendix and "приложени" in unit_title.lower() and re.search(r"\d", unit_title) and re.search(r"\d", first_line) and re.findall(r"\d+[а-яa-z]?", unit_title.lower()) != re.findall(r"\d+[а-яa-z]?", first_line.lower()))
                ):
                    if first_line not in unit_title and len(unit_title) < 120:
                        unit_title = f"{unit_title} — {first_line}"
                        section_title = unit_title
                    curr_lines.append(blk_text)
                    curr_pages.extend(blk_pages)
                    continue
                else:
                    _start_new_unit(first_line)
                    curr_lines.append(blk_text)
                    curr_pages.extend(blk_pages)
                    continue

            # Track order body directive ("ПРИКАЗЫВАЮ:")
            if ORDER_DIRECTIVE_TRIGGER_RE.search(blk_text) and unit_id == "u01_main":
                in_order_body = True

            # 3. Stage 3 Special Zone: Table (`zone="table"`)
            if blk.zone == ZoneType.TABLE.value or blk.block_type == BlockType.TABLE.value:
                _flush_clause_buffer()
                _flush_form_buffer()
                table_chunks = self._slice_table_block(
                    block=blk,
                    doc_id=doc_id,
                    unit_id=unit_id,
                    unit_title=unit_title,
                    section_id=section_id,
                    section_num=section_num,
                    section_title=section_title,
                    active_clause_no=curr_clause_no,
                    next_id_fn=lambda: _next_chunk_id(unit_id),
                )
                if table_chunks:
                    chunks.extend(table_chunks)
                    unit_has_content = True
                continue

            # 4. Stage 3 Special Zone: Form (`zone="form"`)
            if blk.zone == ZoneType.FORM.value or blk.block_type == BlockType.FORM.value:
                _flush_clause_buffer()
                blk_tid = blk.table_id or str((blk.metadata or {}).get("table_id") or "")
                if blk.sheet_name and not form_title_header:
                    form_title_header = f"Лист: {blk.sheet_name}"
                if blk.table_rows:
                    # Form represented as table rows (e.g. Excel smeta with formulas)
                    if blk.table_headers:
                        hdr_line = " | ".join(str(c).strip() for c in blk.table_headers if str(c).strip())
                        if hdr_line and not form_title_header:
                            form_title_header = hdr_line
                    for r in blk.table_rows:
                        row_str = " | ".join(
                            str(c).strip()
                            for c in r
                            if c is not None and str(c).strip() and str(c).strip() != "None"
                        )
                        if row_str:
                            form_buffer_lines.append(row_str)
                            form_buffer_tids.append(blk_tid)
                elif blk_text:
                    added_lines = [ln.strip() for ln in blk_text.split("\n") if ln.strip()]
                    form_buffer_lines.extend(added_lines)
                    form_buffer_tids.extend([blk_tid] * len(added_lines))
                form_buffer_pages.extend(blk_pages)
                continue
            elif form_buffer_lines:
                _flush_form_buffer()

            # 5. Normative Paragraph / Heading / Clause / Sub-item (Level 3, 4, 5)
            if not blk_text:
                continue

            # Check Level 3: Parent Section Header (`section_id`)
            if is_section_header(
                first_line,
                is_bold=blk.is_bold,
                style_name=blk.style_name,
                heading_level=blk.heading_level,
                in_order_body=in_order_body,
            ):
                _flush_clause_buffer()
                sec_counter += 1
                m_exp = EXPLICIT_SECTION_RE.match(first_line) or ROMAN_SECTION_RE.match(first_line)
                if m_exp:
                    section_num = m_exp.group(1).strip()
                    section_title = m_exp.group(2).strip() or first_line
                else:
                    parsed_sec = parse_clause_number(first_line)
                    if parsed_sec:
                        section_num, section_title = parsed_sec[0], parsed_sec[1].strip()
                    else:
                        section_num = str(sec_counter)
                        section_title = first_line
                section_id = f"{doc_id}::{unit_id}::s{sec_counter:02d}"
                # If the block had extra lines after the section header line, keep them in the buffer
                rest_lines = [ln.strip() for ln in blk_text.split("\n")[1:] if ln.strip()]
                curr_lines.append(first_line)
                curr_Lines_rest = rest_lines
                if curr_Lines_rest:
                    curr_lines.extend(curr_Lines_rest)
                curr_pages.extend(blk_pages)
                continue

            # Check Level 4: Child Clause (`clause_no`)
            parsed_clause = parse_clause_number(first_line)
            if parsed_clause is not None:
                c_num, _ = parsed_clause
                # Flush previous clause buffer if it already had a clause or substantial preamble
                if curr_clause_no or (curr_lines and sum(len(x) for x in curr_lines) >= MIN_CLAUSE_CHARS):
                    _flush_clause_buffer()
                curr_clause_no = c_num
                curr_zone = ZoneType.NORMATIVE.value
                curr_block_type = BlockType.PARAGRAPH.value
                curr_lines.append(blk_text)
                curr_pages.extend(blk_pages)
                curr_clause_pages.extend(blk_pages)
                continue

            # Level 5: Sub-item (`а)`, `б)`, `1)`, `-`, `•`) or continuation paragraph -> accumulate in active clause
            if SUBITEM_RE.match(first_line):
                curr_lines.append(blk_text)
                curr_pages.extend(blk_pages)
                if curr_clause_no:
                    curr_clause_pages.extend(blk_pages)
                continue

            # Ordinary unnumbered paragraph (preamble, continuation of clause, or sign-off)
            curr_lines.append(blk_text)
            curr_pages.extend(blk_pages)
            if curr_clause_no:
                curr_clause_pages.extend(blk_pages)

        # Final flushes at end of document
        _flush_clause_buffer()
        _flush_form_buffer()
        _flush_approval_buffer()

        return chunks

    # -------------------------------------------------------------------
    # STAGE 3 Helpers: Table Slicing (8 rows) & Form Packing (380-560 chars)
    # -------------------------------------------------------------------
    def _slice_table_block(
        self,
        block: ParsedBlock,
        doc_id: str,
        unit_id: str,
        unit_title: str,
        section_id: str,
        section_num: str,
        section_title: str,
        active_clause_no: str,
        next_id_fn: Any,
    ) -> list[ChunkRecord]:
        """
        Stage 3 Table Slicer:
        - Slices tables into batches of `TABLE_SLICE_ROWS = 8` rows
        - Repeats `table_headers` in every slice
        - Forward-fills and repeats parent hierarchy levels (Финансодержатель -> ЦФО 1..4) in every slice
        """
        headers = [str(h).strip() for h in (block.table_headers or []) if h is not None]
        rows = list(block.table_rows or [])

        # If table was passed only as Markdown/text in block.text without populated table_rows, parse it
        if not rows and block.text:
            md_lines = [ln.strip() for ln in block.text.split("\n") if ln.strip()]
            pipe_lines = [ln for ln in md_lines if "|" in ln]
            if len(pipe_lines) >= 2:
                headers, rows = self._parse_markdown_table_lines(pipe_lines)
            else:
                # Single-block fallback
                cid = next_id_fn()
                blk_tid = block.table_id or str((block.metadata or {}).get("table_id") or "")
                return [
                    ChunkRecord(
                        chunk_id=cid,
                        doc_id=doc_id,
                        unit_id=unit_id,
                        unit_title=unit_title,
                        section_id=section_id,
                        section_num=section_num,
                        section_title=section_title,
                        clause_no=active_clause_no or "Таблица",
                        table_id=blk_tid,
                        zone=ZoneType.TABLE.value,
                        block_type=BlockType.TABLE.value,
                        clean_text=block.text.strip(),
                        raw_text=block.text.strip(),
                        page_numbers=block.page_numbers or [block.page_num or 1],
                        metadata={"table_id": blk_tid} if blk_tid else {},
                    )
                ]

        if not headers and rows:
            headers = [str(c).strip() for c in rows[0]]
            rows = rows[1:]

        header_line = " | ".join(h for h in headers if h and h != "None")
        hierarchy_prefix = " -> ".join(
            str(p).strip() for p in (block.parent_hierarchy or []) if p and str(p).strip() and str(p).strip() != "None"
        )

        # Forward-fill hierarchical tree columns (e.g. Финансодержатель -> ЦФО 1..4 in columns 0..3)
        is_cfo_tree = any("цфо" in h.lower() or "финансодержатель" in h.lower() for h in headers)
        running_tree_vals: list[str] = [""] * min(len(headers), 5)

        formatted_rows: list[tuple[str, str]] = []
        for raw_row in rows:
            cells = [
                "" if (c is None or str(c).strip() == "None") else str(c).strip()
                for c in raw_row
            ]
            if not any(cells):
                continue

            row_hierarchy = hierarchy_prefix
            if is_cfo_tree and len(cells) >= 2:
                for col_i in range(min(len(cells), len(running_tree_vals))):
                    if cells[col_i]:
                        running_tree_vals[col_i] = cells[col_i]
                        # Reset deeper levels when a higher parent changes
                        for deeper_i in range(col_i + 1, len(running_tree_vals)):
                            if not cells[deeper_i]:
                                running_tree_vals[deeper_i] = ""
                    elif running_tree_vals[col_i]:
                        cells[col_i] = running_tree_vals[col_i]
                active_tree = [v for v in running_tree_vals if v]
                if active_tree:
                    row_hierarchy = " -> ".join(dict.fromkeys(active_tree))

            # Format row with column names when compact, or pipe-separated
            row_str = " | ".join(c for c in cells if c)
            formatted_rows.append((row_hierarchy, row_str))

        if not formatted_rows and header_line:
            formatted_rows = [(hierarchy_prefix, header_line)]

        slices_out: list[ChunkRecord] = []
        total_rows = len(formatted_rows)
        total_slices = max(1, (total_rows + TABLE_SLICE_ROWS - 1) // TABLE_SLICE_ROWS)

        for slice_idx, start_r in enumerate(range(0, total_rows, TABLE_SLICE_ROWS), start=1):
            batch = formatted_rows[start_r : start_r + TABLE_SLICE_ROWS]
            slice_lines: list[str] = []

            if block.sheet_name:
                slice_lines.append(f"[Лист: {block.sheet_name}]")
            # Determine hierarchy banner for this batch
            batch_hierarchies = [h for h, _ in batch if h]
            top_hier = batch_hierarchies[0] if batch_hierarchies else hierarchy_prefix
            if top_hier:
                slice_lines.append(f"[Иерархия: {top_hier}]")
            if header_line:
                slice_lines.append(f"Шапка таблицы: {header_line}")

            for _, r_str in batch:
                slice_lines.append(r_str)

            slice_text = "\n".join(slice_lines).strip()
            cid = next_id_fn()
            base_cl = active_clause_no or "Таблица"
            cl_label = f"{base_cl} (табл. ч. {slice_idx})" if total_slices > 1 else base_cl

            blk_tid = block.table_id or str((block.metadata or {}).get("table_id") or "")
            eff_sheet = block.sheet_name or str((block.metadata or {}).get("sheet_name") or "")
            slice_meta: dict[str, Any] = {
                "table_headers": headers,
                "parent_hierarchy": top_hier,
                "sheet_name": eff_sheet,
                "slice_index": slice_idx,
                "slices_total": total_slices,
            }
            if blk_tid:
                slice_meta["table_id"] = blk_tid

            slices_out.append(
                ChunkRecord(
                    chunk_id=cid,
                    doc_id=doc_id,
                    unit_id=unit_id,
                    unit_title=unit_title,
                    section_id=section_id,
                    section_num=section_num,
                    section_title=section_title,
                    clause_no=cl_label,
                    table_id=blk_tid,
                    zone=ZoneType.TABLE.value,
                    block_type=BlockType.TABLE.value,
                    clean_text=slice_text,
                    raw_text=slice_text,
                    page_numbers=block.page_numbers or [block.page_num or 1],
                    metadata=slice_meta,
                )
            )

        return slices_out

    def _pack_form_lines(
        self,
        lines: list[str],
        form_title: str,
        doc_id: str,
        unit_id: str,
        unit_title: str,
        section_id: str,
        section_num: str,
        section_title: str,
        clause_no: str,
        pages: list[int],
        next_id_fn: Any,
        line_table_ids: list[str] | None = None,
    ) -> list[ChunkRecord]:
        """
        Stage 3 Form Packer:
        Packs form lines (Excel smeta formulas, KVR/KOSGU, application blanks) into
        compact 380-560 char blocks (`MIN_TABLE_FORM_CHARS=380`, `MAX_TABLE_FORM_CHARS=560`)
        preserving the sheet/form header in every packed chunk.
        """
        clean_pairs: list[tuple[str, str]] = []
        for idx_ln, ln in enumerate(lines):
            if not ln or not ln.strip() or ln.strip() == "None":
                continue
            cleaned_ln = re.sub(r"\bNone\b", "", ln).strip(" |")
            if cleaned_ln:
                tid_ln = (
                    line_table_ids[idx_ln]
                    if (line_table_ids is not None and idx_ln < len(line_table_ids))
                    else ""
                )
                clean_pairs.append((cleaned_ln, tid_ln))

        if not clean_pairs:
            return []

        header_prefix = f"[Форма: {form_title.strip()}]" if form_title else ""
        header_len = (len(header_prefix) + 1) if header_prefix else 0
        body_budget = max(MAX_TABLE_FORM_CHARS - header_len, 220)

        batches: list[list[tuple[str, str]]] = []
        curr_batch: list[tuple[str, str]] = []
        curr_len = 0

        for ln, tid_ln in clean_pairs:
            # If a single line exceeds body_budget, split it on word boundaries
            if len(ln) > body_budget:
                sub_segs = _split_into_atomic_segments(ln, body_budget)
            else:
                sub_segs = [ln]

            for seg in sub_segs:
                sep = 1 if curr_batch else 0
                if curr_batch and (curr_len + sep + len(seg) > body_budget):
                    batches.append(curr_batch)
                    curr_batch = [(seg, tid_ln)]
                    curr_len = len(seg)
                else:
                    curr_batch.append((seg, tid_ln))
                    curr_len += sep + len(seg)

        if curr_batch:
            if batches and curr_len < MIN_CLAUSE_CHARS and (
                sum(len(x) for x, _ in batches[-1]) + len(batches[-1]) + curr_len <= body_budget
            ):
                batches[-1].extend(curr_batch)
            else:
                batches.append(curr_batch)

        out_chunks: list[ChunkRecord] = []
        total_b = len(batches)
        for idx, batch in enumerate(batches, start=1):
            body_str = "\n".join(seg for seg, _ in batch).strip()
            full_str = f"{header_prefix}\n{body_str}".strip() if header_prefix and not body_str.startswith(header_prefix) else body_str
            cid = next_id_fn()
            cl_label = f"{clause_no} (ч. {idx})" if total_b > 1 else clause_no

            batch_tids = [t for _, t in batch if t]
            primary_tid = batch_tids[-1] if batch_tids else ""
            unique_tids = list(dict.fromkeys(batch_tids))
            f_meta: dict[str, Any] = {
                "form_title": form_title,
                "MIN_TABLE_FORM_CHARS": MIN_TABLE_FORM_CHARS,
            }
            if primary_tid:
                f_meta["table_id"] = primary_tid
            if len(unique_tids) > 1:
                f_meta["table_ids"] = unique_tids

            out_chunks.append(
                ChunkRecord(
                    chunk_id=cid,
                    doc_id=doc_id,
                    unit_id=unit_id,
                    unit_title=unit_title,
                    section_id=section_id,
                    section_num=section_num,
                    section_title=section_title,
                    clause_no=cl_label,
                    table_id=primary_tid,
                    zone=ZoneType.FORM.value,
                    block_type=BlockType.FORM.value,
                    clean_text=full_str,
                    raw_text=full_str,
                    page_numbers=pages,
                    metadata=f_meta,
                )
            )
        return out_chunks

    # -------------------------------------------------------------------
    # STAGE 4: Bilateral Min/Max Size Control
    # -------------------------------------------------------------------
    def _stage4_size_control(
        self,
        chunks: list[ChunkRecord],
        doc_meta: DocumentMeta,
    ) -> list[ChunkRecord]:
        """
        Stage 4:
        1) Split long clauses (>880 chars for normative, >560 chars for table/form) with continuation headers
        2) Run 3-pass short chunk merger (<160 chars)
        3) Re-index synthetic chunk_ids sequentially per unit_id so IDs are clean (doc_id::unit_id::c001...)
        """
        after_split: list[ChunkRecord] = []
        for ch in chunks:
            after_split.extend(_split_long_chunk(ch))

        if self.merge_short:
            after_merge = _merge_short_chunks(after_split, min_chars=MIN_CLAUSE_CHARS, max_chars=MAX_CLAUSE_CHARS)
        else:
            after_merge = after_split

        # Re-number synthetic chunk_id deterministically per unit_id so every chunk has unique <doc_id>::<unit_id>::cNNN
        unit_counters: dict[str, int] = {}
        final_stage4: list[ChunkRecord] = []
        for ch in after_merge:
            if ch.unit_id == "u00_meta" or ch.zone == ZoneType.META.value:
                ch.chunk_id = f"{doc_meta.doc_id}::u00_meta::c000"
                ch.point_uuid = make_point_uuid(ch.chunk_id)
                final_stage4.append(ch)
                continue

            cnt = unit_counters.get(ch.unit_id, 0) + 1
            unit_counters[ch.unit_id] = cnt
            new_cid = f"{doc_meta.doc_id}::{ch.unit_id}::c{cnt:03d}"
            ch.chunk_id = new_cid
            ch.point_uuid = make_point_uuid(new_cid)
            ch.char_count = len(ch.clean_text)
            ch.content_hash = compute_content_hash(ch.clean_text)
            final_stage4.append(ch)

        return final_stage4

    # -------------------------------------------------------------------
    # STAGE 5: Clean ParentSectionRecord Assembly (Before Stage 6 Overlap!)
    # -------------------------------------------------------------------
    def _stage5_build_parent_sections(
        self,
        chunks: list[ChunkRecord],
        doc_meta: DocumentMeta,
    ) -> list[ParentSectionRecord]:
        """
        Stage 5:
        Assemble clean ParentSectionRecord objects strictly AFTER Stage 4 size control
        and BEFORE Stage 6 bilateral overlap (zero overlap duplication).
        Segments cleanly if a section exceeds MAX_PARENT_SECTION_CHARS = 12,000.
        """
        section_groups: dict[str, list[ChunkRecord]] = {}
        section_order: list[str] = []

        for ch in chunks:
            if ch.zone in NON_ANSWER_ZONES:
                continue
            sec_id = ch.section_id or f"{doc_meta.doc_id}::{ch.unit_id}::s00"
            ch.section_id = sec_id
            if sec_id not in section_groups:
                section_groups[sec_id] = []
                section_order.append(sec_id)
            section_groups[sec_id].append(ch)

        parent_records: list[ParentSectionRecord] = []

        for sec_id in section_order:
            sec_chunks = section_groups[sec_id]
            if not sec_chunks:
                continue

            first_ch = sec_chunks[0]
            sec_header = ""
            if first_ch.section_num and first_ch.section_num != "0" and first_ch.section_title:
                sec_header = f"{first_ch.section_num}. {first_ch.section_title}"
            elif first_ch.section_title:
                sec_header = first_ch.section_title

            # Segment into <= MAX_PARENT_SECTION_CHARS (12,000) buckets
            batches: list[list[ChunkRecord]] = []
            curr_b: list[ChunkRecord] = []
            curr_chars = len(sec_header) + 1 if sec_header else 0

            for ch in sec_chunks:
                ch_len = len(ch.clean_text) + 2
                if curr_b and (curr_chars + ch_len > MAX_PARENT_SECTION_CHARS):
                    batches.append(curr_b)
                    curr_b = [ch]
                    curr_chars = (len(sec_header) + 1 if sec_header else 0) + ch_len
                else:
                    curr_b.append(ch)
                    curr_chars += ch_len
            if curr_b:
                batches.append(curr_b)

            edo_number = str((doc_meta.metadata or {}).get("edo_number") or getattr(doc_meta, "edo_number", "") or "").strip()
            edo_date = str((doc_meta.metadata or {}).get("edo_date") or getattr(doc_meta, "edo_date", "") or "").strip()
            is_edo_stamp = bool((doc_meta.metadata or {}).get("is_edo_stamp", getattr(doc_meta, "is_edo_stamp", False)))

            total_parts = len(batches)
            for p_idx, batch in enumerate(batches, start=1):
                actual_sec_id = sec_id if total_parts == 1 else f"{sec_id}_p{p_idx}"
                actual_sec_title = (
                    first_ch.section_title
                    if total_parts == 1
                    else f"{first_ch.section_title} (ч. {p_idx})"
                )

                unique_sec_pages = sorted({int(p) for ch in batch for p in (ch.page_numbers or []) if p})
                is_multipage = len(unique_sec_pages) > 1

                text_pieces: list[str] = []
                if sec_header and not batch[0].clean_text.startswith(sec_header[:30]):
                    text_pieces.append(sec_header)

                batch_chunk_ids: list[str] = []
                batch_pages: list[int] = []
                clause_page_map: dict[str, list[int]] = {}
                chunk_pages: list[dict[str, Any]] = []
                active_page: int | None = None

                def _register_clause_pages(cl_key: str, pages_list: list[int]) -> None:
                    ck = cl_key.strip()
                    if not ck or ck == "META" or not pages_list:
                        return
                    clause_page_map[ck] = sorted(set(clause_page_map.get(ck, [])) | {int(x) for x in pages_list if x})
                    # Also register clean base clause key without 'п. ' or '(ч. N)' / '(табл. ч. N)'
                    base_ck = re.sub(r"^(?:п\.|пункт)\s*", "", ck, flags=re.IGNORECASE).strip()
                    base_ck = re.sub(r"\s*\((?:табл\.\s*)?ч\.\s*\d+\)\s*$", "", base_ck, flags=re.IGNORECASE).strip()
                    if base_ck and base_ck != ck and "," not in base_ck and base_ck not in {"б/н", "Таблица", "Форма"}:
                        clause_page_map[base_ck] = sorted(
                            set(clause_page_map.get(base_ck, [])) | {int(x) for x in pages_list if x}
                        )

                for ch in batch:
                    ch.section_id = actual_sec_id
                    ch_pages = sorted({int(p) for p in (ch.page_numbers or []) if p}) or [1]
                    batch_chunk_ids.append(ch.chunk_id)
                    batch_pages.extend(ch_pages)
                    chunk_pages.append(
                        {
                            "chunk_id": ch.chunk_id,
                            "clause_no": ch.clause_no,
                            "page_numbers": list(ch_pages),
                        }
                    )

                    ch_cpm = (ch.metadata or {}).get("clause_page_map")
                    if isinstance(ch_cpm, dict) and ch_cpm:
                        for c_k, p_vals in ch_cpm.items():
                            if p_vals:
                                _register_clause_pages(str(c_k), [int(x) for x in p_vals if x])
                    if ch.clause_no:
                        _register_clause_pages(ch.clause_no, ch_pages)

                    if not is_multipage:
                        text_pieces.append(ch.clean_text)
                        continue

                    # Multi-page section: annotate exact page transitions subtly as '[Стр. N]'
                    if len(ch_pages) > 1 and isinstance(ch_cpm, dict) and ch_cpm:
                        annotated_lines: list[str] = []
                        for ln in ch.clean_text.split("\n"):
                            parsed_ln = parse_clause_number(ln.strip())
                            if parsed_ln is not None:
                                ln_cl = parsed_ln[0]
                                ln_pages = ch_cpm.get(ln_cl)
                                if ln_pages:
                                    p0 = int(ln_pages[0])
                                    if active_page != p0:
                                        annotated_lines.append(f"[Стр. {p0}]")
                                        active_page = int(ln_pages[-1])
                            elif active_page is None:
                                annotated_lines.append(f"[Стр. {ch_pages[0]}]")
                                active_page = ch_pages[0]
                            annotated_lines.append(ln)
                        active_page = ch_pages[-1]
                        text_pieces.append("\n".join(annotated_lines))
                    else:
                        if active_page != ch_pages[0]:
                            text_pieces.append(f"[Стр. {ch_pages[0]}]\n{ch.clean_text}")
                        else:
                            text_pieces.append(ch.clean_text)
                        active_page = ch_pages[-1]

                clean_parent_text = "\n\n".join(text_pieces).strip()
                if len(clean_parent_text) > MAX_PARENT_SECTION_CHARS:
                    clean_parent_text = _extract_head_overlap(
                        clean_parent_text,
                        min_chars=1000,
                        max_chars=MAX_PARENT_SECTION_CHARS,
                        target_chars=MAX_PARENT_SECTION_CHARS,
                    )

                parent_records.append(
                    ParentSectionRecord(
                        section_id=actual_sec_id,
                        doc_id=doc_meta.doc_id,
                        unit_id=first_ch.unit_id,
                        unit_title=first_ch.unit_title,
                        section_num=first_ch.section_num,
                        section_title=actual_sec_title,
                        clean_text=clean_parent_text,
                        char_count=len(clean_parent_text),
                        chunk_ids=batch_chunk_ids,
                        page_numbers=sorted(set(batch_pages)),
                        metadata={
                            "file_name": doc_meta.file_name,
                            "doc_number": doc_meta.doc_number,
                            "doc_date": doc_meta.doc_date,
                            "edo_number": edo_number,
                            "edo_date": edo_date,
                            "is_edo_stamp": is_edo_stamp,
                            "clause_page_map": clause_page_map,
                            "chunk_pages": chunk_pages,
                        },
                    )
                )

        return parent_records

    # -------------------------------------------------------------------
    # STAGE 6: 12% Bilateral Word-Boundary Overlap & Unit Isolation
    # -------------------------------------------------------------------
    def _stage6_apply_bilateral_overlap(
        self,
        chunks: list[ChunkRecord],
    ) -> list[ChunkRecord]:
        """
        Stage 6:
        - Applies 12% bilateral word-boundary overlap (35-95 chars for paragraph, 24-48 chars for table/form)
        - Prepends `...<tail of i-1>` and appends `<head of i+1>...` into `raw_text` while keeping `clean_text` pure
        - Strictly isolates `prev_id`, `next_id`, and overlap within the same `unit_id` (and excludes NON_ANSWER_ZONES)
        """
        unit_groups: dict[str, list[ChunkRecord]] = {}
        for ch in chunks:
            if ch.zone in NON_ANSWER_ZONES or ch.unit_id == "u00_meta":
                ch.prev_id = None
                ch.next_id = None
                ch.overlap_prefix = ""
                ch.overlap_suffix = ""
                ch.raw_text = ch.clean_text
                continue
            unit_groups.setdefault(ch.unit_id, []).append(ch)

        for _, u_chunks in unit_groups.items():
            count = len(u_chunks)
            for i, ch in enumerate(u_chunks):
                is_tbl_form = ch.zone in {ZoneType.TABLE.value, ZoneType.FORM.value} or ch.block_type in {
                    BlockType.TABLE.value,
                    BlockType.FORM.value,
                }
                min_ov = TABLE_OVERLAP_MIN_CHARS if is_tbl_form else PARA_OVERLAP_MIN_CHARS
                max_ov = TABLE_OVERLAP_MAX_CHARS if is_tbl_form else PARA_OVERLAP_MAX_CHARS

                # Previous chunk within the SAME unit_id
                if i > 0:
                    prev_ch = u_chunks[i - 1]
                    ch.prev_id = prev_ch.chunk_id
                    target_prev = int(round(len(prev_ch.clean_text) * CHUNK_OVERLAP_RATIO))
                    tail_slice = _extract_tail_overlap(
                        prev_ch.clean_text,
                        min_chars=min_ov,
                        max_chars=max_ov,
                        target_chars=target_prev,
                    )
                    ch.overlap_prefix = f"...{tail_slice}" if tail_slice else ""
                else:
                    ch.prev_id = None
                    ch.overlap_prefix = ""

                # Next chunk within the SAME unit_id
                if i + 1 < count:
                    next_ch = u_chunks[i + 1]
                    ch.next_id = next_ch.chunk_id
                    target_next = int(round(len(next_ch.clean_text) * CHUNK_OVERLAP_RATIO))
                    head_slice = _extract_head_overlap(
                        next_ch.clean_text,
                        min_chars=min_ov,
                        max_chars=max_ov,
                        target_chars=target_next,
                    )
                    ch.overlap_suffix = f"{head_slice}..." if head_slice else ""
                else:
                    ch.next_id = None
                    ch.overlap_suffix = ""

                # Assemble raw_text with bilateral overlap while keeping clean_text untouched
                raw_parts: list[str] = []
                if ch.overlap_prefix:
                    raw_parts.append(ch.overlap_prefix)
                raw_parts.append(ch.clean_text)
                if ch.overlap_suffix:
                    raw_parts.append(ch.overlap_suffix)
                ch.raw_text = "\n".join(raw_parts)

        return chunks

    # -------------------------------------------------------------------
    # STAGE 7: Enriched Bracketed `embed_text` & `content_hash`
    # -------------------------------------------------------------------
    def _stage7_build_embed_text(
        self,
        chunks: list[ChunkRecord],
        doc_meta: DocumentMeta,
    ) -> list[ChunkRecord]:
        """
        Stage 7:
        Populates roles (12 ROLE_PATTERNS), topics, glossary terms, content_hash, and exact bracketed `embed_text`:
        [Документ: <Название> (№ <номер> от <дата>) | Редакция/Статус: <status_ru> | Годы: <applicable_years> | Часть: <unit_title> | Раздел <N>: <section_title> | Пункт: <clause_no> | Роли: <roles> | Темы: <topics>]
        [Термины: <АББР (расшифровка)>; ...]
        <raw_text>
        """
        doc_title = doc_meta.title or doc_meta.file_name or doc_meta.doc_id
        doc_num = doc_meta.doc_number or "б/н"
        doc_date = doc_meta.doc_date or "б/д"
        edo_number = str((doc_meta.metadata or {}).get("edo_number") or getattr(doc_meta, "edo_number", "") or "").strip()
        edo_date = str((doc_meta.metadata or {}).get("edo_date") or getattr(doc_meta, "edo_date", "") or "").strip()
        is_edo_stamp = bool((doc_meta.metadata or {}).get("is_edo_stamp", getattr(doc_meta, "is_edo_stamp", False)))
        edo_header_seg = f"Зарег. в ЭДО {edo_date or 'б/д'} № {edo_number} | " if edo_number else ""
        status_ru = STATUS_RU_MAP.get(doc_meta.status, doc_meta.status)
        years_str = (
            ", ".join(str(y) for y in doc_meta.applicable_years)
            if doc_meta.applicable_years
            else "не ограничено"
        )

        for ch in chunks:
            # Tag roles, topics, and glossary terms on clean_text + context
            context_probe = f"{ch.unit_title} {ch.section_title} {ch.clean_text}"
            ch.roles = self.glossary.extract_roles(context_probe)
            ch.topics = self.glossary.extract_topics(context_probe)
            ch.terms = self.glossary.extract_terms(ch.clean_text, auto_learn=False)

            if doc_meta.also_in_files and not ch.also_in_files:
                ch.also_in_files = list(doc_meta.also_in_files)

            ch.content_hash = compute_content_hash(ch.clean_text)
            ch.char_count = len(ch.clean_text)
            ch.point_uuid = make_point_uuid(ch.chunk_id)

            sec_num_label = ch.section_num or "0"
            sec_title_label = ch.section_title or "Основной текст"
            clause_label = ch.clause_no or "б/н"
            roles_str = ", ".join(ch.roles) if ch.roles else "общие"
            topics_str = ", ".join(ch.topics) if ch.topics else "общие положения"
            terms_str = "; ".join(ch.terms) if ch.terms else "нет"

            ch.embed_text = (
                f"[Документ: {doc_title} (№ {doc_num} от {doc_date}) | "
                f"{edo_header_seg}"
                f"Редакция/Статус: {status_ru} | Годы: {years_str} | "
                f"Часть: {ch.unit_title} | Раздел {sec_num_label}: {sec_title_label} | "
                f"Пункт: {clause_label} | Роли: {roles_str} | Темы: {topics_str}]\n"
                f"[Термины: {terms_str}]\n"
                f"{ch.raw_text}"
            )

            # Enrich chunk metadata for Qdrant payload filtering & UI inspection
            ch.metadata.update(
                {
                    "file_name": doc_meta.file_name,
                    "doc_number": doc_meta.doc_number,
                    "doc_date": doc_meta.doc_date,
                    "edo_number": edo_number,
                    "edo_date": edo_date,
                    "is_edo_stamp": is_edo_stamp,
                    "doc_type": doc_meta.doc_type,
                    "title": doc_meta.title,
                    "status": doc_meta.status,
                    "applicable_years": list(doc_meta.applicable_years),
                    "supersedes": list(doc_meta.supersedes),
                    "superseded_by": list(doc_meta.superseded_by),
                    "amends": list(doc_meta.amends),
                    "amended_by": list(doc_meta.amended_by),
                    "version_note": doc_meta.version_note,
                    "quality_warning": doc_meta.quality_warning,
                }
            )
            if ch.table_id:
                ch.metadata["table_id"] = ch.table_id

        return chunks


# Alias class names and module-level entrypoint function
HierarchicalChunkerV3 = HierarchicalChunker
ChunkerV3 = HierarchicalChunker


def chunk_document(
    blocks: list[ParsedBlock] | DocumentMeta,
    doc_meta: DocumentMeta | list[ParsedBlock] | None = None,
    file_name: str = "",
    doc_id: str = "",
    glossary: GlossaryManager | None = None,
) -> tuple[list[ChunkRecord], list[ParentSectionRecord]]:
    """Module-level convenience function running the 7-stage V3 Hierarchical Chunker."""
    chunker = HierarchicalChunker(glossary=glossary)
    return chunker.chunk_document(blocks=blocks, doc_meta=doc_meta, file_name=file_name, doc_id=doc_id)


chunk_blocks = chunk_document
run_chunker_v3 = chunk_document
