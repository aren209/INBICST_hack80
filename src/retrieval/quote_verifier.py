"""Extraction and verification of verbatim document quotes in LLM answers.

The prompt asks the model to format every verbatim quote as a two-line Markdown blockquote:

    > «точный текст из документа»
    > — Название документа, № 123, п. 4.5, стр. 7

`extract_quotes()` parses those blocks out of the (complete) answer and `verify_quotes()` checks each
quote against the retrieved context (child chunks, parent sections, amendments, table previews) with
normalization (case, ё, quotes/dashes, hyphenation, whitespace, punctuation) plus a token-level fuzzy
match, so the UI can mark quotes as «Проверено в документе» / «Не найдено дословно».

Verified quotes are additionally *enriched* deterministically from the retrieved context: the quote is
located in the best matching context text (parent section preferred), the enclosing numbered clause
(or paragraph for unnumbered prose) is cut out as `clause_text` (bounded) with `clause_highlights`
(char spans of the quoted fragments inside it), and `source_title` / `doc_number` / `clause` / `page`
are taken from the matched chunk metadata instead of the model's source line. Placeholder numbers
(`[___]`, `б/н`, `___`) are never returned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable

FUZZY_THRESHOLD = 0.9
MIN_FRAGMENT_TOKENS = 3
MAX_CLAUSE_CHARS = 1500

_OPEN_QUOTES = "«“„\"'‘"
_BQ_LINE = re.compile(r"^\s{0,6}>\s?(.*)$")
_SOURCE_LINE = re.compile(r"^\s*[—–-]{1,2}\s*(.+)$")
_ELLIPSIS_SPLIT = re.compile(r"\s*(?:\.{3,}|…|\[\s*(?:\.{3}|…)\s*\]|\(\s*(?:\.{3}|…)\s*\))\s*")
_DOC_NUM = re.compile(r"(?:№|(?<![A-Za-zА-Яа-яЁё])N(?=\s*\d))\s*([0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё./\-]*)")
_CLAUSE = re.compile(r"(?<![А-Яа-яЁёA-Za-z])(?:пп?\.|пункт|раздел)\s*([^,;|]+)", re.IGNORECASE)
_SHEET = re.compile(r"(?<![А-Яа-яЁёA-Za-z])лист\s*:\s*([^,;|]+)", re.IGNORECASE)
_PAGE = re.compile(r"(?<![А-Яа-яЁёA-Za-z])(?:стр\.|с\.)\s*([0-9]+(?:\s*[–\-,]\s*[0-9]+)*)", re.IGNORECASE)
_WORD = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
_GLUE_GAP = re.compile(r"^(?:[\-\u00ad][ \t]*\n\s*|\u00ad)$")

_KIND_PRIORITY: dict[str, int] = {"parent": 3, "chunk": 2, "table": 1, "tool": 0}
_TECH_FIELD_TOKEN_RE = re.compile(
    r"\b(?:table_id|matched_count|returned_count|total_table_rows|delivery_mode|tool_query|inline_parent|SQLite)\b",
    re.IGNORECASE,
)

# Placeholder document numbers left by templates / OCR: «№ [___]», «№ ___», «б/н», «без номера»…
_PLACEHOLDER_NUM = re.compile(
    r"^(?:\[[\s_.…-]*\]|[_\s.…–—-]+|б\s*/\s*н|бн|без\s+номера|n/?a|[xх]{2,}|\?+)$", re.IGNORECASE
)
_PLACEHOLDER_IN_TEXT = re.compile(
    r"\s*(?:№|(?<![A-Za-zА-Яа-яЁё])N)\s*(?:\[[\s_.…-]*\]|_{2,}|б\s*/\s*н\.?|без\s+номера)", re.IGNORECASE
)
_BARE_PLACEHOLDER = re.compile(r"\[\s*_+\s*\]|_{3,}")


@dataclass
class ParsedQuote:
    text: str
    source_raw: str = ""
    source_title: str = ""
    doc_number: str = ""
    clause: str = ""
    page: str = ""


@dataclass
class ContextDoc:
    doc_id: str
    title: str = ""
    doc_number: str = ""
    texts: list[str] = field(default_factory=list)
    # Optional per-text metadata, parallel to `texts`: kind (chunk/parent/table/tool), title, doc_number,
    # clause_no, pages, clause_page_map
    metas: list[dict[str, Any]] = field(default_factory=list)
    _spans: list[list[tuple[str, int, int]]] | None = None
    _tokens: list[list[str]] | None = None

    def spans(self) -> list[list[tuple[str, int, int]]]:
        if self._spans is None:
            self._spans = [token_spans(x) for x in self.texts]
        return self._spans

    def tokens(self) -> list[list[str]]:
        if self._tokens is None:
            self._tokens = [[t for t, _, _ in sp] for sp in self.spans()]
        return self._tokens

    def meta(self, i: int) -> dict[str, Any]:
        return self.metas[i] if i < len(self.metas) and isinstance(self.metas[i], dict) else {}


# ----------------------------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------------------------
def token_spans(text: str) -> list[tuple[str, int, int]]:
    """Normalized word tokens with their char spans in the original text.

    Lowercase, ё→е, hyphenated line breaks / soft hyphens glued, punctuation/quotes/dashes dropped.
    """
    if not text:
        return []
    src = str(text)
    out: list[tuple[str, int, int]] = []
    for m in _WORD.finditer(src):
        tok = m.group(0).lower().replace("ё", "е")
        if out and _GLUE_GAP.match(src[out[-1][2] : m.start()]):
            prev, start, _ = out[-1]
            out[-1] = (prev + tok, start, m.end())
        else:
            out.append((tok, m.start(), m.end()))
    return out


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, ё→е, glue hyphenated line breaks, drop punctuation/quotes/dashes → word tokens."""
    return [t for t, _, _ in token_spans(text)]


def clean_doc_number(value: Any) -> str:
    """Return a real document number or "" for empty/placeholder values (`[___]`, `б/н`, `___`)."""
    v = str(value or "").strip().strip(".,;:").strip()
    if not v or _PLACEHOLDER_NUM.match(v) or "___" in v or "[" in v or "]" in v:
        return ""
    return v


def clean_clause(value: Any) -> str:
    """Normalize a clause locator ("п. 2.5" → "2.5"); drop placeholders like «б/н» or chunk-part labels."""
    v = str(value or "").strip()
    if not v or re.search(r"б\s*/\s*н|_{2,}|\[", v, re.IGNORECASE):
        return ""
    v = re.sub(r"^(?:пп?\.|пункт[а-я]*)\s*", "", v, flags=re.IGNORECASE).strip()
    v = re.sub(r"\s*\(ч\.\s*\d+\)\s*$", "", v).strip().rstrip(".")
    return v


def strip_placeholders(text: str) -> str:
    """Remove placeholder numbers like «№ [___]» / «№ б/н» from a free-text source line."""
    s = _PLACEHOLDER_IN_TEXT.sub("", str(text or ""))
    s = _BARE_PLACEHOLDER.sub("", s)
    s = re.sub(r"\s*,(?:\s*,)+", ",", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" ,;·|")


# ----------------------------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------------------------
def _parse_source(src: str) -> dict[str, str]:
    src = strip_placeholders(src.strip().strip("*_ ").strip())
    out = {"source_raw": src, "source_title": "", "doc_number": "", "clause": "", "page": ""}
    if not src:
        return out
    if m := _DOC_NUM.search(src):
        out["doc_number"] = clean_doc_number(m.group(1))
    if m := _CLAUSE.search(src):
        out["clause"] = clean_clause(m.group(1))
    if not out["clause"] and (m := _SHEET.search(src)):
        sheet_val = m.group(1).strip(" ,;·|")
        if sheet_val:
            out["clause"] = f"Лист: {sheet_val}"
    if m := _PAGE.search(src):
        out["page"] = re.sub(r"\s+", "", m.group(1))
    # Title = document name incl. number/date, without the clause/sheet/page locator
    cut = len(src)
    for rx in (_CLAUSE, _SHEET, _PAGE):
        if m := rx.search(src):
            cut = min(cut, m.start())
    title = src[:cut].strip(" ,;·|—–-")
    out["source_title"] = title.strip("«»\"' ") or src
    return out


def _split_quote_block(lines: list[str]) -> ParsedQuote | None:
    """Turn the inner lines of one blockquote into a ParsedQuote (None if it is not a doc quote)."""
    body = [ln.strip() for ln in lines]
    while body and not body[0]:
        body.pop(0)
    if not body:
        return None
    first = body[0].lstrip("*_ ")
    if not first or first[0] not in _OPEN_QUOTES:
        return None

    src = ""
    quote_lines: list[str] = []
    for i, ln in enumerate(body):
        m = _SOURCE_LINE.match(ln.lstrip("*_ "))
        if i > 0 and m:
            src = m.group(1)
            break
        quote_lines.append(ln)
    text = " ".join(q for q in quote_lines if q).strip()
    # Inline source on the same line: «…» — Источник
    if not src:
        m = re.search(r"[»”\"]\s*[—–]\s*(.+)$", text)
        if m:
            src = m.group(1)
            text = text[: m.start() + 1]
    text = text.strip("*_ ")
    if text and text[0] in _OPEN_QUOTES:
        text = text[1:]
    close = max(text.rfind("»"), text.rfind("”"), text.rfind('"'))
    if close >= 0 and close >= len(text) - 3:
        text = text[:close]
    text = text.strip()
    if not text:
        return None
    return ParsedQuote(text=text, **_parse_source(src))


def extract_quotes(answer: str) -> list[ParsedQuote]:
    """Find all `> «…»` / `> — источник` blocks in a Markdown answer, in order of appearance."""
    quotes: list[ParsedQuote] = []
    block: list[str] = []

    def _flush() -> None:
        if block:
            q = _split_quote_block(block)
            if q is not None:
                quotes.append(q)
            block.clear()

    for line in str(answer or "").splitlines():
        m = _BQ_LINE.match(line)
        if m:
            inner = m.group(1)
            # A new «…» line after a completed source line starts a new quote inside the same blockquote
            if block and inner.strip()[:1] in _OPEN_QUOTES and any(
                _SOURCE_LINE.match(b.strip()) for b in block[1:]
            ):
                _flush()
            block.append(inner)
        else:
            _flush()
    _flush()
    return quotes


# ----------------------------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------------------------
def _find_seq(hay: list[str], needle: list[str]) -> int:
    n = len(needle)
    if n == 0 or n > len(hay):
        return -1
    first = needle[0]
    for i, tok in enumerate(hay):
        if tok == first and hay[i : i + n] == needle:
            return i
    return -1


def _contains_seq(hay: list[str], needle: list[str]) -> bool:
    return _find_seq(hay, needle) >= 0


def _fuzzy_locate(hay: list[str], needle: list[str]) -> tuple[float, int, int]:
    """Best SequenceMatcher match of `needle` against windows of `hay` anchored on matching tokens.

    Returns (score, start_token, end_token) — score is the share of quote tokens matched in order.
    """
    n = len(needle)
    if n == 0 or not hay:
        return 0.0, -1, -1
    anchors = set(needle[: min(4, n)]) | set(needle[-min(4, n):])
    needle_set = set(needle)
    if len(needle_set & set(hay)) < max(1, int(len(needle_set) * 0.6)):
        return 0.0, -1, -1
    best, best_span = 0.0, (-1, -1)
    tail_anchor = set(needle[-min(4, n):])
    for i, tok in enumerate(hay):
        if tok not in anchors:
            continue
        # window may start here (head anchor) or end here (tail anchor)
        starts = [i]
        if tok in tail_anchor:
            starts.append(max(0, i - n + 1))
        for s in starts:
            window = hay[s : s + n + max(2, n // 8)]
            sm = SequenceMatcher(None, window, needle, autojunk=False)
            blocks = [b for b in sm.get_matching_blocks() if b.size]
            # ratio relative to the quote length (window may be slightly longer)
            score = sum(b.size for b in blocks) / n
            if score > best and blocks:
                best = score
                best_span = (s + blocks[0].a, s + blocks[-1].a + blocks[-1].size)
                if best >= 0.999:
                    return best, best_span[0], best_span[1]
    return best, best_span[0], best_span[1]


def _fuzzy_score(hay: list[str], needle: list[str]) -> float:
    return _fuzzy_locate(hay, needle)[0]


def _locate(hay: list[str], needle: list[str]) -> tuple[float, int, int]:
    i = _find_seq(hay, needle)
    if i >= 0:
        return 1.0, i, i + len(needle)
    return _fuzzy_locate(hay, needle)


def _fragment_match(frag: list[str], docs: list[ContextDoc]) -> tuple[float, ContextDoc | None]:
    best, best_doc = 0.0, None
    for d in docs:
        for toks in d.tokens():
            if _contains_seq(toks, frag):
                return 1.0, d
        for toks in d.tokens():
            sc = _fuzzy_score(toks, frag)
            if sc > best:
                best, best_doc = sc, d
    return best, best_doc


def _doc_preference(q: ParsedQuote, docs: list[ContextDoc]) -> list[ContextDoc]:
    """Search the document named in the source line first (same doc_number / title), then the rest."""
    def rank(d: ContextDoc) -> int:
        if q.doc_number and d.doc_number and q.doc_number.lower() == d.doc_number.lower():
            return 0
        if q.source_title and d.title and (
            q.source_title.lower()[:40] in d.title.lower() or d.title.lower()[:40] in q.source_title.lower()
        ):
            return 1
        return 2

    return sorted(docs, key=rank)


# ----------------------------------------------------------------------------------------------
# Enrichment: full clause text + metadata of the matched chunk
# ----------------------------------------------------------------------------------------------
# Numbered clause heads at a line start: "2.5.", "2.5.1", "п. 3.2", "**4. Порядок…**" (single level needs a dot)
_CLAUSE_HEAD = re.compile(
    r"^[ \t>*_#]*(?:п\.\s*)?(?:(?P<multi>\d{1,2}(?:\.\d{1,3}){1,3})\.?(?=[\s)])|(?P<single>\d{1,2})\.(?=[ \t]+\S))",
    re.MULTILINE,
)
_PAGE_MARK = re.compile(r"\[\s*Стр\.?\s*(\d+)\s*\]", re.IGNORECASE)
_CONT_PREFIX = re.compile(r"^[ \t]*\(Продолжение[^\n]*\)[ \t]*$", re.MULTILINE | re.IGNORECASE)
_PARA_BREAK = re.compile(r"\n[ \t]*\n|(?<=[.!?…;])[ \t]*\n")
_SENT_BREAK = re.compile(r"(?<=[.!?…;])\s+|\n")
_SIMPLE_CLAUSE = re.compile(r"^\d{1,2}(?:\.\d{1,3}){0,3}$")


def _clause_heads(text: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group("multi") or m.group("single")) for m in _CLAUSE_HEAD.finditer(text)]


def _is_descendant(num: str, parent: str) -> bool:
    return num.startswith(parent + ".")


def _bounded_window(text: str, cs: int, ce: int, qs: int, qe: int, limit: int) -> tuple[int, int]:
    """Shrink [cs, ce) to ≤ `limit` chars around the quote [qs, qe), cutting at sentence boundaries."""
    if ce - cs <= limit:
        return cs, ce
    if qe - qs >= limit:
        return qs, qe
    spare = limit - (qe - qs)
    ws = max(cs, qs - spare // 2)
    we = min(ce, ws + limit)
    ws = max(cs, we - limit)
    if ws > cs:  # start at the first full sentence inside the window
        m = _SENT_BREAK.search(text, ws, qs)
        ws = m.end() if m else qs
    if we < ce:  # end at the last full sentence inside the window
        last = None
        for m in _SENT_BREAK.finditer(text, qe, we):
            last = m
        we = last.start() if last else qe
    return ws, max(we, qe)


def _clause_bounds(text: str, qs: int, qe: int) -> tuple[str, int, int]:
    """Locate the numbered clause containing the quote; falls back to the enclosing paragraph."""
    heads = _clause_heads(text)
    before = [h for h in heads if h[0] <= qs]
    if before:
        start, num = before[-1]
        end, last_num = len(text), num
        for pos, n in heads:
            if pos <= start or _is_descendant(n, num):
                continue
            if pos >= qe:
                end = pos
                break
            last_num = n  # the quote continues into the next clause
        label = num if last_num == num else f"{num}–{last_num}"
        return label, start, end
    # Unnumbered prose: paragraph bounded by blank lines / sentence-final line breaks
    ps = 0
    for m in _PARA_BREAK.finditer(text, 0, qs):
        ps = m.end()
    m = _PARA_BREAK.search(text, qe)
    pe = m.start() if m else len(text)
    return "", ps, max(pe, qe)


def _clean_clause_text(text: str) -> str:
    s = _CONT_PREFIX.sub("", text)
    s = _PAGE_MARK.sub("", s)
    s = re.sub(r"\*\*|__", "", s)
    s = re.sub(r"^[ \t]*(?:-{3,}|={3,})[ \t]*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"(\w)[\-\u00ad][ \t]*\n[ \t]*(\w)", r"\1\2", s)
    s = s.replace("\u00ad", "")
    # soft PDF line wraps: a line continuing in lower case is joined with a space
    s = re.sub(r"(?<![.!?:;\n])[ \t]*\n[ \t]*(?=[a-zа-яё(«\"0-9])", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def _page_at(text: str, pos: int) -> str:
    page = ""
    for m in _PAGE_MARK.finditer(text, 0, pos):
        page = m.group(1)
    return page


def _resolve_page(
    text: str,
    qs: int,
    qe: int,
    meta: dict[str, Any],
    clause: str,
    model_page: str,
    doc_id: str = "",
) -> str:
    p1, p2 = _page_at(text, qs + 1), _page_at(text, qe)
    if p1:
        return p1 if not p2 or p2 == p1 else f"{p1}–{p2}"
    pages: list[int] = []
    cpm = meta.get("clause_page_map")
    if clause and isinstance(cpm, dict):
        v = cpm.get(clause) or cpm.get(clause.split("–")[0])
        if isinstance(v, list):
            pages = [int(x) for x in v if str(x).isdigit()]
    if not pages:
        pages = [int(x) for x in (meta.get("pages") or []) if str(x).isdigit()]
    pages = sorted(set(pages))
    if len(pages) == 1:
        return str(pages[0])
    if model_page and model_page.split("–")[0].split("-")[0].strip().isdigit() and (
        not pages or int(model_page.split("–")[0].split("-")[0]) in pages
    ):
        return model_page
    if pages and pages[-1] - pages[0] <= 1:
        return f"{pages[0]}–{pages[-1]}"
    if (
        not pages
        and doc_id
        and not doc_id.startswith("tool_result_")
        and meta.get("is_single_sheet")
    ):
        return "1"
    return ""


def _highlights(clause_text: str, fragments: list[list[str]]) -> list[list[int]]:
    sp = token_spans(clause_text)
    toks = [t for t, _, _ in sp]
    out: list[list[int]] = []
    for frag in fragments:
        sc, i0, i1 = _locate(toks, frag)
        if sc >= FUZZY_THRESHOLD and 0 <= i0 < i1 <= len(sp):
            out.append([sp[i0][1], sp[i1 - 1][2]])
    out.sort()
    merged: list[list[int]] = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _clean_doc_title(raw_title: str) -> str:
    title = strip_placeholders(str(raw_title or "")).strip()
    return re.sub(r"\.(pdf|docx?|xlsx?|xlsm|rtf)$", "", title, flags=re.IGNORECASE).strip()


def _enrich(
    fragments: list[list[str]],
    q: ParsedQuote,
    candidates: list[ContextDoc],
    threshold: float,
) -> dict[str, Any]:
    """Find the quote in the best context text and return clause text + chunk metadata."""
    best: tuple[tuple[int, int, float], ContextDoc, int, list[tuple[float, int, int]]] | None = None
    for d in candidates:
        for ti, sp in enumerate(d.spans()):
            kind = str(d.meta(ti).get("kind") or "")
            if not sp:
                continue
            toks = d.tokens()[ti]
            locs = [_locate(toks, f) for f in fragments]
            if not locs or min(l[0] for l in locs) < threshold or any(l[1] < 0 for l in locs):
                continue
            score = sum(l[0] for l in locs) / len(locs)
            rank = (_KIND_PRIORITY.get(kind, 1), 1 if score >= 0.999 else 0, score)
            if best is None or rank > best[0]:
                best = (rank, d, ti, locs)
        if best is not None:
            break  # candidates are ordered: matched document first
    if best is None:
        return {}
    _, d, ti, locs = best
    text, sp, meta = d.texts[ti], d.spans()[ti], d.meta(ti)
    kind = str(meta.get("kind") or "")
    sheet_name = str(meta.get("sheet_name") or "").strip()
    qs = min(sp[l[1]][1] for l in locs)
    qe = max(sp[l[2] - 1][2] for l in locs)
    if kind in {"table", "tool"} and sheet_name:
        clause = ""
        ps = 0
        for m in _PARA_BREAK.finditer(text, 0, qs):
            ps = m.end()
        m = _PARA_BREAK.search(text, qe)
        pe = m.start() if m else len(text)
        cs, ce = ps, max(pe, qe)
    else:
        clause, cs, ce = _clause_bounds(text, qs, qe)
    ws, we = _bounded_window(text, cs, ce, qs, qe, MAX_CLAUSE_CHARS)
    clause_text = _clean_clause_text(text[ws:we])
    if len(clause_text) > MAX_CLAUSE_CHARS + 200:  # cleaning never grows text; safety net only
        clause_text = clause_text[:MAX_CLAUSE_CHARS].rstrip() + "…"
    if ws > cs:
        clause_text = "… " + clause_text
    if we < ce:
        clause_text = clause_text + " …"
    if not clause:
        cl = clean_clause(meta.get("clause_no"))
        clause = cl if _SIMPLE_CLAUSE.match(cl) else ""
    if not clause and kind in {"table", "tool"} and sheet_name:
        clause = f"Лист: {sheet_name}"
    title = _clean_doc_title(str(meta.get("title") or d.title or ""))
    out: dict[str, Any] = {
        "source_title": title,
        "doc_number": clean_doc_number(meta.get("doc_number") or d.doc_number),
        "doc_date": str(meta.get("doc_date") or ""),
        "clause": clause,
        "page": _resolve_page(text, qs, qe, meta, clause, q.page, doc_id=d.doc_id),
        "clause_text": clause_text,
        "clause_highlights": _highlights(clause_text, fragments),
        "matched_doc_id": d.doc_id,
    }
    if sheet_name:
        out["sheet_name"] = sheet_name
    return out


def build_source_label(title: str, doc_number: str = "", clause: str = "", page: str = "") -> str:
    parts = [title.strip()] if title and title.strip() else []
    dn = str(doc_number or "").strip()
    if dn and not re.search(re.escape(dn), title or ""):
        if re.match(r"^(?:№|приказ\b|распоряжен|постановлен|указ\b|N\s*\d)", dn, re.IGNORECASE):
            parts.append(dn)
        else:
            parts.append(f"№ {dn}")
    if clause:
        cl_s = str(clause).strip()
        if re.match(r"^(?:лист\s*:|пп?\.\s*|раздел\b|табл\.)", cl_s, re.IGNORECASE):
            parts.append(cl_s)
        else:
            parts.append(f"п. {cl_s}")
    if page:
        parts.append(f"стр. {page}")
    return ", ".join(parts)


def verify_quotes(
    answer: str,
    context_docs: Iterable[ContextDoc],
    threshold: float = FUZZY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Extract quotes from `answer`, check each against the retrieved context and enrich verified ones."""
    parsed = extract_quotes(answer)
    if not parsed:
        return []
    docs = [d for d in context_docs if d.texts]
    results: list[dict[str, Any]] = []
    for q in parsed:
        fragments = [normalize_tokens(f) for f in _ELLIPSIS_SPLIT.split(q.text)]
        fragments = [f for f in fragments if f]
        long_frags = [f for f in fragments if len(f) >= MIN_FRAGMENT_TOKENS] or fragments
        ordered = _doc_preference(q, docs)
        scores: list[float] = []
        matched_doc: ContextDoc | None = None
        for frag in long_frags:
            sc, d = _fragment_match(frag, ordered)
            scores.append(sc)
            if d is not None and matched_doc is None:
                matched_doc = d
        score = min(scores) if scores else 0.0
        verified = bool(long_frags) and score >= threshold
        res: dict[str, Any] = {
            "text": q.text,
            "source_raw": q.source_raw,
            "source_title": q.source_title,
            "doc_number": clean_doc_number(q.doc_number),
            "doc_date": "",
            "clause": q.clause,
            "page": q.page,
            "verified": verified,
            "score": round(score, 3),
            "matched_doc_id": matched_doc.doc_id if (matched_doc and verified) else "",
            "clause_text": "",
            "clause_highlights": [],
            "metadata_source": "model",
        }
        if verified and matched_doc is not None:
            cands = [matched_doc] + [d for d in ordered if d is not matched_doc]
            enriched = _enrich(long_frags, q, cands, threshold)
            if enriched:
                res.update({k: v for k, v in enriched.items() if k != "source_title" or v})
                if not res["source_title"]:
                    res["source_title"] = q.source_title
                res["metadata_source"] = "context"
            else:
                res["source_title"] = q.source_title or matched_doc.title
                res["doc_number"] = res["doc_number"] or clean_doc_number(matched_doc.doc_number)
        res["source_label"] = build_source_label(res["source_title"], res["doc_number"], res["clause"], res["page"])
        results.append(res)
    return results


def _clean_table_preview_text(raw_text: str) -> str:
    """Strip internal schema lines (table_id, SQLite, etc.) from markdown previews before indexing."""
    lines: list[str] = []
    for ln in str(raw_text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if _TECH_FIELD_TOKEN_RE.search(s) or re.search(r"::tbl_\d+", s):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def _format_table_rows_lines(rows: Iterable[Any], columns: Iterable[Any] = ()) -> list[str]:
    """Render structured table row dicts as clean human-readable lines without JSON keys."""
    col_names: list[str] = []
    for c in columns or ():
        if isinstance(c, str) and c.strip():
            col_names.append(c.strip())
        elif isinstance(c, dict) and str(c.get("name") or "").strip():
            col_names.append(str(c["name"]).strip())

    out: list[str] = []
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        keys = [k for k in col_names if k in row] + [k for k in row if k not in col_names]
        cells: list[str] = []
        for k in keys:
            v_str = str(row.get(k) or "").strip()
            if not v_str:
                continue
            cells.append(f"{k}: {v_str}")
        if cells:
            out.append("; ".join(cells))
    return out


def _format_table_formulas_lines(formulas: Iterable[Any]) -> list[str]:
    """Render structured Excel formula dicts as clean human-readable lines."""
    out: list[str] = []
    for item in formulas or ():
        if not isinstance(item, dict):
            continue
        row_lbl = str(item.get("row_label") or "").strip()
        col = str(item.get("column") or "").strip()
        formula = str(item.get("formula") or "").strip()
        cell_txt = str(item.get("cell_text") or item.get("cell") or "").strip()
        parts: list[str] = []
        if row_lbl:
            parts.append(row_lbl)
        if col and formula:
            parts.append(f"{col}: {formula}")
        elif formula:
            parts.append(formula)
        line = " — ".join(parts)
        if line and cell_txt and formula not in cell_txt:
            line += f" ({cell_txt})"
        elif not line and cell_txt:
            line = cell_txt
        if line:
            out.append(line)
    return out


def _format_aggregation_lines(agg_res: Any) -> list[str]:
    """Render aggregation_result into readable lines without technical keys."""
    if agg_res is None:
        return []
    if isinstance(agg_res, dict):
        # Skip raw passport dump from get_schema (it has table_id / columns / preview_rows)
        if "table_id" in agg_res or "preview_rows" in agg_res:
            return []
        lines = [f"{k}: {v}" for k, v in agg_res.items() if str(v or "").strip()]
        return lines
    if isinstance(agg_res, list):
        return [str(x).strip() for x in agg_res if str(x or "").strip()]
    return [f"Итог: {agg_res}"]


def _parse_tool_json_payload(raw_txt: str) -> dict[str, Any] | None:
    """Parse tool JSON output (handling truncated JSON gracefully when possible)."""
    s = str(raw_txt or "").strip()
    if not s.startswith("{"):
        return None
    try:
        loaded = json.loads(s)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        extracted: dict[str, Any] = {}
        for key in ("table_id", "table_title", "file_name", "sheet_name"):
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', s)
            if m:
                extracted[key] = m.group(1)
        m_prev = re.search(r'"markdown_preview"\s*:\s*"((?:\\.|[^"\\])*)"', s)
        if m_prev:
            try:
                extracted["markdown_preview"] = json.loads(f'"{m_prev.group(1)}"')
            except Exception:
                pass
        return extracted or None


def context_docs_from_search(search_resp: Any, extra_texts: Iterable[str] = ()) -> list[ContextDoc]:
    """Collect every text the LLM could have seen (grouped by doc_id) from a SearchResponse."""
    by_doc: dict[str, ContextDoc] = {}
    seen_texts: set[tuple[str, int]] = set()

    def _add(doc_id: str, title: str, doc_number: str, text: str, meta: dict[str, Any]) -> None:
        text = (text or "").strip()
        if meta.get("kind") in {"chunk", "parent"}:
            # Synthetic «(Продолжение п. X: …)» headers are chunking artifacts, not document text: drop them so
            # quotes spanning a chunk boundary still match the real, contiguous clause text.
            text = re.sub(r"\n{3,}", "\n\n", _CONT_PREFIX.sub("", text)).strip()
        if not text:
            return
        key = (doc_id, hash(text))
        if key in seen_texts:
            return
        seen_texts.add(key)
        d = by_doc.get(doc_id)
        if d is None:
            d = by_doc[doc_id] = ContextDoc(
                doc_id=doc_id,
                title=_clean_doc_title(title),
                doc_number=clean_doc_number(doc_number),
            )
        else:
            if not d.title and title:
                d.title = _clean_doc_title(title)
            if not d.doc_number and doc_number:
                d.doc_number = clean_doc_number(doc_number)
        d.texts.append(text)
        d.metas.append(meta)

    tables_by_id: dict[str, Any] = {}
    tables_by_doc: dict[str, list[Any]] = {}
    for tp in getattr(search_resp, "linked_tables", None) or []:
        tid = str(getattr(tp, "table_id", "") or "").strip()
        did = str(getattr(tp, "doc_id", "") or "").strip()
        if tid:
            tables_by_id[tid] = tp
        if did:
            tables_by_doc.setdefault(did, []).append(tp)

    hits = list(getattr(search_resp, "hits", None) or []) + list(getattr(search_resp, "amendment_chunks", None) or [])
    doc_info: dict[str, dict[str, Any]] = {}
    table_hints: dict[str, dict[str, Any]] = {}
    sheet_pages: dict[tuple[str, str], list[int]] = {}

    for h in hits:
        h_doc_id = str(getattr(h, "doc_id", "") or getattr(h, "file_name", "") or "unknown")
        h_title = str(getattr(h, "title", "") or getattr(h, "file_name", "") or "")
        h_num = str(getattr(h, "doc_number", "") or "")
        h_date = str(getattr(h, "doc_date", "") or "")
        h_meta = getattr(h, "metadata", None) or {}
        h_tid = str(getattr(h, "table_id", "") or h_meta.get("table_id") or "").strip()
        h_sheet = str(getattr(h, "sheet_name", "") or h_meta.get("sheet_name") or "").strip()
        if not h_sheet and h_tid in tables_by_id:
            h_sheet = str(getattr(tables_by_id[h_tid], "sheet_name", "") or "").strip()
        h_pages = [int(p) for p in (getattr(h, "page_numbers", None) or []) if str(p).isdigit()]

        if h_doc_id not in doc_info:
            doc_info[h_doc_id] = {
                "title": h_title,
                "doc_number": h_num,
                "doc_date": h_date,
                "file_name": str(getattr(h, "file_name", "") or ""),
            }
        if h_sheet and h_pages:
            sheet_pages.setdefault((h_doc_id, h_sheet.lower()), []).extend(h_pages)
        if h_tid:
            hint = table_hints.setdefault(
                h_tid,
                {
                    "doc_id": h_doc_id,
                    "title": h_title,
                    "doc_number": h_num,
                    "doc_date": h_date,
                    "sheet_name": h_sheet,
                    "pages": [],
                },
            )
            if h_sheet and not hint["sheet_name"]:
                hint["sheet_name"] = h_sheet
            hint["pages"].extend(h_pages)

    def _is_single_sheet_doc(doc_id: str, tid: str, tp: Any) -> bool:
        if not doc_id or doc_id.startswith("tool_result_"):
            return False
        if tid and re.search(r"::tbl_0*([2-9]|\d{2,})$", tid):
            return False
        doc_tbls = tables_by_doc.get(doc_id, [])
        if len(doc_tbls) > 1:
            return False
        fmt = str(getattr(tp, "file_format", "") or "").lower().lstrip(".") if tp else ""
        fname = str(getattr(tp, "file_name", "") or doc_info.get(doc_id, {}).get("file_name") or "").lower()
        sh = str(getattr(tp, "sheet_name", "") or "").strip() if tp else ""
        return bool(sh or fmt in {"xlsx", "xls", "xlsm"} or fname.endswith((".xlsx", ".xls", ".xlsm")))

    def _resolve_table_meta(tid: str, payload_title: str = "", payload_sheet: str = "", payload_fname: str = "") -> tuple[str, dict[str, Any]]:
        tp = tables_by_id.get(tid)
        t_hint = table_hints.get(tid, {})
        doc_id = (
            (str(getattr(tp, "doc_id", "") or "").strip() if tp else "")
            or str(t_hint.get("doc_id") or "").strip()
            or (tid.split("::", 1)[0] if "::" in tid else "")
        )
        d_info = doc_info.get(doc_id, {})
        title = (
            str(d_info.get("title") or "").strip()
            or (str(getattr(tp, "table_title", "") or getattr(tp, "section_title", "") or getattr(tp, "file_name", "") or "").strip() if tp else "")
            or str(payload_title or "").strip()
            or str(payload_fname or "").strip()
            or str(t_hint.get("title") or "").strip()
        )
        num = (
            (str(getattr(tp, "doc_number", "") or "").strip() if tp else "")
            or str(d_info.get("doc_number") or "").strip()
            or str(t_hint.get("doc_number") or "").strip()
        )
        date = (
            (str(getattr(tp, "doc_date", "") or "").strip() if tp else "")
            or str(d_info.get("doc_date") or "").strip()
            or str(t_hint.get("doc_date") or "").strip()
        )
        sheet = (
            (str(getattr(tp, "sheet_name", "") or "").strip() if tp else "")
            or str(payload_sheet or "").strip()
            or str(t_hint.get("sheet_name") or "").strip()
        )
        pages = sorted(set(int(p) for p in (t_hint.get("pages") or []) if str(p).isdigit()))
        if not pages and doc_id and sheet:
            pages = sorted(set(int(p) for p in sheet_pages.get((doc_id, sheet.lower()), []) if str(p).isdigit()))
        single_sheet = _is_single_sheet_doc(doc_id, tid, tp)
        if not pages and single_sheet:
            pages = [1]
        meta_out: dict[str, Any] = {
            "title": title,
            "doc_number": num,
            "doc_date": date,
            "table_id": tid,
            "pages": pages,
            "is_single_sheet": single_sheet,
        }
        if sheet:
            meta_out["sheet_name"] = sheet
        return doc_id, meta_out

    for h in hits:
        doc_id = str(getattr(h, "doc_id", "") or getattr(h, "file_name", "") or "unknown")
        title = str(getattr(h, "title", "") or getattr(h, "file_name", "") or "")
        num = str(getattr(h, "doc_number", "") or "")
        h_meta = getattr(h, "metadata", None) or {}
        h_tid = str(getattr(h, "table_id", "") or h_meta.get("table_id") or "").strip()
        sheet_name = str(getattr(h, "sheet_name", "") or h_meta.get("sheet_name") or "").strip()
        if not sheet_name and h_tid in tables_by_id:
            sheet_name = str(getattr(tables_by_id[h_tid], "sheet_name", "") or "").strip()

        base: dict[str, Any] = {
            "title": title,
            "doc_number": num,
            "doc_date": getattr(h, "doc_date", "") or "",
        }
        if sheet_name:
            base["sheet_name"] = sheet_name
        if h_tid:
            base["table_id"] = h_tid

        tp_for_hit = tables_by_id.get(h_tid) if h_tid else None
        h_pages = list(getattr(h, "page_numbers", None) or [])
        single_sheet = _is_single_sheet_doc(doc_id, h_tid, tp_for_hit) if (h_tid or sheet_name) else False
        if not h_pages and single_sheet:
            h_pages = [1]

        chunk_meta: dict[str, Any] = {
            **base,
            "kind": "table" if h_tid else "chunk",
            "clause_no": getattr(h, "clause_no", "") or "",
            "pages": h_pages,
            "is_single_sheet": single_sheet,
        }
        _add(doc_id, title, num, getattr(h, "clean_text", ""), chunk_meta)
        ps = getattr(h, "parent_section", None)
        if ps is not None:
            ps_meta = getattr(ps, "metadata", None) or {}
            ps_sheet = str(ps_meta.get("sheet_name") or sheet_name or "").strip()
            ps_dict: dict[str, Any] = {
                **base,
                "kind": "parent",
                "pages": list(getattr(ps, "page_numbers", None) or []),
                "clause_page_map": ps_meta.get("clause_page_map"),
            }
            if ps_sheet:
                ps_dict["sheet_name"] = ps_sheet
            _add(doc_id, title, num, getattr(ps, "clean_text", ""), ps_dict)
        _add(doc_id, title, num, getattr(h, "parent_text", ""), {**base, "kind": "parent"})
        _add(doc_id, title, num, getattr(h, "raw_text", ""), {**chunk_meta})

    for t in getattr(search_resp, "table_context", None) or []:
        tid = str(getattr(t, "table_id", "") or "").strip()
        resolved_doc_id, tbl_meta = _resolve_table_meta(
            tid,
            payload_title=str(getattr(t, "table_title", "") or ""),
        )
        target_doc_id = resolved_doc_id or tid or "table"
        t_meta = {**tbl_meta, "kind": "table"}
        row_lines = _format_table_rows_lines(
            getattr(t, "rows", None) or [],
            getattr(t, "columns", None) or [],
        )
        formula_lines = _format_table_formulas_lines(getattr(t, "formulas", None) or [])
        agg_lines = _format_aggregation_lines(getattr(t, "aggregation_result", None))
        md_prev = _clean_table_preview_text(getattr(t, "markdown_preview", "") or "")
        for block_txt in (
            "\n".join(row_lines),
            "\n".join(formula_lines),
            "\n".join(agg_lines),
            md_prev,
        ):
            if block_txt.strip():
                _add(target_doc_id, tbl_meta["title"], tbl_meta["doc_number"], block_txt, dict(t_meta))

    for i, txt in enumerate(extra_texts):
        parsed = _parse_tool_json_payload(txt)
        if parsed is not None:
            tid = str(parsed.get("table_id") or "").strip()
            resolved_doc_id, tbl_meta = _resolve_table_meta(
                tid,
                payload_title=str(parsed.get("table_title") or ""),
                payload_sheet=str(parsed.get("sheet_name") or ""),
                payload_fname=str(parsed.get("file_name") or ""),
            )
            target_doc_id = resolved_doc_id or f"tool_result_{i}"
            tool_meta = {**tbl_meta, "kind": "tool"}
            row_lines = _format_table_rows_lines(
                parsed.get("rows") or parsed.get("preview_rows") or [],
                parsed.get("columns") or [],
            )
            formula_lines = _format_table_formulas_lines(parsed.get("formulas") or [])
            agg_lines = _format_aggregation_lines(parsed.get("aggregation_result"))
            md_prev = _clean_table_preview_text(str(parsed.get("markdown_preview") or ""))
            added_any = False
            for block_txt in (
                "\n".join(row_lines),
                "\n".join(formula_lines),
                "\n".join(agg_lines),
                md_prev,
            ):
                if block_txt.strip():
                    _add(target_doc_id, tbl_meta["title"], tbl_meta["doc_number"], block_txt, dict(tool_meta))
                    added_any = True
            if not added_any:
                fallback_hdr = tbl_meta["title"] or tbl_meta.get("sheet_name") or ""
                if fallback_hdr:
                    _add(target_doc_id, tbl_meta["title"], tbl_meta["doc_number"], fallback_hdr, dict(tool_meta))
        else:
            clean_extra = _clean_table_preview_text(txt)
            if clean_extra:
                _add(f"tool_result_{i}", "", "", clean_extra, {"kind": "tool"})
    return list(by_doc.values())
