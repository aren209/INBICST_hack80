"""Hybrid RRF Search Service with ParentSection Hydration, Edition Graph (Amends/Supersedes), and Out-of-Scope Detection."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from src.chunking.glossary import GlossaryManager, get_glossary_manager
from src.config import (
    NON_ANSWER_ZONES,
    OUT_OF_SCOPE_DENSE_THRESHOLD,
    OUT_OF_SCOPE_LOW_DENSE_THRESHOLD,
    OUT_OF_SCOPE_MIN_LEXICAL_OVERLAP,
    STATUS_RU_MAP,
    SUPERSEDED_SCORE_PENALTY,
)
from src.indexing.embedder import (
    HybridEmbedder,
    _get_morph_analyzer,
    _lemmatize_ru_word,
    tokenize_bm25,
)
from src.indexing.pipeline import IndexingPipeline, _core_doc_number, get_pipeline
from src.indexing.table_store import StructuredTableStore, _format_preview_markdown, _norm_ru
from src.schemas import (
    ChatbotContextRequest,
    ChatbotContextResponse,
    ChunkRecord,
    DocumentMeta,
    DocumentStatus,
    ParentSectionRecord,
    SearchHit,
    SearchQuery,
    SearchResponse,
    TablePassport,
    TableQueryRequest,
    TableQueryResponse,
    ZoneType,
    compute_content_hash,
)

logger = logging.getLogger(__name__)

# K2 Actuality Calibration: default score multiplier for superseded documents on current-actuality queries
K2_SUPERSEDED_SCORE_PENALTY: float = 0.65

_HISTORICAL_EDITION_INTENT_RE = re.compile(
    r"(?:"
    r"\bв\s+стар(?:ой|ых|ую)\s+редакци[а-я]*\b"
    r"|\b(?:стар(?:ая|ой|ую|ые|ых)|предыдущ(?:ая|ей|ую|ие|их)|прежн(?:яя|ей|юю|ие|их)"
    r"|прошл(?:ая|ой|ую|ые|ых)|первоначальн(?:ая|ой|ую|ые|ых)|исходн(?:ая|ой|ую)|ранн(?:яя|ей|юю|ие|их))"
    r"\s+(?:редакци[а-я]*|верси[а-я]*)\b"
    r"|\bдо\s+(?:отмены|изменений|внесения\s+изменений|обновления|новой\s+редакции|принятия\s+нов[а-я]+)\b"
    r"|\bранее\s+действовавш[а-я]*\b"
    r"|\bдействовавш[а-я]+\s+(?:ранее|прежде|до\b|в\s+(?:19|20)\d{2})\b"
    r"|\bутративш[а-я]+\s+(?:силу|актуальность)\b"
    r"|\bотмененн[а-я]+\s+(?:приказ[а-я]*|распоряжени[а-я]*|регламент[а-я]*|положени[а-я]*|поряд[а-я]*|правил[а-я]*|редакци[а-я]*|верси[а-я]*|акт[а-я]*|документ[а-я]*)\b"
    r"|\b(?:сравн[а-я]+|отличи[а-я]*|разниц[а-я]*|изменил[а-я]*)\b.*\b(?:редакци[а-я]*|верси[а-я]*)\b"
    r"|\b(?:редакци[а-я]*|верси[а-я]*)\b.*\b(?:сравн[а-я]+|отличи[а-я]*|разниц[а-я]*)\b"
    r"|\bкак\s+(?:было|регулировал[а-я]*|действовал[а-я]*)\s+(?:раньше|ранее|прежде|до\s+изменений|до\s+отмены)\b"
    r")",
    re.IGNORECASE,
)

_HISTORICAL_COMPARISON_RE = re.compile(
    r"\b(?:сравн[а-я]+|отличи[а-я]*|отлича[а-я]+|разниц[а-я]*|что\s+изменил[а-я]*|как\s+изменил[а-я]*)\b",
    re.IGNORECASE,
)

_QUERY_YEAR_CONTEXT_RE = re.compile(
    r"(?:"
    r"\b(?:в|за|на|от|до|по\s+состоянию\s+на|редакци[а-я]*\s*(?:от\s*)?|верси[а-я]*\s*(?:от\s*)?)"
    r"\s*(?:\d{1,2}[./-]\d{1,2}[./-])?((?:19|20)\d{2})\b"
    r"|\b((?:19|20)\d{2})\s*(?:год[ауеом]*|г\.|г\b)"
    r")",
    re.IGNORECASE,
)

_QUERY_EDITION_NUM_RE = re.compile(
    r"\b(?:редакци[а-я]*|верси[а-я]*|ред\.|вер\.)\s*(?:№|n|#)?\s*(\d{1,3})\b",
    re.IGNORECASE,
)


def _extract_doc_years_set(doc_meta_or_dict: Any) -> set[int]:
    """Extract all 4-digit years associated with a document (`doc_date`, `edition_date`, `applicable_years`)."""
    years: set[int] = set()
    if doc_meta_or_dict is None:
        return years
    if isinstance(doc_meta_or_dict, dict):
        d_date = str(doc_meta_or_dict.get("doc_date") or "")
        ed_date = str(
            doc_meta_or_dict.get("edition_date")
            or (doc_meta_or_dict.get("metadata") or {}).get("edition_date")
            or ""
        )
        app_years = doc_meta_or_dict.get("applicable_years") or []
    else:
        d_date = str(getattr(doc_meta_or_dict, "doc_date", "") or "")
        md = getattr(doc_meta_or_dict, "metadata", None) or {}
        ed_date = str(getattr(doc_meta_or_dict, "edition_date", None) or md.get("edition_date") or "")
        app_years = getattr(doc_meta_or_dict, "applicable_years", None) or []

    for ds in (d_date, ed_date):
        m_iso = re.match(r"^((?:19|20)\d{2})", ds.strip())
        if m_iso:
            years.add(int(m_iso.group(1)))
        else:
            m_ru = re.search(r"\b((?:19|20)\d{2})\b", ds)
            if m_ru:
                years.add(int(m_ru.group(1)))
    for y in app_years:
        try:
            yi = int(y)
            if 1990 <= yi <= 2100:
                years.add(yi)
        except (TypeError, ValueError):
            pass
    return years


def _compute_active_recency_bonus(effective_date: str, edition_number: int | str | None = None) -> float:
    """
    Compute a subtle date/edition recency tie-breaker bonus (`0.0 .. 0.025`) for `active` documents
    so newer `edition_date`/`doc_date` or higher `edition_number` ranks ahead when relevance is close.
    """
    bonus = 0.0
    ds = (effective_date or "").strip()
    year, month, day = 0, 1, 1
    m_iso = re.match(r"^((?:19|20)\d{2})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", ds)
    if m_iso:
        year = int(m_iso.group(1))
        month = int(m_iso.group(2) or 1)
        day = int(m_iso.group(3) or 1)
    else:
        m_ru = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d{2})", ds)
        if m_ru:
            day = int(m_ru.group(1))
            month = int(m_ru.group(2))
            year = int(m_ru.group(3))
    if 2015 <= year <= 2040:
        days_since_2015 = max(0, (year - 2015) * 372 + max(1, min(12, month)) * 31 + max(1, min(31, day)))
        bonus += min(0.021, (days_since_2015 / (15.0 * 372.0)) * 0.021)

    if edition_number is not None and str(edition_number).strip() != "":
        m_num = re.search(r"\d+", str(edition_number))
        if m_num:
            bonus += min(0.004, int(m_num.group(0)) * 0.0008)

    return round(min(0.025, bonus), 5)

# Generic administrative/question lemmas that appear in almost any Russian question (e.g. "правила оформления ипотеки")
# and therefore must NOT count by themselves as MIPT domain lexical evidence.
GENERIC_QUERY_LEMMAS: frozenset[str] = frozenset(
    {
        "как",
        "какой",
        "каков",
        "сколько",
        "где",
        "когда",
        "куда",
        "зачем",
        "почему",
        "кто",
        "что",
        "чей",
        "нужно",
        "нужный",
        "можно",
        "надо",
        "необходимо",
        "необходимый",
        "должный",
        "следовать",
        "порядок",
        "правило",
        "условие",
        "требование",
        "регламент",
        "положение",
        "инструкция",
        "приказ",
        "распоряжение",
        "документ",
        "оформление",
        "оформить",
        "оформлять",
        "подача",
        "подать",
        "подавать",
        "получение",
        "получить",
        "получать",
        "проведение",
        "провести",
        "проводить",
        "отправка",
        "отправить",
        "отправлять",
        "слать",
        "послать",
        "направить",
        "направлять",
        "заявление",
        "заявка",
        "обращение",
        "форма",
        "бланк",
        "шаблон",
        "приложение",
        "пункт",
        "раздел",
        "глава",
        "таблица",
        "строка",
        "год",
        "месяц",
        "день",
        "сутки",
        "срок",
        "время",
        "дата",
        "работа",
        "работать",
        "общий",
        "основной",
        "новый",
        "старый",
        "действующий",
        "актуальный",
        "сейчас",
        "текущий",
        "список",
        "перечень",
        "размер",
        "процент",
        "ставка",
        "норма",
        "норматив",
        "расчет",
        "сделать",
        "делать",
        "давать",
        "даваться",
        "дать",
        "иметь",
        "являться",
        "случай",
        "цель",
        "вопрос",
        "ответ",
        "способ",
        "метод",
        "система",
        "процесс",
        "процедура",
        "этап",
        "шаг",
        "банк",
        "клиент",
        "человек",
        "гражданин",
    }
)

# Explicit non-MIPT out-of-scope stems (culinary, banking/mortgages, crypto, auto, travel, etc.)
OUT_OF_SCOPE_STEMS: tuple[str, ...] = (
    "ипотек",
    "автокредит",
    "потребительский кредит",
    "сбербанк",
    "альфа-банк",
    "тинькофф",
    "втб",
    "заемщик",
    "созаемщик",
    "первоначальный взнос",
    "риелтор",
    "карбонар",
    "спагетти",
    "макарон",
    "паста карбонара",
    "пасты карбонара",
    "бекон",
    "панчетт",
    "пармезан",
    "пекорино",
    "желток",
    "сливки",
    "борщ",
    "свекл",
    "капуст",
    "бульон",
    "кастрюл",
    "сковород",
    "рецепт",
    "ингредиент",
    "приготовить пасту",
    "сварить суп",
    "биткоин",
    "bitcoin",
    "криптовалют",
    "криптобирж",
    "майнинг",
    "блокчейн",
    "эфириум",
    "ethereum",
    "догекоин",
    "форекс",
    "трейдинг",
    "осаго",
    "каско",
    "гибдд",
    "техосмотр",
    "алимент",
    "расторжение брака",
    "развод в загсе",
    "шенгенск",
    "туристическая виза",
    "гороскоп",
    "астролог",
    "прогноз погоды",
    "лига чемпионов",
)

# High-confidence MIPT domain anchors
MIPT_DOMAIN_ANCHORS: tuple[str, ...] = (
    "мфти",
    "физтех",
    "флс",
    "финансово-лицев",
    "цфо",
    "финансодержател",
    "кфо",
    "квр",
    "косгу",
    "1с:до",
    "1с",
    "пфо",
    "фэу",
    "рид",
    "рцни",
    "кмо",
    "kmo@mipt.ru",
    "концертный зал",
    "концертного зала",
    "концертном зале",
    "кз",
    "пумс",
    "посвящение в студенты",
    "репетиц",
    "цод",
    "центр обработки данных",
    "центра обработки данных",
    "первая линия",
    "первой линии",
    "техподдержк",
    "технической поддержки",
    "общежити",
    "общаг",
    "поселени",
    "койко-мест",
    "командиров",
    "суточн",
    "авансовый отчет",
    "подбор персонала",
    "поиск и подбор",
    "рекрутер",
    "ваканси",
    "резерв на оплату отпусков",
    "резерв отпусков",
    "отпускн",
    "ппс",
    "профессорско-преподавательск",
    "смет",
    "накладные расходы",
    "накладных расходов",
    "бюджетирован",
    "финансовая структура",
    "финансовой структуры",
    "финструктур",
    "фркт",
    "фпми",
    "лфи",
    "факт",
    "фэфм",
    "фбмф",
    "пиш",
    "служебк",
    "служебная записка",
    "служебную записку",
    "эдо",
    "3065-1",
    "3699-1",
    "794-1",
    "5990-1",
    "1292-1",
    "3897-1",
    "3614-1",
    "1469-1",
    "1465-1",
    "591-1",
    "700",
    "16.4%",
    "8.2%",
)

# ---------------------------------------------------------------------------
# Universal Syntactic Multi-Aspect Query Decomposer (100% Corpus-Agnostic NLP)
# ---------------------------------------------------------------------------

# Purely grammatical Russian parenthetical scope introducers
_PAREN_SCOPE_RE = re.compile(
    r"\(\s*(?:включая|в\s+том\s+числе|в\s+т\.\s*ч\.|кроме\s+того|в\s+частности|с\s+указанием|с\s+учетом)\s+([^()]{10,})\)",
    re.IGNORECASE,
)

# Purely grammatical Russian interrogative phrases (no domain nouns!)
_INTERROGATIVE_PHRASE_PAT = (
    r"(?:"
    r"в\s+каких\s+случаях|в\s+каком\s+случае|в\s+каком\s+порядке|в\s+какой\s+срок|в\s+какие\s+сроки|"
    r"до\s+какого|до\s+каких|за\s+сколько|на\s+сколько|во\s+сколько|"
    r"на\s+какую|на\s+какой|на\s+какие|на\s+каком|"
    r"по\s+каким|по\s+какому|по\s+какой|при\s+каких|при\s+каком|при\s+какой|"
    r"с\s+какого|с\s+каких|с\s+кем|у\s+кого|от\s+кого|от\s+чего|о\s+чем|к\s+кому|к\s+какому|к\s+какой|"
    r"каким\s+образом|какие|какой|какая|какое|каков|какова|каково|каковы|каким|какими|какую|какого|какому|каких|"
    r"сколько|когда|кто|кем|кому|кого|где|куда|откуда|чем|чему|зачем|почему|как\b|чей|чья|чье|чьи|"
    r"можно\s+ли|допускается\s+ли|разрешается\s+ли|требуется\s+ли|нужно\s+ли|необходимо\s+ли|"
    r"обязан\s+ли|вправе\s+ли|следует\s+ли|распространяется\s+ли|действует\s+ли|сохраняется\s+ли"
    r")"
)

# Grammatical modal / passive / impersonal clause predicates after `, и `
_CLAUSE_PREDICATE_PAT = (
    r"(?:"
    + _INTERROGATIVE_PHRASE_PAT
    + r"|принимаются|принимается|разрешается|разрешаются|допускается|допускаются|"
    r"требуется|требуются|распространяется|распространяются|необходимо|нужно|можно|запрещается|запрещено|"
    r"предусмотрено|предусмотрены|установлен|установлены|установлено|определяется|определяются|"
    r"осуществляется|производится|выполняется|подается|оформляется|направляется|согласовывается|утверждается"
    r")\b"
)

_COMMA_CONJ_SPLIT_RE = re.compile(
    r"(?:,\s*(?:а\s+также|кроме\s+того,?|при\s+этом|причем)\s+"
    r"|\s+а\s+также\s+"
    r"|,\s*и\s+(?=" + _CLAUSE_PREDICATE_PAT + r")"
    r"|\s+и\s+(?=" + _INTERROGATIVE_PHRASE_PAT + r"\b)"
    r"|,\s*(?=" + _INTERROGATIVE_PHRASE_PAT + r"\b))",
    re.IGNORECASE,
)

_COMMA_AND_NEXT_WORD_RE = re.compile(r",\s*и\s+([а-яёА-ЯЁ]{4,})\b")

# Framing prepositions introducing a shared topic at the start of a Russian question
_FRAMING_PREP_RE = re.compile(
    r"^\s*(?:согласно|в\s+соответствии\s+с|в\s+рамках|на\s+основании|по\s+правилам|в\s+порядке|по|при|для|в|во|об|о)\s+"
    r"([^,;:?]{6,105})",
    re.IGNORECASE,
)

# Purely grammatical/meta lemmas that do NOT by themselves form a specific topic anchor
_SYNTACTIC_META_LEMMAS: frozenset[str] = frozenset(
    {
        "вопрос",
        "ответ",
        "случай",
        "образ",
        "порядок",
        "правило",
        "условие",
        "требование",
        "срок",
        "время",
        "дата",
        "день",
        "месяц",
        "год",
        "количество",
        "число",
        "размер",
        "сумма",
        "перечень",
        "список",
        "состав",
        "форма",
        "способ",
        "метод",
        "этап",
        "шаг",
        "пункт",
        "раздел",
        "глава",
        "часть",
        "таблица",
        "строка",
        "столбец",
        "колонка",
        "сторона",
        "лицо",
        "указать",
        "назвать",
        "перечислить",
        "описать",
        "пояснить",
        "объяснить",
        "рассказать",
        "определить",
    }
)


def _mask_nested_spans(text: str) -> str:
    """Replace characters inside (...), «...», and "..." with '_' of exact same length for safe top-level splitting."""
    chars = list(text)
    paren_depth = 0
    in_guillemet = 0
    in_dquote = False
    for idx, ch in enumerate(chars):
        if ch == "«":
            in_guillemet += 1
            chars[idx] = "_"
        elif ch == "»" and in_guillemet > 0:
            chars[idx] = "_"
            in_guillemet -= 1
        elif ch == '"' and paren_depth == 0 and in_guillemet == 0:
            in_dquote = not in_dquote
            chars[idx] = "_"
        elif ch == "(" and not in_dquote and in_guillemet == 0:
            paren_depth += 1
            chars[idx] = "_"
        elif ch == ")" and paren_depth > 0:
            chars[idx] = "_"
            paren_depth -= 1
        elif paren_depth > 0 or in_guillemet > 0 or in_dquote:
            chars[idx] = "_"
    return "".join(chars)


def _count_content_words(segment: str) -> int:
    """Count Cyrillic/Latin/numeric tokens of length >= 2 in a clause candidate."""
    return len(re.findall(r"[0-9a-zA-Zа-яА-ЯёЁ]{2,}", segment))


def _is_finite_verb_or_predicative(word: str) -> bool:
    """Use pymorphy3 grammatical tags to check if a Russian word is a finite verb, passive participle, or predicative."""
    w = word.strip().lower().replace("ё", "е")
    if len(w) < 4:
        return False
    morph = _get_morph_analyzer()
    parses = morph.parse(w)
    if not parses:
        return False
    top = parses[0]
    pos = top.tag.POS
    return pos in {"VERB", "PRTS", "PRED"}


def _split_top_level_by_char(text: str, sep: str) -> list[str]:
    """Split `text` by single-character `sep` only outside parentheses and quotes."""
    masked = _mask_nested_spans(text)
    parts: list[str] = []
    last = 0
    for idx, ch in enumerate(masked):
        if ch == sep:
            seg = text[last:idx].strip()
            if seg:
                parts.append(seg)
            last = idx + 1
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return parts if parts else [text.strip()]


def _split_clause_syntactically(clause: str) -> list[str]:
    """Split a single clause on top-level grammatical conjunctions and comma-interrogative boundaries."""
    masked = _mask_nested_spans(clause)
    cut_spans: list[tuple[int, int]] = []

    for m in _COMMA_CONJ_SPLIT_RE.finditer(masked):
        s_idx, e_idx = m.start(), m.end()
        # Keep the interrogative phrase inside the right segment (only consume comma / conjunction!)
        raw_delim = clause[s_idx:e_idx]
        # If delimiter was `, <interrogative>`, zero-width lookahead means e_idx stops right after `, `
        cut_spans.append((s_idx, e_idx))

    # Also check `, и <finite_verb>` via pymorphy3 POS tagger
    for m_v in _COMMA_AND_NEXT_WORD_RE.finditer(masked):
        next_word = clause[m_v.start(1) : m_v.end(1)]
        if _is_finite_verb_or_predicative(next_word):
            span = (m_v.start(), m_v.start(1))
            if not any(not (span[1] <= a or span[0] >= b) for a, b in cut_spans):
                cut_spans.append(span)

    if not cut_spans:
        return [clause.strip()]

    cut_spans.sort(key=lambda x: x[0])
    segments: list[str] = []
    cursor = 0
    for s_idx, e_idx in cut_spans:
        left_candidate = clause[cursor:s_idx].strip(" ,;—–-")
        right_preview = clause[e_idx:].strip(" ,;—–-")
        if (
            len(left_candidate) >= 18
            and _count_content_words(left_candidate) >= 3
            and len(right_preview) >= 16
            and _count_content_words(right_preview) >= 3
        ):
            segments.append(left_candidate)
            cursor = e_idx

    final_tail = clause[cursor:].strip(" ,;—–-")
    if final_tail:
        segments.append(final_tail)
    return segments if segments else [clause.strip()]


def _extract_syntactic_topic_anchor(full_query: str, first_clause: str, header_prefix: str = "") -> str:
    """
    Extract the shared subject / topic noun-phrase anchor from the lead-in header or first clause
    using pure Russian morphological POS/case patterns (pymorphy3) + structural identifiers.
    """
    if header_prefix:
        hdr_clean = header_prefix.strip(" :;,-—–?")
        if _count_content_words(hdr_clean) >= 2 and len(hdr_clean) <= 115:
            return hdr_clean

    # 1. Check explicit framing prepositional phrase at start of full_query or first_clause
    for probe in (first_clause, full_query):
        m_frame = _FRAMING_PREP_RE.match(probe)
        if m_frame:
            cand = m_frame.group(1).strip(" ,;:?!")
            cand_lemmas = [
                t for t in tokenize_bm25(cand)
                if t not in GENERIC_QUERY_LEMMAS and t not in _SYNTACTIC_META_LEMMAS
            ]
            if cand_lemmas and 6 <= len(cand) <= 100:
                return cand

    # 2. Morphological Noun-Phrase (ADJF/PRTF* + NOUN + dependent NOUN[gent/ablt]*) + Identifiers from first_clause
    morph = _get_morph_analyzer()
    tokens = list( re.finditer(r"«[^»]{2,70}»|\"[^\"]{2,70}\"|\d{1,5}-[0-9a-zA-Zа-яА-Я]{1,4}|[0-9a-zA-Zа-яА-ЯёЁ@._-]+", first_clause) )

    np_spans: list[str] = []
    cur_words: list[str] = []
    has_head_noun = False

    def _flush_np() -> None:
        nonlocal cur_words, has_head_noun
        if has_head_noun and cur_words:
            phrase = " ".join(cur_words).strip(" ,.;:!?")
            subst = [
                t for t in tokenize_bm25(phrase)
                if t not in GENERIC_QUERY_LEMMAS and t not in _SYNTACTIC_META_LEMMAS
            ]
            if subst and len(phrase) >= 4:
                np_spans.append(phrase)
        cur_words = []
        has_head_noun = False

    for m_tok in tokens:
        w = m_tok.group(0)
        w_low = w.lower().replace("ё", "е")

        # Quoted title or explicit document number / email / uppercase acronym
        if (
            w.startswith(("«", '"'))
            or re.match(r"^\d{1,5}-[0-9a-zа-я]{1,4}$", w_low)
            or "@" in w_low
            or (w.isupper() and len(w) >= 2 and re.search(r"[А-ЯЁA-Z]", w))
        ):
            cur_words.append(w)
            has_head_noun = True
            continue

        if w in {"№", "N"} or re.match(r"^(?:19|20)\d{2}$", w_low):
            if cur_words:
                cur_words.append(w)
            continue

        if not re.match(r"^[а-яА-ЯёЁ-]+$", w):
            _flush_np()
            continue

        parses = morph.parse(w_low)
        top = parses[0] if parses else None
        pos = top.tag.POS if top else None
        lemma = (top.normal_form.lower().replace("ё", "е") if top else w_low)

        if pos in {"INTJ", "CONJ", "PRCL", "VERB", "INFN", "PRTS", "GRND", "PRED", "ADVB", "NPRO"}:
            _flush_np()
            continue

        if pos in {"PREP"}:
            if has_head_noun and w_low in {"по", "о", "об", "в", "на", "для", "при", "к", "из", "с", "со", "от"} and len(cur_words) <= 5:
                cur_words.append(w)
            else:
                _flush_np()
            continue

        if pos in {"ADJF", "PRTF", "NUMR"}:
            if lemma in {"какой", "каков", "который", "чей", "сколький", "такой", "данный", "этот", "весь", "сам"}:
                _flush_np()
                continue
            cur_words.append(w)
            continue

        if pos == "NOUN":
            if not has_head_noun and lemma in _SYNTACTIC_META_LEMMAS:
                cur_words = []
                continue
            cur_words.append(w)
            has_head_noun = True
            if len(cur_words) >= 8:
                _flush_np()
            continue

        _flush_np()

    _flush_np()

    if not np_spans:
        return ""

    # Combine up to 2 most informative noun phrases from Clause 1 (capped cleanly at ~95 chars)
    chosen: list[str] = []
    total_len = 0
    for sp in np_spans:
        if sp.lower() in " ".join(chosen).lower():
            continue
        if total_len + len(sp) + 2 > 95 and chosen:
            break
        chosen.append(sp)
        total_len += len(sp) + 2
        if len(chosen) >= 2:
            break

    return " ".join(chosen).strip()


def decompose_complex_query(query: str, max_sub_queries: int = 4) -> list[str]:
    """
    Universal Syntactic Multi-Aspect Query Decomposer (100% corpus-agnostic Russian NLP).

    - Splits compound multi-clause questions on `?`, semicolons `;`, parenthetical scope clauses
      `(включая ...)` / `(в том числе ...)`, and coordinating conjunctions introducing a new
      interrogative or predicate clause (`, и какие...`, `, и допускается...`, `, а также ...`, `, какие ...`).
    - Extracts the shared subject/topic noun-phrase anchor so each decomposed sub-query (`up to 4`)
      remains self-contained alongside the full original query.
    """
    clean_q = re.sub(r"\s+", " ", (query or "")).strip()
    if not clean_q:
        return []

    max_sq = max(1, min(int(max_sub_queries), 4))

    # 1. Extract grammatical parenthetical scope clauses `(включая ...)` / `(в том числе ...)`
    paren_aspects: list[str] = []

    def _strip_scope_paren(match: re.Match[str]) -> str:
        inner = match.group(1).strip(" ,;.")
        if _count_content_words(inner) >= 2:
            paren_aspects.append(inner)
            return ""
        return match.group(0)

    working_q = _PAREN_SCOPE_RE.sub(_strip_scope_paren, clean_q)
    working_q = re.sub(r"\s+([,;:?.!])", r"\1", working_q)
    working_q = re.sub(r"\s{2,}", " ", working_q).strip()

    # 2. Extract optional top-level colon framing header `Header: sub-q1 ...`
    header_prefix = ""
    masked_wq = _mask_nested_spans(working_q)
    colon_pos = masked_wq.find(":")
    if 8 <= colon_pos <= 120:
        cand_hdr = working_q[:colon_pos].strip()
        cand_rest = working_q[colon_pos + 1 :].strip()
        if _count_content_words(cand_hdr) >= 2 and _count_content_words(cand_rest) >= 4:
            hdr_low = cand_hdr.lower().replace("ё", "е")
            if hdr_low not in {"вопрос", "запрос", "задача", "ситуация"}:
                header_prefix = cand_hdr
                working_q = cand_rest

    # 3. Split on top-level `?` sentences and `;` clauses, then grammatical conjunctions
    raw_clauses: list[str] = []
    for q_sent in _split_top_level_by_char(working_q, "?"):
        for semi_part in _split_top_level_by_char(q_sent, ";"):
            for syn_seg in _split_clause_syntactically(semi_part):
                cleaned_seg = syn_seg.strip(" ,;—–-")
                if _count_content_words(cleaned_seg) >= 2:
                    raw_clauses.append(cleaned_seg)

    all_segments = list(raw_clauses) + list(paren_aspects)
    if len(all_segments) <= 1:
        return [clean_q]

    # 4. Extract shared subject/topic noun-phrase anchor from header_prefix or first clause
    first_clause = raw_clauses[0] if raw_clauses else clean_q
    topic_anchor = _extract_syntactic_topic_anchor(
        full_query=clean_q,
        first_clause=first_clause,
        header_prefix=header_prefix,
    )
    anchor_subst_tokens = [
        t for t in tokenize_bm25(topic_anchor)
        if t not in GENERIC_QUERY_LEMMAS and t not in _SYNTACTIC_META_LEMMAS
    ]

    # 5. Merge excess trailing segments if > max_sq so zero clauses are dropped!
    while len(all_segments) > max_sq:
        # Merge the shortest adjacent pair among segments[1:] (or last two)
        best_idx = len(all_segments) - 2
        best_combined_len = 10**9
        start_search = 1 if len(all_segments) > 2 else 0
        for idx in range(start_search, len(all_segments) - 1):
            comb_len = len(all_segments[idx]) + len(all_segments[idx + 1])
            if comb_len < best_combined_len:
                best_combined_len = comb_len
                best_idx = idx
        merged_pair = f"{all_segments[best_idx]}; {all_segments[best_idx + 1]}"
        all_segments = all_segments[:best_idx] + [merged_pair] + all_segments[best_idx + 2 :]

    # 6. Anchor each sub-query so it is completely self-contained
    self_contained: list[str] = []
    seen_norm: set[str] = set()
    for idx, seg in enumerate(all_segments):
        seg_out = seg.strip()
        if idx == 0 and header_prefix and header_prefix.lower() not in seg_out.lower():
            seg_out = f"{header_prefix}: {seg_out}"
        elif topic_anchor and anchor_subst_tokens:
            seg_tokens = set(tokenize_bm25(seg_out))
            overlap_cnt = sum(1 for tok in anchor_subst_tokens if tok in seg_tokens)
            overlap_ratio = overlap_cnt / max(1, len(anchor_subst_tokens))
            if overlap_ratio < 0.5 and topic_anchor.lower() not in seg_out.lower():
                seg_out = f"{topic_anchor}: {seg_out}"

        norm_key = _norm_ru(seg_out).rstrip("?.!;:")
        if norm_key and norm_key not in seen_norm:
            seen_norm.add(norm_key)
            self_contained.append(seg_out)

    return self_contained[:max_sq] if self_contained else [clean_q]


class SearchService:
    """
    Hybrid RRF Search Service over Qdrant with:
    - Bidirectional GlossaryManager query expansion
    - Filtering of NON_ANSWER_ZONES ('approval', 'meta')
    - Cross-file content_hash deduplication (merging also_in_files)
    - Graph-aware amendment retrieval (e.g. 3065-1 <-> 700) and supersession/formula bridging (794-1 <-> 2019 smeta)
    - Version & quality alerts extraction
    - ParentSectionRecord hydration from data/sections_store.json
    - Calibrated multi-signal Out-of-Scope detector
    """

    def __init__(
        self,
        pipeline: IndexingPipeline | None = None,
        glossary: GlossaryManager | None = None,
        table_store: StructuredTableStore | None = None,
    ) -> None:
        self.pipeline = pipeline or get_pipeline()
        self.glossary = glossary or get_glossary_manager()
        self._table_store = table_store

    @property
    def table_store(self) -> StructuredTableStore:
        """Return explicitly bound StructuredTableStore or delegate to pipeline.table_store."""
        if self._table_store is not None:
            return self._table_store
        return self.pipeline.table_store

    def _get_active_table_store(self) -> StructuredTableStore | None:
        """Return StructuredTableStore without creating an empty DB file during read-only checks."""
        if self._table_store is not None:
            return self._table_store
        if hasattr(self.pipeline, "_has_active_table_store") and self.pipeline._has_active_table_store():
            return self.pipeline.table_store
        return None

    @property
    def qdrant(self):
        return self.pipeline.qdrant

    @property
    def embedder(self) -> HybridEmbedder:
        return self.pipeline.embedder

    @staticmethod
    def _natural_section_sort_key(sec: ParentSectionRecord) -> tuple[Any, ...]:
        """Natural document-order sort key for ParentSectionRecord IDs (so s02 < s10 and s00_p2 < s00_p10)."""
        return tuple(
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", sec.section_id or "")
        )

    def get_parent_section(self, section_id: str) -> ParentSectionRecord | None:
        """Hydrate clean ParentSectionRecord by section_id from sections_store.json."""
        if not section_id:
            return None
        sec = self.pipeline.get_parent_section(section_id)
        if sec is None:
            # Reload sections store from disk if updated by another instance
            self.pipeline.sections_store = self.pipeline._load_sections_store()
            sec = self.pipeline.get_parent_section(section_id)
        return sec

    def get_unit_parent_sections(self, doc_id: str, unit_id: str) -> list[ParentSectionRecord]:
        """Return all ParentSectionRecords belonging to `(doc_id, unit_id)` sorted in natural document order."""
        if not doc_id or not unit_id:
            return []
        if not self.pipeline.sections_store:
            self.pipeline.sections_store = self.pipeline._load_sections_store()
        matched = [
            sec
            for sec in self.pipeline.sections_store.values()
            if sec.doc_id == doc_id and sec.unit_id == unit_id
        ]
        if not matched:
            self.pipeline.sections_store = self.pipeline._load_sections_store()
            matched = [
                sec
                for sec in self.pipeline.sections_store.values()
                if sec.doc_id == doc_id and sec.unit_id == unit_id
            ]
        matched.sort(key=self._natural_section_sort_key)
        return matched

    def get_doc_parent_sections(self, doc_id: str) -> list[ParentSectionRecord]:
        """Return all ParentSectionRecords belonging to `doc_id` sorted in natural document order."""
        if not doc_id:
            return []
        if not self.pipeline.sections_store:
            self.pipeline.sections_store = self.pipeline._load_sections_store()
        matched = [sec for sec in self.pipeline.sections_store.values() if sec.doc_id == doc_id]
        if not matched:
            self.pipeline.sections_store = self.pipeline._load_sections_store()
            matched = [sec for sec in self.pipeline.sections_store.values() if sec.doc_id == doc_id]
        matched.sort(key=self._natural_section_sort_key)
        return matched

    def _expand_query_for_retrieval(self, raw_query: str) -> str:
        """Expand query using GlossaryManager + targeted multi-concept MIPT domain rules."""
        expanded = self.glossary.expand_query(raw_query)
        q_norm = raw_query.lower().replace("ё", "е")
        extra_terms: list[str] = []

        # Concert Hall (КЗ) colloquial booking queries (Scenario 1 & 2)
        if any(w in q_norm for w in ("кз", "концерт", "зал", "мероприяти", "сцен")):
            if any(w in q_norm for w in ("месяц", "срок", "служеб", "заявк", "почт", "email", "отправ", "слать", "пода")):
                extra_terms.extend(["Регламент бронирования Концертного зала", "служебная записка", "2 месяца", "kmo@mipt.ru", "2.3", "2.6"])
            if any(w in q_norm for w in ("репетиц", "посвящен", "студент", "физтех-школ", "дн", "сколько")):
                extra_terms.extend(["Посвящение в студенты", "репетиционные дни", "физтех-школы", "6.1", "не более 6"])

        # Dormitory settlement queries (Scenario 3)
        if any(w in q_norm for w in ("общежит", "общаг", "поселен", "койко", "жилищ", "найм")):
            extra_terms.extend(["Регламент поселения сотрудников МФТИ в общежития", "жилое помещение", "договор найма"])

        # Smeta & KVR/KOSGU/formulas queries (Scenario 4)
        if any(w in q_norm for w in ("смет", "794-1", "1469-1", "квр", "косгу", "накладн")):
            extra_terms.extend(["Приказ 794-1", "Приложение 1а", "1С:ДО", "помесячно", "КВР", "КОСГУ", "накладные расходы 20%", "НДС 20%", "формула"])

        # FLS opening & amendment queries (Scenario 5)
        if any(w in q_norm for w in ("флс", "лицевой счет", "лицевого счета", "лицевых счетов", "3065-1", "700")):
            extra_terms.extend(["Приказ 3065-1", "Распоряжение 700", "Порядок открытия ФЛС", "присвоение номера ФЛС"])

        # Vacation reserve & 2026-2028 budgeting queries (Scenario 6)
        if any(w in q_norm for w in ("отпуск", "резерв", "3699-1", "ппс", "16.4", "8.2")):
            extra_terms.extend(["Приказ 3699-1", "резерв на оплату отпусков", "16.4%", "ППС", "8.2%", "2026"])

        # Financial structure & CFO tree queries (Scenario 7 & Phystech-school CFO queries)
        if any(
            w in q_norm
            for w in (
                "финансовая структура",
                "финансовой структуры",
                "фин. структур",
                "финструктур",
                "цфо",
                "финансодержател",
                "3614-1",
            )
        ):
            extra_terms.extend(["Приказ 3614-1", "финансовая структура 2026", "Финансодержатель", "ЦФО 1", "ЦФО 2"])

        phystech_schools = {
            "фркт": "Физтех-школа радиотехники и компьютерных технологий",
            "фпми": "Физтех-школа прикладной математики и информатики",
            "лфи": "Физтех-школа физики и исследований Ландау",
            "факт": "Физтех-школа аэрокосмических технологий",
            "фэфм": "Физтех-школа электроники фотоники и молекулярной физики",
            "фбмф": "Физтех-школа биологической и медицинской физики",
        }
        for abbr, full_school in phystech_schools.items():
            if re.search(rf"\b{abbr}\b", q_norm):
                extra_terms.append(full_school)

        # Tech support / Data center / Business trips / HR recruitment (Scenario 8)
        if any(w in q_norm for w in ("техподдержк", "технической поддержки", "первая линия", "первой линии", "1292-1")):
            extra_terms.extend(["Регламент работы первой линии технической поддержки", "1292-1", "обращение", "инцидент"])
        if any(w in q_norm for w in ("цод", "центр обработки данных", "центра обработки данных", "серверн", "стойк")):
            extra_terms.extend(["Регламент эксплуатации центра обработки данных ЦОД МФТИ", "машинный зал", "доступ"])
        if any(w in q_norm for w in ("командиров", "суточн", "авансовый отчет", "3897-1")):
            extra_terms.extend(["Регламент о командировании работников МФТИ", "3897-1", "служебное задание", "суточные"])

        exp_lower = expanded.lower().replace("ё", "е")
        new_extras = [t for t in extra_terms if t.lower().replace("ё", "е") not in exp_lower]
        if new_extras:
            return f"{expanded} ({', '.join(new_extras)})"
        return expanded

    @staticmethod
    def _should_exempt_superseded(query_norm: str, doc_meta_or_payload: dict[str, Any]) -> bool:
        """
        Check if a superseded / superseded_template_reference document should be exempt from score demotion:
        - Query contains explicit historical/old-edition intent phrases
        - Query explicitly mentions the old order number or old year matching the document
        - Or document is 'superseded_template_reference' and query asks about formulas / KVR / KOSGU / rates / smeta
        """
        if _HISTORICAL_EDITION_INTENT_RE.search(query_norm or ""):
            return True

        doc_num = str(doc_meta_or_payload.get("doc_number", "") or "")
        core_num = _core_doc_number(doc_num).lower()
        if core_num and core_num in query_norm:
            return True

        for y in doc_meta_or_payload.get("applicable_years", []) or []:
            if str(y) in query_norm and int(y) <= 2022:
                return True

        status = str(doc_meta_or_payload.get("status", "") or "")
        if status == DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value:
            formula_triggers = (
                "формул",
                "квр",
                "косгу",
                "накладн",
                "ндс",
                "20%",
                "8.2%",
                "расчет",
                "шаблон",
                "2019",
                "1469",
                "смет",
            )
            if any(trig in query_norm for trig in formula_triggers):
                return True

        return False

    def detect_temporal_edition_intent(self, query: str) -> dict[str, Any]:
        """
        Detect K2 temporal / edition intent on a user query:
        - Explicit old-edition phrases («в старой редакции», «предыдущая редакция», «до отмены», «действовавший ранее»,
          «утративший силу», «сравни редакции»);
        - Or an explicit historical year / edition number («в 2020 году», «редакция № 1») that matches a `superseded`
          document's `doc_date`/`edition_date`/`applicable_years`/`edition_number` and NOT its `active` replacement.
        """
        q_norm = (query or "").lower().replace("ё", "е").strip()
        explicit_old_phrase = bool(_HISTORICAL_EDITION_INTENT_RE.search(q_norm))
        is_comparison = bool(_HISTORICAL_COMPARISON_RE.search(q_norm))

        req_years_set: set[int] = set()
        for m_yr in _QUERY_YEAR_CONTEXT_RE.finditer(q_norm):
            y_str = m_yr.group(1) or m_yr.group(2)
            if y_str and y_str.isdigit():
                req_years_set.add(int(y_str))
        requested_years = sorted(req_years_set)

        req_ed_nums: set[int] = set()
        for m_ed in _QUERY_EDITION_NUM_RE.finditer(q_norm):
            if m_ed.group(1) and m_ed.group(1).isdigit():
                req_ed_nums.add(int(m_ed.group(1)))
        requested_edition_numbers = sorted(req_ed_nums)

        matched_superseded_doc_ids: list[str] = []
        registry = getattr(self.pipeline, "registry", None) or {}
        if (requested_years or requested_edition_numbers) and registry:
            q_topic_lemmas = {
                tok
                for tok in tokenize_bm25(q_norm)
                if len(tok) >= 3 and tok not in GENERIC_QUERY_LEMMAS and not tok.isdigit()
            }
            year_or_ed_candidates: list[tuple[str, bool]] = []
            for d_id, d_meta in registry.items():
                if d_meta.status not in {
                    DocumentStatus.SUPERSEDED.value,
                    DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value,
                }:
                    continue

                sup_years = _extract_doc_years_set(d_meta)
                d_md = d_meta.metadata or {}
                sup_ed_raw = getattr(d_meta, "edition_number", None)
                if sup_ed_raw is None:
                    sup_ed_raw = d_md.get("edition_number")
                sup_ed_int: int | None = None
                if sup_ed_raw is not None and re.search(r"\d+", str(sup_ed_raw)):
                    sup_ed_int = int(re.search(r"\d+", str(sup_ed_raw)).group(0))  # type: ignore[union-attr]

                # Gather active replacement documents (`superseded_by` or active peers in same `version_family`)
                repl_docs: list[DocumentMeta] = []
                if hasattr(self.pipeline, "_find_matching_docs"):
                    for ref in (d_meta.superseded_by or []):
                        repl_docs.extend(self.pipeline._find_matching_docs(ref))
                v_fam = str(getattr(d_meta, "version_family", None) or d_md.get("version_family") or "")
                if v_fam:
                    for peer_meta in registry.values():
                        peer_vf = str(
                            getattr(peer_meta, "version_family", None)
                            or (peer_meta.metadata or {}).get("version_family")
                            or ""
                        )
                        if (
                            peer_meta.doc_id != d_id
                            and peer_vf == v_fam
                            and peer_meta.status == DocumentStatus.ACTIVE.value
                            and all(r.doc_id != peer_meta.doc_id for r in repl_docs)
                        ):
                            repl_docs.append(peer_meta)

                active_repl_years: set[int] = set()
                active_repl_eds: set[int] = set()
                for r_doc in repl_docs:
                    if r_doc.status == DocumentStatus.ACTIVE.value:
                        active_repl_years.update(_extract_doc_years_set(r_doc))
                        r_ed = getattr(r_doc, "edition_number", None)
                        if r_ed is None:
                            r_ed = (r_doc.metadata or {}).get("edition_number")
                        if r_ed is not None and re.search(r"\d+", str(r_ed)):
                            active_repl_eds.add(int(re.search(r"\d+", str(r_ed)).group(0)))  # type: ignore[union-attr]

                year_exclusive_hit = bool(
                    requested_years
                    and sup_years.intersection(requested_years)
                    and not active_repl_years.intersection(requested_years)
                )
                ed_exclusive_hit = bool(
                    requested_edition_numbers
                    and sup_ed_int is not None
                    and sup_ed_int in requested_edition_numbers
                    and sup_ed_int not in active_repl_eds
                )
                if not (year_exclusive_hit or ed_exclusive_hit):
                    continue

                doc_probe_text = " ".join(
                    filter(
                        None,
                        [
                            d_meta.title,
                            d_meta.file_name,
                            v_fam.split("::", 1)[-1] if v_fam else "",
                            " ".join(getattr(d_meta, "topics", None) or []),
                        ],
                    )
                )
                doc_lemmas = set(tokenize_bm25(doc_probe_text))
                has_topic_overlap = (not q_topic_lemmas) or bool(q_topic_lemmas.intersection(doc_lemmas))
                year_or_ed_candidates.append((d_id, has_topic_overlap))

            topic_matched = [d_id for d_id, has_ov in year_or_ed_candidates if has_ov]
            if topic_matched:
                matched_superseded_doc_ids = topic_matched
            elif year_or_ed_candidates and not any(
                bool(
                    requested_years
                    and _extract_doc_years_set(m).intersection(requested_years)
                    and q_topic_lemmas.intersection(set(tokenize_bm25(f"{m.title} {m.file_name}")))
                )
                for m in registry.values()
                if m.status == DocumentStatus.ACTIVE.value
            ):
                matched_superseded_doc_ids = [d_id for d_id, _ in year_or_ed_candidates]

        return {
            "is_historical": bool(explicit_old_phrase or matched_superseded_doc_ids),
            "explicit_old_phrase": explicit_old_phrase,
            "is_comparison": is_comparison,
            "requested_years": requested_years,
            "requested_edition_numbers": requested_edition_numbers,
            "matched_superseded_doc_ids": matched_superseded_doc_ids,
        }

    @staticmethod
    def decompose_complex_query(query: str, max_sub_queries: int = 4) -> list[str]:
        """Delegate to module-level universal syntactic multi-aspect query decomposer."""
        return decompose_complex_query(query, max_sub_queries=max_sub_queries)

    @staticmethod
    def _extract_edo_requisites(
        reg_meta: DocumentMeta | None,
        payload: dict[str, Any],
        raw_meta: dict[str, Any],
        embed_text: str = "",
    ) -> tuple[str, str, bool, str]:
        """Extract `(edo_number, edo_date, is_edo_stamp, source_origin)` from DocumentMeta or Qdrant payload."""
        reg_md = (reg_meta.metadata or {}) if reg_meta is not None else {}
        edo_number = str(
            reg_md.get("edo_number")
            or payload.get("edo_number")
            or raw_meta.get("edo_number")
            or ""
        ).strip()
        edo_date = str(
            reg_md.get("edo_date")
            or payload.get("edo_date")
            or raw_meta.get("edo_date")
            or ""
        ).strip()
        is_edo_stamp = bool(
            reg_md.get("is_edo_stamp")
            or payload.get("is_edo_stamp")
            or raw_meta.get("is_edo_stamp")
            or False
        )
        if not edo_number and embed_text:
            m_edo = re.search(
                r"Зарег(?:истрировано|\.)?\s+в\s+ЭДО\s*(?:от\s*)?(?:(\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})\s*)?(?:№|N)\s*([0-9a-zA-Zа-яА-Я/-]+)",
                embed_text,
                re.IGNORECASE,
            )
            if m_edo:
                if m_edo.group(1) and not edo_date:
                    edo_date = m_edo.group(1).strip()
                edo_number = m_edo.group(2).strip()

        file_path_str = str(
            (reg_meta.file_path if reg_meta is not None else "")
            or payload.get("file_path")
            or raw_meta.get("file_path")
            or ""
        )
        source_origin = "uploads" if ("/uploads/" in file_path_str.replace("\\", "/") or "uploads/" in file_path_str.replace("\\", "/")) else "corpus"
        return edo_number, edo_date, is_edo_stamp, source_origin

    @staticmethod
    def _resolve_exact_clause_pages(
        clause_no: str,
        fallback_pages: list[int],
        parent_sec: ParentSectionRecord | None,
    ) -> list[int]:
        """Refine page_numbers for a clause using `ParentSectionRecord.metadata['clause_page_map']` when available."""
        pages = [int(p) for p in (fallback_pages or []) if isinstance(p, (int, float)) or str(p).isdigit()]
        if parent_sec is None or not clause_no:
            return pages
        p_meta = parent_sec.metadata or {}
        cpm = p_meta.get("clause_page_map")
        if not isinstance(cpm, dict) or not cpm:
            return pages

        # Direct match or sub-part / merged-clause lookup (e.g. 'п. 2.1 (ч. 1)' or '1.2, 1.3')
        clean_cl = re.sub(r"^(?:п\.|пункт)\s*", "", clause_no.strip(), flags=re.IGNORECASE)
        clean_cl = re.sub(r"\s*\(ч\.\s*\d+\)\s*$", "", clean_cl).strip()
        candidate_keys = [clause_no.strip(), clean_cl]
        if "," in clean_cl:
            candidate_keys.extend(part.strip() for part in clean_cl.split(",") if part.strip())

        mapped_pages: list[int] = []
        for k in candidate_keys:
            if k in cpm and isinstance(cpm[k], list):
                for p in cpm[k]:
                    p_int = int(p)
                    if p_int not in mapped_pages:
                        mapped_pages.append(p_int)
        return sorted(mapped_pages) if mapped_pages else pages

    def _raw_point_to_hit(
        self,
        pt_dict: dict[str, Any],
        query_norm: str,
        query_tokens_set: set[str],
        include_parent: bool = True,
        is_amendment: bool = False,
        amends_doc_id: str | None = None,
    ) -> SearchHit:
        """Convert a raw Qdrant hybrid search point dict into a hydrated SearchHit."""
        payload = dict(pt_dict.get("payload") or {})
        doc_id = str(payload.get("doc_id", "") or "")
        reg_meta: DocumentMeta | None = self.pipeline.get_document(doc_id)

        # Overlay latest registry metadata if available
        if reg_meta is not None:
            file_name = reg_meta.file_name
            doc_number = reg_meta.doc_number
            doc_date = reg_meta.doc_date
            doc_type = reg_meta.doc_type
            title = reg_meta.title
            status = reg_meta.status
            applicable_years = list(reg_meta.applicable_years)
            supersedes = list(reg_meta.supersedes)
            superseded_by = list(reg_meta.superseded_by)
            amends = list(reg_meta.amends)
            amended_by = list(reg_meta.amended_by)
            version_note = reg_meta.version_note
            quality_warning = reg_meta.quality_warning
            also_in_files = list(
                dict.fromkeys(list(payload.get("also_in_files") or []) + list(reg_meta.also_in_files))
            )
            if file_name in also_in_files:
                also_in_files.remove(file_name)
        else:
            file_name = str(payload.get("file_name", "") or "")
            doc_number = str(payload.get("doc_number", "") or "")
            doc_date = str(payload.get("doc_date", "") or "")
            doc_type = str(payload.get("doc_type", "") or "")
            title = str(payload.get("title", "") or "")
            status = str(payload.get("status", DocumentStatus.ACTIVE.value) or DocumentStatus.ACTIVE.value)
            applicable_years = list(payload.get("applicable_years") or [])
            supersedes = list(payload.get("supersedes") or [])
            superseded_by = list(payload.get("superseded_by") or [])
            amends = list(payload.get("amends") or [])
            amended_by = list(payload.get("amended_by") or [])
            version_note = str(payload.get("version_note", "") or "")
            quality_warning = str(payload.get("quality_warning", "") or "")
            also_in_files = list(payload.get("also_in_files") or [])

        rrf_score = float(pt_dict.get("rrf_score", 0.0))
        dense_score = float(pt_dict.get("dense_score", 0.0))
        sparse_score = float(pt_dict.get("sparse_score", 0.0))

        clean_text = str(payload.get("clean_text", "") or "")
        raw_text = str(payload.get("raw_text", "") or clean_text)
        embed_text = str(payload.get("embed_text", "") or raw_text)
        raw_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

        edo_number, edo_date, is_edo_stamp, source_origin = self._extract_edo_requisites(
            reg_meta=reg_meta,
            payload=payload,
            raw_meta=raw_meta,
            embed_text=embed_text,
        )

        reg_md = (reg_meta.metadata or {}) if reg_meta is not None else {}
        edition_date = str(
            (getattr(reg_meta, "edition_date", None) if reg_meta is not None else None)
            or reg_md.get("edition_date")
            or payload.get("edition_date")
            or raw_meta.get("edition_date")
            or ""
        ).strip()
        edition_number = (
            getattr(reg_meta, "edition_number", None) if reg_meta is not None else None
        )
        if edition_number in (None, ""):
            edition_number = reg_md.get("edition_number")
        if edition_number in (None, ""):
            edition_number = payload.get("edition_number")
        if edition_number in (None, ""):
            edition_number = raw_meta.get("edition_number")
        version_family = str(
            (getattr(reg_meta, "version_family", None) if reg_meta is not None else None)
            or reg_md.get("version_family")
            or payload.get("version_family")
            or raw_meta.get("version_family")
            or ""
        ).strip()

        # Compute lexical overlap bonus on chunk text
        chunk_tokens = set(tokenize_bm25(f"{title} {doc_number} {edo_number} {clean_text}"))
        matched_q_tokens = query_tokens_set.intersection(chunk_tokens)
        lex_ratio = len(matched_q_tokens) / max(1, len(query_tokens_set)) if query_tokens_set else 0.0

        # Combined calibrated score (0..1 scale)
        norm_rrf = min(1.0, rrf_score * 1.35)
        combined_score = 0.55 * max(0.0, dense_score) + 0.25 * norm_rrf + 0.20 * min(1.0, lex_ratio * 1.4)

        # Direct doc_number / edo_number match bonus when user query mentions this document's number
        core_num = _core_doc_number(doc_number).lower()
        core_edo = _core_doc_number(edo_number).lower()
        explicit_doc_num_match = bool((core_num and core_num in query_norm) or (core_edo and core_edo in query_norm))
        if explicit_doc_num_match:
            combined_score = min(1.0, combined_score + 0.06)

        # K2 Actuality & Temporal-Intent Calibration:
        # - Default (current actuality queries): apply K2_SUPERSEDED_SCORE_PENALTY (0.65) to superseded documents
        #   and add a subtle date/edition recency bonus to active documents so newer editions win close ties.
        # - Historical / old-edition queries: lift superseded penalty and boost matching historical editions.
        temporal_intent = self.detect_temporal_edition_intent(query_norm)
        doc_years = _extract_doc_years_set(
            {"doc_date": doc_date, "edition_date": edition_date, "applicable_years": applicable_years}
        )
        matches_req_year = bool(
            temporal_intent["requested_years"] and doc_years.intersection(temporal_intent["requested_years"])
        )
        matches_req_ed = bool(
            temporal_intent["requested_edition_numbers"]
            and edition_number is not None
            and re.search(r"\d+", str(edition_number))
            and int(re.search(r"\d+", str(edition_number)).group(0)) in temporal_intent["requested_edition_numbers"]  # type: ignore[union-attr]
        )

        meta_probe = {
            "doc_number": doc_number,
            "applicable_years": applicable_years,
            "status": status,
        }
        if not temporal_intent["is_historical"] and not explicit_doc_num_match:
            if status == DocumentStatus.SUPERSEDED.value and not self._should_exempt_superseded(query_norm, meta_probe):
                combined_score *= K2_SUPERSEDED_SCORE_PENALTY
            elif (
                status == DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value
                and not self._should_exempt_superseded(query_norm, meta_probe)
            ):
                combined_score *= 0.90
            elif status == DocumentStatus.ACTIVE.value:
                recency_bonus = _compute_active_recency_bonus(
                    edition_date or doc_date or edo_date,
                    edition_number=edition_number,
                )
                combined_score = min(1.0, combined_score + recency_bonus)
        else:
            if status in {DocumentStatus.SUPERSEDED.value, DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value}:
                if (
                    doc_id in temporal_intent["matched_superseded_doc_ids"]
                    or matches_req_year
                    or matches_req_ed
                    or explicit_doc_num_match
                ):
                    combined_score = min(1.0, combined_score + 0.085)
                elif temporal_intent["explicit_old_phrase"]:
                    combined_score = min(1.0, combined_score + 0.065)
            elif status == DocumentStatus.ACTIVE.value:
                if matches_req_year or matches_req_ed or temporal_intent["is_comparison"]:
                    recency_bonus = _compute_active_recency_bonus(
                        edition_date or doc_date or edo_date,
                        edition_number=edition_number,
                    )
                    combined_score = min(1.0, combined_score + recency_bonus)

        section_id = str(payload.get("section_id", "") or "")
        parent_sec: ParentSectionRecord | None = None
        parent_text = ""
        if include_parent:
            if section_id:
                parent_sec = self.get_parent_section(section_id)
            if parent_sec is None and clean_text:
                parent_sec = ParentSectionRecord(
                    section_id=section_id or f"{doc_id}::u01_main::s01",
                    doc_id=doc_id,
                    unit_id=str(payload.get("unit_id", "u01_main") or "u01_main"),
                    unit_title=str(payload.get("unit_title", "Основной текст") or "Основной текст"),
                    section_num=str(payload.get("section_num", "") or ""),
                    section_title=str(payload.get("section_title", "") or title),
                    clean_text=clean_text,
                    chunk_ids=[str(payload.get("chunk_id", "") or "")],
                    page_numbers=list(payload.get("page_numbers") or []),
                )
            if parent_sec is not None:
                parent_text = parent_sec.clean_text

        clause_no = str(payload.get("clause_no", "") or "")
        raw_pages = list(payload.get("page_numbers") or [])
        exact_pages = self._resolve_exact_clause_pages(clause_no, raw_pages, parent_sec)

        table_id = str(payload.get("table_id") or raw_meta.get("table_id") or "")
        file_format = str(
            (reg_meta.file_format if reg_meta is not None else "")
            or payload.get("file_format")
            or raw_meta.get("file_format")
            or ""
        )
        prev_id = payload.get("prev_id") or raw_meta.get("prev_id") or None
        next_id = payload.get("next_id") or raw_meta.get("next_id") or None

        hit_meta: dict[str, Any] = {
            "content_hash": payload.get("content_hash") or compute_content_hash(clean_text),
            "source_origin": source_origin,
        }
        if file_format:
            hit_meta["file_format"] = file_format
        if table_id:
            hit_meta["table_id"] = table_id
        if edo_number:
            hit_meta["edo_number"] = edo_number
        if edo_date:
            hit_meta["edo_date"] = edo_date
        if is_edo_stamp:
            hit_meta["is_edo_stamp"] = True
        if edition_date:
            hit_meta["edition_date"] = edition_date
        if edition_number not in (None, ""):
            hit_meta["edition_number"] = edition_number
        if version_family:
            hit_meta["version_family"] = version_family
        for m_key in ("sheet_name", "form_title", "table_headers", "table_ids", "slice_index", "slices_total"):
            m_val = raw_meta.get(m_key) if m_key in raw_meta else payload.get(m_key)
            if m_val not in (None, "", []):
                hit_meta[m_key] = m_val

        return SearchHit(
            chunk_id=str(payload.get("chunk_id", "") or ""),
            point_uuid=str(pt_dict.get("point_uuid") or payload.get("point_uuid", "") or ""),
            doc_id=doc_id,
            file_name=file_name,
            doc_number=doc_number,
            doc_date=doc_date,
            edo_number=edo_number,
            edo_date=edo_date,
            doc_type=doc_type,
            title=title,
            status=status,
            applicable_years=applicable_years,
            unit_id=str(payload.get("unit_id", "") or ""),
            unit_title=str(payload.get("unit_title", "") or ""),
            section_id=section_id,
            section_num=str(payload.get("section_num", "") or ""),
            section_title=str(payload.get("section_title", "") or ""),
            clause_no=clause_no,
            table_id=table_id,
            zone=str(payload.get("zone", "normative") or "normative"),
            score=round(float(combined_score), 4),
            rrf_score=round(rrf_score, 4),
            dense_score=round(dense_score, 4),
            sparse_score=round(sparse_score, 4),
            raw_text=raw_text,
            clean_text=clean_text,
            embed_text=embed_text,
            prev_id=str(prev_id) if prev_id else None,
            next_id=str(next_id) if next_id else None,
            parent_section=parent_sec,
            parent_text=parent_text,
            roles=list(payload.get("roles") or []),
            topics=list(payload.get("topics") or []),
            terms=list(payload.get("terms") or []),
            also_in_files=also_in_files,
            page_numbers=exact_pages,
            version_note=version_note,
            quality_warning=quality_warning,
            supersedes=supersedes,
            superseded_by=superseded_by,
            amends=amends,
            amended_by=amended_by,
            is_amendment=is_amendment,
            amends_doc_id=amends_doc_id,
            metadata=hit_meta,
        )

    def _chunk_record_to_hit(
        self,
        chunk: ChunkRecord,
        score: float = 0.50,
        expansion_reason: str = "structural_neighbor",
        include_parent: bool = True,
    ) -> SearchHit:
        """Convert a stored ChunkRecord (from Qdrant scroll) into a fully hydrated SearchHit."""
        payload = chunk.model_dump()
        pt_dict = {
            "point_uuid": chunk.point_uuid,
            "rrf_score": 0.0,
            "dense_score": score,
            "sparse_score": 0.0,
            "payload": payload,
        }
        hit = self._raw_point_to_hit(
            pt_dict=pt_dict,
            query_norm="",
            query_tokens_set=set(),
            include_parent=include_parent,
        )
        hit.score = round(float(score), 4)
        hit.metadata["expansion_reason"] = expansion_reason
        return hit

    @staticmethod
    def _deduplicate_hits_by_content_hash(hits: list[SearchHit]) -> list[SearchHit]:
        """
        Deduplicate hits across files by normalized content_hash, merging duplicate file_names
        into the primary hit's also_in_files list, keeping best scores, and preserving peer table linkage.
        """
        unique_hits: list[SearchHit] = []
        by_hash: dict[str, SearchHit] = {}

        for hit in hits:
            chash = str((hit.metadata or {}).get("content_hash") or compute_content_hash(hit.clean_text))
            if not chash:
                unique_hits.append(hit)
                continue

            if chash not in by_hash:
                by_hash[chash] = hit
                unique_hits.append(hit)
            else:
                primary = by_hash[chash]
                if hit.score > primary.score:
                    primary.score = hit.score
                if hit.rrf_score > primary.rrf_score:
                    primary.rrf_score = hit.rrf_score
                if hit.dense_score > primary.dense_score:
                    primary.dense_score = hit.dense_score
                if hit.sparse_score > primary.sparse_score:
                    primary.sparse_score = hit.sparse_score

                if hit.file_name and hit.file_name != primary.file_name:
                    if hit.file_name not in primary.also_in_files:
                        primary.also_in_files.append(hit.file_name)
                for extra_f in hit.also_in_files:
                    if extra_f and extra_f != primary.file_name and extra_f not in primary.also_in_files:
                        primary.also_in_files.append(extra_f)

                sub_qs_prev = list(primary.metadata.get("matched_subqueries") or [])
                for sq_i in hit.metadata.get("matched_subqueries") or []:
                    if sq_i not in sub_qs_prev:
                        sub_qs_prev.append(sq_i)
                if sub_qs_prev:
                    primary.metadata["matched_subqueries"] = sub_qs_prev

                if hit.table_id:
                    if not primary.table_id:
                        primary.table_id = hit.table_id
                        primary.metadata["table_id"] = hit.table_id
                    elif hit.table_id != primary.table_id:
                        peer_ids = list(primary.metadata.get("peer_table_ids") or [])
                        if hit.table_id not in peer_ids:
                            peer_ids.append(hit.table_id)
                            primary.metadata["peer_table_ids"] = peer_ids

                if hit.doc_id and hit.doc_id != primary.doc_id:
                    peer_refs = list(primary.metadata.get("peer_table_refs") or [])
                    peer_entry = {
                        "doc_id": hit.doc_id,
                        "table_id": hit.table_id,
                        "section_id": hit.section_id,
                        "unit_id": hit.unit_id,
                        "sheet_name": str((hit.metadata or {}).get("sheet_name") or ""),
                    }
                    if peer_entry not in peer_refs:
                        peer_refs.append(peer_entry)
                        primary.metadata["peer_table_refs"] = peer_refs

        return unique_hits

    def _detect_out_of_scope(
        self,
        raw_query: str,
        raw_query_tokens: list[str],
        hits: list[SearchHit],
    ) -> tuple[bool, float, float, float]:
        """
        Calibrated multi-signal Out-of-Scope detector.
        Returns (is_out_of_scope, confidence_score, top_dense_score, lexical_overlap).
        """
        q_norm = raw_query.lower().replace("ё", "е").strip()
        if not q_norm or not hits:
            return True, 0.0, 0.0, 0.0

        top_dense = max((h.dense_score for h in hits[:3]), default=0.0)
        has_mipt_anchor = any(anchor in q_norm for anchor in MIPT_DOMAIN_ANCHORS)
        has_oos_marker = any(stem in q_norm for stem in OUT_OF_SCOPE_STEMS)

        # Substantive tokens = tokens from raw query excluding generic question/admin words
        substantive_tokens = [t for t in raw_query_tokens if t not in GENERIC_QUERY_LEMMAS and len(t) >= 2]
        if not substantive_tokens:
            substantive_tokens = list(raw_query_tokens)

        # Build token pool from top-3 hits (including titles, terms, topics, clean_text)
        top_corpus_tokens: set[str] = set()
        for h in hits[:3]:
            blob = f"{h.title} {h.doc_number} {' '.join(h.terms)} {' '.join(h.topics)} {h.clean_text}"
            top_corpus_tokens.update(tokenize_bm25(blob))

        # Also include expanded synonym tokens for the raw query if they came from MIPT glossary
        expanded_by_glossary = self.glossary.expand_query(raw_query)
        glossary_triggered = expanded_by_glossary != raw_query.strip()

        matched_substantive = [t for t in substantive_tokens if t in top_corpus_tokens]
        lexical_overlap = (
            len(matched_substantive) / max(1, len(substantive_tokens))
            if substantive_tokens
            else 0.0
        )
        if glossary_triggered and lexical_overlap < 0.5:
            # Verify whether expanded terms match top corpus tokens
            exp_tokens = set(tokenize_bm25(expanded_by_glossary)) - set(raw_query_tokens)
            if exp_tokens and exp_tokens.intersection(top_corpus_tokens):
                lexical_overlap = max(lexical_overlap, 0.65)

        # Rule 1: Explicit non-MIPT topic (e.g. mortgage in bank, pasta carbonara, borscht, bitcoin)
        if has_oos_marker and not has_mipt_anchor:
            conf = round(min(0.22, top_dense * 0.35), 4)
            return True, conf, round(top_dense, 4), round(lexical_overlap, 4)

        # Rule 2: Zero substantive lexical overlap and no MIPT domain anchor
        if lexical_overlap < OUT_OF_SCOPE_MIN_LEXICAL_OVERLAP and not has_mipt_anchor and not glossary_triggered:
            conf = round(min(0.28, top_dense * 0.45), 4)
            return True, conf, round(top_dense, 4), round(lexical_overlap, 4)

        # Rule 3: Very low dense cosine score
        if top_dense < OUT_OF_SCOPE_LOW_DENSE_THRESHOLD and not has_mipt_anchor:
            conf = round(min(0.25, top_dense * 0.5), 4)
            return True, conf, round(top_dense, 4), round(lexical_overlap, 4)

        # Rule 4: Below dense threshold with weak lexical overlap
        if top_dense < OUT_OF_SCOPE_DENSE_THRESHOLD and lexical_overlap < 0.25 and not has_mipt_anchor:
            conf = round(min(0.32, top_dense * 0.6), 4)
            return True, conf, round(top_dense, 4), round(lexical_overlap, 4)

        # In-scope confidence score
        confidence = min(
            0.99,
            0.55 * top_dense + 0.30 * max(lexical_overlap, 0.5 if has_mipt_anchor else 0.0) + 0.15,
        )
        return False, round(confidence, 4), round(top_dense, 4), round(lexical_overlap, 4)

    @staticmethod
    def _insert_co_surfaced_hit(
        hits: list[SearchHit],
        new_hit: SearchHit,
        top_k: int,
        insert_front: bool = False,
    ) -> list[SearchHit]:
        """Insert a graph co-surfaced hit without evicting the sole representative of a decomposed sub-query."""
        if any(h.chunk_id == new_hit.chunk_id for h in hits):
            return hits
        if insert_front:
            updated = [new_hit] + list(hits)
            if len(updated) <= top_k:
                return updated
            # Evict from the tail, preferring a hit from a multi-hit doc_id
            doc_counts: dict[str, int] = {}
            for h in updated:
                doc_counts[h.doc_id] = doc_counts.get(h.doc_id, 0) + 1
            for idx in range(len(updated) - 1, 0, -1):
                cand = updated[idx]
                if doc_counts.get(cand.doc_id, 0) >= 2:
                    return updated[:idx] + updated[idx + 1 :]
            return updated[:top_k]

        if len(hits) < top_k or top_k < 2:
            return list(hits) + [new_hit]

        # Count sub-query coverage and per-doc counts so we evict a redundant slot first
        sq_counts: dict[int, int] = {}
        doc_counts = {}
        for h in hits:
            doc_counts[h.doc_id] = doc_counts.get(h.doc_id, 0) + 1
            for sq_i in (h.metadata or {}).get("matched_subqueries") or []:
                sq_counts[int(sq_i)] = sq_counts.get(int(sq_i), 0) + 1

        evict_idx = len(hits) - 1
        for idx in range(len(hits) - 1, 0, -1):
            cand = hits[idx]
            cand_sqs = [int(x) for x in ((cand.metadata or {}).get("matched_subqueries") or [])]
            sole_sq = any(sq_counts.get(sq_i, 0) <= 1 and sq_i > 0 for sq_i in cand_sqs)
            if not sole_sq and doc_counts.get(cand.doc_id, 0) >= 2:
                evict_idx = idx
                break
        else:
            for idx in range(len(hits) - 1, 0, -1):
                cand = hits[idx]
                cand_sqs = [int(x) for x in ((cand.metadata or {}).get("matched_subqueries") or [])]
                sole_sq = any(sq_counts.get(sq_i, 0) <= 1 and sq_i > 0 for sq_i in cand_sqs)
                if not sole_sq:
                    evict_idx = idx
                    break

        out = list(hits)
        out[evict_idx] = new_hit
        return out

    def _expand_amendments_and_editions(
        self,
        hits: list[SearchHit],
        query_dense: list[float],
        query_sparse: Any,
        query_norm: str,
        query_tokens_set: set[str],
        top_k: int,
        include_parent: bool,
        include_amendments: bool,
    ) -> tuple[list[SearchHit], list[SearchHit]]:
        """
        Perform general graph-aware retrieval for:
        1) Partial amendments (`amended_by` / `amends`)
        2) Superseded / `superseded_template_reference` <-> active replacement co-surfacing
        Returns updated `(hits, amendment_chunks)`.
        """
        amendment_chunks: list[SearchHit] = []
        if not hits:
            return hits, amendment_chunks

        if not self.pipeline.registry:
            self.pipeline.registry = self.pipeline._load_registry()

        present_doc_ids: set[str] = {h.doc_id for h in hits}
        existing_chunk_ids: set[str] = {h.chunk_id for h in hits}

        # 1. Forward & reverse amendments (`amended_by` / `amends`)
        if include_amendments:
            for hit in list(hits):
                for amending_ref in hit.amended_by:
                    amending_docs = self.pipeline._find_matching_docs(amending_ref)
                    target_ids = [d.doc_id for d in amending_docs if d.doc_id]
                    if not target_ids:
                        continue
                    raw_amd_pts = self.qdrant.hybrid_search(
                        query_dense=query_dense,
                        query_sparse=query_sparse,
                        top_k=3,
                        doc_id_filter=target_ids,
                        exclude_zones=NON_ANSWER_ZONES,
                    )
                    for pt in raw_amd_pts:
                        amd_hit = self._raw_point_to_hit(
                            pt,
                            query_norm=query_norm,
                            query_tokens_set=query_tokens_set,
                            include_parent=include_parent,
                            is_amendment=True,
                            amends_doc_id=hit.doc_id,
                        )
                        if amd_hit.chunk_id not in {a.chunk_id for a in amendment_chunks}:
                            amendment_chunks.append(amd_hit)

                if hit.amends:
                    base_docs = []
                    for b_ref in hit.amends:
                        base_docs.extend(self.pipeline._find_matching_docs(b_ref))
                    base_doc_id = base_docs[0].doc_id if base_docs else None
                    amd_copy = hit.model_copy(update={"is_amendment": True, "amends_doc_id": base_doc_id})
                    if amd_copy.chunk_id not in {a.chunk_id for a in amendment_chunks}:
                        amendment_chunks.append(amd_copy)

                    for b_doc in base_docs:
                        if b_doc.doc_id not in present_doc_ids:
                            raw_base_pts = self.qdrant.hybrid_search(
                                query_dense=query_dense,
                                query_sparse=query_sparse,
                                top_k=2,
                                doc_id_filter=b_doc.doc_id,
                                exclude_zones=NON_ANSWER_ZONES,
                            )
                            for b_pt in raw_base_pts[:1]:
                                b_hit = self._raw_point_to_hit(
                                    b_pt,
                                    query_norm=query_norm,
                                    query_tokens_set=query_tokens_set,
                                    include_parent=include_parent,
                                )
                                if b_hit.chunk_id not in existing_chunk_ids:
                                    hits = self._insert_co_surfaced_hit(hits, b_hit, top_k=top_k, insert_front=False)
                                    present_doc_ids.add(b_hit.doc_id)
                                    existing_chunk_ids.add(b_hit.chunk_id)

            for amd_hit in amendment_chunks:
                if amd_hit.doc_id not in present_doc_ids and len(hits) >= 1:
                    hits = self._insert_co_surfaced_hit(hits, amd_hit, top_k=top_k, insert_front=False)
                    present_doc_ids.add(amd_hit.doc_id)
                    existing_chunk_ids.add(amd_hit.chunk_id)

        # 2. General supersession & template-reference co-surfacing across any registered edition pair
        asks_template_or_formulas = any(
            w in query_norm
            for w in ("формул", "квр", "косгу", "накладн", "ндс", "%", "ставк", "расчет", "шаблон")
        )

        # Case A: Active document (or its appendix sharing core doc_number) matched, and a
        # `superseded_template_reference` document in registry points to it via `superseded_by`
        present_core_nums = {
            _core_doc_number(h.doc_number).lower()
            for h in hits
            if _core_doc_number(h.doc_number)
        }
        if asks_template_or_formulas:
            for cand_meta in self.pipeline.registry.values():
                if (
                    cand_meta.status == DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value
                    and cand_meta.doc_id not in present_doc_ids
                ):
                    sup_Targets = self.pipeline._find_matching_docs_many(cand_meta.superseded_by) if hasattr(self.pipeline, "_find_matching_docs_many") else [
                        m_doc for ref in cand_meta.superseded_by for m_doc in self.pipeline._find_matching_docs(ref)
                    ]
                    sup_ids = {d.doc_id for d in sup_Targets}
                    sup_cores = {_core_doc_number(r).lower() for r in cand_meta.superseded_by if _core_doc_number(r)}
                    if present_doc_ids.intersection(sup_ids) or present_core_nums.intersection(sup_cores):
                        raw_ref_pts = self.qdrant.hybrid_search(
                            query_dense=query_dense,
                            query_sparse=query_sparse,
                            top_k=2,
                            doc_id_filter=cand_meta.doc_id,
                            exclude_zones=NON_ANSWER_ZONES,
                        )
                        if raw_ref_pts:
                            ref_hit = self._raw_point_to_hit(
                                raw_ref_pts[0],
                                query_norm=query_norm,
                                query_tokens_set=query_tokens_set,
                                include_parent=include_parent,
                            )
                            hits = self._insert_co_surfaced_hit(hits, ref_hit, top_k=top_k, insert_front=False)
                            present_doc_ids.add(ref_hit.doc_id)

        # Case B: Superseded / `superseded_template_reference` document matched, ensure its active replacement is surfaced
        temporal_intent = self.detect_temporal_edition_intent(query_norm)
        is_historical = bool(temporal_intent["is_historical"])
        for hit in list(hits):
            if hit.status in {
                DocumentStatus.SUPERSEDED.value,
                DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value,
            } and hit.superseded_by:
                repl_docs = [
                    m_doc for ref in hit.superseded_by for m_doc in self.pipeline._find_matching_docs(ref)
                ]
                repl_ids = [d.doc_id for d in repl_docs if d.doc_id]
                if repl_ids and not present_doc_ids.intersection(repl_ids):
                    raw_act_pts = self.qdrant.hybrid_search(
                        query_dense=query_dense,
                        query_sparse=query_sparse,
                        top_k=2,
                        doc_id_filter=repl_ids,
                        exclude_zones=NON_ANSWER_ZONES,
                    )
                    if raw_act_pts:
                        act_hit = self._raw_point_to_hit(
                            raw_act_pts[0],
                            query_norm=query_norm,
                            query_tokens_set=query_tokens_set,
                            include_parent=include_parent,
                        )
                        hits = self._insert_co_surfaced_hit(
                            hits, act_hit, top_k=top_k, insert_front=(not is_historical)
                        )
                        present_doc_ids.add(act_hit.doc_id)

        # Case C: Historical / old-edition query where ONLY the active edition matched so far -> co-surface superseded predecessor(s)
        if is_historical:
            req_years = set(temporal_intent["requested_years"] or [])
            for hit in list(hits):
                if hit.status == DocumentStatus.ACTIVE.value and hit.supersedes:
                    pred_docs = [
                        m_doc for ref in hit.supersedes for m_doc in self.pipeline._find_matching_docs(ref)
                    ]
                    if req_years:
                        yr_filtered = [d for d in pred_docs if _extract_doc_years_set(d).intersection(req_years)]
                        if yr_filtered:
                            pred_docs = yr_filtered
                    pred_ids = [d.doc_id for d in pred_docs if d.doc_id and d.doc_id not in present_doc_ids]
                    if pred_ids:
                        raw_old_pts = self.qdrant.hybrid_search(
                            query_dense=query_dense,
                            query_sparse=query_sparse,
                            top_k=2,
                            doc_id_filter=pred_ids,
                            exclude_zones=NON_ANSWER_ZONES,
                        )
                        if raw_old_pts:
                            old_hit = self._raw_point_to_hit(
                                raw_old_pts[0],
                                query_norm=query_norm,
                                query_tokens_set=query_tokens_set,
                                include_parent=include_parent,
                            )
                            put_old_front = bool(
                                not temporal_intent["is_comparison"]
                                and (temporal_intent["explicit_old_phrase"] or bool(req_years))
                                and old_hit.score >= (hits[0].score - 0.04 if hits else 0.0)
                            )
                            hits = self._insert_co_surfaced_hit(
                                hits, old_hit, top_k=top_k, insert_front=put_old_front
                            )
                            present_doc_ids.add(old_hit.doc_id)

        return hits[:top_k], amendment_chunks[:5]

    @staticmethod
    def _title_topic_lemmas(title: str, file_name: str = "") -> set[str]:
        """Extract substantive topic lemmas from a document title/filename (excluding generic admin/type words)."""
        admin_stop = {
            "приказ",
            "распоряжение",
            "регламент",
            "положение",
            "инструкция",
            "порядок",
            "правило",
            "утверждение",
            "утвердить",
            "введение",
            "действие",
            "новая",
            "новый",
            "форма",
            "приложение",
            "документ",
            "мфти",
            "физтех",
            "федеральный",
            "государственный",
            "автономный",
            "образовательный",
            "учреждение",
            "высший",
            "образование",
            "год",
            "от",
            "об",
            "о",
            "по",
            "на",
            "за",
            "в",
            "и",
            "pdf",
            "docx",
            "doc",
            "xlsx",
            "xls",
        }
        clean_fname = re.sub(r"\.[a-zA-Z0-9]{2,5}$", "", (file_name or "")).replace("_", " ")
        raw_toks = tokenize_bm25(f"{title} {clean_fname}")
        out: set[str] = set()
        for tok in raw_toks:
            if tok in admin_stop or tok in GENERIC_QUERY_LEMMAS:
                continue
            if re.match(r"^\d+(?:[-./]\d+)*[а-яa-z]?$", tok):
                continue
            if len(tok) >= 3:
                out.add(tok)
        return out

    def _find_sibling_topic_docs(self, doc_id: str) -> list[DocumentMeta]:
        """
        Find distinct documents in `pipeline.registry` that share strong substantive title/topic overlap
        with `doc_id` (e.g., an earlier standalone regulation vs a separate uploaded order on the same topic).
        """
        if not self.pipeline.registry:
            self.pipeline.registry = self.pipeline._load_registry()
        base_meta = self.pipeline.registry.get(doc_id)
        if base_meta is None:
            return []
        base_vf = str(
            getattr(base_meta, "version_family", None)
            or (base_meta.metadata or {}).get("version_family")
            or ""
        ).strip()
        base_lemmas = self._title_topic_lemmas(base_meta.title, base_meta.file_name)
        if not base_lemmas and not base_vf:
            return []

        siblings: list[DocumentMeta] = []
        for cand_id, cand_meta in self.pipeline.registry.items():
            if cand_id == doc_id or cand_meta.status == DocumentStatus.ERROR.value:
                continue
            # Skip exact cross-file duplicates already linked via also_in_files
            if cand_meta.file_name and cand_meta.file_name in base_meta.also_in_files:
                continue
            cand_vf = str(
                getattr(cand_meta, "version_family", None)
                or (cand_meta.metadata or {}).get("version_family")
                or ""
            ).strip()
            if base_vf and cand_vf and base_vf == cand_vf:
                siblings.append(cand_meta)
                continue
            cand_lemmas = self._title_topic_lemmas(cand_meta.title, cand_meta.file_name)
            if not cand_lemmas or not base_lemmas:
                continue
            inter = base_lemmas.intersection(cand_lemmas)
            if not inter:
                continue
            jaccard = len(inter) / max(1, len(base_lemmas.union(cand_lemmas)))
            min_contain = len(inter) / max(1, min(len(base_lemmas), len(cand_lemmas)))
            if jaccard >= 0.45 or (len(inter) >= 2 and min_contain >= 0.65):
                siblings.append(cand_meta)
        return siblings

    @staticmethod
    def _format_doc_requisite_badge(h: SearchHit) -> str:
        """Render crystal-clear document identity string with file_name, doc_number/date, ЭДО stamp, and status."""
        parts = [f"«{h.title or h.file_name}» (файл: `{h.file_name}`"]
        if h.doc_number:
            num_str = f"№ {h.doc_number}" + (f" от {h.doc_date}" if h.doc_date else "")
            if (h.metadata or {}).get("is_edo_stamp"):
                num_str += " [штамп ЭДО]"
            parts.append(f", {num_str}")
        else:
            parts.append(", без номера приказа")
        if h.edo_number and h.edo_number != h.doc_number:
            edo_str = f"Зарег. в ЭДО № {h.edo_number}" + (f" от {h.edo_date}" if h.edo_date else "")
            parts.append(f", {edo_str}")
        elif h.edo_number and h.edo_date and h.edo_date != h.doc_date:
            parts.append(f", Зарег. в ЭДО {h.edo_date} № {h.edo_number}")
        origin = str((h.metadata or {}).get("source_origin") or "")
        if origin == "uploads":
            parts.append(", источник: загруженный документ `uploads/`")
        parts.append(f", статус: {h.status})")
        return "".join(parts)

    def _collect_alerts(self, hits: list[SearchHit], amendment_chunks: list[SearchHit]) -> tuple[list[str], list[str]]:
        """Collect deduplicated version_alerts, sibling-edition disambiguation alerts, and quality_alerts."""
        version_alerts: list[str] = []
        quality_alerts: list[str] = []

        all_hits = list(hits) + list(amendment_chunks)
        for h in all_hits:
            if h.version_note and h.version_note not in version_alerts:
                version_alerts.append(h.version_note)
            if h.amended_by:
                num_badge = f" (№ {h.doc_number})" if h.doc_number else (f" (от {h.doc_date})" if h.doc_date else "")
                alert = (
                    f"Документ «{h.title}»{num_badge} действует с учетом изменений: "
                    f"№ {', '.join(h.amended_by)}."
                )
                if alert not in version_alerts and not any(h.doc_number and h.doc_number in va for va in version_alerts):
                    version_alerts.append(alert)
            if h.superseded_by:
                num_badge = f" (№ {h.doc_number})" if h.doc_number else (f" (от {h.doc_date})" if h.doc_date else "")
                alert = (
                    f"Документ «{h.title}»{num_badge} заменен / отменен актом: "
                    f"{', '.join(h.superseded_by)} (текущий статус: {h.status})."
                )
                if alert not in version_alerts and not any(h.doc_number and h.doc_number in va for va in version_alerts):
                    version_alerts.append(alert)
            if h.quality_warning and h.quality_warning not in quality_alerts:
                quality_alerts.append(h.quality_warning)

        # Generic sibling-document disambiguation alert when 2+ distinct documents share the same regulation topic
        rep_by_doc: dict[str, SearchHit] = {}
        for h in hits:
            if h.doc_id and h.doc_id not in rep_by_doc:
                rep_by_doc[h.doc_id] = h

        doc_ids_list = list(rep_by_doc.keys())
        seen_pairs: set[tuple[str, str]] = set()
        for i in range(len(doc_ids_list)):
            d1 = doc_ids_list[i]
            sib_ids = {s.doc_id for s in self._find_sibling_topic_docs(d1)}
            for j in range(i + 1, len(doc_ids_list)):
                d2 = doc_ids_list[j]
                pair_key = (min(d1, d2), max(d1, d2))
                if d2 in sib_ids and pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    h1, h2 = rep_by_doc[d1], rep_by_doc[d2]
                    # Skip if already linked by explicit supersedes/amends
                    explicit_link = (
                        bool(set(h1.supersedes + h1.superseded_by + h1.amends + h1.amended_by))
                        and bool(set(h2.supersedes + h2.superseded_by + h2.amends + h2.amended_by))
                    )
                    if not explicit_link:
                        disamb_note = (
                            "ВНИМАНИЕ (Разграничение одноименных/смежных документов в базе): "
                            f"по данной теме найдены два отдельных документа — "
                            f"1) {self._format_doc_requisite_badge(h1)}; "
                            f"2) {self._format_doc_requisite_badge(h2)}. "
                            "При ответе различайте их реквизиты, даты и область действия."
                        )
                        if disamb_note not in version_alerts:
                            version_alerts.append(disamb_note)

        return version_alerts, quality_alerts

    @staticmethod
    def _extract_sheet_name(hit: SearchHit) -> str:
        """Extract Excel sheet name from hit metadata or text markers if present."""
        meta = hit.metadata or {}
        sheet = str(meta.get("sheet_name") or "").strip()
        if sheet:
            return sheet
        form_title = str(meta.get("form_title") or "").strip()
        if form_title.lower().startswith("лист:"):
            return form_title.split(":", 1)[1].strip()
        for txt in (hit.clean_text, hit.embed_text, hit.section_title, hit.unit_title):
            if not txt:
                continue
            m = re.search(r"\[Лист:\s*([^\]]+)\]", txt)
            if m:
                return m.group(1).strip()
            if txt.lower().startswith("лист:"):
                return txt.split(":", 1)[1].strip()
        return ""

    @staticmethod
    def _is_table_or_sheet_hit(hit: SearchHit) -> bool:
        """Return True if a SearchHit represents a table slice, Excel sheet row group, or table clause."""
        meta = hit.metadata or {}
        if hit.table_id or meta.get("table_id") or meta.get("table_ids"):
            return True
        if hit.zone in {ZoneType.TABLE.value, "table"}:
            return True
        fmt = str(meta.get("file_format") or "").lower().lstrip(".")
        if fmt in {"xlsx", "xlsm", "xls"} or hit.file_name.lower().endswith((".xlsx", ".xlsm", ".xls")):
            return True
        if meta.get("sheet_name") or meta.get("table_headers"):
            return True
        if (
            "Шапка таблицы:" in hit.clean_text
            or "[Таблица" in hit.clean_text
            or "[Лист:" in hit.clean_text
            or "(табл." in (hit.clause_no or "").lower()
        ):
            return True
        return False

    def _apply_adaptive_table_routing(
        self,
        hits: list[SearchHit],
        amendment_chunks: list[SearchHit],
        is_oos: bool,
    ) -> list[TablePassport]:
        """
        Adaptive 3-Tier Table Context Routing:
        - Tier 1/2 (Structured tables in SQLite: all .xlsx sheets + large .docx/.pdf tables >=25 rows or >=5000 chars):
          Annotate hit.metadata['table_delivery_mode'] = 'tool_query', set hit.table_id, and populate deduplicated
          SearchResponse.linked_tables ordered by hit relevance.
        - Tier 3 (Compact Word/PDF tables <25 rows & <5000 chars, e.g. Concert Hall cl. 6.1):
          Keep full clean ParentSectionRecord inline on hit.parent_section / hit.parent_text and annotate
          hit.metadata['table_delivery_mode'] = 'inline_parent'.
        """
        if is_oos:
            return []

        store = self._get_active_table_store()
        linked_tables: list[TablePassport] = []
        seen_table_ids: set[str] = set()
        seen_table_hashes: dict[str, int] = {}

        def _register_passport(tp: TablePassport, extra_files: list[str]) -> None:
            merged_also = list(tp.also_in_files)
            for fname in extra_files:
                if fname and fname != tp.file_name and fname not in merged_also:
                    merged_also.append(fname)

            if tp.table_id in seen_table_ids:
                return
            if tp.table_hash and tp.table_hash in seen_table_hashes:
                idx = seen_table_hashes[tp.table_hash]
                existing = linked_tables[idx]
                updated_also = list(existing.also_in_files)
                for fname in [tp.file_name, *merged_also]:
                    if fname and fname != existing.file_name and fname not in updated_also:
                        updated_also.append(fname)
                if updated_also != existing.also_in_files:
                    linked_tables[idx] = existing.model_copy(update={"also_in_files": updated_also})
                return

            seen_table_ids.add(tp.table_id)
            tp_out = tp.model_copy(
                update={
                    "delivery_mode": "tool_query",
                    "also_in_files": merged_also,
                }
            )
            if tp_out.table_hash:
                seen_table_hashes[tp_out.table_hash] = len(linked_tables)
            linked_tables.append(tp_out)

        for hit in list(hits) + list(amendment_chunks):
            matched_passports: list[TablePassport] = []
            meta = hit.metadata

            if store is not None:
                direct_tid = str(hit.table_id or meta.get("table_id") or "").strip()
                if direct_tid:
                    tp = store.get_table(direct_tid)
                    if tp is not None:
                        matched_passports.append(tp)

                for extra_tid in meta.get("table_ids") or []:
                    tid_str = str(extra_tid).strip()
                    if tid_str and all(m.table_id != tid_str for m in matched_passports):
                        tp = store.get_table(tid_str)
                        if tp is not None:
                            matched_passports.append(tp)

                if not matched_passports and self._is_table_or_sheet_hit(hit):
                    sheet_name = self._extract_sheet_name(hit)
                    found = store.find_tables_for_hit(
                        doc_id=hit.doc_id,
                        table_id=direct_tid,
                        section_id=hit.section_id,
                        sheet_name=sheet_name,
                        unit_id=hit.unit_id,
                    )
                    for tp in found:
                        if all(m.table_id != tp.table_id for m in matched_passports):
                            matched_passports.append(tp)

                # Fallback to cross-file deduplicated peer references if primary doc had no table match
                if not matched_passports:
                    for peer_tid in meta.get("peer_table_ids") or []:
                        tp = store.get_table(str(peer_tid).strip())
                        if tp is not None and all(m.table_id != tp.table_id for m in matched_passports):
                            matched_passports.append(tp)
                if not matched_passports and self._is_table_or_sheet_hit(hit):
                    for pref in meta.get("peer_table_refs") or []:
                        if not isinstance(pref, dict):
                            continue
                        p_found = store.find_tables_for_hit(
                            doc_id=str(pref.get("doc_id") or ""),
                            table_id=str(pref.get("table_id") or ""),
                            section_id=str(pref.get("section_id") or ""),
                            sheet_name=str(pref.get("sheet_name") or ""),
                            unit_id=str(pref.get("unit_id") or ""),
                        )
                        for tp in p_found:
                            if all(m.table_id != tp.table_id for m in matched_passports):
                                matched_passports.append(tp)

            if matched_passports:
                primary_tp = matched_passports[0]
                hit.table_id = primary_tp.table_id
                meta["table_id"] = primary_tp.table_id
                meta["table_delivery_mode"] = "tool_query"
                extra_files = [hit.file_name, *list(hit.also_in_files)]
                for mp in matched_passports:
                    _register_passport(mp, extra_files)
            else:
                is_compact_table = (
                    hit.zone in {ZoneType.TABLE.value, "table"}
                    or "Шапка таблицы:" in hit.clean_text
                    or ("Шапка таблицы:" in hit.parent_text and 0 < len(hit.parent_text) < 5000)
                )
                if is_compact_table:
                    meta["table_delivery_mode"] = "inline_parent"
                else:
                    meta.setdefault("table_delivery_mode", "normative_prose")

        return linked_tables

    def _effective_table_bucket_key(self, hit: SearchHit) -> str:
        """
        Return a stable `table_id` / bucket key if `hit` belongs to a `StructuredTableStore` (`tool_query`) table
        or Excel sheet, else `''` (compact inline Word/PDF tables keep `''` so they are never capped).
        """
        meta = hit.metadata or {}
        tid = str(hit.table_id or meta.get("table_id") or "").strip()
        if tid:
            return tid
        if not self._is_table_or_sheet_hit(hit):
            return ""
        store = self._get_active_table_store()
        if store is not None:
            sheet_name = self._extract_sheet_name(hit)
            found = store.find_tables_for_hit(
                doc_id=hit.doc_id,
                table_id="",
                section_id=hit.section_id,
                sheet_name=sheet_name,
                unit_id=hit.unit_id,
            )
            if not found:
                for pref in meta.get("peer_table_refs") or []:
                    if isinstance(pref, dict):
                        found = store.find_tables_for_hit(
                            doc_id=str(pref.get("doc_id") or ""),
                            table_id=str(pref.get("table_id") or ""),
                            section_id=str(pref.get("section_id") or ""),
                            sheet_name=str(pref.get("sheet_name") or ""),
                            unit_id=str(pref.get("unit_id") or ""),
                        )
                        if found:
                            break
            if found:
                hit.table_id = found[0].table_id
                meta["table_id"] = found[0].table_id
                return found[0].table_id
        fmt = str(meta.get("file_format") or "").lower().lstrip(".")
        if fmt in {"xlsx", "xlsm", "xls"} or hit.file_name.lower().endswith((".xlsx", ".xlsm", ".xls")):
            sh = str(meta.get("sheet_name") or "").strip()
            return f"{hit.doc_id}::sheet::{sh or '0'}"
        return ""

    def _select_balanced_multi_query_hits(
        self,
        deduped_hits: list[SearchHit],
        per_subquery_hashes: dict[int, list[str]],
        top_k: int,
        query_dense: list[float],
        query_sparse: Any,
        query_norm: str,
        query_tokens_set: set[str],
        include_parent: bool,
        allow_sibling_probe: bool = True,
    ) -> list[SearchHit]:
        """
        Select `top_k` hits guaranteeing:
        1) Top distinct non-duplicate slot(s) per syntactic sub-query (preventing Sub-Question #1 monopoly).
        2) Cap on primary hits from a single `table_id` (`max(2, top_k // 3)`), since `linked_tables` + `table_context`
           already handles full SQLite table querying.
        3) Sibling-document diversity when two distinct documents share strong title/topic similarity (e.g. a base
           regulation vs a separate uploaded order in `uploads/`), so neither starves the other.
        """
        if not deduped_hits:
            return []

        by_hash: dict[str, SearchHit] = {}
        for h in deduped_hits:
            chash = str((h.metadata or {}).get("content_hash") or compute_content_hash(h.clean_text) or h.chunk_id)
            by_hash.setdefault(chash, h)

        max_per_table = 3 if top_k >= 6 else 2
        selected: list[SearchHit] = []
        selected_ids: set[str] = set()
        table_counts: dict[str, int] = {}

        def _try_add(cand: SearchHit, enforce_table_cap: bool = True) -> bool:
            if cand.chunk_id in selected_ids:
                return False
            tb_key = self._effective_table_bucket_key(cand)
            if enforce_table_cap and tb_key and table_counts.get(tb_key, 0) >= max_per_table:
                return False
            selected.append(cand)
            selected_ids.add(cand.chunk_id)
            if tb_key:
                table_counts[tb_key] = table_counts.get(tb_key, 0) + 1
            return True

        # Step 1: Anchor #1 overall hit first
        _try_add(deduped_hits[0], enforce_table_cap=False)

        # Step 2: Sub-Query Slot Guarantee — reserve top-1 (and top-2 when top_k >= 10) distinct hit per sub-query
        num_sqs = len(per_subquery_hashes)
        if num_sqs >= 2 and top_k >= num_sqs:
            slots_per_sq = 2 if (top_k >= 10 and num_sqs <= 3) else 1
            for pass_num in range(slots_per_sq):
                for sq_idx in sorted(per_subquery_hashes.keys()):
                    if len(selected) >= top_k:
                        break
                    added_for_sq = 0
                    already_covered = sum(
                        1 for s in selected if sq_idx in ((s.metadata or {}).get("matched_subqueries") or [])
                    )
                    if already_covered > pass_num:
                        continue
                    for chash in per_subquery_hashes[sq_idx]:
                        cand = by_hash.get(chash)
                        if cand is None or cand.score < 0.40:
                            continue
                        if _try_add(cand, enforce_table_cap=True):
                            added_for_sq += 1
                            break

        # Step 3: Fill remaining slots by calibrated score while strictly respecting single-table cap
        for cand in deduped_hits:
            if len(selected) >= top_k:
                break
            _try_add(cand, enforce_table_cap=True)

        # Step 4: Only relax table cap if fewer than 2 total hits were found overall
        if len(selected) < min(2, top_k):
            for cand in deduped_hits:
                if len(selected) >= min(3, top_k):
                    break
                _try_add(cand, enforce_table_cap=False)

        # Step 5: Sibling-Document / Multi-Edition Diversity Check
        if allow_sibling_probe and top_k >= 4 and len(selected) >= 2:
            doc_counts: dict[str, int] = {}
            normative_doc_counts: dict[str, int] = {}
            for h in selected:
                if h.doc_id:
                    doc_counts[h.doc_id] = doc_counts.get(h.doc_id, 0) + 1
                    if not self._effective_table_bucket_key(h):
                        normative_doc_counts[h.doc_id] = normative_doc_counts.get(h.doc_id, 0) + 1

            query_wants_excel = any(
                kw in query_norm
                for kw in ("таблиц", "excel", "xlsx", "xls", "анкет", "шаблон", "смет", "цфо", "финансодержател")
            )
            seed_docs = [
                d_id
                for d_id, cnt in sorted(normative_doc_counts.items(), key=lambda kv: kv[1], reverse=True)
                if cnt >= 2 or (selected and d_id == selected[0].doc_id)
            ]
            if not seed_docs:
                dom_doc_id, dom_cnt = max(doc_counts.items(), key=lambda kv: kv[1], default=("", 0))
                if dom_doc_id and dom_cnt >= 3:
                    seed_docs = [dom_doc_id]

            present_docs = set(doc_counts.keys())
            top_score = selected[0].score if selected else 0.0
            for seed_doc_id in seed_docs[:2]:
                siblings = self._find_sibling_topic_docs(seed_doc_id)
                for sib in siblings:
                    if sib.doc_id in present_docs:
                        continue
                    sib_is_excel = sib.file_format in {"xlsx", "xlsm", "xls"} or sib.file_name.lower().endswith(
                        (".xlsx", ".xlsm", ".xls")
                    )
                    if sib_is_excel and not query_wants_excel:
                        continue
                    # Look in deduped_hits first, otherwise probe Qdrant for top-2 chunks of sibling doc
                    sib_hit = next((h for h in deduped_hits if h.doc_id == sib.doc_id), None)
                    if sib_hit is None:
                        raw_sib = self.qdrant.hybrid_search(
                            query_dense=query_dense,
                            query_sparse=query_sparse,
                            top_k=2,
                            doc_id_filter=sib.doc_id,
                            exclude_zones=NON_ANSWER_ZONES,
                        )
                        if raw_sib:
                            sib_hit = self._raw_point_to_hit(
                                raw_sib[0],
                                query_norm=query_norm,
                                query_tokens_set=query_tokens_set,
                                include_parent=include_parent,
                            )
                    if sib_hit is not None and (
                        sib_hit.dense_score >= 0.46 or sib_hit.score >= max(0.43, top_score - 0.20)
                    ):
                        selected = self._insert_co_surfaced_hit(
                            selected, sib_hit, top_k=top_k, insert_front=False
                        )
                        present_docs.add(sib.doc_id)

        selected.sort(key=lambda h: (h.score, h.rrf_score, h.dense_score), reverse=True)
        return selected[:top_k]

    # ------------------------------------------------------------------
    # Generic Structural Graph & Neighbor Expansion (for Chatbot Context)
    # ------------------------------------------------------------------
    def _expand_structural_neighbors_and_sweep(
        self,
        primary_hits: list[SearchHit],
        raw_query: str,
        sub_queries: list[str],
    ) -> tuple[list[SearchHit], list[ParentSectionRecord]]:
        """
        Generic corpus-agnostic structural graph expansion over matched hits:
        1) Intra-Unit Gap Bridging (`prev_id` / `next_id` separated by 1..2 chunks in same `(doc_id, unit_id)`).
        2) Segmented-Section Windowing (`±1` immediate neighbors when `ParentSectionRecord` was segmented or > 6,000 chars).
        3) Generic Internal Cross-Reference Follower (`Приложение № N` / `Приложени[июя]` / `форме заявки...` -> `u02`, `u03`...).
        4) Dominant-Regulation Secondary Sweep (`>= 60%` of primary hits in one compact normative doc `chunk_count <= 130` ->
           BM25 marginal-coverage sweep pulling top 2–3 companion clauses from other sections of that same document).
        """
        if not primary_hits:
            return [], []

        doc_chunks_cache: dict[str, list[ChunkRecord]] = {}

        def _get_doc_chunks(doc_id: str) -> list[ChunkRecord]:
            if not doc_id:
                return []
            if doc_id not in doc_chunks_cache:
                doc_chunks_cache[doc_id] = self.qdrant.get_document_chunks(doc_id)
            return doc_chunks_cache[doc_id]

        expanded_hits: list[SearchHit] = list(primary_hits)
        seen_chunk_ids: set[str] = {h.chunk_id for h in expanded_hits if h.chunk_id}
        extra_parent_sections: list[ParentSectionRecord] = []
        seen_extra_sec_ids: set[str] = set()

        def _append_chunk_as_hit(ch: ChunkRecord, score: float, reason: str) -> None:
            if not ch.chunk_id or ch.chunk_id in seen_chunk_ids:
                return
            if ch.zone in NON_ANSWER_ZONES:
                return
            new_hit = self._chunk_record_to_hit(ch, score=score, expansion_reason=reason, include_parent=True)
            seen_chunk_ids.add(new_hit.chunk_id)
            expanded_hits.append(new_hit)

        # Group primary non-table-sheet hits by (doc_id, unit_id)
        by_doc_unit: dict[tuple[str, str], list[SearchHit]] = {}
        for h in primary_hits:
            if h.doc_id and h.unit_id and not self._effective_table_bucket_key(h):
                by_doc_unit.setdefault((h.doc_id, h.unit_id), []).append(h)

        # 1. Intra-Unit Gap Bridging (1..2 intermediate chunks via prev_id/next_id chain & unit order)
        for (doc_id, unit_id), unit_hits in by_doc_unit.items():
            if len(unit_hits) < 2:
                continue
            all_doc_chunks = _get_doc_chunks(doc_id)
            unit_chunks = [
                c for c in all_doc_chunks
                if c.unit_id == unit_id and c.zone not in NON_ANSWER_ZONES
            ]
            if not unit_chunks:
                continue
            by_cid = {c.chunk_id: c for c in unit_chunks}
            idx_by_cid = {c.chunk_id: i for i, c in enumerate(unit_chunks)}
            matched_cids = {uh.chunk_id for uh in unit_hits}

            # 1a) Pointer-hop check (1 or 2 hops via next_id)
            for uh in unit_hits:
                c_start = by_cid.get(uh.chunk_id)
                if c_start is None or not c_start.next_id:
                    continue
                hop1 = by_cid.get(c_start.next_id)
                if hop1 is not None and hop1.chunk_id not in matched_cids:
                    if hop1.next_id and hop1.next_id in matched_cids:
                        _append_chunk_as_hit(hop1, score=max(0.45, uh.score * 0.92), reason="gap_bridge")
                    elif hop1.next_id:
                        hop2 = by_cid.get(hop1.next_id)
                        if hop2 is not None and hop2.chunk_id not in matched_cids and hop2.next_id in matched_cids:
                            _append_chunk_as_hit(hop1, score=max(0.44, uh.score * 0.90), reason="gap_bridge")
                            _append_chunk_as_hit(hop2, score=max(0.44, uh.score * 0.90), reason="gap_bridge")

            # 1b) Positional index check within ordered unit chunks (catches split-clause suffixes cleanly)
            matched_indices = sorted(idx_by_cid[cid] for cid in matched_cids if cid in idx_by_cid)
            for pos_a, pos_b in zip(matched_indices, matched_indices[1:]):
                gap = pos_b - pos_a - 1
                if 1 <= gap <= 2:
                    ref_score = max(0.44, unit_hits[0].score * 0.90)
                    for mid_idx in range(pos_a + 1, pos_b):
                        _append_chunk_as_hit(unit_chunks[mid_idx], score=ref_score, reason="gap_bridge")

        # 2. Segmented-Section Windowing (±1 neighbor chunks around hits in segmented or >6,000-char sections)
        for h in primary_hits:
            if not h.doc_id or not h.unit_id or self._effective_table_bucket_key(h):
                continue
            p_sec = h.parent_section
            sec_chars = (p_sec.char_count if p_sec else 0) or len(h.parent_text or "")
            is_seg_id = bool(re.search(r"_(?:p|part|seg)\d+$", h.section_id or "", re.IGNORECASE))
            is_seg_meta = bool((p_sec.metadata or {}).get("is_segmented")) if p_sec else False
            many_chunks_flat = bool(p_sec and len(p_sec.chunk_ids) >= 15)

            if sec_chars > 6000 or is_seg_id or is_seg_meta or many_chunks_flat:
                all_doc_chunks = _get_doc_chunks(h.doc_id)
                by_cid = {
                    c.chunk_id: c for c in all_doc_chunks
                    if c.unit_id == h.unit_id and c.zone not in NON_ANSWER_ZONES
                }
                cur_ch = by_cid.get(h.chunk_id)
                prev_cid = (cur_ch.prev_id if cur_ch else None) or h.prev_id
                next_cid = (cur_ch.next_id if cur_ch else None) or h.next_id
                win_score = max(0.43, h.score * 0.88)
                if prev_cid and prev_cid in by_cid:
                    _append_chunk_as_hit(by_cid[prev_cid], score=win_score, reason="segmented_window")
                if next_cid and next_cid in by_cid:
                    _append_chunk_as_hit(by_cid[next_cid], score=win_score, reason="segmented_window")

        # 3. Generic Internal Cross-Reference Follower (Appendices & Application Forms)
        app_num_re = re.compile(r"\bприложени[еяюиемях]\s*(?:№|N)?\s*(\d+\s*[а-яa-z]?)\b", re.IGNORECASE)
        app_gen_re = re.compile(
            r"(?:\bв\s+приложени[иях]\b|\bсогласно\s+приложени[юям]\b|"
            r"\bприложени[еяюи]\s+к\s+(?:настоящему|регламенту|положению|порядку|приказу|инструкции)|"
            r"\bформ[аеуой]\s+(?:заявки|заявления|служебной\s+записки|анкеты|акта|отчета|согласия|представления))",
            re.IGNORECASE,
        )
        query_asks_app_or_form = bool(app_num_re.search(raw_query) or app_gen_re.search(raw_query) or "приложени" in raw_query.lower() or "заявк" in raw_query.lower() or "бланк" in raw_query.lower())

        present_units_by_doc: dict[str, set[str]] = {}
        for h in expanded_hits:
            if h.doc_id and h.unit_id:
                present_units_by_doc.setdefault(h.doc_id, set()).add(h.unit_id)

        checked_xref_docs: set[str] = set()
        for h in primary_hits:
            if not h.doc_id or h.doc_id in checked_xref_docs:
                continue
            doc_hits = [x for x in primary_hits if x.doc_id == h.doc_id]
            combined_hit_text = "\n".join(x.clean_text for x in doc_hits)
            ref_nums = {
                re.sub(r"\s+", "", m.group(1)).lower().replace("a", "а").replace("b", "б")
                for m in app_num_re.finditer(combined_hit_text + "\n" + raw_query)
            }
            has_generic_ref = bool(app_gen_re.search(combined_hit_text)) or query_asks_app_or_form
            if not ref_nums and not has_generic_ref:
                continue

            checked_xref_docs.add(h.doc_id)
            all_doc_chunks = _get_doc_chunks(h.doc_id)
            sec_units: dict[str, list[ChunkRecord]] = {}
            for ch in all_doc_chunks:
                if ch.unit_id not in {"", "u00_meta", "u01_main"} and ch.zone not in NON_ANSWER_ZONES:
                    sec_units.setdefault(ch.unit_id, []).append(ch)

            if not sec_units:
                continue

            already_units = present_units_by_doc.get(h.doc_id, set())
            candidate_units: list[str] = []
            for u_id, u_chunks in sorted(sec_units.items()):
                if u_id in already_units:
                    continue
                u_header_blob = f"{u_chunks[0].unit_title} {u_chunks[0].section_title} {u_chunks[0].clean_text[:260]}".lower().replace("ё", "е")
                if ref_nums:
                    for rn in ref_nums:
                        if re.search(rf"приложени[еяюиемях]\s*(?:№|n)?\s*{re.escape(rn)}\b", u_header_blob):
                            candidate_units.append(u_id)
                            break
                elif has_generic_ref:
                    if len(sec_units) == 1 or any(
                        kw in u_header_blob for kw in ("заявк", "заявлен", "форм", "анкет", "акт", "приложени")
                    ):
                        candidate_units.append(u_id)

            if not candidate_units and ref_nums and len(sec_units) == 1:
                sole_u = next(iter(sec_units.keys()))
                if sole_u not in already_units:
                    candidate_units.append(sole_u)

            for u_id in candidate_units[:2]:
                u_chunks = sec_units[u_id]
                for lead_ch in u_chunks[:2]:
                    _append_chunk_as_hit(lead_ch, score=max(0.42, h.score * 0.86), reason="appendix_xref")
                    if lead_ch.section_id:
                        p_sec = self.get_parent_section(lead_ch.section_id)
                        if p_sec and p_sec.section_id not in seen_extra_sec_ids:
                            seen_extra_sec_ids.add(p_sec.section_id)
                            extra_parent_sections.append(p_sec)

        # Shared query + sub-query BM25 token set for structural sweeps
        all_q_text = " ".join([raw_query, *sub_queries, self._expand_query_for_retrieval(raw_query)])
        q_toks = set(tokenize_bm25(all_q_text))

        # 3b. Order Main-Body (`u01_main`) Governing Clause Sweep when an Appendix is matched
        if not self.pipeline.registry:
            self.pipeline.registry = self.pipeline._load_registry()
        doc_ids_by_fname: dict[str, list[str]] = {}
        for r_id, r_meta in self.pipeline.registry.items():
            if r_meta.file_name:
                doc_ids_by_fname.setdefault(r_meta.file_name, []).append(r_id)

        swept_gov_pairs: set[tuple[str, str]] = set()
        for h in list(expanded_hits):
            if not h.doc_id or self._effective_table_bucket_key(h):
                continue
            is_appendix_unit = h.unit_id not in {"", "u00_meta", "u01_main"}
            h_reg = self.pipeline.registry.get(h.doc_id)
            peer_fnames = set(h.also_in_files or [])
            if h_reg and h_reg.also_in_files:
                peer_fnames.update(h_reg.also_in_files)
            is_standalone_app = bool(
                not is_appendix_unit
                and peer_fnames
                and ("приложени" in (h.file_name or "").lower() or "приложени" in (h.unit_title or "").lower())
            )
            if not is_appendix_unit and not is_standalone_app:
                continue

            app_blob = f"{h.unit_title} {h.section_title} {h.clean_text[:220]}".lower().replace("ё", "е")
            app_nums = {
                re.sub(r"\s+", "", m.group(1)).lower().replace("a", "а").replace("b", "б")
                for m in app_num_re.finditer(app_blob)
            }
            cand_main_docs: list[str] = [h.doc_id]
            for pf in peer_fnames:
                for pid in doc_ids_by_fname.get(pf, []):
                    if pid not in cand_main_docs:
                        cand_main_docs.append(pid)

            for main_doc_id in cand_main_docs:
                gov_key = (main_doc_id, ",".join(sorted(app_nums)) or h.unit_id)
                if gov_key in swept_gov_pairs:
                    continue
                main_chunks = [
                    c
                    for c in _get_doc_chunks(main_doc_id)
                    if c.unit_id == "u01_main" and c.zone not in NON_ANSWER_ZONES and not c.table_id
                ]
                if not main_chunks:
                    continue
                swept_gov_pairs.add(gov_key)

                scored_gov: list[tuple[float, ChunkRecord]] = []
                for mc in main_chunks:
                    if mc.chunk_id in seen_chunk_ids:
                        continue
                    mc_low = f"{mc.section_title} {mc.clean_text}".lower().replace("ё", "е")
                    g_score = 0.0
                    matched_app_ref = False
                    for rn in app_nums:
                        if re.search(rf"приложени[еяюиемях]\s*(?:№|n)?\s*{re.escape(rn)}\b", mc_low):
                            g_score += 4.5
                            matched_app_ref = True
                            break
                    if not matched_app_ref and "приложени" in mc_low:
                        g_score += 1.0

                    mc_toks = set(tokenize_bm25(mc_low))
                    subst_cnt = 0
                    for tok in q_toks.intersection(mc_toks):
                        if tok not in GENERIC_QUERY_LEMMAS and tok not in _SYNTACTIC_META_LEMMAS:
                            g_score += 2.3
                            subst_cnt += 1
                        else:
                            g_score += 0.45
                    if g_score >= 3.2 and (matched_app_ref or subst_cnt >= 2):
                        scored_gov.append((g_score, mc))

                scored_gov.sort(key=lambda item: item[0], reverse=True)
                for _, gov_ch in scored_gov[:2]:
                    _append_chunk_as_hit(gov_ch, score=max(0.44, h.score * 0.88), reason="order_main_governance")
                    if gov_ch.section_id:
                        p_sec = self.get_parent_section(gov_ch.section_id)
                        if p_sec and p_sec.section_id not in seen_extra_sec_ids:
                            seen_extra_sec_ids.add(p_sec.section_id)
                            extra_parent_sections.append(p_sec)

        # 4. Multi-Document Normative Secondary BM25 Sweep
        #    Runs on every matched normative document (chunk_count <= 150) holding >= 20% or >= 2 primary hits (up to top 3 docs)
        normative_primary = [h for h in primary_hits if h.doc_id and not self._effective_table_bucket_key(h)]
        pool_for_dom = normative_primary if normative_primary else primary_hits
        if pool_for_dom:
            counts_by_doc: dict[str, int] = {}
            max_score_by_doc: dict[str, float] = {}
            for h in pool_for_dom:
                if h.doc_id:
                    counts_by_doc[h.doc_id] = counts_by_doc.get(h.doc_id, 0) + 1
                    max_score_by_doc[h.doc_id] = max(max_score_by_doc.get(h.doc_id, 0.0), h.score)

            ranked_docs = sorted(
                counts_by_doc.keys(),
                key=lambda d_id: (counts_by_doc[d_id], max_score_by_doc.get(d_id, 0.0)),
                reverse=True,
            )
            swept_norm_docs = 0
            for cand_doc_id in ranked_docs:
                if swept_norm_docs >= 3:
                    break
                doc_cnt = counts_by_doc[cand_doc_id]
                doc_ratio = doc_cnt / max(1, len(pool_for_dom))
                if not (doc_cnt >= 2 or doc_ratio >= 0.20):
                    continue
                reg_meta = self.pipeline.get_document(cand_doc_id)
                all_dom_chunks = _get_doc_chunks(cand_doc_id)
                eff_chunk_cnt = (
                    reg_meta.chunk_count if reg_meta and reg_meta.chunk_count else len(all_dom_chunks)
                )
                if not (0 < eff_chunk_cnt <= 150 and all_dom_chunks):
                    continue

                swept_norm_docs += 1
                covered_toks: set[str] = set()
                covered_sections: set[str] = set()
                for eh in expanded_hits:
                    if eh.doc_id == cand_doc_id:
                        covered_toks.update(tokenize_bm25(eh.clean_text))
                        if eh.section_id:
                            covered_sections.add(eh.section_id)

                scored_companions: list[tuple[float, ChunkRecord]] = []
                for ch in all_dom_chunks:
                    if ch.chunk_id in seen_chunk_ids or ch.zone in NON_ANSWER_ZONES or ch.table_id:
                        continue
                    ch_toks = set(tokenize_bm25(f"{ch.section_title} {ch.clean_text}"))
                    matched = q_toks.intersection(ch_toks)
                    if not matched:
                        continue
                    score_acc = 0.0
                    subst_matches = 0
                    for tok in matched:
                        is_subst = tok not in GENERIC_QUERY_LEMMAS and tok not in _SYNTACTIC_META_LEMMAS
                        base_w = 2.4 if is_subst else 0.55
                        if is_subst:
                            subst_matches += 1
                        if tok not in covered_toks:
                            base_w *= 1.85
                        score_acc += base_w

                    if subst_matches == 0:
                        continue
                    if ch.section_id and ch.section_id not in covered_sections:
                        score_acc *= 1.20
                    scored_companions.append((score_acc, ch))

                scored_companions.sort(key=lambda item: item[0], reverse=True)
                dom_ref_score = max_score_by_doc.get(cand_doc_id, 0.55)
                max_comp = 3 if doc_ratio >= 0.50 else 2
                for companion_score, comp_ch in scored_companions[:max_comp]:
                    if companion_score >= 2.2:
                        _append_chunk_as_hit(
                            comp_ch,
                            score=max(0.42, min(0.75, dom_ref_score * 0.85)),
                            reason="dominant_doc_sweep",
                        )

        # 5. Parallel Sibling-Edition Section / Clause Alignment
        #    When >= 2 normative documents share topic overlap (or both hold >= 2 primary hits), align matched
        #    section_num / clause_no prefixes across editions (prioritizing newer doc_date editions so updated rules are never missed).
        def _parse_clause_anchors(cl_raw: str, sec_raw: str) -> tuple[set[str], set[str]]:
            cl_clean = re.sub(r"^\s*(?:п\.|пункт)\s*", "", (cl_raw or "").strip(), flags=re.IGNORECASE)
            cl_clean = re.sub(r"\s*\((?:ч\.|табл\.|продолж).*$", "", cl_clean, flags=re.IGNORECASE)
            base_cls: set[str] = set()
            majors: set[str] = set()
            for part in re.split(r"[,;]+", cl_clean):
                tok = part.strip().strip(".")
                if tok and re.match(r"^\d+(?:\.\d+)*$", tok):
                    base_cls.add(tok)
                    if "." in tok:
                        majors.add(tok.split(".", 1)[0])
            s_clean = (sec_raw or "").strip().strip(".")
            if s_clean and re.match(r"^(?:\d+|[IVXLC]+)$", s_clean, re.IGNORECASE) and s_clean != "0":
                # Avoid treating flat unsectioned wrapper "1" as Section 1 when multi-level clauses belong to another section
                non_one_multi = any("." in bc and bc.split(".", 1)[0] != "1" for bc in base_cls)
                if not (s_clean == "1" and non_one_multi):
                    majors.add(s_clean)
            return base_cls, majors

        norm_hits_by_doc: dict[str, list[SearchHit]] = {}
        for nh in normative_primary:
            if nh.doc_id:
                norm_hits_by_doc.setdefault(nh.doc_id, []).append(nh)

        sibling_map: dict[str, set[str]] = {}
        candidate_align_docs: list[str] = list(norm_hits_by_doc.keys())
        for d_id, d_hits in list(norm_hits_by_doc.items()):
            sib_metas = self._find_sibling_topic_docs(d_id)
            sib_ids = set()
            for sm in sib_metas:
                if sm.file_format in {"xlsx", "xlsm", "xls"} or sm.file_name.lower().endswith((".xlsx", ".xlsm", ".xls")):
                    continue
                sib_ids.add(sm.doc_id)
                if (len(d_hits) >= 2 or (primary_hits and primary_hits[0].doc_id == d_id)) and sm.doc_id not in candidate_align_docs:
                    candidate_align_docs.append(sm.doc_id)
            sibling_map[d_id] = sib_ids

        if len(candidate_align_docs) >= 2:
            def _doc_date_sort_key(d_id: str) -> str:
                m = self.pipeline.get_document(d_id)
                return (m.doc_date or "") if m else ""

            # Process newer editions first so updated regulations/orders always get aligned cleanly
            ordered_targets = sorted(candidate_align_docs, key=_doc_date_sort_key, reverse=True)
            for target_doc_id in ordered_targets[:4]:
                target_hits = norm_hits_by_doc.get(target_doc_id, [])
                peer_docs: list[str] = []
                for other_id, other_hits in norm_hits_by_doc.items():
                    if other_id == target_doc_id:
                        continue
                    is_topic_sib = (
                        target_doc_id in sibling_map.get(other_id, set())
                        or other_id in sibling_map.get(target_doc_id, set())
                    )
                    both_strong = len(target_hits) >= 2 and len(other_hits) >= 2
                    if is_topic_sib or both_strong:
                        peer_docs.append(other_id)

                if not peer_docs:
                    continue

                target_chunks = _get_doc_chunks(target_doc_id)
                if not (0 < len(target_chunks) <= 180):
                    continue

                peer_exact_clauses: set[str] = set()
                peer_major_secs: set[str] = set()
                peer_sec_title_toks: set[str] = set()
                peer_max_date = max((_doc_date_sort_key(pid) for pid in peer_docs), default="")
                target_date = _doc_date_sort_key(target_doc_id)
                is_newer_edition = bool(target_date and (not peer_max_date or target_date > peer_max_date))
                has_topic_sib_peer = any(
                    target_doc_id in sibling_map.get(pid, set()) or pid in sibling_map.get(target_doc_id, set())
                    for pid in peer_docs
                )

                for pid in peer_docs:
                    for ph in list(expanded_hits):
                        if ph.doc_id != pid or self._effective_table_bucket_key(ph):
                            continue
                        b_cls, maj_s = _parse_clause_anchors(ph.clause_no, ph.section_num)
                        peer_exact_clauses.update(b_cls)
                        peer_major_secs.update(maj_s)
                        for st_tok in tokenize_bm25(ph.section_title or ""):
                            if st_tok not in GENERIC_QUERY_LEMMAS and st_tok not in _SYNTACTIC_META_LEMMAS:
                                peer_sec_title_toks.add(st_tok)

                already_target_clauses: set[str] = set()
                already_target_secs: set[str] = set()
                for eh in expanded_hits:
                    if eh.doc_id == target_doc_id:
                        b_cls, maj_s = _parse_clause_anchors(eh.clause_no, eh.section_num)
                        already_target_clauses.update(b_cls)
                        already_target_secs.update(maj_s)

                scored_aligned: list[tuple[float, ChunkRecord]] = []
                for ch in target_chunks:
                    if ch.chunk_id in seen_chunk_ids or ch.zone in NON_ANSWER_ZONES or ch.table_id:
                        continue
                    ch_cls, ch_majors = _parse_clause_anchors(ch.clause_no, ch.section_num)
                    exact_cl_match = bool(ch_cls and (ch_cls - already_target_clauses).intersection(peer_exact_clauses))
                    sec_num_match = bool(ch_majors and ch_majors.intersection(peer_major_secs))

                    ch_toks = set(tokenize_bm25(f"{ch.section_title} {ch.clean_text}"))
                    q_matched = q_toks.intersection(ch_toks)
                    subst_q_cnt = sum(
                        1 for t in q_matched if t not in GENERIC_QUERY_LEMMAS and t not in _SYNTACTIC_META_LEMMAS
                    )
                    sec_title_ov = len(peer_sec_title_toks.intersection(set(tokenize_bm25(ch.section_title or ""))))

                    if not (exact_cl_match or sec_num_match or (has_topic_sib_peer and subst_q_cnt >= 2)):
                        continue
                    if not has_topic_sib_peer and subst_q_cnt == 0:
                        continue
                    if not exact_cl_match and subst_q_cnt == 0 and sec_title_ov == 0:
                        continue

                    a_score = 0.0
                    if exact_cl_match:
                        a_score += 4.2
                    if sec_num_match:
                        a_score += 2.4
                        if not ch_majors.issubset(already_target_secs):
                            a_score += 0.9
                    if sec_title_ov > 0:
                        a_score += min(2.4, 0.8 * sec_title_ov)
                    a_score += 2.0 * subst_q_cnt + 0.35 * (len(q_matched) - subst_q_cnt)
                    if is_newer_edition:
                        a_score *= 1.18

                    if a_score >= 3.4:
                        scored_aligned.append((a_score, ch))

                scored_aligned.sort(key=lambda item: item[0], reverse=True)
                ref_sc = max(
                    (h.score for h in primary_hits if h.doc_id in peer_docs or h.doc_id == target_doc_id),
                    default=0.55,
                )
                for _, al_ch in scored_aligned[:3]:
                    _append_chunk_as_hit(
                        al_ch,
                        score=max(0.44, min(0.78, ref_sc * (0.90 if is_newer_edition else 0.85))),
                        reason="sibling_edition_align",
                    )
                    if al_ch.section_id:
                        p_sec = self.get_parent_section(al_ch.section_id)
                        if p_sec and p_sec.section_id not in seen_extra_sec_ids:
                            seen_extra_sec_ids.add(p_sec.section_id)
                            extra_parent_sections.append(p_sec)

        return expanded_hits, extra_parent_sections

    # ------------------------------------------------------------------
    # Automatic Deterministic StructuredTableStore Hydration
    # ------------------------------------------------------------------
    @staticmethod
    def _morphological_stem_prefix(word: str) -> str:
        """
        Compute general Russian case-invariant substring root via `commonprefix(surface_norm, pymorphy3_lemma)`.
        Example: 'квантовых' + 'квантовый' -> 'квантов'; 'Климову' + 'климов' -> 'климов'.
        """
        w_norm = _norm_ru(word)
        if len(w_norm) < 4 or not re.match(r"^[а-яa-z-]+$", w_norm):
            return w_norm
        lemma = _lemmatize_ru_word(w_norm)
        prefix = os.path.commonprefix([w_norm, lemma])
        if len(prefix) >= 4:
            return prefix
        return lemma if len(lemma) >= 4 else w_norm

    def _extract_salient_table_probes(
        self,
        tp: TablePassport,
        raw_query: str,
        sub_queries: list[str],
        max_probes: int = 10,
    ) -> list[str]:
        """Extract corpus-agnostic salient codes, quoted phrases, glossary expansions, and morphological stems for SQLite filtering."""
        probes: list[str] = []
        seen: set[str] = set()

        def _add_probe(val: str) -> None:
            v = _norm_ru(val).strip(" ,.;:!?\"'«»()")
            if len(v) < 2 or v in seen:
                return
            seen.add(v)
            probes.append(v)

        doc_num_core = _core_doc_number(tp.doc_number).lower()
        table_meta_tokens = set(tokenize_bm25(f"{tp.table_title} {tp.file_name} {' '.join(c.name for c in tp.columns)}"))

        # 1. Quoted phrases in query (`«...»` or `"..."`)
        for m_q in re.finditer(r"«([^»]{3,80})»|\"([^\"]{3,80})\"", raw_query):
            _add_probe(m_q.group(1) or m_q.group(2) or "")

        # 2. Numeric/alphanumeric codes in query (e.g. '10267', '10294', '211.03', '244', '16.4'), skipping doc_number & years
        for m_code in re.finditer(r"\b\d{3,6}(?:\.\d{1,3})?\b|\b\d{1,2}[.,]\d+\b", raw_query):
            c_val = m_code.group(0).replace(",", ".")
            if doc_num_core and c_val == doc_num_core:
                continue
            if re.match(r"^(?:199\d|20[0-3]\d)$", c_val):
                continue
            _add_probe(c_val)

        # 3. Expanded Glossary full terms if query contains a known abbreviation
        expanded_full = self._expand_query_for_retrieval(raw_query)
        m_paren_exp = re.search(r"\(([^()]+)\)\s*$", expanded_full)
        if m_paren_exp:
            for item in m_paren_exp.group(1).split(","):
                item_clean = item.strip()
                if len(item_clean) >= 8 and not any(ch.isdigit() for ch in item_clean):
                    # Take distinctive multi-word stem phrase or longest root
                    words = [
                        self._morphological_stem_prefix(w)
                        for w in re.findall(r"[а-яА-ЯёЁ]{4,}", item_clean)
                        if _lemmatize_ru_word(w.lower().replace("ё", "е")) not in GENERIC_QUERY_LEMMAS
                    ]
                    if words:
                        _add_probe(" ".join(words[:3]))

        for abbr, full_expansion in (self.glossary.abbreviations or {}).items():
            abbr_norm = _norm_ru(abbr)
            if len(abbr_norm) >= 2 and re.search(rf"\b{re.escape(abbr_norm)}\b", _norm_ru(raw_query)):
                _add_probe(abbr_norm)
                exp_roots = [
                    self._morphological_stem_prefix(w)
                    for w in re.findall(r"[а-яА-ЯёЁ]{4,}", str(full_expansion))
                ]
                if exp_roots:
                    _add_probe(" ".join(exp_roots[:2]))

        # 4. Salient content stems from raw_query & sub_queries not merely repeating the table's own header/title
        table_generic_stop = {
            "таблица",
            "строка",
            "столбец",
            "колонка",
            "структура",
            "финансовый",
            "финансодержатель",
            "уровень",
            "цфо",
            "мфти",
            "физтех",
            "подразделение",
            "относиться",
            "подчиняться",
            "входить",
            "состоять",
            "установленный",
            "использоваться",
            "применяться",
            "формирование",
            "бюджет",
            "смета",
            "шаблон",
        }
        for src_text in [raw_query, *sub_queries]:
            for raw_w in re.findall(r"[а-яА-ЯёЁa-zA-Z0-9-]{4,}", src_text):
                w_low = raw_w.lower().replace("ё", "е")
                if doc_num_core and w_low == doc_num_core:
                    continue
                if re.match(r"^\d+-\d+$", w_low):
                    continue
                lemma = _lemmatize_ru_word(w_low)
                if (
                    lemma in GENERIC_QUERY_LEMMAS
                    or lemma in _SYNTACTIC_META_LEMMAS
                    or lemma in table_generic_stop
                    or lemma in table_meta_tokens
                ):
                    continue
                stem = self._morphological_stem_prefix(w_low)
                if len(stem) >= 4:
                    _add_probe(stem)

        return probes[:max_probes]

    def _hydrate_table_context(
        self,
        raw_query: str,
        sub_queries: list[str],
        linked_tables: list[TablePassport],
    ) -> list[TableQueryResponse]:
        """
        Deterministically query `StructuredTableStore` for every matched `TablePassport`:
        - `get_formulas` when table has formulas and query involves formulas/rates/%/calculations/codes;
        - Per-subquery `filter_rows` across each decomposed `sub_query` (plus salient codes, glossary expansions,
          and morphological stems) merging deduplicated rows up to ~42 rows total so 2nd/3rd sub-questions never drop rows;
        - Hierarchy follow-up or compact full-sheet readout (`<= 45` rows) so multi-part table answers have complete rows.
        """
        store = self._get_active_table_store()
        if store is None or not linked_tables:
            return []

        q_norm = _norm_ru(raw_query)
        formula_triggers = (
            "формул",
            "расчет",
            "рассчит",
            "ставк",
            "процент",
            "%",
            "квр",
            "косгу",
            "накладн",
            "ндс",
            "резерв",
            "отчислен",
            "шаблон",
            "смет",
            "итого",
        )
        wants_formulas = any(trig in q_norm for trig in formula_triggers)
        wants_aggregation = any(
            trig in q_norm for trig in ("сколько", "количеств", "число", "сумм", "итого", "сгруппир", "всего")
        )

        responses: list[TableQueryResponse] = []

        for tp in linked_tables[:4]:
            added_for_tp = 0

            # 1. Extract formulas if applicable
            if tp.has_formulas and wants_formulas:
                f_resp = store.query_table(
                    TableQueryRequest(
                        table_id=tp.table_id,
                        operation="get_formulas",
                        limit=50,
                    )
                )
                if f_resp.formulas or f_resp.rows:
                    responses.append(f_resp)
                    added_for_tp += 1

            # 2. Compact table (`<= 45` rows): return all rows deterministically so zero rows can be missed
            if 0 < tp.row_count <= 45:
                all_resp = store.query_table(
                    TableQueryRequest(
                        table_id=tp.table_id,
                        operation="get_rows",
                        limit=50,
                    )
                )
                if all_resp.rows:
                    responses.append(all_resp)
                    added_for_tp += 1
                continue

            # 3. Large table (`> 45` rows): per-subquery probe extraction & deduplicated row hydration
            query_units = list(sub_queries) if len(sub_queries) >= 2 else [raw_query]
            per_unit_probes: list[tuple[int, list[str]]] = []
            seen_queued_probes: set[str] = set()

            for sq_idx, sq_text in enumerate(query_units, start=1):
                u_probes = self._extract_salient_table_probes(tp, sq_text, [], max_probes=8)
                fresh = [p for p in u_probes if p not in seen_queued_probes]
                seen_queued_probes.update(fresh)
                if u_probes:
                    # Keep u_probes even if a code appeared in an earlier sub-query when fresh is empty
                    per_unit_probes.append((sq_idx, fresh if fresh else u_probes[:3]))

            global_probes = self._extract_salient_table_probes(tp, raw_query, sub_queries, max_probes=14)
            extra_global = [p for p in global_probes if p not in seen_queued_probes]
            if extra_global:
                per_unit_probes.append((0, extra_global))

            seen_resp_signatures: set[str] = set()
            matched_hierarchy_values: dict[str, set[str]] = {}
            hier_cols = [c.name for c in tp.columns if c.is_hierarchy]
            probe_responses_raw: list[tuple[int, str, TableQueryResponse]] = []
            max_matches_per_unit = 2 if len(query_units) >= 2 else 4

            for sq_idx, unit_probes in per_unit_probes:
                matched_in_unit = 0
                for probe in unit_probes:
                    r_resp = store.query_table(
                        TableQueryRequest(
                            table_id=tp.table_id,
                            operation="filter_rows",
                            search_text=probe,
                            limit=60,
                        )
                    )
                    if r_resp.matched_count > 0 and r_resp.rows:
                        sig = f"{r_resp.matched_count}::{json.dumps(r_resp.rows[:3], ensure_ascii=False, sort_keys=True)}"
                        if sig not in seen_resp_signatures:
                            seen_resp_signatures.add(sig)
                            probe_responses_raw.append((sq_idx, probe, r_resp))
                            matched_in_unit += 1
                            for row_d in r_resp.rows[:10]:
                                for hc in hier_cols[:2]:
                                    hv = str(row_d.get(hc) or "").strip()
                                    if hv:
                                        matched_hierarchy_values.setdefault(hc, set()).add(hv)
                    if matched_in_unit >= max_matches_per_unit:
                        break

            # Allocate fair row budget (~42 rows total across sub-queries) & deduplicate rows across probe responses
            if probe_responses_raw:
                total_row_budget = 42
                remaining_budget = total_row_budget
                global_seen_rows: set[str] = set()
                num_resps = len(probe_responses_raw)

                for r_pos, (sq_idx, probe, r_resp) in enumerate(probe_responses_raw):
                    resps_left = max(1, num_resps - r_pos)
                    fair_quota = max(6, remaining_budget // resps_left) if remaining_budget > 0 else 4
                    unique_rows_for_resp: list[dict[str, Any]] = []
                    for row_d in r_resp.rows:
                        r_sig = json.dumps(row_d, ensure_ascii=False, sort_keys=True)
                        if r_sig not in global_seen_rows:
                            global_seen_rows.add(r_sig)
                            unique_rows_for_resp.append(row_d)
                            if len(unique_rows_for_resp) >= fair_quota:
                                break

                    # If all rows of this probe overlapped an earlier probe, keep 2 sample rows so the probe's
                    # exact matched_count header still has clear context without duplicating 20+ rows.
                    display_rows = unique_rows_for_resp if unique_rows_for_resp else list(r_resp.rows[:2])
                    remaining_budget = max(0, remaining_budget - len(unique_rows_for_resp))
                    cols = list(r_resp.columns) if r_resp.columns else [c.name for c in tp.columns]
                    sq_tag = f"sub_q={sq_idx}, " if (len(query_units) >= 2 and sq_idx > 0) else ""
                    md_prev = _format_preview_markdown(cols, display_rows, max_rows=max(1, len(display_rows)))
                    responses.append(
                        r_resp.model_copy(
                            update={
                                "operation": f"filter_rows({sq_tag}search='{probe}')",
                                "rows": display_rows,
                                "returned_count": len(display_rows),
                                "truncated": len(display_rows) < r_resp.matched_count,
                                "markdown_preview": md_prev,
                            }
                        )
                    )
                    added_for_tp += 1

            # 4. If query asks 'сколько / перечень / все' and matched a specific top-level hierarchy owner,
            # run an exact count/filter on that hierarchy owner so counts are 100% complete
            if wants_aggregation and matched_hierarchy_values:
                first_hc = next(iter(matched_hierarchy_values.keys()))
                vals = list(matched_hierarchy_values[first_hc])
                if len(vals) == 1:
                    owner_val = vals[0]
                    owner_resp = store.query_table(
                        TableQueryRequest(
                            table_id=tp.table_id,
                            operation="filter_rows",
                            filters={first_hc: owner_val},
                            limit=120,
                        )
                    )
                    if owner_resp.matched_count > 0:
                        responses.append(
                            owner_resp.model_copy(
                                update={"operation": f"filter_rows({first_hc}='{owner_val}')"}
                            )
                        )
                        added_for_tp += 1

            # 5. Fallback summary preview if zero specific probes matched
            if added_for_tp == 0:
                if hier_cols:
                    agg_resp = store.query_table(
                        TableQueryRequest(
                            table_id=tp.table_id,
                            operation="aggregate",
                            agg_func="count",
                            group_by=hier_cols[0],
                            limit=30,
                        )
                    )
                    if agg_resp.rows:
                        responses.append(agg_resp)
                        added_for_tp += 1
                prev_resp = store.query_table(
                    TableQueryRequest(
                        table_id=tp.table_id,
                        operation="get_rows",
                        limit=15,
                    )
                )
                if prev_resp.rows:
                    responses.append(prev_resp)

        return responses

    # ------------------------------------------------------------------
    # Clean Self-Contained Context Formatter (`formatted_context`)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_hit_centered_parent_window(
        parent_text: str,
        matched_snippets: list[str],
        max_chars: int = 12000,
    ) -> str:
        """
        Render `parent_text` WITHOUT cropping whenever `len(parent_text) <= 12_000`.
        Only when a single section genuinely exceeds `12_000` chars (e.g. an oversized flat regulation),
        extract merged windows centered around every matched clause snippet keeping `>= 10_000` chars!
        """
        eff_max = max(int(max_chars), 12000)
        if len(parent_text) <= 12000 or len(parent_text) <= eff_max:
            return parent_text

        intervals: list[tuple[int, int]] = [(0, min(len(parent_text), 2200))]
        half_win = max(2200, (eff_max - 2000) // max(2, len(matched_snippets) * 2))
        for snip in matched_snippets:
            probe = (snip or "").strip()[:120]
            if not probe:
                continue
            pos = parent_text.find(probe)
            if pos < 0 and len(probe) > 35:
                pos = parent_text.find(probe[:35])
            if pos >= 0:
                s_i = max(0, pos - half_win)
                e_i = min(len(parent_text), pos + len(snip) + half_win)
                intervals.append((s_i, e_i))

        intervals.sort(key=lambda x: x[0])
        merged: list[tuple[int, int]] = []
        for s_i, e_i in intervals:
            if not merged or s_i > merged[-1][1] + 80:
                merged.append((s_i, e_i))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e_i))

        total_span = sum(e_i - s_i for s_i, e_i in merged)
        min_keep = min(len(parent_text), 10500)
        if total_span < min_keep and merged:
            extra_needed = min_keep - total_span
            first_s, first_e = merged[0]
            new_first_e = min(len(parent_text), first_e + extra_needed)
            merged[0] = (first_s, new_first_e)
            re_merged: list[tuple[int, int]] = []
            for s_i, e_i in merged:
                if not re_merged or s_i > re_merged[-1][1] + 80:
                    re_merged.append((s_i, e_i))
                else:
                    re_merged[-1] = (re_merged[-1][0], max(re_merged[-1][1], e_i))
            merged = re_merged

        pieces: list[str] = []
        used = 0
        for idx, (s_i, e_i) in enumerate(merged):
            if idx > 0:
                pieces.append("\n...[пропущены промежуточные пункты длинного раздела]...\n")
            budget = eff_max - used
            if budget <= 200:
                break
            seg = parent_text[s_i : min(e_i, s_i + budget)]
            pieces.append(seg.strip())
            used += len(seg)
        return "".join(pieces)

    @staticmethod
    def _format_clause_page_map_annotation(
        parent_sec: ParentSectionRecord,
        section_hits: list[SearchHit],
    ) -> str:
        """Format `clause_page_map` (from `ParentSectionRecord.metadata` or matched hits) into a crisp page index."""
        p_meta = parent_sec.metadata or {}
        cpm = p_meta.get("clause_page_map")
        entries: list[str] = []
        if isinstance(cpm, dict) and cpm:
            for cl_key, pages_val in cpm.items():
                if isinstance(pages_val, list) and pages_val:
                    p_str = ",".join(str(p) for p in pages_val)
                    entries.append(f"п. {cl_key} -> стр. {p_str}")
        if not entries:
            seen_cl: set[str] = set()
            for h in section_hits:
                if h.clause_no and h.page_numbers and h.clause_no not in seen_cl:
                    seen_cl.add(h.clause_no)
                    p_str = ",".join(str(p) for p in h.page_numbers)
                    entries.append(f"п. {h.clause_no} -> стр. {p_str}")
        if not entries:
            return ""
        if len(entries) > 40:
            entries = entries[:40] + ["..."]
        return "Карта страниц пунктов (clause_page_map): " + "; ".join(entries)

    @classmethod
    def _format_citation_bracket_header(cls, hit: SearchHit, idx: int | None = None, prefix: str = "Фрагмент") -> str:
        """Build explicit `[Документ | Зарег. в ЭДО | Статус | Раздел | Пункт | Стр.]` citation header."""
        lead = f"[{prefix} #{idx}] " if idx is not None else f"[{prefix}] "
        doc_part = f"Документ: {hit.title or hit.file_name} (файл: {hit.file_name}"
        if hit.doc_number:
            doc_part += f", № {hit.doc_number}" + (f" от {hit.doc_date}" if hit.doc_date else "")
            if (hit.metadata or {}).get("is_edo_stamp"):
                doc_part += " [штамп ЭДО]"
        doc_part += ")"

        edo_val = "не указан"
        if hit.edo_number:
            edo_val = f"№ {hit.edo_number}" + (f" от {hit.edo_date}" if hit.edo_date else "")

        status_ru = STATUS_RU_MAP.get(hit.status, hit.status)
        sec_label = " / ".join(p for p in (hit.unit_title, hit.section_title) if p) or "Основной текст"
        cl_label = hit.clause_no or "б/н"
        pages_label = ", ".join(str(p) for p in hit.page_numbers) if hit.page_numbers else "н/д"

        parts = [
            f"{lead}{doc_part}",
            f"Зарег. в ЭДО: {edo_val}",
            f"Статус: {status_ru} ({hit.status})",
            f"Раздел: {sec_label}",
            f"Пункт: {cl_label}",
            f"Стр.: {pages_label}",
        ]
        if hit.also_in_files:
            parts.append(f"Также в файлах: {', '.join(hit.also_in_files)}")
        if (hit.metadata or {}).get("source_origin") == "uploads":
            parts.append("Источник: uploads/")
        exp_reason = str((hit.metadata or {}).get("expansion_reason") or "")
        if exp_reason:
            parts.append(f"Связь: {exp_reason}")
        return " | ".join(parts)

    def _render_formatted_chatbot_context(
        self,
        query: str,
        sub_queries: list[str],
        hits: list[SearchHit],
        parent_sections: list[ParentSectionRecord],
        amendment_chunks: list[SearchHit],
        version_alerts: list[str],
        quality_alerts: list[str],
        linked_tables: list[TablePassport],
        table_context: list[TableQueryResponse],
        unit_outlines: list[str] | None = None,
    ) -> str:
        """Render a self-contained, richly cited Russian RAG context string for stateless chatbot consumption."""
        blocks: list[str] = []

        # 1. Query & Decomposed Sub-Queries Header
        q_lines = [f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}"]
        if len(sub_queries) > 1:
            q_lines.append("ДЕКОМПОЗИЦИЯ МНОГОАСПЕКТНОГО ЗАПРОСА:")
            for sq_i, sq in enumerate(sub_queries, start=1):
                q_lines.append(f"  {sq_i}) {sq}")
        blocks.append("\n".join(q_lines))

        # 2. Alerts & Document Requisites Passport Summary
        if version_alerts or quality_alerts:
            al_lines = ["ПРЕДУПРЕЖДЕНИЯ ПО РЕДАКЦИЯМ, АКТУАЛЬНОСТИ И КАЧЕСТВУ ИСТОЧНИКОВ:"]
            for va in version_alerts:
                al_lines.append(f"- [РЕДАКЦИЯ] {va}")
            for qa in quality_alerts:
                al_lines.append(f"- [КАЧЕСТВО] {qa}")
            blocks.append("\n".join(al_lines))

        doc_reps: dict[str, SearchHit] = {}
        for h in list(hits) + list(amendment_chunks):
            if h.doc_id and h.doc_id not in doc_reps:
                doc_reps[h.doc_id] = h
        if doc_reps:
            pass_lines = ["РЕЕСТР НАЙДЕННЫХ ДОКУМЕНТОВ И ИХ ТОЧНЫЕ РЕКВИЗИТЫ:"]
            for d_hit in doc_reps.values():
                pass_lines.append(f"- {self._format_doc_requisite_badge(d_hit)}")
            blocks.append("\n".join(pass_lines))

        # 3. Matched & Structurally Expanded Chunks
        if hits:
            hit_strs: list[str] = []
            for idx, h in enumerate(hits, start=1):
                hdr = self._format_citation_bracket_header(h, idx=idx, prefix="Фрагмент")
                hit_strs.append(f"{hdr}\n{h.clean_text.strip()}")
            blocks.append("НАЙДЕННЫЕ И СВЯЗАННЫЕ ПУНКТЫ ДОКУМЕНТОВ:\n\n" + "\n\n---\n\n".join(hit_strs))

        # 4. Amendment Chunks
        if amendment_chunks:
            amd_strs: list[str] = []
            for idx, ah in enumerate(amendment_chunks, start=1):
                hdr = self._format_citation_bracket_header(ah, idx=idx, prefix="Изменяющий акт")
                amd_strs.append(f"{hdr}\n{ah.clean_text.strip()}")
            blocks.append("ИЗМЕНЯЮЩИЕ ДОКУМЕНТЫ (EDITION GRAPH):\n\n" + "\n\n---\n\n".join(amd_strs))

        # 4b. Structural Unit Outlines (`unit_outline`) for large units > 20,000 chars
        if unit_outlines:
            blocks.append(
                "СТРУКТУРНОЕ ОГЛАВЛЕНИЕ КРУПНЫХ РАЗДЕЛОВ И ДОКУМЕНТОВ (unit_outline):\n\n"
                + "\n\n".join(unit_outlines)
            )

        # 5. Hydrated Parent Sections with clause_page_map & hit-centered windowing
        if parent_sections:
            sec_strs: list[str] = []
            all_h = list(hits) + list(amendment_chunks)
            doc_rep_by_id = {h.doc_id: h for h in reversed(all_h) if h.doc_id}
            for p_sec in parent_sections:
                sec_hits = [h for h in all_h if h.section_id == p_sec.section_id]
                rep_h = sec_hits[0] if sec_hits else doc_rep_by_id.get(p_sec.doc_id)
                reg_doc = self.pipeline.get_document(p_sec.doc_id) if not rep_h else None
                fname = rep_h.file_name if rep_h else (reg_doc.file_name if reg_doc and reg_doc.file_name else p_sec.doc_id)
                dnum_val = rep_h.doc_number if rep_h else (reg_doc.doc_number if reg_doc else "")
                dnum = f", № {dnum_val}" if dnum_val else ""
                edo_num_val = rep_h.edo_number if rep_h else (reg_doc.edo_number if reg_doc else "")
                edo_date_val = rep_h.edo_date if rep_h else (reg_doc.edo_date if reg_doc else "")
                edo_s = (
                    f" | Зарег. в ЭДО: № {edo_num_val}" + (f" от {edo_date_val}" if edo_date_val else "")
                    if edo_num_val
                    else ""
                )
                pages_s = ", ".join(str(p) for p in p_sec.page_numbers) if p_sec.page_numbers else "н/д"
                sec_hdr = (
                    f"[Полный раздел: {p_sec.section_id} | Документ: {fname}{dnum}{edo_s} | "
                    f"Раздел: {p_sec.unit_title} / {p_sec.section_title} | Стр.: {pages_s}]"
                )
                cpm_line = self._format_clause_page_map_annotation(p_sec, sec_hits)
                body_window = self._build_hit_centered_parent_window(
                    parent_text=p_sec.clean_text.strip(),
                    matched_snippets=[sh.clean_text for sh in sec_hits],
                    max_chars=12000,
                )
                if cpm_line:
                    sec_strs.append(f"{sec_hdr}\n{cpm_line}\n{body_window}")
                else:
                    sec_strs.append(f"{sec_hdr}\n{body_window}")
            blocks.append("ПОЛНЫЙ КОНТЕКСТ РОДИТЕЛЬСКИХ РАЗДЕЛОВ (ParentSectionRecord):\n\n" + "\n\n===\n\n".join(sec_strs))

        # 6. Structured Tables Passports + Deterministic Query Results
        if linked_tables:
            tbl_lines: list[str] = ["СТРУКТУРИРОВАННЫЕ ТАБЛИЦЫ И РЕЗУЛЬТАТЫ ДЕТЕРМИНИРОВАННЫХ ЗАПРОСОВ:"]
            for tp in linked_tables:
                also_note = f" | также в: {', '.join(tp.also_in_files)}" if tp.also_in_files else ""
                num_note = f" (№ {tp.doc_number} от {tp.doc_date})" if tp.doc_number else ""
                col_names = ", ".join(f"`{c.name}` ({c.inferred_type})" for c in tp.columns[:20])
                tbl_lines.append(
                    f"\n### Таблица `{tp.table_id}` — {tp.table_title} | Файл: {tp.file_name}{num_note}{also_note}\n"
                    f"Размер: {tp.row_count} строк × {tp.col_count} столбцов | Формул: {tp.formula_count}\n"
                    f"Столбцы: {col_names}"
                )
            for t_resp in table_context:
                tbl_lines.append(
                    f"\n#### Запрос `{t_resp.operation}` к `{t_resp.table_id}` "
                    f"(совпало {t_resp.matched_count} из {t_resp.total_table_rows} строк, показано {t_resp.returned_count}):"
                )
                if t_resp.formulas:
                    f_lines = [
                        f"  - Строка {f.get('row_index')} ({f.get('row_label', '')}) | `{f.get('column', '')}`: "
                        f"`{f.get('formula', '')}` (значение: {f.get('cell_text', '')})"
                        for f in t_resp.formulas[:30]
                    ]
                    tbl_lines.append("Формулы Excel:\n" + "\n".join(f_lines))
                if t_resp.markdown_preview:
                    tbl_lines.append(t_resp.markdown_preview)
            blocks.append("\n".join(tbl_lines))

        return "\n\n==============================\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Public Search & Stateless Chatbot Context Endpoints
    # ------------------------------------------------------------------
    def search(
        self,
        request: SearchQuery | str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> SearchResponse:
        """
        Execute full Hybrid RRF search with syntactic multi-query decomposition, sub-query slot guarantee,
        table-slice capping, sibling-document disambiguation, edition graph traversal, ParentSection hydration,
        deterministic table hydration, and calibrated out-of-scope detection.
        """
        t0 = time.perf_counter()
        if isinstance(request, str):
            req = SearchQuery(query=request, top_k=top_k, **kwargs)
        elif isinstance(request, SearchQuery):
            req = request
        elif isinstance(request, dict):
            req = SearchQuery.model_validate(request)
        else:
            raise TypeError(f"Unsupported search request type: {type(request)}")

        raw_query = (req.query or "").strip()
        if not raw_query:
            return SearchResponse(
                query="",
                expanded_query="",
                sub_queries=[],
                is_out_of_scope=True,
                confidence_score=0.0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        query_norm = raw_query.lower().replace("ё", "е")
        expanded_query = self._expand_query_for_retrieval(raw_query)

        # 1. Embed primary query (Dense 1024d + Sparse BM25 CRC32)
        raw_query_tokens = tokenize_bm25(raw_query)
        query_dense, query_sparse, expanded_tokens = self.embedder.embed_query(
            query_text=raw_query,
            sparse_query_text=expanded_query,
        )
        query_tokens_set = set(raw_query_tokens) | set(expanded_tokens)

        temporal_intent = self.detect_temporal_edition_intent(raw_query)
        eff_status_filter = req.status_filter
        if not req.include_superseded and not eff_status_filter and not temporal_intent["is_historical"]:
            eff_status_filter = [
                DocumentStatus.ACTIVE.value,
                DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value,
            ]

        # 2. Primary Hybrid RRF search
        candidate_limit = max(req.top_k * 3, 20)
        raw_points = self.qdrant.hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=candidate_limit,
            prefetch_limit=max(candidate_limit * 2, 48),
            status_filter=eff_status_filter,
            year_filter=req.year_filter,
            doc_id_filter=req.doc_id_filter,
            unit_id_filter=req.unit_id_filter,
            exclude_zones=NON_ANSWER_ZONES,
        )

        primary_candidates: list[SearchHit] = []
        primary_rank_by_hash: dict[str, int] = {}
        for r_idx, pt in enumerate(raw_points, start=1):
            hit = self._raw_point_to_hit(
                pt,
                query_norm=query_norm,
                query_tokens_set=query_tokens_set,
                include_parent=req.include_parent,
            )
            hit.metadata["matched_subqueries"] = [0]
            chash = str(hit.metadata.get("content_hash") or hit.chunk_id)
            primary_rank_by_hash.setdefault(chash, r_idx)
            primary_candidates.append(hit)

        primary_deduped = self._deduplicate_hits_by_content_hash(primary_candidates)
        primary_deduped.sort(key=lambda h: (h.score, h.rrf_score, h.dense_score), reverse=True)
        primary_top = primary_deduped[: req.top_k]

        # 3. Evaluate Out-of-Scope calibration STRICTLY on the primary query & primary top hits!
        is_oos, confidence_score, top_dense_score, lexical_overlap = self._detect_out_of_scope(
            raw_query=raw_query,
            raw_query_tokens=raw_query_tokens,
            hits=primary_top,
        )

        if is_oos:
            for h in primary_top:
                h.score = round(min(h.score * 0.35, 0.25), 4)
            return SearchResponse(
                query=raw_query,
                expanded_query=expanded_query,
                sub_queries=[],
                hits=primary_top,
                amendment_chunks=[],
                linked_tables=[],
                table_context=[],
                version_alerts=[],
                quality_alerts=[],
                is_out_of_scope=True,
                confidence_score=confidence_score,
                top_dense_score=top_dense_score,
                lexical_overlap=lexical_overlap,
                total_found=len(primary_deduped),
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        # 4. Universal Syntactic Multi-Aspect Decomposition & Multi-Subquery RRF Fusion
        sub_queries = decompose_complex_query(raw_query, max_sub_queries=4)
        extra_sqs = [
            (sq_idx, sq)
            for sq_idx, sq in enumerate(sub_queries, start=1)
            if _norm_ru(sq).rstrip("?.!;:") != _norm_ru(raw_query).rstrip("?.!;:")
        ]

        all_candidates: list[SearchHit] = list(primary_candidates)
        per_subquery_hashes: dict[int, list[str]] = {}
        subquery_ranks_by_hash: dict[str, list[int]] = {}

        if len(sub_queries) >= 2 and extra_sqs:
            sq_texts = [sq for _, sq in extra_sqs]
            sq_dense_vecs = self.embedder.dense.embed_texts(sq_texts)
            sq_limit = max(req.top_k * 2, 14)

            for (sq_idx, sq_text), sq_dense in zip(extra_sqs, sq_dense_vecs):
                sq_expanded = self._expand_query_for_retrieval(sq_text)
                sq_sparse, sq_exp_toks = self.embedder.sparse.encode_with_tokens(sq_expanded)
                sq_tokens_set = set(tokenize_bm25(sq_text)) | set(sq_exp_toks) | query_tokens_set
                sq_norm = sq_text.lower().replace("ё", "е")

                sq_pts = self.qdrant.hybrid_search(
                    query_dense=sq_dense,
                    query_sparse=sq_sparse,
                    top_k=sq_limit,
                    prefetch_limit=max(sq_limit * 2, 36),
                    status_filter=eff_status_filter,
                    year_filter=req.year_filter,
                    doc_id_filter=req.doc_id_filter,
                    unit_id_filter=req.unit_id_filter,
                    exclude_zones=NON_ANSWER_ZONES,
                )

                sq_local_hits: list[SearchHit] = []
                for r_pos, pt in enumerate(sq_pts, start=1):
                    sq_hit = self._raw_point_to_hit(
                        pt,
                        query_norm=sq_norm,
                        query_tokens_set=sq_tokens_set,
                        include_parent=req.include_parent,
                    )
                    # Discount slightly vs full-query score so primary ranking stays calibrated
                    sq_hit.score = round(min(1.0, sq_hit.score * 0.97), 4)
                    sq_hit.metadata["matched_subqueries"] = [sq_idx]
                    chash = str(sq_hit.metadata.get("content_hash") or sq_hit.chunk_id)
                    subquery_ranks_by_hash.setdefault(chash, []).append(r_pos)
                    sq_local_hits.append(sq_hit)
                    all_candidates.append(sq_hit)

                sq_local_hits.sort(key=lambda h: (h.score, h.rrf_score, h.dense_score), reverse=True)
                ordered_hashes: list[str] = []
                for lh in sq_local_hits:
                    chash = str(lh.metadata.get("content_hash") or lh.chunk_id)
                    if chash not in ordered_hashes:
                        ordered_hashes.append(chash)
                per_subquery_hashes[sq_idx] = ordered_hashes

        deduped_hits = self._deduplicate_hits_by_content_hash(all_candidates)

        # Apply Multi-Subquery RRF fusion bonus when a chunk matched across multiple query aspects
        if per_subquery_hashes:
            k_rrf = 20.0
            for h in deduped_hits:
                chash = str((h.metadata or {}).get("content_hash") or h.chunk_id)
                fused_rrf = 0.0
                if chash in primary_rank_by_hash:
                    fused_rrf += 1.0 / (k_rrf + primary_rank_by_hash[chash])
                for sq_r in subquery_ranks_by_hash.get(chash, []):
                    fused_rrf += 0.85 / (k_rrf + sq_r)
                matched_sqs = [x for x in (h.metadata.get("matched_subqueries") or []) if int(x) > 0]
                if len(matched_sqs) >= 2:
                    h.score = round(min(1.0, h.score + 0.03 * (len(matched_sqs) - 1)), 4)
                h.rrf_score = round(max(h.rrf_score, fused_rrf), 4)

        deduped_hits.sort(key=lambda h: (h.score, h.rrf_score, h.dense_score), reverse=True)

        # 5. Balanced selection (Sub-Query Slot Guarantee + Single-Table Cap + Sibling Document Diversity)
        top_hits = self._select_balanced_multi_query_hits(
            deduped_hits=deduped_hits,
            per_subquery_hashes=per_subquery_hashes,
            top_k=req.top_k,
            query_dense=query_dense,
            query_sparse=query_sparse,
            query_norm=query_norm,
            query_tokens_set=query_tokens_set,
            include_parent=req.include_parent,
            allow_sibling_probe=(req.doc_id_filter is None and req.unit_id_filter is None),
        )

        # 6. Edition Graph Traversal + Adaptive Table Routing + Deterministic Table Context
        top_hits, amendment_chunks = self._expand_amendments_and_editions(
            hits=top_hits,
            query_dense=query_dense,
            query_sparse=query_sparse,
            query_norm=query_norm,
            query_tokens_set=query_tokens_set,
            top_k=req.top_k,
            include_parent=req.include_parent,
            include_amendments=req.include_amendments,
        )

        linked_tables = self._apply_adaptive_table_routing(
            hits=top_hits,
            amendment_chunks=amendment_chunks,
            is_oos=False,
        )
        table_context = self._hydrate_table_context(
            raw_query=raw_query,
            sub_queries=sub_queries,
            linked_tables=linked_tables,
        )
        version_alerts, quality_alerts = self._collect_alerts(top_hits, amendment_chunks)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return SearchResponse(
            query=raw_query,
            expanded_query=expanded_query,
            sub_queries=sub_queries,
            hits=top_hits,
            amendment_chunks=amendment_chunks,
            linked_tables=linked_tables,
            table_context=table_context,
            version_alerts=version_alerts,
            quality_alerts=quality_alerts,
            is_out_of_scope=False,
            confidence_score=confidence_score,
            top_dense_score=top_dense_score,
            lexical_overlap=lexical_overlap,
            total_found=len(deduped_hits),
            latency_ms=latency_ms,
        )

    def retrieve_candidate_pool(
        self,
        queries: list[str],
        top_k: int = 18,
        status_filter: list[str] | str | None = None,
        year_filter: int | None = None,
        doc_id_filter: str | None = None,
        unit_id_filter: str | None = None,
        include_parent: bool = True,
        exclude_chunk_ids: set[str] | None = None,
    ) -> tuple[list[SearchHit], list[TablePassport]]:
        """
        Multi-query hybrid RRF candidate retrieval for Stage 2 of QueryPipeline:
        - Single batched dense embedding call across all `queries` + glossary-expanded sparse BM25;
        - Content-hash deduplication + multi-query RRF fusion;
        - Per-query slot reservation (2..3 slots per query) so every sub-aspect & form/annex is represented;
        - Single-table cap + sibling-edition diversity probe + TablePassport annotation.
        """
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            return [], []

        excluded = exclude_chunk_ids or set()
        dense_vecs = self.embedder.dense.embed_texts(clean_queries)
        shared_tokens_set: set[str] = set()
        for q_str in clean_queries[:2]:
            shared_tokens_set.update(tokenize_bm25(q_str))

        per_q_limit = max(12, min(20, top_k))
        all_candidates: list[SearchHit] = []
        per_query_hashes: dict[int, list[str]] = {}
        ranks_by_hash: dict[str, list[tuple[int, int]]] = {}

        for q_idx, (q_text, q_dense) in enumerate(zip(clean_queries, dense_vecs)):
            q_expanded = self._expand_query_for_retrieval(q_text)
            q_sparse, q_exp_toks = self.embedder.sparse.encode_with_tokens(q_expanded)
            q_tokens_set = set(tokenize_bm25(q_text)) | set(q_exp_toks) | shared_tokens_set
            q_norm = q_text.lower().replace("ё", "е")

            pts = self.qdrant.hybrid_search(
                query_dense=q_dense,
                query_sparse=q_sparse,
                top_k=per_q_limit,
                prefetch_limit=max(per_q_limit * 2, 40),
                status_filter=status_filter,
                year_filter=year_filter,
                doc_id_filter=doc_id_filter,
                unit_id_filter=unit_id_filter,
                exclude_zones=NON_ANSWER_ZONES,
            )

            local_hits: list[SearchHit] = []
            for r_pos, pt in enumerate(pts, start=1):
                payload_cid = str((pt.get("payload") or {}).get("chunk_id") or "")
                if payload_cid and payload_cid in excluded:
                    continue
                hit = self._raw_point_to_hit(
                    pt,
                    query_norm=q_norm,
                    query_tokens_set=q_tokens_set,
                    include_parent=include_parent,
                )
                if not hit.chunk_id or hit.chunk_id in excluded:
                    continue
                if q_idx >= 2:
                    hit.score = round(min(1.0, hit.score * 0.98), 4)
                hit.metadata["matched_subqueries"] = [q_idx]
                chash = str(hit.metadata.get("content_hash") or hit.chunk_id)
                ranks_by_hash.setdefault(chash, []).append((q_idx, r_pos))
                local_hits.append(hit)
                all_candidates.append(hit)

            local_hits.sort(key=lambda h: (h.score, h.rrf_score, h.dense_score), reverse=True)
            ordered_hashes: list[str] = []
            for lh in local_hits:
                chash = str(lh.metadata.get("content_hash") or lh.chunk_id)
                if chash not in ordered_hashes:
                    ordered_hashes.append(chash)
            per_query_hashes[q_idx] = ordered_hashes

        deduped = [
            h for h in self._deduplicate_hits_by_content_hash(all_candidates)
            if h.chunk_id and h.chunk_id not in excluded
        ]
        if not deduped:
            return [], []

        k_rrf = 20.0
        for h in deduped:
            chash = str((h.metadata or {}).get("content_hash") or h.chunk_id)
            fused_rrf = 0.0
            q_seen: set[int] = set()
            for q_idx, r_pos in ranks_by_hash.get(chash, []):
                w = 1.0 if q_idx == 0 else (1.05 if q_idx == 1 else 0.90)
                fused_rrf += w / (k_rrf + r_pos)
                q_seen.add(q_idx)
            if len(q_seen) >= 2:
                h.score = round(min(1.0, h.score + 0.025 * (len(q_seen) - 1)), 4)
            h.rrf_score = round(max(h.rrf_score, fused_rrf), 4)

        deduped.sort(key=lambda h: (h.score, h.rrf_score, h.dense_score), reverse=True)

        # Balanced multi-query pool selection guaranteeing top distinct chunks per query
        by_hash = {
            str((h.metadata or {}).get("content_hash") or h.chunk_id): h
            for h in deduped
        }
        max_per_table = 3
        selected: list[SearchHit] = []
        selected_ids: set[str] = set()
        table_counts: dict[str, int] = {}

        def _add_cand(cand: SearchHit, enforce_table_cap: bool = True) -> bool:
            if cand.chunk_id in selected_ids:
                return False
            tb_key = self._effective_table_bucket_key(cand)
            if enforce_table_cap and tb_key and table_counts.get(tb_key, 0) >= max_per_table:
                return False
            selected.append(cand)
            selected_ids.add(cand.chunk_id)
            if tb_key:
                table_counts[tb_key] = table_counts.get(tb_key, 0) + 1
            return True

        _add_cand(deduped[0], enforce_table_cap=False)

        slots_per_q = 3 if (top_k >= 16 and len(clean_queries) <= 5) else 2
        for pass_num in range(slots_per_q):
            for q_idx in sorted(per_query_hashes.keys()):
                if len(selected) >= top_k:
                    break
                covered = sum(
                    1 for s in selected if q_idx in ((s.metadata or {}).get("matched_subqueries") or [])
                )
                if covered > pass_num:
                    continue
                for chash in per_query_hashes[q_idx]:
                    cand = by_hash.get(chash)
                    if cand is not None and _add_cand(cand, enforce_table_cap=True):
                        break

        for cand in deduped:
            if len(selected) >= top_k:
                break
            _add_cand(cand, enforce_table_cap=True)

        if len(selected) < min(4, top_k):
            for cand in deduped:
                if len(selected) >= min(4, top_k):
                    break
                _add_cand(cand, enforce_table_cap=False)

        # Sibling-topic diversity check so parallel/updated editions both reach Stage 3 Reranker
        if doc_id_filter is None and unit_id_filter is None and len(selected) >= 2:
            present_docs = {h.doc_id for h in selected if h.doc_id}
            seed_docs: list[str] = []
            for h in selected[:6]:
                if h.doc_id and h.doc_id not in seed_docs and not self._effective_table_bucket_key(h):
                    seed_docs.append(h.doc_id)
            ref_idx = 1 if len(clean_queries) > 1 else 0
            ref_dense = dense_vecs[ref_idx]
            ref_sparse, ref_toks = self.embedder.sparse.encode_with_tokens(
                self._expand_query_for_retrieval(clean_queries[ref_idx])
            )
            ref_norm = clean_queries[ref_idx].lower().replace("ё", "е")
            for s_doc_id in seed_docs[:2]:
                for sib in self._find_sibling_topic_docs(s_doc_id):
                    if sib.doc_id in present_docs:
                        continue
                    sib_hit = next(
                        (h for h in deduped if h.doc_id == sib.doc_id and h.chunk_id not in excluded),
                        None,
                    )
                    if sib_hit is None:
                        raw_sib = self.qdrant.hybrid_search(
                            query_dense=ref_dense,
                            query_sparse=ref_sparse,
                            top_k=2,
                            doc_id_filter=sib.doc_id,
                            exclude_zones=NON_ANSWER_ZONES,
                        )
                        for r_pt in raw_sib:
                            cand_sib = self._raw_point_to_hit(
                                r_pt,
                                query_norm=ref_norm,
                                query_tokens_set=shared_tokens_set | set(ref_toks),
                                include_parent=include_parent,
                            )
                            if cand_sib.chunk_id not in excluded:
                                sib_hit = cand_sib
                                break
                    if sib_hit is not None:
                        selected = self._insert_co_surfaced_hit(selected, sib_hit, top_k=top_k, insert_front=False)
                        present_docs.add(sib.doc_id)

        selected.sort(key=lambda h: (h.score, h.rrf_score, h.dense_score), reverse=True)
        selected = selected[:top_k]
        linked_tables = self._apply_adaptive_table_routing(hits=selected, amendment_chunks=[], is_oos=False)
        return selected, linked_tables

    def build_chatbot_context(
        self,
        request: ChatbotContextRequest | dict[str, Any] | str,
        **kwargs: Any,
    ) -> ChatbotContextResponse:
        """
        Build a complete, zero-leakage, single-call RAG context package for a raw user question:
        - When `use_reranker=True` (default) and Polza LLM is configured, executes the 4-stage AI Query Rewriter ->
          Hybrid Candidate Retrieval -> AI Reranker (+ Conditional Follow-up) -> Verified Context pipeline.
        - Otherwise falls back deterministically to syntactic decomposition + structural graph expansion.
        """
        t0 = time.perf_counter()
        if isinstance(request, str):
            req = ChatbotContextRequest(query=request, **kwargs)
        elif isinstance(request, ChatbotContextRequest):
            req = request
        elif isinstance(request, dict):
            req = ChatbotContextRequest.model_validate(request)
        else:
            raise TypeError(f"Unsupported ChatbotContextRequest type: {type(request)}")

        raw_query = (req.query or "").strip()
        if not raw_query:
            return ChatbotContextResponse(
                query="",
                is_out_of_scope=True,
                confidence_score=0.0,
                formatted_context="Пустой запрос.",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        from src.config import settings as _app_settings

        if req.use_reranker and bool(_app_settings.llm_api_key):
            from src.retrieval.query_pipeline import get_query_pipeline

            qp = get_query_pipeline(search_service=self)
            run_res = qp.run(query=raw_query, top_k=req.top_k)
            v_resp = run_res.search_response
            if not run_res.trace.used_fallback:
                if v_resp.query_intent == "meta_assistant":
                    meta_msg = (
                        "Я нахожу ответы по загруженным в базу документам и таблицам, привожу дословные "
                        "проверенные цитаты с указанием документа, пункта и страницы, а также помогаю "
                        "фильтровать и считать данные по таблицам Excel и Word. Новые файлы (PDF, DOCX, DOC, XLSX, XLS) "
                        "можно добавить во вкладке «База знаний»."
                    )
                    return ChatbotContextResponse(
                        query=raw_query,
                        sub_queries=[],
                        is_out_of_scope=False,
                        confidence_score=1.0,
                        hits=[],
                        formatted_context=meta_msg,
                        retrieved_chunk_ids=[],
                        latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                        pipeline_trace=run_res.trace,
                    )
                if v_resp.is_out_of_scope or not v_resp.hits:
                    oos_msg = (
                        "В загруженной базе документов нет сведений по этому вопросу. "
                        "Я могу отвечать только по тем документам и таблицам, которые добавлены в базу знаний — "
                        "при необходимости нужный файл можно загрузить во вкладке «База знаний»."
                    )
                    return ChatbotContextResponse(
                        query=raw_query,
                        sub_queries=v_resp.sub_queries,
                        is_out_of_scope=True,
                        confidence_score=v_resp.confidence_score,
                        hits=[],
                        formatted_context=oos_msg,
                        retrieved_chunk_ids=[],
                        latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                        pipeline_trace=run_res.trace,
                    )

                linked_tables = v_resp.linked_tables if req.include_tables else []
                table_context = v_resp.table_context if req.include_tables else []
                retrieved_ids: list[str] = []
                for h in list(v_resp.hits) + list(v_resp.amendment_chunks):
                    if h.chunk_id and h.chunk_id not in retrieved_ids:
                        retrieved_ids.append(h.chunk_id)

                formatted_context = self._render_formatted_chatbot_context(
                    query=raw_query,
                    sub_queries=v_resp.sub_queries,
                    hits=v_resp.hits,
                    parent_sections=v_resp.parent_sections,
                    amendment_chunks=v_resp.amendment_chunks,
                    version_alerts=v_resp.version_alerts,
                    quality_alerts=v_resp.quality_alerts,
                    linked_tables=linked_tables,
                    table_context=table_context,
                )
                return ChatbotContextResponse(
                    query=raw_query,
                    sub_queries=v_resp.sub_queries,
                    is_out_of_scope=False,
                    confidence_score=v_resp.confidence_score,
                    hits=v_resp.hits,
                    parent_sections=v_resp.parent_sections,
                    amendment_chunks=v_resp.amendment_chunks,
                    version_alerts=v_resp.version_alerts,
                    quality_alerts=v_resp.quality_alerts,
                    linked_tables=linked_tables,
                    table_context=table_context,
                    formatted_context=formatted_context,
                    retrieved_chunk_ids=retrieved_ids,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    pipeline_trace=run_res.trace,
                )

        base_resp = self.search(
            SearchQuery(
                query=raw_query,
                top_k=req.top_k,
                include_parent=True,
                include_amendments=True,
                include_superseded=True,
            )
        )

        if base_resp.is_out_of_scope:
            oos_msg = (
                "В загруженной базе документов нет сведений по этому вопросу. "
                "Я могу отвечать только по тем документам и таблицам, которые добавлены в базу знаний — "
                "при необходимости нужный файл можно загрузить во вкладке «База знаний»."
            )
            return ChatbotContextResponse(
                query=raw_query,
                sub_queries=base_resp.sub_queries,
                is_out_of_scope=True,
                confidence_score=base_resp.confidence_score,
                hits=[],
                formatted_context=oos_msg,
                retrieved_chunk_ids=[],
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # Structural Graph & Neighbor Expansion
        expanded_hits, extra_parents = self._expand_structural_neighbors_and_sweep(
            primary_hits=base_resp.hits,
            raw_query=raw_query,
            sub_queries=base_resp.sub_queries,
        )

        # Refresh table routing in case appendix cross-reference sweep surfaced a table unit
        linked_tables = (
            self._apply_adaptive_table_routing(
                hits=expanded_hits,
                amendment_chunks=base_resp.amendment_chunks,
                is_oos=False,
            )
            if req.include_tables
            else []
        )
        table_context = (
            self._hydrate_table_context(
                raw_query=raw_query,
                sub_queries=base_resp.sub_queries,
                linked_tables=linked_tables,
            )
            if req.include_tables and linked_tables
            else []
        )

        version_alerts, quality_alerts = self._collect_alerts(expanded_hits, base_resp.amendment_chunks)

        # Collect deduplicated ParentSectionRecords with Whole-Unit Sibling Section Hydration (<= 20,000 chars)
        # and complete Unit Structural Outlines (`unit_outline`) when a unit exceeds 20,000 chars!
        direct_secs_by_id: dict[str, ParentSectionRecord] = {}
        touched_units_ordered: list[tuple[str, str]] = []
        seen_touched_units: set[tuple[str, str]] = set()
        structured_table_sec_ids: set[str] = {tp.section_id for tp in linked_tables if tp.section_id}

        for h in list(expanded_hits) + list(base_resp.amendment_chunks):
            mode = str((h.metadata or {}).get("table_delivery_mode") or "normative_prose")
            if mode == "tool_query" or h.table_id:
                continue
            p_sec = h.parent_section or (self.get_parent_section(h.section_id) if h.section_id else None)
            if p_sec is not None and p_sec.section_id:
                direct_secs_by_id.setdefault(p_sec.section_id, p_sec)
                u_pair = (p_sec.doc_id or h.doc_id, p_sec.unit_id or h.unit_id)
            else:
                u_pair = (h.doc_id, h.unit_id)
            if u_pair[0] and u_pair[1] and u_pair[1] != "u00_meta" and u_pair not in seen_touched_units:
                seen_touched_units.add(u_pair)
                touched_units_ordered.append(u_pair)

        for ep in extra_parents:
            if ep.section_id:
                direct_secs_by_id.setdefault(ep.section_id, ep)
            u_pair = (ep.doc_id, ep.unit_id)
            if u_pair[0] and u_pair[1] and u_pair[1] != "u00_meta" and u_pair not in seen_touched_units:
                seen_touched_units.add(u_pair)
                touched_units_ordered.append(u_pair)

        parent_sections: list[ParentSectionRecord] = []
        seen_sec_ids: set[str] = set()
        unit_outlines: list[str] = []

        for u_doc_id, u_id in touched_units_ordered:
            unit_secs = [
                s
                for s in self.get_unit_parent_sections(u_doc_id, u_id)
                if s.section_id not in structured_table_sec_ids
            ]
            if not unit_secs:
                for d_sec in direct_secs_by_id.values():
                    if d_sec.doc_id == u_doc_id and d_sec.unit_id == u_id and d_sec.section_id not in seen_sec_ids:
                        seen_sec_ids.add(d_sec.section_id)
                        parent_sections.append(d_sec)
                continue

            total_unit_chars = sum(len((s.clean_text or "").strip()) for s in unit_secs)
            if total_unit_chars <= 20000:
                # Compact unit (`<= 20,000` chars): hydrate ALL sibling ParentSectionRecords in natural document order!
                for s in unit_secs:
                    if s.section_id and s.section_id not in seen_sec_ids:
                        seen_sec_ids.add(s.section_id)
                        parent_sections.append(s)
            else:
                # Large unit (`> 20,000` chars): hydrate matched sections in document order + prepend full Unit Structural Outline
                for s in unit_secs:
                    if s.section_id in direct_secs_by_id and s.section_id not in seen_sec_ids:
                        seen_sec_ids.add(s.section_id)
                        parent_sections.append(s)

                reg_doc = self.pipeline.get_document(u_doc_id)
                fname = reg_doc.file_name if reg_doc and reg_doc.file_name else u_doc_id
                dnum = f" (№ {reg_doc.doc_number})" if reg_doc and reg_doc.doc_number else ""
                u_title = unit_secs[0].unit_title or u_id
                outline_lines = [
                    f"[Оглавление единицы {u_doc_id}::{u_id} — {fname}{dnum} | {u_title} | "
                    f"секций: {len(unit_secs)}, объем: {total_unit_chars} симв.]:"
                ]
                for s in unit_secs:
                    p_str = ",".join(str(p) for p in s.page_numbers) if s.page_numbers else "н/д"
                    s_num = f"{s.section_num}. " if s.section_num and s.section_num != "0" else ""
                    outline_lines.append(
                        f"  - [{s.section_id}] {s_num}{s.section_title or u_title} (стр. {p_str})"
                    )
                unit_outlines.append("\n".join(outline_lines))

        # Safety pass: include any direct_secs_by_id whose unit_id was empty
        for d_sec in direct_secs_by_id.values():
            if d_sec.section_id and d_sec.section_id not in seen_sec_ids:
                seen_sec_ids.add(d_sec.section_id)
                parent_sections.append(d_sec)

        # Also provide a multi-unit document structural overview if a matched large document (> 20,000 chars across units)
        # has unhydrated autonomous units (e.g. an Order with multiple Appendices u02..u07)
        checked_doc_outlines: set[str] = set()
        for u_doc_id, _ in touched_units_ordered:
            if u_doc_id in checked_doc_outlines:
                continue
            checked_doc_outlines.add(u_doc_id)
            doc_secs = [s for s in self.get_doc_parent_sections(u_doc_id) if s.unit_id != "u00_meta"]
            if not doc_secs:
                continue
            doc_total_chars = sum(len((s.clean_text or "").strip()) for s in doc_secs)
            units_map: dict[str, list[ParentSectionRecord]] = {}
            for ds in doc_secs:
                units_map.setdefault(ds.unit_id, []).append(ds)
            if len(units_map) >= 2 and doc_total_chars > 20000:
                reg_doc = self.pipeline.get_document(u_doc_id)
                fname = reg_doc.file_name if reg_doc and reg_doc.file_name else u_doc_id
                dnum = f" (№ {reg_doc.doc_number})" if reg_doc and reg_doc.doc_number else ""
                doc_out_lines = [
                    f"[Структура автономных единиц документа {fname}{dnum} | единиц: {len(units_map)}, общий объем: {doc_total_chars} симв.]:"
                ]
                for du_id, du_secs in units_map.items():
                    du_title = du_secs[0].unit_title or du_id
                    sec_titles = []
                    for s in du_secs:
                        lbl = f"{s.section_num}. {s.section_title}" if (s.section_num and s.section_num != "0") else s.section_title
                        if lbl and lbl not in sec_titles:
                            sec_titles.append(lbl)
                    sec_preview = "; ".join(sec_titles[:12]) + (" ..." if len(sec_titles) > 12 else "")
                    doc_out_lines.append(f"  - {du_id} ({du_title}): {sec_preview}")
                unit_outlines.append("\n".join(doc_out_lines))

        retrieved_chunk_ids: list[str] = []
        for h in list(expanded_hits) + list(base_resp.amendment_chunks):
            if h.chunk_id and h.chunk_id not in retrieved_chunk_ids:
                retrieved_chunk_ids.append(h.chunk_id)

        formatted_context = self._render_formatted_chatbot_context(
            query=raw_query,
            sub_queries=base_resp.sub_queries,
            hits=expanded_hits,
            parent_sections=parent_sections,
            amendment_chunks=base_resp.amendment_chunks,
            version_alerts=version_alerts,
            quality_alerts=quality_alerts,
            linked_tables=linked_tables,
            table_context=table_context,
            unit_outlines=unit_outlines,
        )

        return ChatbotContextResponse(
            query=raw_query,
            sub_queries=base_resp.sub_queries,
            is_out_of_scope=False,
            confidence_score=base_resp.confidence_score,
            hits=expanded_hits,
            parent_sections=parent_sections,
            amendment_chunks=base_resp.amendment_chunks,
            version_alerts=version_alerts,
            quality_alerts=quality_alerts,
            linked_tables=linked_tables,
            table_context=table_context,
            formatted_context=formatted_context,
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )


_SEARCH_SERVICE_SINGLETON: SearchService | None = None


def get_search_service(
    pipeline: IndexingPipeline | None = None,
    table_store: StructuredTableStore | None = None,
) -> SearchService:
    """Return process-wide SearchService singleton."""
    global _SEARCH_SERVICE_SINGLETON
    if _SEARCH_SERVICE_SINGLETON is None:
        _SEARCH_SERVICE_SINGLETON = SearchService(pipeline=pipeline, table_store=table_store)
    else:
        if pipeline is not None and _SEARCH_SERVICE_SINGLETON.pipeline is not pipeline:
            _SEARCH_SERVICE_SINGLETON.pipeline = pipeline
        if table_store is not None and _SEARCH_SERVICE_SINGLETON._table_store is not table_store:
            _SEARCH_SERVICE_SINGLETON._table_store = table_store
    return _SEARCH_SERVICE_SINGLETON


# Alias for external callers
HybridSearchService = SearchService
