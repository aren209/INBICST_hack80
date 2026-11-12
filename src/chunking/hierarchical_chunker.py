from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.chunking.glossary import GlossaryManager
from src.config import (
    CHUNK_OVERLAP_RATIO,
    MAX_CLAUSE_CHARS,
    MAX_OVERLAP_CHARS,
    MAX_TABLE_FORM_CHARS,
    MIN_CLAUSE_CHARS,
    MIN_OVERLAP_CHARS,
)
from src.ingestion.parsers import ParsedBlock
from src.ingestion.registry import DocumentMetadata


@dataclass
class ParentSectionRecord:
    section_id: str
    doc_id: str
    doc_title: str
    doc_number: str
    doc_date: str
    unit_id: str
    unit_title: str
    section_no: str
    section_title: str
    text: str
    clause_ids: list[str] = field(default_factory=list)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    file_name: str
    doc_title: str
    doc_type: str
    doc_number: str
    doc_date: str
    effective_from: str
    status: str  # "active" | "superseded" | "draft"
    supersedes: list[str]
    superseded_by: list[str]
    related_docs: list[str]
    version_note: str
    quality_warning: str
    unit_id: str
    unit_title: str
    section_id: str
    section_no: str
    section_title: str
    clause_no: str
    parent_id: str
    prev_id: str
    next_id: str
    zone: str  # "body" | "table" | "form" | "approval" | "meta"
    roles: list[str]
    topic_tags: list[str]
    page_from: int
    page_to: int
    raw_text: str
    embed_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ROLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Руководитель проекта", re.compile(r"руководител\w+\s+(?:научного\s+)?проекта", re.I)),
    ("Руководитель подразделения / ЦФО", re.compile(r"руководител\w+\s+(?:структурного\s+)?(?:подразделения|ЦФО)", re.I)),
    ("ПФО", re.compile(r"(?:\bПФО\b|планово-финансов\w+\s+отдел)", re.I)),
    ("ФЭУ", re.compile(r"(?:\bФЭУ\b|финансово-экономическ\w+\s+управлени)", re.I)),
    ("Бухгалтерия", re.compile(r"(?:бухгалтер\w+|главн\w+\s+бухгалтер)", re.I)),
    ("УИТ", re.compile(r"(?:\bУИТ\b|управлени\w+\s+информационных\s+технологий|перв\w+\s+лини\w+\s+техподдержки)", re.I)),
    ("Отдел кадров / HR", re.compile(r"(?:отдел\w*\s+подбора|управлени\w+\s+по\s+работе\s+с\s+персоналом|кадров\w+\s+служб)", re.I)),
    ("Командируемый работник / обучающийся", re.compile(r"(?:командируем\w+|работник|обучающ\w+)", re.I)),
    ("Ответственный за электрохозяйство", re.compile(r"(?:ответственн\w+\s+за\s+электрохозяйство|электротехническ\w+\s+персонал)", re.I)),
    ("Администратор ЦОД", re.compile(r"(?:администратор\w*\s+ЦОД|ответственн\w+\s+за\s+ЦОД)", re.I)),
    ("Жилищная комиссия / Профком", re.compile(r"(?:жилищн\w+\s+комисси|профком|отдел\s+поселения|общежити)", re.I)),
    ("Проректор", re.compile(r"проректор", re.I)),
]


def extract_roles(text: str) -> list[str]:
    found: list[str] = []
    for role_name, pat in ROLE_PATTERNS:
        if pat.search(text):
            found.append(role_name)
    return found


class HierarchicalChunker:
    """
    Implements V3 (Parent-Child) hierarchical chunking:
    file -> unit -> section -> clause -> item
    """

    UNIT_HEADER_RE = re.compile(
        r"^(?:"
        r"Приложение\s*(?:№|N)?\s*[0-9а-яА-Яa-zA-Z]+(?:\s+к\s+(?:приказу|распоряжению|регламенту|положению))?"
        r"|УТВЕРЖДЕН[АОЫ]?\s+приказом"
        r"|РЕГЛАМЕНТ\b"
        r"|ПОЛОЖЕНИЕ\b"
        r"|ИНСТРУКЦИЯ\b"
        r"|ПОРЯДОК\b"
        r"|КАРТОЧКА\s+ПРОЕКТА\b"
        r"|ЛИСТ\s+СОГЛАСОВАНИЯ\b"
        r")",
        re.IGNORECASE,
    )

    # Known section titles in MIPT regulations (even when unnumbered or OCR-garbled)
    KNOWN_SECTION_TITLES_RE = re.compile(
        r"^(?:(\d{1,2}|[IVX]{1,4})[\.\)]?\s*)?("
        r"Общие\s+положения"
        r"|Термины(?:,\s*определения)?\s+и\s+сокращения"
        r"|Определения\s+и\s+сокращения"
        r"|Порядок\s+[а-яА-ЯёЁ\s,-]{4,220}"
        r"|Правила\s+[а-яА-ЯёЁ\s,-]{4,220}"
        r"|Требования\s+к\s+[а-яА-ЯёЁ\s,-]{4,220}"
        r"|Ответственность(?:\s+сторон|\s+за\s+нарушение)?"
        r"|Заключительные\s+положения"
        r"|Организация\s+[а-яА-ЯёЁ\s,-]{4,220}"
        r"|Регламент\s+работы\s+[а-яА-ЯёЁ\s,-]{4,220}"
        r"|Сроки\s+и\s+условия\s+[а-яА-ЯёЁ\s,-]{4,220}"
        r"|Оформление\s+[а-яА-ЯёЁ\s,-]{4,220}"
        r"|Финансовое\s+обеспечение\s+[а-яА-ЯёЁ\s,-]{4,220}"
        r"|Возмещение\s+расходов\s+[а-яА-ЯёЁ\s,-]{4,220}"
        r")$",
        re.IGNORECASE,
    )

    # Section heading: e.g. "1. Общие положения" or "Раздел 2. Порядок открытия ФЛС"
    SECTION_RE = re.compile(
        r"^(?:Раздел\s+)?(\d{1,2}|[IVX]{1,4})\.?\s+([А-ЯЁ][а-яА-ЯёЁ0-9\s,()«»\"':/–—-]{3,220})$"
    )

    # Первое слово заголовка раздела. Нужно, чтобы длинный «2. Порядок открытия...»
    # не принимали за пункт 2, но «2. Настоящий Регламент...» осталось пунктом.
    SECTION_TITLE_STARTERS = {
        "общие", "порядок", "правила", "требования", "организация", "сроки",
        "оформление", "финансовое", "возмещение", "ответственность",
        "заключительные", "термины", "определения", "цели", "задачи", "состав",
        "структура", "контроль", "учет", "учёт", "взаимодействие", "обеспечение",
        "эксплуатация", "бронирование", "поселение", "командирование", "подбор",
        "бюджет", "бюджетирование", "формирование", "корректировка",
        "согласование", "утверждение", "предмет", "область", "обязанности",
        "права", "условия", "оплата", "материальная", "дисциплинарная",
        "техника", "охрана", "пожарная", "электробезопасность", "доступ",
        "режим", "пропускной", "информационная", "безопасность",
        "персональные", "данные", "делопроизводство", "документооборот",
        "архив", "закупки", "инвестиции", "заимствование", "гарантии",
        "риски", "мониторинг", "отчетность", "отчётность", "аудит", "налоги",
        "бухгалтерский", "управленческий", "кадры", "персонал", "найм",
        "адаптация", "обучение", "развитие", "мотивация", "премирование",
        "социальная", "жилищная", "общежитие", "питание", "транспорт",
        "связь", "ит", "цод", "порядок",
    }

    # Multi-level clause: e.g. "1.1.", "2.3.1", "2.6." (protecting against dates like 25.11.2021)
    MULTI_CLAUSE_RE = re.compile(
        r"^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\.?(?:\s+|$)(.+)$"
    )

    # Single-level numbered clause inside an Order or flat Regulation: e.g. "1. Утвердить...", "17. Установить..."
    SINGLE_CLAUSE_RE = re.compile(r"^(\d{1,2})\.\s+([А-ЯЁ0-9«\"].+)$")

    # Sub-item: "а)", "б)", "1)", "-", "•", "1.1.1."
    SUBITEM_RE = re.compile(r"^(?:[а-яёa-z]\)|\d{1,2}\)|[-•–])\s+.+$")

    DATE_PREFIX_RE = re.compile(r"^\d{2}\.\d{2}\.20\d{2}\b")

    def __init__(
        self,
        glossary: GlossaryManager | None = None,
        min_clause_chars: int = MIN_CLAUSE_CHARS,
        max_clause_chars: int = MAX_CLAUSE_CHARS,
        max_table_form_chars: int = MAX_TABLE_FORM_CHARS,
        overlap_ratio: float = CHUNK_OVERLAP_RATIO,
    ) -> None:
        self.glossary = glossary or GlossaryManager()
        self.min_clause_chars = min_clause_chars
        self.max_clause_chars = max_clause_chars
        self.max_table_form_chars = max_table_form_chars
        self.overlap_ratio = overlap_ratio

    @staticmethod
    def _is_generic_clause_no(clause_no: str) -> bool:
        low = clause_no.strip().lower()
        return (
            low in {"преамбула", "вводная часть", "паспорт"}
            or low.startswith("раздел ")
        )

    @classmethod
    def _combine_clause_no(cls, first: str, second: str) -> str:
        if cls._is_generic_clause_no(first) and not cls._is_generic_clause_no(second):
            return second
        if cls._is_generic_clause_no(second) and not cls._is_generic_clause_no(first):
            return first
        if first == second:
            return first
        if second.startswith(f"{first} (ч."):
            return first
        if second.startswith(first + "."):
            return f"{first} ({second})"
        return f"{first}, {second}"

    @staticmethod
    def _extract_tail_overlap(
        text: str,
        ratio: float,
        min_chars: int = MIN_OVERLAP_CHARS,
        max_chars: int = MAX_OVERLAP_CHARS,
    ) -> str:
        """Возвращает последние ~ratio символов текста с выравниванием по границе слова."""
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) < 20 or ratio <= 0:
            return ""
        target = int(round(len(clean) * ratio))
        target = max(min_chars, min(max_chars, target))
        if target >= len(clean):
            target = max(15, int(len(clean) * 0.35))
        start = max(0, len(clean) - target)
        if start > 0 and not clean[start - 1].isspace():
            # Сдвигаем к началу следующего слова, чтобы не резать слово пополам
            next_space = clean.find(" ", start)
            if next_space != -1 and len(clean) - next_space >= 15:
                start = next_space + 1
            else:
                prev_space = clean.rfind(" ", 0, start)
                start = (prev_space + 1) if prev_space != -1 else start
        return clean[start:].strip()

    @staticmethod
    def _extract_head_overlap(
        text: str,
        ratio: float,
        min_chars: int = MIN_OVERLAP_CHARS,
        max_chars: int = MAX_OVERLAP_CHARS,
    ) -> str:
        """Возвращает первые ~ratio символов текста с выравниванием по границе слова."""
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) < 20 or ratio <= 0:
            return ""
        target = int(round(len(clean) * ratio))
        target = max(min_chars, min(max_chars, target))
        if target >= len(clean):
            target = max(15, int(len(clean) * 0.35))
        end = min(len(clean), target)
        if end < len(clean) and not clean[end].isspace():
            next_space = clean.find(" ", end)
            if next_space != -1 and next_space <= max_chars + 20:
                end = next_space
            else:
                prev_space = clean.rfind(" ", 0, end)
                if prev_space >= 15:
                    end = prev_space
        return clean[:end].strip()

    def _build_embed_text(self, meta: DocumentMetadata, chunk: ChunkRecord) -> str:
        if chunk.zone == "meta":
            return self.glossary.expand_text(
                f"[Документ: {meta.doc_title} | Статус: {meta.status} | Паспорт версий и коллизий]\n{chunk.raw_text}"
            )
        breadcrumb = (
            f"[Документ: {meta.doc_title} (№ {meta.doc_number} от {meta.doc_date or 'б/д'}) | "
            f"Статус: {meta.status} | Часть: {chunk.unit_title} | "
            f"Раздел {chunk.section_no}: {chunk.section_title} | Пункт: {chunk.clause_no}"
        )
        if chunk.roles:
            breadcrumb += f" | Роли: {', '.join(chunk.roles)}"
        if meta.topic_tags:
            breadcrumb += f" | Темы: {', '.join(meta.topic_tags)}"
        breadcrumb += "]"

        expansions = self.glossary.extract_expansions(chunk.raw_text)
        terms_block = f"\n[Термины: {'; '.join(expansions)}]" if expansions else ""
        return f"{breadcrumb}{terms_block}\n{chunk.raw_text}"

    def _merge_short_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        """Объединяет слишком короткие чанки (< min_clause_chars) с соседними в рамках раздела/части."""
        if len(chunks) <= 1 or self.min_clause_chars <= 0:
            return chunks

        meta_chunks = [c for c in chunks if c.zone == "meta"]
        work = [c for c in chunks if c.zone != "meta"]
        if len(work) <= 1:
            return meta_chunks + work

        def merge_two(left: ChunkRecord, right: ChunkRecord) -> ChunkRecord:
            combined_raw = f"{left.raw_text}\n{right.raw_text}".strip()
            combined_roles = extract_roles(combined_raw)
            keep_sec_from = (
                right
                if self._is_generic_clause_no(left.clause_no)
                and not self._is_generic_clause_no(right.clause_no)
                else left
            )
            left.section_id = keep_sec_from.section_id
            left.section_no = keep_sec_from.section_no
            left.section_title = keep_sec_from.section_title
            left.parent_id = keep_sec_from.parent_id
            left.clause_no = self._combine_clause_no(left.clause_no, right.clause_no)
            left.page_from = min(left.page_from, right.page_from)
            left.page_to = max(left.page_to, right.page_to)
            left.raw_text = combined_raw
            left.roles = combined_roles
            return left

        # Pass 1: объединяем короткий чанк со следующим в том же разделе (или unit для вводных заголовков)
        i = 0
        forward_merged: list[ChunkRecord] = []
        while i < len(work):
            curr = work[i]
            max_limit = (
                self.max_table_form_chars
                if curr.zone in ("table", "form")
                else self.max_clause_chars
            )
            while (
                len(curr.raw_text) < self.min_clause_chars
                and i + 1 < len(work)
                and work[i + 1].unit_id == curr.unit_id
                and work[i + 1].zone == curr.zone
                and (
                    work[i + 1].section_id == curr.section_id
                    or self._is_generic_clause_no(curr.clause_no)
                    or len(curr.raw_text) < 100
                )
                and len(curr.raw_text) + 1 + len(work[i + 1].raw_text) <= max_limit + 140
            ):
                curr = merge_two(curr, work[i + 1])
                i += 1
            forward_merged.append(curr)
            i += 1

        # Pass 2: если в конце раздела/части остался короткий «хвост» (< min_clause_chars),
        # присоединяем его к предыдущему чанку того же раздела/части.
        pass2_work: list[ChunkRecord] = []
        for curr in forward_merged:
            max_limit = (
                self.max_table_form_chars
                if curr.zone in ("table", "form")
                else self.max_clause_chars
            )
            if (
                len(curr.raw_text) < self.min_clause_chars
                and pass2_work
                and pass2_work[-1].unit_id == curr.unit_id
                and pass2_work[-1].zone == curr.zone
                and (
                    pass2_work[-1].section_id == curr.section_id
                    or len(curr.raw_text) < 110
                )
                and len(pass2_work[-1].raw_text) + 1 + len(curr.raw_text) <= max_limit + 180
            ):
                pass2_work[-1] = merge_two(pass2_work[-1], curr)
            else:
                pass2_work.append(curr)

        # Pass 3: если после внутриюнитного слияния остались одиночные короткие штампы
        # («УТВЕРЖДЕН приказом...», «к приказу ОТ», заголовки перед таблицей) < min_clause_chars,
        # сливаем их со следующим (или предыдущим) чанком документа, сохраняя реквизиты основного чанка.
        final_work: list[ChunkRecord] = []
        idx = 0
        while idx < len(pass2_work):
            curr = pass2_work[idx]
            if len(curr.raw_text) < self.min_clause_chars and idx + 1 < len(pass2_work):
                nxt = pass2_work[idx + 1]
                nxt.raw_text = f"{curr.raw_text}\n{nxt.raw_text}".strip()
                nxt.roles = extract_roles(nxt.raw_text)
                nxt.page_from = min(curr.page_from, nxt.page_from)
                nxt.page_to = max(curr.page_to, nxt.page_to)
                if not self._is_generic_clause_no(curr.clause_no) and curr.zone == nxt.zone:
                    nxt.clause_no = self._combine_clause_no(curr.clause_no, nxt.clause_no)
                idx += 1
                continue
            if len(curr.raw_text) < self.min_clause_chars and final_work:
                prev = final_work[-1]
                prev.raw_text = f"{prev.raw_text}\n{curr.raw_text}".strip()
                prev.roles = extract_roles(prev.raw_text)
                prev.page_from = min(prev.page_from, curr.page_from)
                prev.page_to = max(prev.page_to, curr.page_to)
                idx += 1
                continue
            final_work.append(curr)
            idx += 1

        return meta_chunks + final_work

    def _is_section_header(self, block: ParsedBlock) -> tuple[str, str] | None:
        text = block.text.strip()
        if len(text) > 260 or "\n" in text:
            return None
        if self.DATE_PREFIX_RE.match(text):
            return None

        m_known = self.KNOWN_SECTION_TITLES_RE.match(text)
        if m_known:
            sec_no = m_known.group(1) or ""
            sec_title = m_known.group(2).strip(" .:")
            return (sec_no or "б/н", sec_title)

        # Numbered section like "1. Общие положения" (short title without sentence-ending period in middle)
        m_sec = self.SECTION_RE.match(text)
        if m_sec:
            sec_no = m_sec.group(1)
            sec_title = m_sec.group(2).strip()
            # Distinguish a section title from a long order clause ("1. Утвердить прилагаемый Регламент...")
            action_verbs = (
                "утвердить",
                "ввести",
                "признать",
                "установить",
                "возложить",
                "контроль",
                "направить",
                "производить",
                "осуществлять",
                "внести",
                "считать",
                "приказываю",
            )
            first_word = sec_title.split()[0].lower().rstrip(".,:")
            is_section_starter = first_word in self.SECTION_TITLE_STARTERS
            if (
                len(sec_title) <= 220
                and not sec_title.endswith(";")
                and first_word not in action_verbs
                and (
                    is_section_starter
                    or block.is_bold
                    or "Heading" in block.style_name
                    or len(sec_title) <= 65
                )
            ):
                return (sec_no, sec_title)
        return None

    def _match_clause_start(self, text: str) -> tuple[str, str] | None:
        s = text.strip()
        if self.DATE_PREFIX_RE.match(s):
            return None

        m_multi = self.MULTI_CLAUSE_RE.match(s)
        if m_multi:
            c_no = m_multi.group(1)
            parts = [int(p) for p in c_no.split(".") if p.isdigit()]
            # Guard against financial codes like 01.002.12345 or dates
            if parts and parts[0] <= 30 and all(p <= 60 for p in parts):
                return (c_no, s)

        m_single = self.SINGLE_CLAUSE_RE.match(s)
        if m_single:
            c_no = m_single.group(1)
            if int(c_no) <= 50:
                return (c_no, s)

        return None

    def chunk_document(
        self, meta: DocumentMetadata, blocks: list[ParsedBlock]
    ) -> tuple[list[ChunkRecord], list[ParentSectionRecord]]:
        # Learn any new abbreviations from document blocks.
        # Таблицы и OCR-обрывки не используем: там regex «АББР — текст» ловит
        # ячейки ЦФО и рождает мусорные определения.
        for b in blocks:
            if b.zone != "approval" and b.block_type == "paragraph":
                self.glossary.learn_from_text(b.text)

        chunks: list[ChunkRecord] = []
        parent_sections: dict[str, ParentSectionRecord] = {}

        unit_idx = 1
        unit_id = "u01_main"
        unit_title = (
            "Приказ (основной текст)"
            if "приказ" in meta.doc_type.lower()
            else meta.doc_title
        )

        sec_idx = 1
        section_no = "1"
        section_title = "Общие положения и вводная часть"
        section_id = f"{meta.doc_id}::{unit_id}::s{sec_idx:02d}"

        clause_seq = 0
        curr_clause_no = "преамбула"
        curr_lines: list[str] = []
        curr_page_from = 1
        curr_page_to = 1
        curr_zone = "body"

        def register_section(s_id: str, u_id: str, u_title: str, s_no: str, s_title: str) -> None:
            if s_id not in parent_sections:
                parent_sections[s_id] = ParentSectionRecord(
                    section_id=s_id,
                    doc_id=meta.doc_id,
                    doc_title=meta.doc_title,
                    doc_number=meta.doc_number,
                    doc_date=meta.doc_date,
                    unit_id=u_id,
                    unit_title=u_title,
                    section_no=s_no,
                    section_title=s_title,
                    text="",
                    clause_ids=[],
                )

        register_section(section_id, unit_id, unit_title, section_no, section_title)

        def flush_clause() -> None:
            nonlocal clause_seq, curr_lines
            if not curr_lines:
                return
            full_raw = "\n".join(curr_lines).strip()
            curr_lines = []
            if len(full_raw) < 12:
                return

            max_limit = (
                self.max_table_form_chars
                if curr_zone in ("table", "form")
                else self.max_clause_chars
            )

            # Split overly long clauses (> max_limit) along paragraph/subitem lines
            sub_texts: list[str] = []
            if len(full_raw) <= max_limit:
                sub_texts.append(full_raw)
            else:
                lines = full_raw.splitlines()
                header_line = lines[0][:120]
                buf: list[str] = []
                buf_len = 0
                for ln in lines:
                    if buf_len + len(ln) > max_limit and buf:
                        sub_texts.append("\n".join(buf))
                        buf = [f"(Продолжение п. {curr_clause_no}: {header_line})", ln]
                        buf_len = len(buf[0]) + len(ln)
                    else:
                        buf.append(ln)
                        buf_len += len(ln) + 1
                if buf:
                    sub_texts.append("\n".join(buf))

                # Если последний кусок разбиения оказался меньше минимального порога,
                # склеиваем его с предпоследним, чтобы не плодить короткие «хвосты».
                if (
                    len(sub_texts) >= 2
                    and len(sub_texts[-1]) < self.min_clause_chars
                    and len(sub_texts[-2]) + len(sub_texts[-1]) <= max_limit + 180
                ):
                    tail = sub_texts.pop()
                    tail_lines = tail.splitlines()
                    if tail_lines and tail_lines[0].startswith("(Продолжение п."):
                        tail_lines = tail_lines[1:]
                    sub_texts[-1] = sub_texts[-1] + "\n" + "\n".join(tail_lines)

            for part_idx, raw_part in enumerate(sub_texts):
                clause_seq += 1
                c_id = f"{meta.doc_id}::{unit_id}::s{sec_idx:02d}::c{clause_seq:03d}"
                disp_clause = (
                    curr_clause_no
                    if len(sub_texts) == 1
                    else f"{curr_clause_no} (ч. {part_idx + 1})"
                )
                roles = extract_roles(raw_part)
                register_section(
                    section_id, unit_id, unit_title, section_no, section_title
                )
                chunk = ChunkRecord(
                    chunk_id=c_id,
                    doc_id=meta.doc_id,
                    file_name=meta.file_name,
                    doc_title=meta.doc_title,
                    doc_type=meta.doc_type,
                    doc_number=meta.doc_number,
                    doc_date=meta.doc_date,
                    effective_from=meta.effective_from,
                    status=meta.status,
                    supersedes=list(meta.supersedes),
                    superseded_by=list(meta.superseded_by),
                    related_docs=list(meta.related_docs),
                    version_note=meta.version_note,
                    quality_warning=meta.quality_warning,
                    unit_id=unit_id,
                    unit_title=unit_title,
                    section_id=section_id,
                    section_no=section_no,
                    section_title=section_title,
                    clause_no=disp_clause,
                    parent_id=section_id,
                    prev_id="",
                    next_id="",
                    zone=curr_zone,
                    roles=roles,
                    topic_tags=list(meta.topic_tags),
                    page_from=curr_page_from,
                    page_to=curr_page_to,
                    raw_text=raw_part,
                    embed_text="",
                )
                chunks.append(chunk)

        # Emit an authoritative version/warning card chunk if the document has version collisions or scan defects
        if meta.version_note or meta.quality_warning:
            clause_seq += 1
            meta_chunk_id = f"{meta.doc_id}::u00_meta::s00::c{clause_seq:03d}"
            meta_lines = [
                f"Паспорт документа: {meta.doc_title} (№ {meta.doc_number} от {meta.doc_date or 'б/д'}, статус: {meta.status})."
            ]
            if meta.version_note:
                meta_lines.append(f"Сведения о версиях и изменениях: {meta.version_note}")
            if meta.quality_warning:
                meta_lines.append(f"Примечание о полноте исходного документа: {meta.quality_warning}")
            meta_raw = "\n".join(meta_lines)
            meta_embed = self.glossary.expand_text(
                f"[Документ: {meta.doc_title} | Статус: {meta.status} | Паспорт версий и коллизий]\n{meta_raw}"
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=meta_chunk_id,
                    doc_id=meta.doc_id,
                    file_name=meta.file_name,
                    doc_title=meta.doc_title,
                    doc_type=meta.doc_type,
                    doc_number=meta.doc_number,
                    doc_date=meta.doc_date,
                    effective_from=meta.effective_from,
                    status=meta.status,
                    supersedes=list(meta.supersedes),
                    superseded_by=list(meta.superseded_by),
                    related_docs=list(meta.related_docs),
                    version_note=meta.version_note,
                    quality_warning=meta.quality_warning,
                    unit_id="u00_meta",
                    unit_title="Паспорт версий и актуальности документа",
                    section_id=f"{meta.doc_id}::u00_meta::s00",
                    section_no="0",
                    section_title="Актуальность, редакции и примечания",
                    clause_no="паспорт",
                    parent_id=f"{meta.doc_id}::u00_meta::s00",
                    prev_id="",
                    next_id="",
                    zone="meta",
                    roles=extract_roles(meta_raw),
                    topic_tags=list(meta.topic_tags),
                    page_from=1,
                    page_to=1,
                    raw_text=meta_raw,
                    embed_text=meta_embed,
                )
            )

        for block in blocks:
            # Skip approval sheets from polluting normative chunks (keep them tagged as zone="approval" if needed)
            if block.zone == "approval":
                continue

            text = block.text.strip()
            if not text:
                continue

            # Table or Form block -> flush active text clause and emit dedicated table/form chunk
            if block.block_type in ("table", "form"):
                flush_clause()
                # Если таблица/форма идёт до первого раздела, не подписываем её
                # «Общими положениями»: создаём отдельный раздел для цитирования.
                if section_title == "Общие положения и вводная часть":
                    sec_idx += 1
                    section_no = "Т" if block.block_type == "table" else "Ф"
                    section_title = (
                        "Табличная часть"
                        if block.block_type == "table"
                        else "Форма / бланк"
                    )
                    section_id = f"{meta.doc_id}::{unit_id}::s{sec_idx:02d}"
                    register_section(
                        section_id, unit_id, unit_title, section_no, section_title
                    )
                curr_zone = block.zone
                curr_clause_no = (
                    f"таблица (стр. {block.page_no})"
                    if block.block_type == "table"
                    else f"форма/бланк (стр. {block.page_no})"
                )
                curr_page_from = block.page_no
                curr_page_to = block.page_no
                curr_lines = [text]
                flush_clause()
                curr_zone = "body"
                continue

            # Check for new Unit boundary (Приложение № N, Регламент..., Положение..., Инструкция...)
            if len(text) <= 160 and self.UNIT_HEADER_RE.match(text):
                flush_clause()
                unit_idx += 1
                unit_id = f"u{unit_idx:02d}"
                unit_title = text.rstrip(".:")
                sec_idx = 1
                section_no = "1"
                section_title = unit_title
                section_id = f"{meta.doc_id}::{unit_id}::s{sec_idx:02d}"
                register_section(
                    section_id, unit_id, unit_title, section_no, section_title
                )
                curr_clause_no = "вводная часть"
                curr_page_from = block.page_no
                curr_page_to = block.page_no
                continue

            # Check for Section Header
            sec_match = self._is_section_header(block)
            if sec_match is not None:
                flush_clause()
                sec_idx += 1
                section_no, section_title = sec_match
                section_id = f"{meta.doc_id}::{unit_id}::s{sec_idx:02d}"
                register_section(
                    section_id, unit_id, unit_title, section_no, section_title
                )
                curr_clause_no = f"раздел {section_no}"
                curr_page_from = block.page_no
                curr_page_to = block.page_no
                continue

            # Check for Clause start (1.1., 2.3., etc.)
            clause_match = self._match_clause_start(text)
            if clause_match is not None:
                c_no, c_text = clause_match
                # If a top-level number like "1. Общие положения" wasn't caught as a section,
                # check if it's actually a short section heading before subclauses 1.1
                if "." not in c_no and len(c_text) <= 75 and not c_text.endswith(";"):
                    flush_clause()
                    sec_idx += 1
                    section_no = c_no
                    section_title = re.sub(r"^\d+\.\s*", "", c_text).strip(" .:")
                    section_id = f"{meta.doc_id}::{unit_id}::s{sec_idx:02d}"
                    register_section(
                        section_id, unit_id, unit_title, section_no, section_title
                    )
                    curr_clause_no = c_no
                    curr_lines = [c_text]
                    curr_page_from = block.page_no
                    curr_page_to = block.page_no
                    continue

                flush_clause()
                curr_clause_no = c_no
                curr_lines = [c_text]
                curr_page_from = block.page_no
                curr_page_to = block.page_no
                continue

            # Continuation paragraph or sub-item (а), б), -, •) stays attached to current clause
            if not curr_lines:
                curr_page_from = block.page_no
            curr_page_to = block.page_no
            curr_lines.append(text)

        flush_clause()

        # 1. Гарантируем минимальный размер чанка (сливаем короткие фрагменты < MIN_CLAUSE_CHARS)
        chunks = self._merge_short_chunks(chunks)

        # 2. Перенумеровываем chunk_id и собираем чистый текст родительских разделов (без дублей перекрытия)
        for sec in parent_sections.values():
            sec.clause_ids = []
            sec.text = ""

        seq_counter = 0
        for ch in chunks:
            seq_counter += 1
            if ch.zone == "meta":
                ch.chunk_id = f"{meta.doc_id}::u00_meta::s00::c{seq_counter:03d}"
                continue
            sec_suffix = ch.section_id.split("::")[-1] if "::" in ch.section_id else "s01"
            ch.chunk_id = f"{meta.doc_id}::{ch.unit_id}::{sec_suffix}::c{seq_counter:03d}"
            register_section(
                ch.section_id,
                ch.unit_id,
                ch.unit_title,
                ch.section_no,
                ch.section_title,
            )
            parent_sections[ch.section_id].clause_ids.append(ch.chunk_id)
            if parent_sections[ch.section_id].text:
                parent_sections[ch.section_id].text += "\n\n" + ch.raw_text
            else:
                parent_sections[ch.section_id].text = ch.raw_text

        # Удаляем пустые разделы, из которых все короткие вводные чанки перетекли в соседние
        active_sections = [
            sec for sec in parent_sections.values() if sec.text.strip()
        ]

        # 3. Двустороннее перекрытие соседних чанков (~12% конца предыдущего и ~12% начала следующего)
        core_texts = [ch.raw_text for ch in chunks]
        for i, ch in enumerate(chunks):
            if ch.zone == "meta":
                ch.embed_text = self._build_embed_text(meta, ch)
                continue

            is_compact_zone = ch.zone in ("table", "form")
            max_ov = 48 if is_compact_zone else MAX_OVERLAP_CHARS
            min_ov = 24 if is_compact_zone else MIN_OVERLAP_CHARS

            prev_overlap = ""
            if (
                i > 0
                and chunks[i - 1].zone != "meta"
                and chunks[i - 1].unit_id == ch.unit_id
            ):
                prev_overlap = self._extract_tail_overlap(
                    core_texts[i - 1],
                    ratio=self.overlap_ratio,
                    min_chars=min_ov,
                    max_chars=max_ov,
                )

            next_overlap = ""
            if (
                i + 1 < len(chunks)
                and chunks[i + 1].zone != "meta"
                and chunks[i + 1].unit_id == ch.unit_id
            ):
                next_overlap = self._extract_head_overlap(
                    core_texts[i + 1],
                    ratio=self.overlap_ratio,
                    min_chars=min_ov,
                    max_chars=max_ov,
                )

            parts: list[str] = []
            if prev_overlap:
                parts.append(f"...{prev_overlap}")
            parts.append(core_texts[i])
            if next_overlap:
                parts.append(f"{next_overlap}...")

            ch.raw_text = "\n".join(parts)
            ch.roles = extract_roles(ch.raw_text)
            ch.embed_text = self._build_embed_text(meta, ch)

        # 4. Связываем prev_id и next_id между соседними чанками внутри одной единицы (unit_id)
        for i, ch in enumerate(chunks):
            if ch.zone == "meta":
                continue
            if (
                i > 0
                and chunks[i - 1].zone != "meta"
                and chunks[i - 1].unit_id == ch.unit_id
            ):
                ch.prev_id = chunks[i - 1].chunk_id
            if (
                i + 1 < len(chunks)
                and chunks[i + 1].zone != "meta"
                and chunks[i + 1].unit_id == ch.unit_id
            ):
                ch.next_id = chunks[i + 1].chunk_id

        return chunks, active_sections
