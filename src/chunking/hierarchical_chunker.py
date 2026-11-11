from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.chunking.glossary import GlossaryManager
from src.config import MAX_CLAUSE_CHARS
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
        r"|Порядок\s+[а-яА-ЯёЁ\s,-]{4,90}"
        r"|Правила\s+[а-яА-ЯёЁ\s,-]{4,90}"
        r"|Требования\s+к\s+[а-яА-ЯёЁ\s,-]{4,90}"
        r"|Ответственность(?:\s+сторон|\s+за\s+нарушение)?"
        r"|Заключительные\s+положения"
        r"|Организация\s+[а-яА-ЯёЁ\s,-]{4,90}"
        r"|Регламент\s+работы\s+[а-яА-ЯёЁ\s,-]{4,90}"
        r"|Сроки\s+и\s+условия\s+[а-яА-ЯёЁ\s,-]{4,90}"
        r"|Оформление\s+[а-яА-ЯёЁ\s,-]{4,90}"
        r"|Финансовое\s+обеспечение\s+[а-яА-ЯёЁ\s,-]{4,90}"
        r"|Возмещение\s+расходов\s+[а-яА-ЯёЁ\s,-]{4,90}"
        r")$",
        re.IGNORECASE,
    )

    # Section heading: e.g. "1. Общие положения" or "Раздел 2. Порядок открытия ФЛС"
    SECTION_RE = re.compile(
        r"^(?:Раздел\s+)?(\d{1,2}|[IVX]{1,4})\.?\s+([А-ЯЁ][а-яА-ЯёЁ0-9\s,()-]{3,110})$"
    )

    # Multi-level clause: e.g. "1.1.", "2.3.1", "2.6." (protecting against dates like 25.11.2021)
    MULTI_CLAUSE_RE = re.compile(
        r"^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\.?(?:\s+|$)(.+)$"
    )

    # Single-level numbered clause inside an Order or flat Regulation: e.g. "1. Утвердить...", "17. Установить..."
    SINGLE_CLAUSE_RE = re.compile(r"^(\d{1,2})\.\s+([А-ЯЁ0-9«\"].+)$")

    # Sub-item: "а)", "б)", "1)", "-", "•", "1.1.1."
    SUBITEM_RE = re.compile(r"^(?:[а-яёa-z]\)|\d{1,2}\)|[-•–])\s+.+$")

    DATE_PREFIX_RE = re.compile(r"^\d{2}\.\d{2}\.20\d{2}\b")

    def __init__(self, glossary: GlossaryManager | None = None) -> None:
        self.glossary = glossary or GlossaryManager()

    def _is_section_header(self, block: ParsedBlock) -> tuple[str, str] | None:
        text = block.text.strip()
        if len(text) > 130 or "\n" in text:
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
            if (
                len(sec_title) <= 95
                and not sec_title.endswith(";")
                and first_word not in action_verbs
                and (block.is_bold or "Heading" in block.style_name or len(sec_title) <= 65)
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
        # Learn any new abbreviations from document blocks
        for b in blocks:
            if b.zone != "approval":
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

            # Split overly long clauses (> MAX_CLAUSE_CHARS) along paragraph/subitem lines
            sub_texts: list[str] = []
            if len(full_raw) <= MAX_CLAUSE_CHARS:
                sub_texts.append(full_raw)
            else:
                lines = full_raw.splitlines()
                header_line = lines[0][:160]
                buf: list[str] = []
                buf_len = 0
                for ln in lines:
                    if buf_len + len(ln) > MAX_CLAUSE_CHARS and buf:
                        sub_texts.append("\n".join(buf))
                        buf = [f"(Продолжение п. {curr_clause_no}: {header_line})", ln]
                        buf_len = len(buf[0]) + len(ln)
                    else:
                        buf.append(ln)
                        buf_len += len(ln) + 1
                if buf:
                    sub_texts.append("\n".join(buf))

            for part_idx, raw_part in enumerate(sub_texts):
                clause_seq += 1
                c_id = f"{meta.doc_id}::{unit_id}::s{sec_idx:02d}::c{clause_seq:03d}"
                disp_clause = (
                    curr_clause_no
                    if len(sub_texts) == 1
                    else f"{curr_clause_no} (ч. {part_idx + 1})"
                )
                roles = extract_roles(raw_part)
                breadcrumb = (
                    f"[Документ: {meta.doc_title} (№ {meta.doc_number} от {meta.doc_date or 'б/д'}) | "
                    f"Статус: {meta.status} | Часть: {unit_title} | "
                    f"Раздел {section_no}: {section_title} | Пункт: {disp_clause}"
                )
                if roles:
                    breadcrumb += f" | Роли: {', '.join(roles)}"
                if meta.topic_tags:
                    breadcrumb += f" | Темы: {', '.join(meta.topic_tags)}"
                breadcrumb += "]"

                expanded_body = self.glossary.expand_text(raw_part)
                embed_text = f"{breadcrumb}\n{expanded_body}"

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
                    embed_text=embed_text,
                )
                chunks.append(chunk)

                if curr_zone != "approval":
                    register_section(
                        section_id, unit_id, unit_title, section_no, section_title
                    )
                    parent_sections[section_id].clause_ids.append(c_id)
                    if parent_sections[section_id].text:
                        parent_sections[section_id].text += "\n\n" + raw_part
                    else:
                        parent_sections[section_id].text = raw_part

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

        # Link prev_id and next_id within the same section
        for i, ch in enumerate(chunks):
            if i > 0 and chunks[i - 1].section_id == ch.section_id:
                ch.prev_id = chunks[i - 1].chunk_id
            if i + 1 < len(chunks) and chunks[i + 1].section_id == ch.section_id:
                ch.next_id = chunks[i + 1].chunk_id

        return chunks, list(parent_sections.values())
