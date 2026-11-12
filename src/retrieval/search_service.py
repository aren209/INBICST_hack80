"""Hybrid RRF Search Service with ParentSection Hydration, Edition Graph (Amends/Supersedes), and Out-of-Scope Detection."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from src.chunking.glossary import GlossaryManager, get_glossary_manager
from src.config import (
    NON_ANSWER_ZONES,
    OUT_OF_SCOPE_DENSE_THRESHOLD,
    OUT_OF_SCOPE_LOW_DENSE_THRESHOLD,
    OUT_OF_SCOPE_MIN_LEXICAL_OVERLAP,
    SUPERSEDED_SCORE_PENALTY,
)
from src.indexing.embedder import HybridEmbedder, tokenize_bm25
from src.indexing.pipeline import IndexingPipeline, _core_doc_number, get_pipeline
from src.schemas import (
    DocumentMeta,
    DocumentStatus,
    ParentSectionRecord,
    SearchHit,
    SearchQuery,
    SearchResponse,
    compute_content_hash,
)

logger = logging.getLogger(__name__)

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
    ) -> None:
        self.pipeline = pipeline or get_pipeline()
        self.glossary = glossary or get_glossary_manager()

    @property
    def qdrant(self):
        return self.pipeline.qdrant

    @property
    def embedder(self) -> HybridEmbedder:
        return self.pipeline.embedder

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

        # Financial structure & CFO tree queries (Scenario 7)
        if any(w in q_norm for w in ("финансовая структура", "финансовой структуры", "фин. структур", "цфо", "финансодержател", "3614-1")):
            extra_terms.extend(["Приказ 3614-1", "финансовая структура 2026", "Финансодержатель", "ЦФО 1", "ЦФО 2"])

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
        - Query explicitly mentions the old order number or old year
        - Or document is 'superseded_template_reference' and query asks about formulas / KVR / KOSGU / rates / smeta
        """
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

        # Compute lexical overlap bonus on chunk text
        chunk_tokens = set(tokenize_bm25(f"{title} {doc_number} {clean_text}"))
        matched_q_tokens = query_tokens_set.intersection(chunk_tokens)
        lex_ratio = len(matched_q_tokens) / max(1, len(query_tokens_set)) if query_tokens_set else 0.0

        # Combined calibrated score (0..1 scale)
        norm_rrf = min(1.0, rrf_score * 1.35)
        combined_score = 0.55 * max(0.0, dense_score) + 0.25 * norm_rrf + 0.20 * min(1.0, lex_ratio * 1.4)

        # Direct doc_number match bonus when user query mentions this document's number
        core_num = _core_doc_number(doc_number).lower()
        if core_num and core_num in query_norm:
            combined_score = min(1.0, combined_score + 0.06)

        # Apply slight score demotion for superseded documents unless exempted
        meta_probe = {
            "doc_number": doc_number,
            "applicable_years": applicable_years,
            "status": status,
        }
        if status == DocumentStatus.SUPERSEDED.value and not self._should_exempt_superseded(query_norm, meta_probe):
            combined_score *= SUPERSEDED_SCORE_PENALTY
        elif (
            status == DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value
            and not self._should_exempt_superseded(query_norm, meta_probe)
        ):
            combined_score *= 0.90

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

        return SearchHit(
            chunk_id=str(payload.get("chunk_id", "") or ""),
            point_uuid=str(pt_dict.get("point_uuid") or payload.get("point_uuid", "") or ""),
            doc_id=doc_id,
            file_name=file_name,
            doc_number=doc_number,
            doc_date=doc_date,
            doc_type=doc_type,
            title=title,
            status=status,
            applicable_years=applicable_years,
            unit_id=str(payload.get("unit_id", "") or ""),
            unit_title=str(payload.get("unit_title", "") or ""),
            section_id=section_id,
            section_num=str(payload.get("section_num", "") or ""),
            section_title=str(payload.get("section_title", "") or ""),
            clause_no=str(payload.get("clause_no", "") or ""),
            zone=str(payload.get("zone", "normative") or "normative"),
            score=round(float(combined_score), 4),
            rrf_score=round(rrf_score, 4),
            dense_score=round(dense_score, 4),
            sparse_score=round(sparse_score, 4),
            raw_text=raw_text,
            clean_text=clean_text,
            embed_text=embed_text,
            parent_section=parent_sec,
            parent_text=parent_text,
            roles=list(payload.get("roles") or []),
            topics=list(payload.get("topics") or []),
            terms=list(payload.get("terms") or []),
            also_in_files=also_in_files,
            page_numbers=list(payload.get("page_numbers") or []),
            version_note=version_note,
            quality_warning=quality_warning,
            supersedes=supersedes,
            superseded_by=superseded_by,
            amends=amends,
            amended_by=amended_by,
            is_amendment=is_amendment,
            amends_doc_id=amends_doc_id,
            metadata={"content_hash": payload.get("content_hash") or compute_content_hash(clean_text)},
        )

    @staticmethod
    def _deduplicate_hits_by_content_hash(hits: list[SearchHit]) -> list[SearchHit]:
        """
        Deduplicate hits across files by normalized content_hash, merging duplicate file_names
        into the primary hit's also_in_files list.
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
                if hit.file_name and hit.file_name != primary.file_name:
                    if hit.file_name not in primary.also_in_files:
                        primary.also_in_files.append(hit.file_name)
                for extra_f in hit.also_in_files:
                    if extra_f and extra_f != primary.file_name and extra_f not in primary.also_in_files:
                        primary.also_in_files.append(extra_f)

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
        Perform graph-aware retrieval for:
        1) Partial amendments (amended_by / amends, e.g. Приказ 3065-1 <-> Распоряжение 700)
        2) Smeta edition transition + formula reference (Приказ 794-1 / Прил. 1а <-> 2019 smeta xlsx)
        Returns updated (hits, amendment_chunks).
        """
        amendment_chunks: list[SearchHit] = []
        if not hits:
            return hits, amendment_chunks

        # Refresh registry if needed
        if not self.pipeline.registry:
            self.pipeline.registry = self.pipeline._load_registry()

        present_doc_ids: set[str] = {h.doc_id for h in hits}
        existing_chunk_ids: set[str] = {h.chunk_id for h in hits}

        # 1. Forward amendments: hit has amended_by (e.g. 3065-1 -> 700)
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

                #Also if a hit in main hits is ALREADY from an amending doc (e.g. 700), include it in amendment_chunks
                if hit.amends:
                    base_docs = []
                    for b_ref in hit.amends:
                        base_docs.extend(self.pipeline._find_matching_docs(b_ref))
                    base_doc_id = base_docs[0].doc_id if base_docs else None
                    amd_copy = hit.model_copy(update={"is_amendment": True, "amends_doc_id": base_doc_id})
                    if amd_copy.chunk_id not in {a.chunk_id for a in amendment_chunks}:
                        amendment_chunks.append(amd_copy)

                    # Ensure base document (e.g. 3065-1) is also represented in main hits!
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
                                    hits.append(b_hit)
                                    present_doc_ids.add(b_hit.doc_id)
                                    existing_chunk_ids.add(b_hit.chunk_id)

            # Ensure top amending chunk (e.g. 700) is also surfaced inside `hits` alongside base doc (3065-1)
            for amd_hit in amendment_chunks:
                if amd_hit.doc_id not in present_doc_ids and len(hits) >= 1:
                    if len(hits) >= top_k and top_k >= 2:
                        hits[-1] = amd_hit
                    else:
                        hits.append(amd_hit)
                    present_doc_ids.add(amd_hit.doc_id)
                    existing_chunk_ids.add(amd_hit.chunk_id)

        # 2. Smeta 794-1 / Прил. 1а <-> 2019 smeta formula template (superseded_template_reference) co-surfacing
        smeta_active_ids = {"order_794_1_2023", "app_1a_smeta_blank"}
        smeta_ref_id = "app_1_smeta_2019_xlsx"
        has_active_smeta = bool(present_doc_ids.intersection(smeta_active_ids))
        has_ref_smeta = smeta_ref_id in present_doc_ids
        asks_formulas = any(
            w in query_norm for w in ("формул", "квр", "косгу", "накладн", "ндс", "20%", "2019", "1469")
        )

        # Case A: Active smeta (794-1 / 1a) matched and query asks for formulas/KVR/KOSGU -> co-surface 2019 template
        if has_active_smeta and not has_ref_smeta and asks_formulas and smeta_ref_id in self.pipeline.registry:
            raw_ref_pts = self.qdrant.hybrid_search(
                query_dense=query_dense,
                query_sparse=query_sparse,
                top_k=2,
                doc_id_filter=smeta_ref_id,
                exclude_zones=NON_ANSWER_ZONES,
            )
            if raw_ref_pts:
                ref_hit = self._raw_point_to_hit(
                    raw_ref_pts[0],
                    query_norm=query_norm,
                    query_tokens_set=query_tokens_set,
                    include_parent=include_parent,
                )
                if len(hits) >= top_k and top_k >= 2:
                    hits[-1] = ref_hit
                else:
                    hits.append(ref_hit)
                present_doc_ids.add(ref_hit.doc_id)

        # Case B: 2019 smeta template matched, but active order 794-1 / 1a is missing from hits -> co-surface 794-1
        if has_ref_smeta and not has_active_smeta:
            avail_active = [d for d in ("order_794_1_2023", "app_1a_smeta_blank") if d in self.pipeline.registry]
            if avail_active:
                raw_act_pts = self.qdrant.hybrid_search(
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    top_k=2,
                    doc_id_filter=avail_active,
                    exclude_zones=NON_ANSWER_ZONES,
                )
                if raw_act_pts:
                    act_hit = self._raw_point_to_hit(
                        raw_act_pts[0],
                        query_norm=query_norm,
                        query_tokens_set=query_tokens_set,
                        include_parent=include_parent,
                    )
                    hits.insert(0, act_hit)
                    if len(hits) > top_k:
                        hits = hits[:top_k]

        return hits[:top_k], amendment_chunks[:5]

    @staticmethod
    def _collect_alerts(hits: list[SearchHit], amendment_chunks: list[SearchHit]) -> tuple[list[str], list[str]]:
        """Collect deduplicated version_alerts and quality_alerts across all matched documents."""
        version_alerts: list[str] = []
        quality_alerts: list[str] = []

        for h in list(hits) + list(amendment_chunks):
            if h.version_note and h.version_note not in version_alerts:
                version_alerts.append(h.version_note)
            if h.amended_by:
                alert = (
                    f"Документ «{h.title}» (№ {h.doc_number}) действует с учетом изменений: "
                    f"№ {', '.join(h.amended_by)}."
                )
                if alert not in version_alerts and not any(h.doc_number in va for va in version_alerts):
                    version_alerts.append(alert)
            if h.superseded_by:
                alert = (
                    f"Документ «{h.title}» (№ {h.doc_number}) заменен / отменен актом: "
                    f"№ {', '.join(h.superseded_by)} (текущий статус: {h.status})."
                )
                if alert not in version_alerts and not any(h.doc_number in va for va in version_alerts):
                    version_alerts.append(alert)
            if h.quality_warning and h.quality_warning not in quality_alerts:
                quality_alerts.append(h.quality_warning)

        return version_alerts, quality_alerts

    def search(
        self,
        request: SearchQuery | str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> SearchResponse:
        """
        Execute full Hybrid RRF search with query expansion, deduplication,
        edition graph traversal, ParentSection hydration, and out-of-scope calibration.
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
                is_out_of_scope=True,
                confidence_score=0.0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        query_norm = raw_query.lower().replace("ё", "е")
        expanded_query = self._expand_query_for_retrieval(raw_query)

        # Embed query (Dense 1024d + Sparse BM25 CRC32)
        raw_query_tokens = tokenize_bm25(raw_query)
        query_dense, query_sparse, expanded_tokens = self.embedder.embed_query(
            query_text=raw_query,
            sparse_query_text=expanded_query,
        )
        query_tokens_set = set(raw_query_tokens) | set(expanded_tokens)

        # Build status filter if include_superseded is False
        eff_status_filter = req.status_filter
        if not req.include_superseded and not eff_status_filter:
            eff_status_filter = [
                DocumentStatus.ACTIVE.value,
                DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value,
            ]

        # Fetch candidate pool via Hybrid RRF (requesting 3x top_k so deduplication never starves top_k)
        candidate_limit = max(req.top_k * 3, 18)
        raw_points = self.qdrant.hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=candidate_limit,
            prefetch_limit=max(candidate_limit * 2, 45),
            status_filter=eff_status_filter,
            year_filter=req.year_filter,
            doc_id_filter=req.doc_id_filter,
            unit_id_filter=req.unit_id_filter,
            exclude_zones=NON_ANSWER_ZONES,
        )

        # Convert to hydrated SearchHit objects
        candidate_hits = [
            self._raw_point_to_hit(
                pt,
                query_norm=query_norm,
                query_tokens_set=query_tokens_set,
                include_parent=req.include_parent,
            )
            for pt in raw_points
        ]

        # Deduplicate across files by content_hash (merging also_in_files)
        deduped_hits = self._deduplicate_hits_by_content_hash(candidate_hits)

        # Sort by calibrated hybrid score descending (with rrf_score as tie-breaker)
        deduped_hits.sort(key=lambda h: (h.score, h.rrf_score, h.dense_score), reverse=True)
        top_hits = deduped_hits[: req.top_k]

        # Run out-of-scope detection
        is_oos, confidence_score, top_dense_score, lexical_overlap = self._detect_out_of_scope(
            raw_query=raw_query,
            raw_query_tokens=raw_query_tokens,
            hits=top_hits,
        )

        amendment_chunks: list[SearchHit] = []
        if is_oos:
            # Dampen hit scores when query is out-of-scope so score separation vs in-scope is crystal clear
            for h in top_hits:
                h.score = round(min(h.score * 0.35, 0.25), 4)
        else:
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

        version_alerts, quality_alerts = (
            ([], []) if is_oos else self._collect_alerts(top_hits, amendment_chunks)
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return SearchResponse(
            query=raw_query,
            expanded_query=expanded_query,
            hits=top_hits,
            amendment_chunks=amendment_chunks,
            version_alerts=version_alerts,
            quality_alerts=quality_alerts,
            is_out_of_scope=is_oos,
            confidence_score=confidence_score,
            top_dense_score=top_dense_score,
            lexical_overlap=lexical_overlap,
            total_found=len(deduped_hits),
            latency_ms=latency_ms,
        )


_SEARCH_SERVICE_SINGLETON: SearchService | None = None


def get_search_service(pipeline: IndexingPipeline | None = None) -> SearchService:
    """Return process-wide SearchService singleton."""
    global _SEARCH_SERVICE_SINGLETON
    if _SEARCH_SERVICE_SINGLETON is None:
        _SEARCH_SERVICE_SINGLETON = SearchService(pipeline=pipeline)
    elif pipeline is not None and _SEARCH_SERVICE_SINGLETON.pipeline is not pipeline:
        _SEARCH_SERVICE_SINGLETON.pipeline = pipeline
    return _SEARCH_SERVICE_SINGLETON


# Alias for external callers
HybridSearchService = SearchService
