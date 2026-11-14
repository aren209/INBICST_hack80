"""Isolated unit & integration tests for v9 discrepancy_alerts, clean Russian alerts, and dynamic corpus OOS anchors."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.chunking.glossary import GlossaryManager
from src.config import OUT_OF_SCOPE_LOW_DENSE_THRESHOLD
from src.indexing.embedder import tokenize_bm25
from src.indexing.table_store import StructuredTableStore
from src.retrieval.query_pipeline import QueryPipeline, RewrittenQueryPlan
from src.retrieval.search_service import SearchService
from src.schemas import (
    ChunkRerankVerdict,
    DocumentMeta,
    DocumentStatus,
    ParentSectionRecord,
    SearchHit,
    TableColumnSchema,
    TablePassport,
)

FORBIDDEN_ALERT_JARGON = (
    "superseded_template_reference",
    "superseded",
    "active",
    "uploads/",
    "doc_id",
    "chunk_id",
    "OCR",
    "ВНИМАНИЕ (",
    "cached_value",
    "[формула:",
)


def _assert_no_alert_jargon(alerts: list[str]) -> None:
    for msg in alerts:
        for bad in FORBIDDEN_ALERT_JARGON:
            assert bad not in msg, f"Forbidden jargon {bad!r} leaked into alert: {msg!r}"


def _make_fake_pipeline(
    registry: dict[str, DocumentMeta] | None = None,
    sections_store: dict[str, ParentSectionRecord] | None = None,
    table_store: StructuredTableStore | None = None,
    raw_points: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    reg = dict(registry or {})
    secs = dict(sections_store or {})
    pts = list(raw_points or [])

    embedder = MagicMock()
    embedder.embed_query.return_value = ([0.1] * 8, {}, ["тест"])
    embedder.dense.embed_texts.return_value = [[0.1] * 8]
    embedder.sparse.encode_with_tokens.return_value = ({}, ["тест"])

    qdrant = MagicMock()
    qdrant.hybrid_search.side_effect = lambda **kw: [
        p
        for p in pts
        if not kw.get("doc_id_filter")
        or (
            p["payload"]["doc_id"] in kw["doc_id_filter"]
            if isinstance(kw["doc_id_filter"], list)
            else p["payload"]["doc_id"] == kw["doc_id_filter"]
        )
    ]
    qdrant.get_document_chunks.return_value = []

    pipeline = SimpleNamespace(
        registry=reg,
        sections_store=secs,
        table_store=table_store,
        embedder=embedder,
        qdrant=qdrant,
        get_document=lambda doc_id: reg.get(doc_id),
        get_parent_section=lambda sid: secs.get(sid),
        _load_registry=lambda: reg,
        _load_sections_store=lambda: secs,
        _has_active_table_store=lambda: table_store is not None,
        _find_matching_docs=lambda ref: [
            d
            for d in reg.values()
            if ref and (d.doc_id == ref or d.doc_number == ref)
        ],
    )
    return pipeline


def _make_point(
    chunk_id: str,
    doc: DocumentMeta,
    clean_text: str,
    dense_score: float = 0.78,
    rrf_score: float = 0.72,
    clause_no: str = "1.1",
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "point_uuid": f"pt_{chunk_id}",
        "dense_score": dense_score,
        "sparse_score": 0.45,
        "rrf_score": rrf_score,
        "payload": {
            "chunk_id": chunk_id,
            "doc_id": doc.doc_id,
            "file_name": doc.file_name,
            "doc_number": doc.doc_number,
            "doc_date": doc.doc_date,
            "doc_type": doc.doc_type,
            "title": doc.title,
            "status": doc.status,
            "section_id": f"{doc.doc_id}::u01::s01",
            "unit_id": "u01",
            "unit_title": "Основной текст",
            "section_title": doc.title,
            "clause_no": clause_no,
            "clean_text": clean_text,
            "raw_text": clean_text,
            "embed_text": clean_text,
            "page_numbers": [1],
            "version_note": doc.version_note,
            "quality_warning": doc.quality_warning,
            "supersedes": list(doc.supersedes),
            "superseded_by": list(doc.superseded_by),
            "amends": list(doc.amends),
            "amended_by": list(doc.amended_by),
            "metadata": dict(doc.metadata or {}),
        },
    }


def _make_glossary(tmp_path: Path) -> GlossaryManager:
    g_path = tmp_path / "glossary.yaml"
    g_path.write_text("synonyms: []\n", encoding="utf-8")
    return GlossaryManager(glossary_path=g_path)


class TestDiscrepancyAlertsAndJargonCleanup:
    def test_unlinked_active_siblings_populate_discrepancy_alerts_not_version_alerts(
        self, tmp_path: Path
    ) -> None:
        doc1 = DocumentMeta(
            doc_id="doc_dorm_a",
            file_name="Положение_о_поселении_1.pdf",
            file_path="uploads/Положение_о_поселении_1.pdf",
            doc_number="101-1",
            doc_date="15.02.2025",
            title="Положение о поселении сотрудников в общежития",
            status=DocumentStatus.ACTIVE.value,
            metadata={"version_family": "normative::поселение сотрудников в общежития"},
        )
        doc2 = DocumentMeta(
            doc_id="doc_dorm_b",
            file_name="Регламент_поселения_в_общежития_2.docx",
            file_path="uploads/Регламент_поселения_в_общежития_2.docx",
            doc_number="202-1",
            doc_date="20.05.2025",
            title="Регламент поселения сотрудников в общежития",
            status=DocumentStatus.ACTIVE.value,
            metadata={"version_family": "normative::поселение сотрудников в общежития"},
        )
        pts = [
            _make_point("c1", doc1, "Заявка на поселение сотрудников подается заблаговременно."),
            _make_point("c2", doc2, "Оформление документов на поселение сотрудников в общежития."),
        ]
        pipeline = _make_fake_pipeline(registry={doc1.doc_id: doc1, doc2.doc_id: doc2}, raw_points=pts)
        svc = SearchService(pipeline=pipeline, glossary=_make_glossary(tmp_path))

        resp = svc.search("порядок поселения сотрудников в общежития", top_k=5)
        assert not resp.is_out_of_scope
        assert len(resp.discrepancy_alerts) >= 1
        joined_disc = " ".join(resp.discrepancy_alerts)
        assert "101-1" in joined_disc and "202-1" in joined_disc
        assert "действует" in joined_disc
        assert not any("101-1" in va and "202-1" in va for va in resp.version_alerts)
        assert resp.version_alerts == []
        _assert_no_alert_jargon(resp.version_alerts + resp.discrepancy_alerts + resp.quality_alerts)

    def test_superseded_template_reference_alongside_active_doc_emits_discrepancy_alert(
        self, tmp_path: Path
    ) -> None:
        doc_act = DocumentMeta(
            doc_id="order_794_1",
            file_name="Приказ_794_1.docx",
            doc_number="794-1",
            doc_date="28.03.2023",
            title="Приказ о помесячной смете расходов",
            status=DocumentStatus.ACTIVE.value,
            supersedes=["app_smeta_2019"],
            version_note="Приказ № 794-1 действует и вводит помесячную форму сметы.",
        )
        doc_ref = DocumentMeta(
            doc_id="app_smeta_2019",
            file_name="Смета_2019.xlsx",
            doc_date="11.10.2019",
            title="Приложение 1 смета расходов 2019 с формулами",
            status=DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value,
            superseded_by=["order_794_1"],
            version_note=(
                "ВНИМАНИЕ (Смена редакции): Бланк сметы 2019 заменен на app_smeta_2023 "
                "(superseded_template_reference). Заменен документом № order_794_1."
            ),
            quality_warning=(
                "Книга Excel сохранена без пересчета кэшированных значений (cached_value=None); "
                "извлечены формулы [формула: =SUM(B2:B5)] и OCR текст."
            ),
        )
        pts = [
            _make_point("c_act", doc_act, "Форма сметы расходов утверждена приказом и подается по форме 1а."),
            _make_point(
                "c_ref",
                doc_ref,
                "Расчетные формулы сметы расходов и коды КОСГУ накладных затрат.",
                dense_score=0.72,
                rrf_score=0.65,
            ),
        ]
        pipeline = _make_fake_pipeline(registry={doc_act.doc_id: doc_act, doc_ref.doc_id: doc_ref}, raw_points=pts)
        svc = SearchService(pipeline=pipeline, glossary=_make_glossary(tmp_path))

        resp = svc.search("формулы и форма сметы расходов", top_k=5)
        assert not resp.is_out_of_scope
        assert len(resp.discrepancy_alerts) >= 1
        disc_text = " ".join(resp.discrepancy_alerts)
        assert "справочный шаблон" in disc_text
        assert "действует" in disc_text
        assert "794-1" in disc_text
        assert len(resp.quality_alerts) == 1
        assert "скан-копия документа" in resp.quality_alerts[0]
        _assert_no_alert_jargon(resp.version_alerts + resp.discrepancy_alerts + resp.quality_alerts)

    def test_conflicting_numeric_deadlines_percentages_and_limits_between_active_docs(
        self, tmp_path: Path
    ) -> None:
        doc_a = DocumentMeta(
            doc_id="doc_trip_a",
            file_name="Регламент_командировок_А.pdf",
            file_path="uploads/Регламент_командировок_А.pdf",
            doc_number="310-1",
            title="Регламент служебных командировок",
            status=DocumentStatus.ACTIVE.value,
        )
        doc_b = DocumentMeta(
            doc_id="doc_trip_b",
            file_name="Приказ_командировок_Б.docx",
            file_path="uploads/Приказ_командировок_Б.docx",
            doc_number="420-1",
            title="Приказ о порядке отчета по командировкам",
            status=DocumentStatus.ACTIVE.value,
        )
        pts = [
            _make_point(
                "ca1",
                doc_a,
                "Авансовый отчет по командировке сдается в течение 5 рабочих дней; суточные составляют 700 руб., компенсация 15%.",
            ),
            _make_point(
                "cb1",
                doc_b,
                "Авансовый отчет по командировке представляется в течение 10 рабочих дней; суточные 1500 руб., компенсация 20%.",
            ),
        ]
        pipeline = _make_fake_pipeline(registry={doc_a.doc_id: doc_a, doc_b.doc_id: doc_b}, raw_points=pts)
        svc = SearchService(pipeline=pipeline, glossary=_make_glossary(tmp_path))

        resp = svc.search("авансовый отчет по командировке суточные", top_k=5)
        assert not resp.is_out_of_scope
        joined = " ".join(resp.discrepancy_alerts)
        assert "5 рабочих дней" in joined
        assert "10 рабочих дней" in joined
        assert "700 руб" in joined and "1500 руб" in joined
        assert "15%" in joined and "20%" in joined
        assert "310-1" in joined and "420-1" in joined
        _assert_no_alert_jargon(resp.version_alerts + resp.discrepancy_alerts + resp.quality_alerts)

    def test_jargon_sanitized_in_alerts_and_chatbot_context(self, tmp_path: Path) -> None:
        doc_a = DocumentMeta(
            doc_id="doc_scan_1",
            file_name="Порядок_архива.pdf",
            file_path="uploads/Порядок_архива.pdf",
            doc_number="55-1",
            title="Порядок архивного хранения",
            status=DocumentStatus.ACTIVE.value,
            version_note="ВНИМАНИЕ (Обновление): Статус active; подробнее в doc_id=doc_scan_1 из `uploads/`.",
            quality_warning="Распознано через OCR, дублируются пункты (разделены по уникальным синтетическим chunk_id).",
        )
        doc_b = DocumentMeta(
            doc_id="doc_scan_2",
            file_name="Инструкция_архива.pdf",
            doc_number="56-1",
            title="Инструкция архивного хранения",
            status=DocumentStatus.SUPERSEDED.value,
            superseded_by=["doc_scan_1"],
        )
        pts = [
            _make_point("c1", doc_a, "Срок архивного хранения составляет 5 рабочих дней."),
            _make_point("c2", doc_b, "Срок архивного хранения составляет 3 рабочих дня."),
        ]
        pipeline = _make_fake_pipeline(registry={doc_a.doc_id: doc_a, doc_b.doc_id: doc_b}, raw_points=pts)
        svc = SearchService(pipeline=pipeline, glossary=_make_glossary(tmp_path))

        ctx = svc.build_chatbot_context("порядок архивного хранения", top_k=5, use_reranker=False)
        assert len(ctx.discrepancy_alerts) >= 1
        assert len(ctx.version_alerts) >= 1
        assert len(ctx.quality_alerts) >= 1
        assert "[РАСХОЖДЕНИЕ]" in ctx.formatted_context
        _assert_no_alert_jargon(ctx.version_alerts + ctx.discrepancy_alerts + ctx.quality_alerts)

    def test_query_pipeline_supports_both_3_tuple_and_legacy_2_tuple_collect_alerts(
        self, tmp_path: Path
    ) -> None:
        h1 = SearchHit(
            chunk_id="d1::c01",
            doc_id="d1",
            section_id="d1::s01",
            file_name="a.pdf",
            title="Регламент А",
            clean_text="Срок подачи заявки 5 рабочих дней.",
            score=0.85,
        )
        # 1) 3-tuple return from _collect_alerts
        svc3 = MagicMock()
        svc3.get_parent_section.return_value = None
        svc3.get_unit_parent_sections.return_value = []
        svc3._collect_alerts.side_effect = lambda hits, amd: (["v1"], ["d1"], ["q1"])
        qp3 = QueryPipeline(search_service=svc3)
        plan = RewrittenQueryPlan(raw_query="срок подачи заявки", rewritten_primary="срок подачи заявки")
        resp3 = qp3._assemble_verified_search_response(
            raw_query="срок подачи заявки",
            plan=plan,
            kept_hits=[h1],
            candidate_tables={},
            max_rerank_score=0.9,
            total_candidates_count=1,
        )
        assert resp3.version_alerts == ["v1"]
        assert resp3.discrepancy_alerts == ["d1"]
        assert resp3.quality_alerts == ["q1"]

        # 2) Legacy 2-tuple mock compatibility
        svc2 = MagicMock()
        svc2.get_parent_section.return_value = None
        svc2.get_unit_parent_sections.return_value = []
        svc2._collect_alerts.side_effect = lambda hits, amd: (["v_legacy"], ["q_legacy"])
        qp2 = QueryPipeline(search_service=svc2)
        resp2 = qp2._assemble_verified_search_response(
            raw_query="срок подачи заявки",
            plan=plan,
            kept_hits=[h1],
            candidate_tables={},
            max_rerank_score=0.9,
            total_candidates_count=1,
        )
        assert resp2.version_alerts == ["v_legacy"]
        assert resp2.discrepancy_alerts == []
        assert resp2.quality_alerts == ["q_legacy"]


class TestDynamicCorpusAnchorsAndOOS:
    def test_dynamic_anchors_cache_invalidation_and_safe_pipeline_none(self, tmp_path: Path) -> None:
        svc_none = SearchService.__new__(SearchService)
        svc_none.pipeline = None
        svc_none.glossary = _make_glossary(tmp_path)
        svc_none._table_store = None
        assert svc_none._get_dynamic_corpus_anchors() == set()

        svc_mock = SearchService.__new__(SearchService)
        svc_mock.pipeline = MagicMock()
        svc_mock.glossary = _make_glossary(tmp_path)
        svc_mock._table_store = None
        assert svc_mock._get_dynamic_corpus_anchors() == set()

        db_path = tmp_path / "tables.db"
        ts = StructuredTableStore(db_path=db_path)
        list_calls = 0
        orig_list_tables = ts.list_tables

        def _counting_list_tables(doc_id: str | None = None) -> list[TablePassport]:
            nonlocal list_calls
            list_calls += 1
            return orig_list_tables(doc_id=doc_id)

        ts.list_tables = _counting_list_tables  # type: ignore[method-assign]

        doc1 = DocumentMeta(
            doc_id="doc_1",
            file_name="Регламент_осаго.pdf",
            doc_number="912-1",
            title="Приказ и Положение о федеральном государственном регламенте автопарка",
            indexed_at="2026-11-14T10:00:00Z",
        )
        pipeline = _make_fake_pipeline(registry={doc1.doc_id: doc1}, table_store=ts)
        svc = SearchService(pipeline=pipeline, glossary=_make_glossary(tmp_path), table_store=ts)

        anchors1 = svc._get_dynamic_corpus_anchors()
        assert "автопарк" in anchors1
        assert "912-1" in anchors1
        # Generic admin words must NOT be added as anchors
        for admin_word in ("приказ", "положение", "регламент", "федерального", "государственного", "таблица"):
            assert admin_word not in anchors1
        assert list_calls == 1

        # Second call with unchanged registry & table count hits cache (no list_tables rebuild)
        anchors2 = svc._get_dynamic_corpus_anchors()
        assert anchors2 == anchors1
        assert list_calls == 1

        # Updating registry doc title/indexed_at invalidates cache
        pipeline.registry["doc_1"] = DocumentMeta(
            doc_id="doc_1",
            file_name="Регламент_осаго.pdf",
            doc_number="912-1",
            title="Положение о страховании ОСАГО автопарка",
            indexed_at="2026-11-14T11:00:00Z",
        )
        anchors3 = svc._get_dynamic_corpus_anchors()
        assert "осаго" in anchors3
        assert list_calls == 2

        # Adding a table to table_store changes count and invalidates cache
        ts.upsert_table(
            table_id="doc_1::tbl_01",
            doc_id="doc_1",
            headers=["Госномер", "Страховая премия"],
            rows=[],
            sheet_name="Автопарк",
            table_title="Тарифы полисов телеметрии",
        )
        anchors4 = svc._get_dynamic_corpus_anchors()
        assert "телеметрия" in anchors4 or "телеметрии" in anchors4
        assert "госномер" in anchors4
        assert list_calls == 3

    def test_oos_detector_allows_newly_registered_osago_and_sberbank_docs(self, tmp_path: Path) -> None:
        doc_osago = DocumentMeta(
            doc_id="doc_osago",
            file_name="Политика_ОСАГО_автопарка.docx",
            doc_number="845-1",
            title="Регламент обязательного страхования ОСАГО автопарка",
        )
        doc_sber = DocumentMeta(
            doc_id="doc_sber_cards",
            file_name="Зарплатные_карты_Сбербанка.docx",
            doc_number="846-1",
            title="Порядок выпуска зарплатных карт Сбербанка",
        )
        pipeline = _make_fake_pipeline(registry={doc_osago.doc_id: doc_osago, doc_sber.doc_id: doc_sber})
        svc = SearchService(pipeline=pipeline, glossary=_make_glossary(tmp_path))

        # 1) OOS stem "осаго" IS in dynamic_corpus_anchors -> allowed even below low dense threshold due to multi-token anchor
        hit_osago = SearchHit(
            chunk_id="doc_osago::c01",
            doc_id="doc_osago",
            title=doc_osago.title,
            doc_number="845-1",
            clean_text="Оформление полиса ОСАГО для служебного автопарка.",
            dense_score=OUT_OF_SCOPE_LOW_DENSE_THRESHOLD - 0.03,
        )
        q1 = "Какие правила оформления ОСАГО автопарка?"
        is_oos1, _, _, _ = svc._detect_out_of_scope(q1, tokenize_bm25(q1), [hit_osago])
        assert is_oos1 is False

        # 2) OOS stem "сбербанк" IS in dynamic_corpus_anchors -> зарплата/карта Сбербанка allowed
        hit_sber = SearchHit(
            chunk_id="doc_sber_cards::c01",
            doc_id="doc_sber_cards",
            title=doc_sber.title,
            doc_number="846-1",
            clean_text="Выпуск зарплатной карты Сбербанка для новых сотрудников.",
            dense_score=0.52,
        )
        q2 = "Как оформить зарплатную карту Сбербанка?"
        is_oos2, _, _, _ = svc._detect_out_of_scope(q2, tokenize_bm25(q2), [hit_sber])
        assert is_oos2 is False

        # 3) Classic unrelated stem "ипотек" on the SAME corpus still triggers OOS ("ипотека в Сбербанке")
        q3 = "ипотека в Сбербанке"
        is_oos3, _, _, _ = svc._detect_out_of_scope(q3, tokenize_bm25(q3), [hit_sber])
        assert is_oos3 is True

        # 4) Rule 3 safety: single non-doc_number dynamic token with low dense & zero lexical overlap is still rejected
        hit_weak = SearchHit(
            chunk_id="doc_osago::c99",
            doc_id="doc_osago",
            title="Общие сведения",
            clean_text="Разные организационные вопросы.",
            dense_score=0.30,
        )
        q4 = "автопарка"
        is_oos4, _, _, _ = svc._detect_out_of_scope(q4, tokenize_bm25(q4), [hit_weak])
        assert is_oos4 is True

        # 5) Exact doc_number anchor overrides Rule 3 low dense
        q5 = "Поясните документ 845-1"
        is_oos5, _, _, _ = svc._detect_out_of_scope(q5, tokenize_bm25(q5), [hit_weak])
        assert is_oos5 is False

    def test_classic_off_topic_queries_remain_out_of_scope_on_mipt_corpus(self, tmp_path: Path) -> None:
        # Build isolated copy of current MIPT document registry metadata in tmp_path
        live_reg_path = Path(__file__).resolve().parents[1] / "data" / "document_registry.json"
        reg_raw = json.loads(live_reg_path.read_text(encoding="utf-8"))
        docs_raw = reg_raw.get("documents", reg_raw) if isinstance(reg_raw, dict) else {}
        reg: dict[str, DocumentMeta] = {}
        for d_id, d_val in docs_raw.items():
            reg[d_id] = DocumentMeta.model_validate(d_val)

        pipeline = _make_fake_pipeline(registry=reg)
        svc = SearchService(pipeline=pipeline, glossary=_make_glossary(tmp_path))

        generic_hit = SearchHit(
            chunk_id="reg_business_trips::c01",
            doc_id="reg_business_trips",
            title="Регламент о командировании",
            clean_text="Оформление служебной командировки работников МФТИ и сдача авансового отчета.",
            dense_score=0.49,
        )
        for oos_query in (
            "рецепт пасты карбонара",
            "борщ",
            "ипотека в Сбербанке",
            "курс биткоина",
        ):
            is_oos, _, _, _ = svc._detect_out_of_scope(oos_query, tokenize_bm25(oos_query), [generic_hit])
            assert is_oos is True, f"Expected is_out_of_scope=True for {oos_query!r}"
