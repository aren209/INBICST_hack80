"""Unit and live integration tests for the 4-Stage General-Purpose AI Retrieval Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.config import settings
from src.retrieval.query_pipeline import (
    QueryPipeline,
    is_meta_assistant_query,
)
from src.retrieval.rag_agent import OUT_OF_SCOPE_REFUSAL_RU, TableAwareRAGAgent
from src.schemas import (
    AgentAskRequest,
    DocumentMeta,
    ParentSectionRecord,
    SearchHit,
    SearchResponse,
    TableColumnSchema,
    TablePassport,
)

FORBIDDEN_USER_JARGON = (
    "confidence",
    "score",
    "реранкер",
    "чанки",
    "векторная база",
    "rrf",
    "sqlite",
)


def _make_json_completion(payload: dict[str, Any] | str, model: str = "mock-json-model") -> Any:
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    choice = MagicMock()
    choice.message = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = []
    resp = MagicMock()
    resp.model = model
    resp.choices = [choice]
    return resp


def _make_stream_chunks(tokens: list[str], model: str = "mock-stream-model") -> list[Any]:
    chunks: list[Any] = []
    for tok in tokens:
        delta = MagicMock()
        delta.content = tok
        delta.tool_calls = None
        ch = MagicMock()
        ch.delta = delta
        chunk = MagicMock()
        chunk.model = model
        chunk.choices = [ch]
        chunks.append(chunk)
    return chunks


# -----------------------------------------------------------------------------
# 1. Corpus-Agnostic Static Guard (Zero Document Hardcoding)
# -----------------------------------------------------------------------------


def test_query_pipeline_is_100_percent_corpus_agnostic() -> None:
    src_path = Path(__file__).resolve().parents[1] / "src" / "retrieval" / "query_pipeline.py"
    source = src_path.read_text(encoding="utf-8")
    forbidden_corpus_literals = [
        "3065-1",
        "3699-1",
        "2990-1",
        "5990-1",
        "3897-1",
        "3614-1",
        "794-1",
        "1292-1",
        "order_3065_1",
        "reg_concert_hall_booking",
        "reg_business_trips",
        "ФРКТ",
        "ФПМИ",
        "kmo@mipt.ru",
    ]
    for literal in forbidden_corpus_literals:
        assert literal not in source, f"Hardcoded corpus literal {literal!r} found in query_pipeline.py"


# -----------------------------------------------------------------------------
# 2. Stage 1: AI Query Rewriter & Meta-Assistant Intent Detection
# -----------------------------------------------------------------------------


class TestStage1RewriterAndMetaIntent:
    def test_rewrite_preserves_raw_query_first_and_extracts_sub_queries(self) -> None:
        mock_svc = MagicMock()
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.return_value = _make_json_completion(
            {
                "intent": "document_lookup",
                "rewritten_primary": "порядок поселения работников в общежитие и перечень документов",
                "sub_queries": [
                    "регламент предоставления общежития сотрудникам",
                    "перечень документов анкета согласие для поселения и регистрации",
                ],
                "reasoning": "Разговорный запрос переведен в нормативную терминологию.",
            }
        )

        qp = QueryPipeline(search_service=mock_svc, llm_client=mock_llm)
        raw_q = "как пробить общагу новому преподу № 2990-1 и какие бумажки нести"
        plan = qp.rewrite_query_for_retrieval(raw_q)

        assert plan.query_intent == "document_lookup"
        assert plan.used_fallback is False
        assert plan.search_queries[0] == raw_q
        assert "порядок поселения работников в общежитие и перечень документов" in plan.search_queries
        assert len(plan.sub_queries) == 2

    def test_rewrite_graceful_fallback_when_llm_fails(self) -> None:
        mock_svc = MagicMock()
        mock_svc._expand_query_for_retrieval.side_effect = lambda q: q + " регламент"
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = RuntimeError("Polza gateway timeout")

        qp = QueryPipeline(search_service=mock_svc, llm_client=mock_llm)
        raw_q = "Какой размер суточных и в какой срок сдать отчет?"
        plan = qp.rewrite_query_for_retrieval(raw_q)

        assert plan.used_fallback is True
        assert plan.query_intent == "document_lookup"
        assert plan.search_queries[0] == raw_q
        assert len(plan.sub_queries) >= 1

    @pytest.mark.parametrize(
        "meta_q",
        [
            "Что ты умеешь?",
            "кто ты?",
            "Привет!",
            "Здравствуйте",
            "Чем можешь помочь?",
            "Какие документы есть в базе?",
            "Какие файлы загружены в базу знаний?",
            "Как тобой пользоваться?",
        ],
    )
    def test_is_meta_assistant_query_positive(self, meta_q: str) -> None:
        assert is_meta_assistant_query(meta_q) is True

    @pytest.mark.parametrize(
        "domain_q",
        [
            "Какие документы нужны для поселения в общежитие?",
            "Какие документы требуются в паспортном столе для регистрации?",
            "Привет! За сколько месяцев подавать заявку на концертный зал?",
            "Какая завтра погода?",
            "Как испечь яблочный пирог?",
            "Сколько подразделений подчиняется ФРКТ?",
        ],
    )
    def test_is_meta_assistant_query_negative_for_domain_and_oos(self, domain_q: str) -> None:
        assert is_meta_assistant_query(domain_q) is False


# -----------------------------------------------------------------------------
# 3. Stage 3: AI Reranker, Follow-up Sub-Query & Strict Noise Pruning
# -----------------------------------------------------------------------------


def _build_mock_search_service_for_pipeline() -> MagicMock:
    svc = MagicMock()
    svc.pipeline = MagicMock()
    svc.pipeline.registry = {
        "doc_trips": DocumentMeta(
            doc_id="doc_trips",
            file_name="Регламент о командировании.pdf",
            title="Регламент о командировании",
            status="active",
        ),
        "doc_rates": DocumentMeta(
            doc_id="doc_rates",
            file_name="Приказ о нормативах 2026.docx",
            title="Приказ о нормативах 2026",
            doc_number="3699-1",
            doc_date="2025-08-18",
            status="active",
            amends=[],
            amended_by=["700"],
        ),
        "doc_amend": DocumentMeta(
            doc_id="doc_amend",
            file_name="Распоряжение об изменениях.docx",
            title="Распоряжение об изменениях",
            doc_number="700",
            status="active",
            amends=["3699-1"],
        ),
        "doc_noise": DocumentMeta(
            doc_id="doc_noise",
            file_name="Регламент ЦОД.pdf",
            title="Регламент эксплуатации ЦОД",
            status="active",
        ),
    }

    sec_trips_1 = ParentSectionRecord(
        section_id="doc_trips::u01::s01",
        doc_id="doc_trips",
        unit_id="u01",
        unit_title="Регламент командирования",
        section_num="4",
        section_title="Отчетность по командировке",
        clean_text="4.1. Работник обязан в течение 3 рабочих дней сдать авансовый отчет.",
        char_count=68,
        page_numbers=[5],
    )
    sec_trips_sibling = ParentSectionRecord(
        section_id="doc_trips::u01::s02",
        doc_id="doc_trips",
        unit_id="u01",
        unit_title="Регламент командирования",
        section_num="5",
        section_title="Возмещение расходов",
        clean_text="5.1. Суточные возмещаются за каждый день нахождения в командировке.",
        char_count=67,
        page_numbers=[6],
    )
    sec_rates = ParentSectionRecord(
        section_id="doc_rates::u01::s01",
        doc_id="doc_rates",
        unit_id="u01",
        unit_title="Нормативы расходов",
        section_num="1",
        section_title="Суточные расходы",
        clean_text="Норматив суточных расходов составляет 100 рублей в сутки.",
        char_count=57,
        page_numbers=[2],
    )
    sec_noise = ParentSectionRecord(
        section_id="doc_noise::u01::s01",
        doc_id="doc_noise",
        unit_id="u01",
        unit_title="ЦОД",
        section_num="1",
        section_title="Дежурство",
        clean_text="Дежурная смена ЦОД работает круглосуточно (24 часа в сутки).",
        char_count=60,
        page_numbers=[1],
    )
    sec_map = {
        s.section_id: s for s in (sec_trips_1, sec_trips_sibling, sec_rates, sec_noise)
    }
    svc.get_parent_section.side_effect = lambda sid: sec_map.get(sid)
    svc.get_unit_parent_sections.side_effect = lambda doc_id, unit_id: [
        s for s in sec_map.values() if s.doc_id == doc_id and s.unit_id == unit_id
    ]
    svc._expand_edition_graph.side_effect = lambda hits, include_superseded=True: (
        [
            SearchHit(
                chunk_id="doc_amend::c01",
                doc_id="doc_amend",
                section_id="doc_amend::s01",
                file_name="Распоряжение об изменениях.docx",
                title="Распоряжение об изменениях",
                doc_number="700",
                clean_text="Уточнить порядок применения нормативов.",
                amends_doc_id="doc_rates",
                is_amendment=True,
                score=0.7,
            )
        ]
        if any(h.doc_id == "doc_rates" for h in hits)
        else []
    )
    svc._collect_alerts.side_effect = lambda hits, amd: (
        (["Документ действует с учетом изменений № 700"] if amd else []),
        [],
    )
    svc._hydrate_table_context.return_value = []
    return svc


class TestStage3RerankerFollowupAndPruning:
    def test_followup_round_discovers_missing_aspect_and_prunes_orphan_noise(self) -> None:
        svc = _build_mock_search_service_for_pipeline()

        hit_trips = SearchHit(
            chunk_id="doc_trips::c01",
            doc_id="doc_trips",
            unit_id="u01",
            section_id="doc_trips::u01::s01",
            file_name="Регламент о командировании.pdf",
            title="Регламент о командировании",
            clause_no="4.1",
            page_numbers=[5],
            clean_text="Работник обязан в течение 3 рабочих дней сдать авансовый отчет.",
            score=0.81,
        )
        hit_noise = SearchHit(
            chunk_id="doc_noise::c01",
            doc_id="doc_noise",
            unit_id="u01",
            section_id="doc_noise::u01::s01",
            table_id="doc_noise::tbl_01",
            file_name="Регламент ЦОД.pdf",
            title="Регламент эксплуатации ЦОД",
            clause_no="1.2",
            page_numbers=[1],
            clean_text="Дежурная смена ЦОД работает круглосуточно (24 часа в сутки).",
            score=0.79,
            metadata={"table_delivery_mode": "tool_query"},
        )
        hit_rates_followup = SearchHit(
            chunk_id="doc_rates::c01",
            doc_id="doc_rates",
            unit_id="u01",
            section_id="doc_rates::u01::s01",
            file_name="Приказ о нормативах 2026.docx",
            title="Приказ о нормативах 2026",
            doc_number="3699-1",
            clause_no="1.1",
            page_numbers=[2],
            clean_text="Норматив суточных расходов составляет 100 рублей в сутки.",
            score=0.75,
        )
        noise_table = TablePassport(
            table_id="doc_noise::tbl_01",
            doc_id="doc_noise",
            file_name="Регламент ЦОД.pdf",
            table_title="График смен ЦОД",
            row_count=30,
            col_count=3,
            columns=[TableColumnSchema(name="Смена", inferred_type="text")],
        )

        retrieve_calls: list[dict[str, Any]] = []

        def _fake_retrieve(**kwargs: Any) -> tuple[list[SearchHit], list[TablePassport]]:
            retrieve_calls.append(kwargs)
            if len(retrieve_calls) == 1:
                return [hit_noise, hit_trips], [noise_table]
            assert "doc_trips::c01" in (kwargs.get("exclude_chunk_ids") or set())
            assert "doc_noise::c01" in (kwargs.get("exclude_chunk_ids") or set())
            return [hit_rates_followup], []

        svc.retrieve_candidate_pool.side_effect = _fake_retrieve
        svc._apply_adaptive_table_routing.side_effect = lambda hits, amendment_chunks, is_oos: [
            noise_table for h in hits if h.doc_id == "doc_noise"
        ]

        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = [
            # Stage 1 rewrite
            _make_json_completion(
                {
                    "intent": "document_lookup",
                    "rewritten_primary": "размер суточных при командировках и срок сдачи авансового отчета",
                    "sub_queries": ["норматив суточных расходов", "срок сдачи авансового отчета"],
                }
            ),
            # Stage 3 Round 1: keeps hit_trips, marks hit_noise irrelevant, triggers follow-up for per diem rate!
            _make_json_completion(
                {
                    "items": [
                        {"idx": 0, "chunk_id": "doc_noise::c01", "score": 0.08, "verdict": "irrelevant", "reason": "ЦОД, не командировки"},
                        {"idx": 1, "chunk_id": "doc_trips::c01", "score": 0.91, "verdict": "essential", "reason": "Срок авансового отчета"},
                    ],
                    "all_irrelevant_out_of_scope": False,
                    "needs_followup": True,
                    "missing_aspects": ["конкретный размер суточных в рублях"],
                    "followup_queries": ["норматив суточных расходов в сутки руб"],
                }
            ),
            # Stage 3 Round 2: scores newly discovered hit_rates_followup as essential!
            _make_json_completion(
                {
                    "items": [
                        {"idx": 0, "chunk_id": "doc_rates::c01", "score": 0.95, "verdict": "essential", "reason": "100 рублей суточные"},
                    ],
                    "all_irrelevant_out_of_scope": False,
                    "needs_followup": False,
                    "missing_aspects": [],
                    "followup_queries": [],
                }
            ),
        ]

        qp = QueryPipeline(search_service=svc, llm_client=mock_llm)
        statuses: list[tuple[str, str]] = []
        result = qp.run(
            "Какой размер суточных при командировках и в какой срок нужно сдать авансовый отчет?",
            top_k=5,
            on_status=lambda stage, text: statuses.append((stage, text)),
        )

        # Verify status hooks emitted in exact order including follow-up
        assert [s[0] for s in statuses] == ["rewrite", "rerank", "followup"]
        assert "уточняющий дозапрос" in statuses[2][1].lower()

        resp = result.search_response
        trace = result.trace
        assert resp.is_out_of_scope is False
        assert trace.followup_rounds == 1
        assert trace.followup_candidates_added == 1
        # Ordered by rerank score descending: doc_rates (0.95) -> doc_trips (0.91)
        assert [h.doc_id for h in resp.hits] == ["doc_rates", "doc_trips"]
        assert "doc_noise" in trace.pruned_doc_ids
        assert "doc_noise" not in {h.doc_id for h in resp.hits}
        # Orphan linked_table from doc_noise is strictly pruned
        assert resp.linked_tables == []
        # Governing amendment & version alert for kept doc_rates are preserved
        assert [a.doc_id for a in resp.amendment_chunks] == ["doc_amend"]
        assert len(resp.version_alerts) == 1
        # Compact-unit sibling section doc_trips::u01::s02 (total unit <= 20,000 chars) is hydrated!
        sec_ids = {s.section_id for s in resp.parent_sections}
        assert "doc_trips::u01::s01" in sec_ids
        assert "doc_trips::u01::s02" in sec_ids
        assert "doc_noise::u01::s01" not in sec_ids

    def test_borderline_safety_net_vs_strict_out_of_scope_gate(self) -> None:
        svc = _build_mock_search_service_for_pipeline()
        cand_a = SearchHit(
            chunk_id="doc_trips::c01",
            doc_id="doc_trips",
            unit_id="u01",
            section_id="doc_trips::u01::s01",
            file_name="Регламент о командировании.pdf",
            title="Регламент о командировании",
            clean_text="Командировочные расходы возмещаются работникам.",
            score=0.60,
        )
        svc.retrieve_candidate_pool.return_value = ([cand_a], [])
        svc._apply_adaptive_table_routing.return_value = []

        # Case A: Partial signal (score=0.36, all_irrelevant_out_of_scope=False) -> safety net keeps chunk!
        llm_borderline = MagicMock()
        llm_borderline.chat.completions.create.side_effect = [
            _make_json_completion({"intent": "document_lookup", "rewritten_primary": "суточные", "sub_queries": ["суточные"]}),
            _make_json_completion(
                {
                    "items": [{"idx": 0, "chunk_id": "doc_trips::c01", "score": 0.36, "verdict": "irrelevant", "reason": "частично"}],
                    "all_irrelevant_out_of_scope": False,
                    "needs_followup": False,
                    "followup_queries": [],
                }
            ),
        ]
        qp_border = QueryPipeline(search_service=svc, llm_client=llm_borderline)
        res_border = qp_border.run("косвенный вопрос про командировку")
        assert res_border.search_response.is_out_of_scope is False
        assert res_border.trace.borderline_safety_net_used is True
        assert len(res_border.search_response.hits) == 1

        # Case B: Out-of-scope (score=0.05, all_irrelevant_out_of_scope=True) -> 0 hits cleanly!
        llm_oos = MagicMock()
        llm_oos.chat.completions.create.side_effect = [
            _make_json_completion({"intent": "document_lookup", "rewritten_primary": "погода завтра", "sub_queries": ["прогноз погоды"]}),
            _make_json_completion(
                {
                    "items": [{"idx": 0, "chunk_id": "doc_trips::c01", "score": 0.05, "verdict": "irrelevant", "reason": "не по теме"}],
                    "all_irrelevant_out_of_scope": True,
                    "needs_followup": False,
                    "followup_queries": [],
                }
            ),
        ]
        qp_oos = QueryPipeline(search_service=svc, llm_client=llm_oos)
        res_oos = qp_oos.run("Какая завтра погода?")
        assert res_oos.search_response.is_out_of_scope is True
        assert res_oos.search_response.hits == []
        assert res_oos.search_response.linked_tables == []
        assert res_oos.search_response.parent_sections == []


# -----------------------------------------------------------------------------
# 4. Stage 4: TableAwareRAGAgent (`ask` & `ask_stream`), SSE Order & Plain Russian UX
# -----------------------------------------------------------------------------


class TestAgentStreamingAndUserFacingUX:
    def test_ask_stream_emits_status_then_verified_sources_only_then_tokens(self) -> None:
        svc = _build_mock_search_service_for_pipeline()
        hit_good = SearchHit(
            chunk_id="doc_trips::c01",
            doc_id="doc_trips",
            unit_id="u01",
            section_id="doc_trips::u01::s01",
            file_name="Регламент о командировании.pdf",
            title="Регламент о командировании",
            clause_no="4.1",
            page_numbers=[5],
            clean_text="Работник обязан в течение 3 рабочих дней сдать авансовый отчет.",
            score=0.82,
        )
        hit_bad = SearchHit(
            chunk_id="doc_noise::c01",
            doc_id="doc_noise",
            unit_id="u01",
            section_id="doc_noise::u01::s01",
            file_name="Регламент ЦОД.pdf",
            title="Регламент эксплуатации ЦОД",
            clean_text="Дежурная смена ЦОД работает круглосуточно.",
            score=0.80,
        )
        svc.retrieve_candidate_pool.return_value = ([hit_bad, hit_good], [])
        svc._apply_adaptive_table_routing.return_value = []

        mock_llm = MagicMock()
        answer_md = (
            "Авансовый отчет необходимо сдать в течение 3 рабочих дней.\n\n"
            "> «Работник обязан в течение 3 рабочих дней сдать авансовый отчет.»\n"
            "> — Регламент о командировании, п. 4.1, стр. 5"
        )
        mock_llm.chat.completions.create.side_effect = [
            _make_json_completion({"intent": "document_lookup", "rewritten_primary": "срок авансового отчета", "sub_queries": ["авансовый отчет"]}),
            _make_json_completion(
                {
                    "items": [
                        {"idx": 0, "chunk_id": "doc_noise::c01", "score": 0.04, "verdict": "irrelevant", "reason": "шум"},
                        {"idx": 1, "chunk_id": "doc_trips::c01", "score": 0.94, "verdict": "essential", "reason": "точный пункт"},
                    ],
                    "all_irrelevant_out_of_scope": False,
                    "needs_followup": False,
                    "followup_queries": [],
                }
            ),
            _make_stream_chunks([answer_md]),
        ]

        agent = TableAwareRAGAgent(
            search_service=svc,
            llm_client=mock_llm,
            enable_query_pipeline=True,
        )
        events = list(agent.ask_stream(AgentAskRequest(query="В какой срок сдать авансовый отчет?")))
        event_names = [ev[0] for ev in events]

        # Status events MUST precede `sources`, and `sources` MUST contain ONLY verified hit_good!
        assert event_names[:3] == ["status", "status", "sources"]
        sources_payload = events[2][1]
        assert [h["doc_id"] for h in sources_payload["hits"]] == ["doc_trips"]
        assert "token" in event_names
        quotes_payload = next(ev[1] for ev in events if ev[0] == "quotes")
        assert len(quotes_payload["quotes"]) == 1
        assert quotes_payload["quotes"][0]["verified"] is True
        done_payload = events[-1][1]
        assert done_payload["pipeline_trace"]["kept_doc_ids"] == ["doc_trips"]
        assert "doc_noise" in done_payload["pipeline_trace"]["pruned_doc_ids"]

    def test_meta_assistant_and_out_of_scope_have_zero_jargon_and_zero_bogus_sources(self) -> None:
        svc = _build_mock_search_service_for_pipeline()

        # 1) Meta-assistant («Что ты умеешь?») offline fallback -> dynamic registry summary, 0 sources, 0 jargon
        agent_offline = TableAwareRAGAgent(search_service=svc, api_key="")
        meta_resp = agent_offline.ask("Что ты умеешь?")
        assert meta_resp.search_response is not None
        assert meta_resp.search_response.is_out_of_scope is False
        assert meta_resp.search_response.hits == []
        assert meta_resp.quotes == []
        assert "База знаний" in meta_resp.answer
        assert "Регламент о командировании" in meta_resp.answer
        for bad in FORBIDDEN_USER_JARGON:
            assert bad not in meta_resp.answer.lower()

        # Also check ask_stream for meta-assistant
        stream_evs = list(agent_offline.ask_stream("Что ты умеешь?"))
        src_ev = next(d for name, d in stream_evs if name == "sources")
        assert src_ev["hits"] == []
        assert src_ev["is_out_of_scope"] is False
        streamed_text = "".join(d["text"] for name, d in stream_evs if name == "token")
        assert "База знаний" in streamed_text
        for bad in FORBIDDEN_USER_JARGON:
            assert bad not in streamed_text.lower()

        # 2) Out-of-scope («Как испечь пирог?») -> polite plain-Russian refusal, 0 sources, 0 jargon
        svc.search.return_value = SearchResponse(
            query="Как испечь пирог?",
            hits=[
                SearchHit(
                    chunk_id="bogus::1",
                    doc_id="doc_noise",
                    file_name="Регламент ЦОД.pdf",
                    title="Регламент ЦОД",
                    clean_text="шум",
                    score=0.15,
                )
            ],
            is_out_of_scope=True,
            confidence_score=0.11,
        )
        oos_resp = agent_offline.ask("Как испечь пирог?")
        assert oos_resp.answer == OUT_OF_SCOPE_REFUSAL_RU
        assert oos_resp.search_response is not None
        assert oos_resp.search_response.hits == []
        assert oos_resp.quotes == []
        for bad in FORBIDDEN_USER_JARGON:
            assert bad not in oos_resp.answer.lower()

        oos_stream = list(agent_offline.ask_stream("Как испечь пирог?"))
        oos_src = next(d for name, d in oos_stream if name == "sources")
        assert oos_src["hits"] == []
        assert oos_src["is_out_of_scope"] is True
        oos_txt = "".join(d["text"] for name, d in oos_stream if name == "token")
        for bad in FORBIDDEN_USER_JARGON:
            assert bad not in oos_txt.lower()


# -----------------------------------------------------------------------------
# 5. Live End-to-End Integration Tests (Polza AI + Qdrant)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not bool(settings.POLZA_API_KEY), reason="POLZA_API_KEY not configured")
class TestLivePolzaQueryPipeline:
    def test_live_what_can_you_do_meta_assistant(self) -> None:
        agent = TableAwareRAGAgent()
        resp = agent.ask("Что ты умеешь?")
        assert resp.search_response is not None
        assert resp.search_response.is_out_of_scope is False
        assert resp.search_response.hits == []
        assert resp.quotes == []
        assert len(resp.answer) >= 80
        for bad in FORBIDDEN_USER_JARGON:
            assert bad not in resp.answer.lower()

    def test_live_out_of_scope_weather_and_pie_have_zero_sources_and_zero_confidence_jargon(self) -> None:
        agent = TableAwareRAGAgent()
        for q in ("Какая завтра погода?", "Как испечь яблочный пирог?"):
            resp = agent.ask(q)
            assert resp.search_response is not None
            assert resp.search_response.is_out_of_scope is True
            assert resp.search_response.hits == []
            assert resp.quotes == []
            assert "нет сведений" in resp.answer.lower() or "базе" in resp.answer.lower()
            for bad in FORBIDDEN_USER_JARGON:
                assert bad not in resp.answer.lower()

    def test_live_typo_and_noise_pruning_pipeline(self) -> None:
        agent = TableAwareRAGAgent()
        resp = agent.ask("как забронировать зал фое")
        assert resp.search_response is not None
        assert resp.pipeline_trace is not None
        assert resp.search_response.is_out_of_scope is False
        kept_docs = {h.doc_id for h in resp.search_response.hits}
        assert "reg_concert_hall_booking" in kept_docs
        assert "order_5990_1_2026" not in kept_docs
        for bad in FORBIDDEN_USER_JARGON:
            assert bad not in resp.answer.lower()
