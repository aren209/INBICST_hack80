"""Deterministic Unit & Integration Tests for Adaptive 3-Tier Table Architecture & Tool-Calling.

Zero external network or paid API calls (all LLM client interactions mocked).
Uses tmp_path SQLite DB for isolated, deterministic execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.indexing.table_store import (
    OPENAI_TABLE_TOOLS,
    StructuredTableStore,
    compute_table_hash,
    execute_tool_call,
    should_store_structured_table,
)
from src.retrieval.rag_agent import TableAwareRAGAgent
from src.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    ChunkRecord,
    DocumentMeta,
    ParsedBlock,
    ParentSectionRecord,
    SearchHit,
    SearchResponse,
    TableColumnSchema,
    TablePassport,
    TableQueryRequest,
    TableQueryResponse,
    ToolCallTrace,
)


# ===========================================================================
# 1. Adaptive 3-Tier Eligibility (should_store_structured_table)
# ===========================================================================

class TestAdaptive3TierEligibility:
    """Verifies the eligibility rules routing tables to inline_parent vs tool_query."""

    def test_xlsx_always_eligible_regardless_of_size(self):
        # Even tiny 6-row or 1-row sheets must be stored to preserve formulas and types
        assert should_store_structured_table("xlsx", row_count=6, char_count=300) is True
        assert should_store_structured_table("XLSX", row_count=1, char_count=50) is True
        assert should_store_structured_table(".xlsx", row_count=0, char_count=0) is True
        assert should_store_structured_table("xlsm", row_count=10, char_count=500) is True
        assert should_store_structured_table("xls", row_count=20, char_count=1000) is True

    def test_compact_docx_and_pdf_remain_inline(self):
        # Tables under 25 rows AND under 5000 chars stay inline_parent via ParentSectionRecord
        assert should_store_structured_table("docx", row_count=12, char_count=1200) is False
        assert should_store_structured_table("pdf", row_count=20, char_count=2500) is False
        assert should_store_structured_table(".pdf", row_count=24, char_count=4999) is False
        assert should_store_structured_table("docx", row_count=5, char_count=400) is False

    def test_giant_docx_and_pdf_promoted_to_tool_query(self):
        # Promoted if row_count >= 25 OR char_count >= 5000 (e.g. Order 3614-1 Table 4)
        assert should_store_structured_table("docx", row_count=25, char_count=2000) is True
        assert should_store_structured_table("pdf", row_count=100, char_count=8000) is True
        assert should_store_structured_table("docx", row_count=15, char_count=5000) is True
        assert should_store_structured_table("pdf", row_count=10, char_count=6500) is True
        assert should_store_structured_table(".docx", row_count=1900, char_count=150000) is True


# ===========================================================================
# 2. StructuredTableStore (SQLite Backend in tmp_path)
# ===========================================================================

@pytest.fixture
def temp_store(tmp_path: Path) -> StructuredTableStore:
    db_file = tmp_path / "test_tables.db"
    return StructuredTableStore(db_path=db_file)


class TestStructuredTableStore:
    """Tests upsert, schema inference, filtering, aggregations, formula extraction, and deduplication."""

    def test_upsert_and_schema_inference(self, temp_store: StructuredTableStore):
        headers = ["№ п/п", "Финансодержатель", "ЦФО 1 уровня", "Лимит, руб.", "Ставка %", "Формула расчета"]
        rows = [
            ["1", "ФРКТ", "Кафедра радиоэлектроники", "1 500 000,00 руб.", "20% (0.2)", "[формула: =SUM(D2:D10)]"],
            ["2", "ФРКТ", "Лаборатория физики", "2 300 000 руб.", "20%", "[формула: =D3*0.2]"],
            ["3", "ФАКИ", "Кафедра аэрофизики", "950 000", "10%", ""],
            ["4", "ФАКИ", "Центр космических технологий", "3 100 000,50", "15% (0.15)", "[формула: =AVERAGE(D2:D4)]"],
        ]

        passport = temp_store.upsert_table(
            table_id="doc_fin_2026::tbl_01",
            doc_id="doc_fin_2026",
            headers=headers,
            rows=rows,
            file_name="Приказ_Финструктура_2026.xlsx",
            doc_number="3614-1",
            doc_date="11.08.2025",
            file_format="xlsx",
            sheet_name="Фин.структура",
            table_title="Финансовая структура МФТИ на 2026 год",
        )

        assert passport.table_id == "doc_fin_2026::tbl_01"
        assert passport.row_count == 4
        assert passport.col_count == 6
        assert passport.has_formulas is True
        assert passport.formula_count == 3
        assert passport.delivery_mode == "tool_query"
        assert len(passport.columns) == 6

        # Check column types inference
        col_by_name = {c.name: c for c in passport.columns}
        assert col_by_name["Финансодержатель"].is_hierarchy is True
        assert col_by_name["ЦФО 1 уровня"].is_hierarchy is True
        assert col_by_name["Лимит, руб."].inferred_type == "numeric"
        assert col_by_name["Ставка %"].inferred_type == "numeric"
        assert col_by_name["Формула расчета"].has_formulas is True
        assert len(passport.preview_rows) == 4

    def test_cross_file_table_hash_deduplication(self, temp_store: StructuredTableStore):
        # Two identical tables in two different documents (e.g. Word order & Excel appendix)
        headers = ["Код ЦФО", "Наименование", "Сумма"]
        rows = [
            ["101", "Кафедра радио", "1000"],
            ["102", "Кафедра квантов", "2000"],
        ]

        hash1 = compute_table_hash(headers, rows)
        assert hash1 != ""

        # Upsert in doc 1 (Word)
        p1 = temp_store.upsert_table(
            table_id="word_order::tbl_01",
            doc_id="word_order",
            headers=headers,
            rows=rows,
            file_name="Приказ_3614.docx",
            file_format="docx",
        )

        # Upsert in doc 2 (Excel)
        p2 = temp_store.upsert_table(
            table_id="excel_structure::tbl_01",
            doc_id="excel_structure",
            headers=headers,
            rows=rows,
            file_name="Финструктура_2026.xlsx",
            file_format="xlsx",
        )

        assert p1.table_hash == p2.table_hash == hash1

        # Re-fetch both and check also_in_files cross-linking
        refreshed_p1 = temp_store.get_table("word_order::tbl_01")
        refreshed_p2 = temp_store.get_table("excel_structure::tbl_01")

        assert refreshed_p1 is not None and refreshed_p2 is not None
        assert "Финструктура_2026.xlsx" in refreshed_p1.also_in_files
        assert "Приказ_3614.docx" in refreshed_p2.also_in_files

    def test_filter_rows_substring_normalization_and_partial_columns(self, temp_store: StructuredTableStore):
        headers = ["Финансодержатель", "ЦФО 1 уровня", "Сумма"]
        rows = [
            ["ФРКТ", "Кафедра радиоэлектроники", "1000"],
            ["ФРКТ", "Лаборатория прикладной физики", "2000"],
            ["ИНБИКСТ", "Лаборатория нанотехнологий", "3000"],
            ["ФАКТ", "Отдел летных испытаний", "4000"],
        ]
        temp_store.upsert_table("tbl_filter_test", "doc1", headers, rows)

        # Partial column name ('ЦФО 1' -> 'ЦФО 1 уровня') + case/yo-forgiving match
        req = TableQueryRequest(
            table_id="tbl_filter_test",
            operation="filter_rows",
            filters={"ЦФО 1": "радиоэлектроники"},
        )
        res = temp_store.query_table(req)
        assert res.matched_count == 1
        assert res.rows[0]["Финансодержатель"] == "ФРКТ"

        # Search_text across whole row
        req2 = TableQueryRequest(
            table_id="tbl_filter_test",
            operation="filter_rows",
            search_text="инбикст",
        )
        res2 = temp_store.query_table(req2)
        assert res2.matched_count == 1
        assert res2.rows[0]["Сумма"] == "3000"

        # select_columns & limit/offset pagination
        req3 = TableQueryRequest(
            table_id="tbl_filter_test",
            operation="filter_rows",
            filters={"Финансодержатель": "ФРКТ"},
            select_columns=["ЦФО 1 уровня"],
            limit=1,
            offset=1,
        )
        res3 = temp_store.query_table(req3)
        assert res3.matched_count == 2
        assert res3.returned_count == 1
        assert list(res3.rows[0].keys()) == ["ЦФО 1 уровня"]
        assert res3.rows[0]["ЦФО 1 уровня"] == "Лаборатория прикладной физики"

    def test_filter_rows_numeric_comparisons(self, temp_store: StructuredTableStore):
        headers = ["Наименование", "Лимит", "Ставка"]
        rows = [
            ["Проект А", "1 000 000 руб.", "20%"],
            ["Проект Б", "500 000 руб.", "10%"],
            ["Проект В", "2 500 000 руб.", "20%"],
            ["Проект Г", "100 000 руб.", "5%"],
        ]
        temp_store.upsert_table("tbl_num_comp", "doc1", headers, rows)

        # Greater than comparison
        res_gt = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_num_comp",
                operation="filter_rows",
                filters={"Лимит": "> 800000"},
            )
        )
        assert res_gt.matched_count == 2
        names = {r["Наименование"] for r in res_gt.rows}
        assert names == {"Проект А", "Проект В"}

        # Percentage comparison
        res_pct = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_num_comp",
                operation="filter_rows",
                filters={"Ставка": ">= 20"},
            )
        )
        assert res_pct.matched_count == 2

    def test_aggregate_operations_and_group_by(self, temp_store: StructuredTableStore):
        headers = ["Финансодержатель", "Подразделение", "Бюджет"]
        rows = [
            ["ФРКТ", "Кафедра 1", "1 000 000"],
            ["ФРКТ", "Кафедра 2", "2 000 000"],
            ["ФРКТ", "Лаборатория 1", "500 000"],
            ["ФАКИ", "Кафедра А", "800 000"],
            ["ФАКИ", "Кафедра Б", "1 200 000"],
        ]
        temp_store.upsert_table("tbl_agg_test", "doc_agg", headers, rows)

        # 1. Total COUNT
        res_count = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_agg_test",
                operation="aggregate",
                agg_func="count",
            )
        )
        assert res_count.aggregation_result == 5

        # 2. Filtered COUNT
        res_frkt_count = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_agg_test",
                operation="aggregate",
                agg_func="count",
                filters={"Финансодержатель": "ФРКТ"},
            )
        )
        assert res_frkt_count.aggregation_result == 3

        # 3. DISTINCT
        res_dist = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_agg_test",
                operation="aggregate",
                agg_func="distinct",
                agg_column="Финансодержатель",
            )
        )
        assert set(res_dist.aggregation_result) == {"ФРКТ", "ФАКИ"}

        # 4. SUM
        res_sum = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_agg_test",
                operation="aggregate",
                agg_func="sum",
                agg_column="Бюджет",
                filters={"Финансодержатель": "ФРКТ"},
            )
        )
        assert res_sum.aggregation_result == 3500000.0

        # 5. AVG, MIN, MAX
        res_avg = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_agg_test",
                operation="aggregate",
                agg_func="avg",
                agg_column="Бюджет",
            )
        )
        assert res_avg.aggregation_result == 1100000.0

        res_min = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_agg_test",
                operation="aggregate",
                agg_func="min",
                agg_column="Бюджет",
            )
        )
        assert res_min.aggregation_result == 500000.0

        res_max = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_agg_test",
                operation="aggregate",
                agg_func="max",
                agg_column="Бюджет",
            )
        )
        assert res_max.aggregation_result == 2000000.0

        # 6. GROUP BY
        res_grp = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_agg_test",
                operation="aggregate",
                agg_func="sum",
                agg_column="Бюджет",
                group_by="Финансодержатель",
            )
        )
        assert isinstance(res_grp.aggregation_result, dict)
        assert res_grp.aggregation_result["ФРКТ"] == 3500000.0
        assert res_grp.aggregation_result["ФАКИ"] == 2000000.0

    def test_get_formulas_extraction_and_filtering(self, temp_store: StructuredTableStore):
        headers = ["Статья сметы", "КВР", "КОСГУ", "Расчет"]
        rows = [
            ["ФОТ", "111", "211", "100 000 [формула: =C2*C3]"],
            ["Страховые взносы", "119", "213", "30 200 [формула: =SUM(D2:D5)*0.302]"],
            ["Накладные расходы", "244", "226", "20 000 [формула: =D2*0.2]"],
            ["Командировочные", "112", "212", "15 000"],
        ]
        temp_store.upsert_table("tbl_smeta_formulas", "doc_smeta", headers, rows)

        res_formulas = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_smeta_formulas",
                operation="get_formulas",
            )
        )
        assert len(res_formulas.formulas) == 3
        # Ensure zero 'None' strings
        for f in res_formulas.formulas:
            assert f["formula"] != "None"
            assert f["formula"].startswith("=")

        # Filter formulas by column
        res_filtered_f = temp_store.query_table(
            TableQueryRequest(
                table_id="tbl_smeta_formulas",
                operation="get_formulas",
                filters={"Статья сметы": "Накладные"},
            )
        )
        assert len(res_filtered_f.formulas) == 1
        assert res_filtered_f.formulas[0]["formula"] == "=D2*0.2"

    def test_delete_document_tables_and_peer_reconciliation(self, temp_store: StructuredTableStore):
        headers = ["Колонка 1", "Колонка 2"]
        rows = [["А", "Б"]]

        p1 = temp_store.upsert_table("docA::tbl_01", "docA", headers, rows, file_name="fileA.xlsx")
        p2 = temp_store.upsert_table("docB::tbl_01", "docB", headers, rows, file_name="fileB.docx")

        # Prior to deletion, docB references fileA
        refreshed_b = temp_store.get_table("docB::tbl_01")
        assert refreshed_b is not None
        assert "fileA.xlsx" in refreshed_b.also_in_files

        # Delete docA
        deleted = temp_store.delete_document_tables("docA")
        assert deleted == 1
        assert temp_store.get_table("docA::tbl_01") is None

        # docB should now have fileA removed from also_in_files
        after_delete_b = temp_store.get_table("docB::tbl_01")
        assert after_delete_b is not None
        assert "fileA.xlsx" not in after_delete_b.also_in_files


# ===========================================================================
# 3. OpenAI Tool Schema & Tool Calling Execution
# ===========================================================================

class TestOpenAIToolCallingSchemaAndExecution:
    """Verifies valid JSON schemas and deterministic tool dispatching."""

    def test_openai_table_tools_schema_validity(self):
        assert isinstance(OPENAI_TABLE_TOOLS, list)
        tool_names = [t["function"]["name"] for t in OPENAI_TABLE_TOOLS]
        assert "query_structured_table" in tool_names
        assert "get_table_schema" in tool_names

        for tool in OPENAI_TABLE_TOOLS:
            assert tool["type"] == "function"
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            params = fn["parameters"]
            assert params["type"] == "object"
            assert "table_id" in params["required"]

    def test_execute_tool_call_query_structured_table(self, temp_store: StructuredTableStore):
        headers = ["Подразделение", "Количество"]
        rows = [
            ["Лаборатория 1", "10"],
            ["Лаборатория 2", "20"],
            ["Лаборатория 3", "30"],
        ]
        temp_store.upsert_table("tool_exec_tbl", "doc1", headers, rows)

        # Call query_structured_table via execute_tool_call
        args = {
            "table_id": "tool_exec_tbl",
            "operation": "aggregate",
            "agg_func": "sum",
            "agg_column": "Количество",
        }
        raw_res, q_res = execute_tool_call("query_structured_table", args, store=temp_store)

        assert "error" not in raw_res
        assert raw_res["aggregation_result"] == 60.0
        assert q_res is not None
        assert q_res.aggregation_result == 60.0

        # Call get_table_schema via execute_tool_call
        raw_schema, _ = execute_tool_call("get_table_schema", {"table_id": "tool_exec_tbl"}, store=temp_store)
        assert "error" not in raw_schema
        assert raw_schema["table_id"] == "tool_exec_tbl"
        assert raw_schema["row_count"] == 3
        assert len(raw_schema["columns"]) == 2


# ===========================================================================
# 4. TableAwareRAGAgent Loop with Mocked LLM Client
# ===========================================================================

class TestTableAwareRAGAgentLoop:
    """Verifies TableAwareRAGAgent with mocked OpenAI client across all 3 tiers and out-of-scope."""

    def _make_mock_chat_completion(self, content: str = "", tool_calls: list[Any] | None = None, model: str = "mock-model"):
        choice = MagicMock()
        choice.message = MagicMock()
        choice.message.content = content
        choice.message.tool_calls = tool_calls or []
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = model
        return resp

    def test_xl_aggregation_tool_calling_roundtrip(self, temp_store: StructuredTableStore):
        # 1. Populate structured table in temp_store
        headers = ["Финансодержатель", "ЦФО 1 уровня", "Бюджет"]
        rows = [
            ["ФРКТ", "Кафедра радиотехники", "1 000 000"],
            ["ФРКТ", "Лаборатория физики", "2 000 000"],
            ["ФРКТ", "Учебный центр", "3 000 000"],
        ]
        passport = temp_store.upsert_table(
            table_id="fin_structure::tbl_01",
            doc_id="fin_doc",
            headers=headers,
            rows=rows,
            file_name="Финструктура.xlsx",
            file_format="xlsx",
            table_title="Структура ЦФО 2026",
        )

        # 2. Mock SearchService returning SearchHit with linked_tables
        mock_search_service = MagicMock()
        mock_search_hit = SearchHit(
            doc_id="fin_doc",
            unit_id="u01",
            section_id="sec_01",
            chunk_id="chunk_01",
            title="Финансовая структура",
            file_name="Финструктура.xlsx",
            table_id="fin_structure::tbl_01",
            clean_text="Таблица финансовой структуры МФТИ на 2026 год",
            metadata={"table_delivery_mode": "tool_query"},
        )
        mock_search_resp = SearchResponse(
            query="Сколько подразделений подчиняется ФРКТ?",
            hits=[mock_search_hit],
            linked_tables=[passport],
            is_out_of_scope=False,
            confidence_score=0.92,
        )
        mock_search_service.search.return_value = mock_search_resp

        # 3. Create mock OpenAI client:
        # Round 1 returns tool_call: query_structured_table
        # Round 2 returns final Russian answer citing the count
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc123"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "query_structured_table"
        mock_tool_call.function.arguments = json.dumps({
            "table_id": "fin_structure::tbl_01",
            "operation": "aggregate",
            "agg_func": "count",
            "filters": {"Финансодержатель": "ФРКТ"},
        })

        resp_round_1 = self._make_mock_chat_completion(tool_calls=[mock_tool_call])
        resp_round_2 = self._make_mock_chat_completion(
            content="Финансодержателю ФРКТ подчиняется ровно 3 подразделения согласно таблице финансовой структуры."
        )

        mock_llm_client = MagicMock()
        mock_llm_client.chat.completions.create.side_effect = [resp_round_1, resp_round_2]

        agent = TableAwareRAGAgent(
            search_service=mock_search_service,
            table_store=temp_store,
            llm_client=mock_llm_client,
            model="openai/gpt-4o-mini",
        )

        req = AgentAskRequest(
            query="Сколько подразделений подчиняется ФРКТ?",
            enable_table_tools=True,
            max_tool_rounds=4,
        )
        response: AgentAskResponse = agent.ask(req)

        # Assertions:
        assert response.used_table_tools is True
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "query_structured_table"
        assert response.tool_calls[0].matched_count == 3
        assert "3 подразделения" in response.answer
        assert len(response.linked_tables) >= 1
        assert response.linked_tables[0].table_id == "fin_structure::tbl_01"

        # Verify the prompt sent to round 1 included TablePassport and tools
        call_args_1 = mock_llm_client.chat.completions.create.call_args_list[0].kwargs
        assert "tools" in call_args_1
        prompt_content = call_args_1["messages"][1]["content"]
        assert "fin_structure::tbl_01" in prompt_content
        assert "ДОСТУПНЫЕ СТРУКТУРИРОВАННЫЕ ТАБЛИЦЫ В SQLite" in prompt_content

        # Verify round 2 received tool output
        call_args_2 = mock_llm_client.chat.completions.create.call_args_list[1].kwargs
        messages_2 = call_args_2["messages"]
        assert any(m.get("role") == "tool" and m.get("tool_call_id") == "call_abc123" for m in messages_2)

    def test_compact_table_inline_parent_single_pass_no_tools(self, temp_store: StructuredTableStore):
        # Compact table in PDF (e.g. Регламент бронирования КЗ: 6 дней репетиций)
        # delivery_mode = "inline_parent", linked_tables is empty!
        mock_search_service = MagicMock()
        parent_rec = ParentSectionRecord(
            doc_id="reglament_kz",
            unit_id="u01",
            section_id="sec_kz_06",
            section_title="6. Особенности бронирования для традиционных мероприятий",
            clean_text=(
                "Таблица: Количество репетиционных дней перед Посвящением в студенты.\n"
                "| Мероприятие | Дней репетиций |\n"
                "| Посвящение в студенты | 6 дней |\n"
                "| Дни факультетов | 3 дня |\n"
            ),
        )
        mock_search_hit = SearchHit(
            doc_id="reglament_kz",
            unit_id="u01",
            section_id="sec_kz_06",
            chunk_id="chunk_kz_01",
            title="Регламент бронирования концертного зала",
            file_name="Регламент_КЗ.pdf",
            parent_section=parent_rec,
            parent_text=parent_rec.clean_text,
            clean_text="Таблица норм бронирования концертного зала...",
            metadata={"table_delivery_mode": "inline_parent"},
        )
        mock_search_resp = SearchResponse(
            query="Сколько дней дается на репетиции Посвящения в студенты?",
            hits=[mock_search_hit],
            linked_tables=[],  # No structured table needed for compact tables!
            is_out_of_scope=False,
            confidence_score=0.88,
        )
        mock_search_service.search.return_value = mock_search_resp

        # Mock LLM returns answer in 1 pass directly from ParentSectionRecord
        mock_llm_client = MagicMock()
        mock_llm_client.chat.completions.create.return_value = self._make_mock_chat_completion(
            content="Согласно разделу 6 Регламента бронирования КЗ, на Посвящение в студенты выделяется 6 дней репетиций."
        )

        agent = TableAwareRAGAgent(
            search_service=mock_search_service,
            table_store=temp_store,
            llm_client=mock_llm_client,
        )

        response: AgentAskResponse = agent.ask("Сколько дней дается на репетиции Посвящения в студенты?")

        # Assertions:
        assert response.used_table_tools is False
        assert len(response.tool_calls) == 0
        assert "6 дней репетиций" in response.answer

        # LLM was invoked exactly once with allow_tools=False
        assert mock_llm_client.chat.completions.create.call_count == 1
        first_call = mock_llm_client.chat.completions.create.call_args.kwargs
        assert "tools" not in first_call
        assert "6 дней" in first_call["messages"][1]["content"]

    def test_out_of_scope_query_short_circuits_with_zero_tool_calls(self, temp_store: StructuredTableStore):
        mock_search_service = MagicMock()
        mock_search_service.search.return_value = SearchResponse(
            query="Какая погода в Долгопрудном завтра?",
            hits=[],
            is_out_of_scope=True,
            confidence_score=0.12,
        )

        mock_llm_client = MagicMock()

        agent = TableAwareRAGAgent(
            search_service=mock_search_service,
            table_store=temp_store,
            llm_client=mock_llm_client,
        )

        response = agent.ask("Какая погода в Долгопрудном завтра?")

        assert response.used_table_tools is False
        assert len(response.tool_calls) == 0
        assert response.model_used == "out_of_scope_guard"
        assert "в загруженной базе документов нет сведений" in response.answer.lower()
        assert "база знаний" in response.answer.lower()
        assert "confidence" not in response.answer.lower()
        # Zero calls to LLM chat completion
        assert mock_llm_client.chat.completions.create.call_count == 0
