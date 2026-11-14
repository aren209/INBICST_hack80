"""Focused unit tests for citations/table verification, offline fallback formatting,
alert propagation, and background indexing error status handling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing.table_store import StructuredTableStore
from src.retrieval.quote_verifier import (
    build_source_label,
    context_docs_from_search,
    verify_quotes,
)
from src.retrieval.rag_agent import SYSTEM_PROMPT_RU, TableAwareRAGAgent
from src.schemas import (
    AgentAskRequest,
    DocumentMeta,
    ParentSectionRecord,
    SearchHit,
    SearchResponse,
    TableColumnSchema,
    TablePassport,
    TableQueryResponse,
    ToolCallTrace,
)
from src.web.app import (
    TECH_ERROR_STUB_RU,
    TaskState,
    _build_document_view_sync,
    _run_index_task,
    _sanitize_user_index_error,
)


# ---------------------------------------------------------------------------
# 1. Background indexing error handling (_run_index_task in src/web/app.py)
# ---------------------------------------------------------------------------


def test_run_index_task_all_files_failed_sets_failed_status_and_sanitizes(tmp_path: Path) -> None:
    f1 = tmp_path / "broken_scan.pdf"
    f2 = tmp_path / "empty_doc.docx"
    f1.write_bytes(b"%PDF-1.4 broken")
    f2.write_bytes(b"PK\x03\x04 empty")

    class FakePipeline:
        def index_file(self, file_path: Path, **kwargs: Any) -> dict[str, Any]:
            name = Path(file_path).name
            cb = kwargs.get("progress_callback")
            if name == "broken_scan.pdf":
                raw_msg = (
                    'Traceback (most recent call last):\n  File "/app/src/ingestion/loader.py", line 42\n'
                    "RuntimeError: PDF parsing failed in /home/arz/hac2/data/raw/broken_scan.pdf"
                )
                if cb:
                    cb({"stage": "error", "stage_label": "Ошибка", "percent": 50, "message": raw_msg})
                return {
                    "status": "error",
                    "file_name": name,
                    "error_message": raw_msg,
                }
            return {
                "status": "failed",
                "file_name": name,
                "error_message": "Документ не содержит извлекаемого текста.",
            }

    task = TaskState(task_id="task-all-fail", task_type="upload")
    _run_index_task(
        task=task,
        pipeline=FakePipeline(),
        saved_paths=[f1, f2],
        metadata_overrides=None,
        clean_ocr_backend=None,
        effective_force=False,
    )

    assert task.status == "failed"
    assert task.error == TECH_ERROR_STUB_RU
    assert task.message == TECH_ERROR_STUB_RU
    assert task.result is not None
    assert task.result["status"] == "failed"
    assert task.result["error"] == TECH_ERROR_STUB_RU
    assert task.result["total_files"] == 2

    files = task.result["files"]
    assert len(files) == 2
    # File 1 had a Python traceback + path -> sanitized to TECH_ERROR_STUB_RU on both fields
    assert files[0]["error_message"] == TECH_ERROR_STUB_RU
    assert files[0]["message"] == TECH_ERROR_STUB_RU
    assert "Traceback" not in files[0]["error_message"]
    assert "/app/" not in files[0]["error_message"]

    # File 2 had a clean Russian error -> preserved cleanly
    assert files[1]["error_message"] == "Документ не содержит извлекаемого текста."
    assert files[1]["message"] == "Документ не содержит извлекаемого текста."

    done_events = [ev for ev in task.events if ev.get("stage") == "done"]
    assert len(done_events) == 1
    assert done_events[0]["status"] == "failed"
    assert done_events[0]["stage_label"] == "Ошибка"
    assert done_events[0]["percent"] == 100
    assert done_events[0]["message"] == TECH_ERROR_STUB_RU

    # Per-file error events via progress_callback are also sanitized
    error_events = [ev for ev in task.events if ev.get("stage") == "error"]
    assert len(error_events) == 1
    assert TECH_ERROR_STUB_RU in error_events[0]["message"]
    assert "Traceback" not in error_events[0]["message"]
    assert "/app/" not in error_events[0]["message"]


def test_run_index_task_single_clean_russian_failure_preserves_message(tmp_path: Path) -> None:
    f1 = tmp_path / "bad.docx"
    f1.write_bytes(b"corrupt")

    class FakePipeline:
        def index_file(self, file_path: Path, **kwargs: Any) -> dict[str, Any]:
            return {
                "status": "error",
                "file_name": Path(file_path).name,
                "error_message": "Не удалось распознать таблицу в документе.",
            }

    task = TaskState(task_id="task-clean-fail", task_type="upload")
    _run_index_task(task, FakePipeline(), [f1], None, None, False)

    assert task.status == "failed"
    assert task.error == "Не удалось распознать таблицу в документе."
    assert task.message == "Не удалось распознать таблицу в документе."
    assert task.result["status"] == "failed"
    done_ev = [ev for ev in task.events if ev.get("stage") == "done"][0]
    assert done_ev["status"] == "failed"
    assert done_ev["message"] == "Не удалось распознать таблицу в документе."


def test_run_index_task_partial_errors_keeps_completed_with_honest_counts(tmp_path: Path) -> None:
    f_ok = tmp_path / "ok.pdf"
    f_dup = tmp_path / "dup.pdf"
    f_err = tmp_path / "err.pdf"
    for fp in (f_ok, f_dup, f_err):
        fp.write_bytes(b"%PDF")

    class FakePipeline:
        def index_file(self, file_path: Path, **kwargs: Any) -> dict[str, Any]:
            name = Path(file_path).name
            if name == "ok.pdf":
                return {"status": "indexed", "file_name": name, "doc_id": "doc-ok", "chunks_count": 4}
            if name == "dup.pdf":
                return {"status": "duplicate", "file_name": name, "doc_id": "doc-dup"}
            return {
                "status": "error",
                "file_name": name,
                "error_message": "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/missing.bin'",
            }

    task = TaskState(task_id="task-partial", task_type="upload")
    _run_index_task(task, FakePipeline(), [f_ok, f_dup, f_err], None, None, False)

    assert task.status == "completed"
    assert task.error is None
    assert "Успешно обработано файлов" not in task.message
    assert "проиндексировано: 1" in task.message
    assert "дубликатов: 1" in task.message
    assert "ошибок: 1" in task.message
    assert task.result is not None
    assert task.result["total_files"] == 3

    done_ev = [ev for ev in task.events if ev.get("stage") == "done"][0]
    assert done_ev["message"] == task.message

    assert task.result["files"][2]["error_message"] == TECH_ERROR_STUB_RU
    assert task.result["files"][2]["message"] == TECH_ERROR_STUB_RU
    assert _sanitize_user_index_error("Traceback (most recent call last): ValueError") == TECH_ERROR_STUB_RU


# ---------------------------------------------------------------------------
# 2. Prompt rules, offline fallback formatting, and alert propagation
# ---------------------------------------------------------------------------


def test_system_prompt_ru_updated_rules() -> None:
    assert "Всего до 5 цитат" not in SYSTEM_PROMPT_RU
    assert "3–8" in SYSTEM_PROMPT_RU
    assert "Лист:" in SYSTEM_PROMPT_RU
    assert "### Порядок действий" in SYSTEM_PROMPT_RU
    assert "> «точный текст из документа»" in SYSTEM_PROMPT_RU


def _make_fake_search_service(search_resp: SearchResponse) -> Any:
    class FakeSearchService:
        def search(self, *args: Any, **kwargs: Any) -> SearchResponse:
            return search_resp

        def _get_active_table_store(self) -> None:
            return None

    return FakeSearchService()


def _disable_real_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_offline_fallback_procedural_and_non_procedural_with_verifiable_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_real_api_keys(monkeypatch)

    hit1_text = (
        "(Продолжение п. 2.1)\n"
        "[Стр. 3]\n"
        "2.1. Для оформления временного пропуска сотрудник подаёт служебную записку в отдел безопасности не позднее чем за три рабочих дня.\n"
        "2.2. После проверки данных отдел безопасности активирует электронный пропуск и направляет уведомление инициатору."
    )
    hit2_text = (
        "3.1. При утрате пропуска сотрудник обязан немедленно уведомить дежурного администратора и оформить заявку на перевыпуск."
    )
    sr = SearchResponse(
        query="Как оформить пропуск в лабораторию?",
        hits=[
            SearchHit(
                chunk_id="c-1",
                doc_id="doc-pass-1",
                file_name="pass_reglament.pdf",
                title="Регламент пропускного режима",
                doc_number="№ 45-ОД",
                doc_date="2025-02-10",
                clause_no="2.1",
                page_numbers=[3],
                clean_text=hit1_text,
                score=0.92,
            ),
            SearchHit(
                chunk_id="c-2",
                doc_id="doc-pass-2",
                file_name="pass_security.docx",
                title="Инструкция по безопасности",
                doc_number="№ 12-И",
                clause_no="3.1",
                page_numbers=[5],
                clean_text=hit2_text,
                score=0.81,
            ),
        ],
        version_alerts=["Приказ № 45-ОД частично обновлён приказом № 80-ОД."],
        discrepancy_alerts=["В старой памятке указан срок 1 рабочий день, действует норма 3 рабочих дня."],
        quality_alerts=["Качество OCR страницы 3 снижено."],
        confidence_score=0.88,
    )

    agent = TableAwareRAGAgent(
        search_service=_make_fake_search_service(sr),
        table_store=None,
        llm_client=None,
        api_key="",
        enable_query_pipeline=False,
    )

    # Procedural query -> includes ### Порядок действий with parenthetical locators
    resp_proc = agent.ask(AgentAskRequest(query="Как оформить пропуск в лабораторию?", enable_query_pipeline=False))
    ans_proc = resp_proc.answer

    assert "### Краткий ответ" in ans_proc
    assert "### Порядок действий" in ans_proc
    assert "### Подробный ответ и цитаты" in ans_proc
    assert "[pass_reglament.pdf]" not in ans_proc
    assert "[pass_security.docx]" not in ans_proc
    assert "(Продолжение п." not in ans_proc

    # Extract lines in Порядок действий section and verify each numbered step has (...) source locator
    steps_block = ans_proc.split("### Порядок действий", 1)[1].split("### Подробный ответ и цитаты", 1)[0]
    numbered_lines = [ln.strip() for ln in steps_block.splitlines() if ln.strip()[:1].isdigit() and "." in ln[:4]]
    assert 2 <= len(numbered_lines) <= 5
    for step_line in numbered_lines:
        assert step_line.endswith(")") and "(" in step_line

    # Verify alerts propagated cleanly and no double "№ №"
    assert resp_proc.version_alerts == sr.version_alerts
    assert resp_proc.discrepancy_alerts == sr.discrepancy_alerts
    assert resp_proc.quality_alerts == sr.quality_alerts
    assert "расхождения" in ans_proc.lower()
    assert "№ №" not in ans_proc

    # Verify all blockquote quotes pass quote_verifier and _verify_answer_quotes with verified=True
    docs = context_docs_from_search(sr)
    verified = verify_quotes(ans_proc, docs)
    assert len(verified) >= 2
    assert all(q.get("verified") is True for q in verified)
    assert verified[0]["matched_doc_id"] == "doc-pass-1"
    assert resp_proc.quotes[0]["doc_id"] == "doc-pass-1"
    assert resp_proc.quotes[0]["page_num"] == 3

    # Non-procedural query -> omits ### Порядок действий completely
    sr_info = sr.model_copy(update={"query": "Какой срок хранения журналов инструктажа?"})
    agent_info = TableAwareRAGAgent(
        search_service=_make_fake_search_service(sr_info),
        table_store=None,
        llm_client=None,
        api_key="",
        enable_query_pipeline=False,
    )
    resp_info = agent_info.ask(
        AgentAskRequest(query="Какой срок хранения журналов инструктажа?", enable_query_pipeline=False)
    )
    assert "### Краткий ответ" in resp_info.answer
    assert "### Подробный ответ и цитаты" in resp_info.answer
    assert "### Порядок действий" not in resp_info.answer


def test_ask_stream_propagates_alerts_and_clears_on_out_of_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_real_api_keys(monkeypatch)

    sr_in_scope = SearchResponse(
        query="Порядок командирования",
        hits=[
            SearchHit(
                chunk_id="h-1",
                doc_id="doc-trip",
                file_name="trip.pdf",
                title="Положение о командировках",
                doc_number="№ 9-П",
                clause_no="1.2",
                page_numbers=[2],
                clean_text="1.2. Суточные выплачиваются за каждый день нахождения в служебной командировке по утверждённым нормам.",
                score=0.9,
            )
        ],
        version_alerts=["Версия 2025 года заменила редакцию 2023 года."],
        discrepancy_alerts=["Лимит гостиницы в Москве отличается от регионального лимита."],
        quality_alerts=["Таблица приложения распознана частично."],
        confidence_score=0.85,
        is_out_of_scope=False,
    )

    agent = TableAwareRAGAgent(
        search_service=_make_fake_search_service(sr_in_scope),
        table_store=None,
        llm_client=None,
        api_key="",
        enable_query_pipeline=False,
    )
    events = list(agent.ask_stream(AgentAskRequest(query="Порядок командирования", enable_query_pipeline=False)))
    src_events = [payload for ev_type, payload in events if ev_type == "sources"]
    assert len(src_events) == 1
    assert src_events[0]["version_alerts"] == sr_in_scope.version_alerts
    assert src_events[0]["discrepancy_alerts"] == sr_in_scope.discrepancy_alerts
    assert src_events[0]["quality_alerts"] == sr_in_scope.quality_alerts

    # Out-of-scope: both ask() and ask_stream() return empty alert lists
    sr_oos = sr_in_scope.model_copy(update={"hits": [], "is_out_of_scope": True, "confidence_score": 0.1})
    agent_oos = TableAwareRAGAgent(
        search_service=_make_fake_search_service(sr_oos),
        table_store=None,
        llm_client=None,
        api_key="",
        enable_query_pipeline=False,
    )
    resp_oos = agent_oos.ask(AgentAskRequest(query="Рецепт борща", enable_query_pipeline=False))
    assert resp_oos.version_alerts == []
    assert resp_oos.discrepancy_alerts == []
    assert resp_oos.quality_alerts == []

    oos_events = list(agent_oos.ask_stream(AgentAskRequest(query="Рецепт борща", enable_query_pipeline=False)))
    oos_src = [payload for ev_type, payload in oos_events if ev_type == "sources"][0]
    assert oos_src["version_alerts"] == []
    assert oos_src["discrepancy_alerts"] == []
    assert oos_src["quality_alerts"] == []


# ---------------------------------------------------------------------------
# 3. Table Citation & Verification (quote_verifier.py & rag_agent.py)
# ---------------------------------------------------------------------------


def test_table_context_and_formulas_verify_cleanly_with_sheet_and_page(tmp_path: Path) -> None:
    store = StructuredTableStore(db_path=tmp_path / "tables.db")
    passport = store.upsert_table(
        table_id="tbl-tariffs-1",
        doc_id="doc-xls-tariffs",
        headers=["Категория", "Лимит руб", "Итого"],
        rows=[
            ["Командировка РФ", "4500", "4500 [формула: =B2*1.0]"],
            ["Зарубежная", "12000", "12000"],
        ],
        file_name="limits_2025.xlsx",
        doc_number="Приказ № 12-Т",
        doc_date="2025-01-15",
        file_format="xlsx",
        sheet_name="Тарифы",
        table_title="Лимиты суточных расходов",
    )

    q_resp = store.query_table({"table_id": "tbl-tariffs-1", "operation": "filter_rows", "limit": 10})
    q_formulas = store.query_table({"table_id": "tbl-tariffs-1", "operation": "get_formulas", "limit": 10})
    assert len(q_resp.rows) == 2
    assert len(q_formulas.formulas) >= 1

    sr = SearchResponse(
        query="Какой лимит для категории Командировка РФ?",
        hits=[
            SearchHit(
                chunk_id="xls-chunk-1",
                doc_id="doc-xls-tariffs",
                file_name="limits_2025.xlsx",
                title="Лимиты командировочных расходов",
                doc_number="Приказ № 12-Т",
                table_id="tbl-tariffs-1",
                page_numbers=[],
                clean_text="Таблица содержит лимиты суточных расходов по категориям поездок.",
                metadata={"sheet_name": "Тарифы", "is_single_sheet": True},
                score=0.89,
            )
        ],
        linked_tables=[passport],
        table_context=[q_resp, q_formulas],
    )

    docs = context_docs_from_search(sr)
    table_entries = [
        (doc, meta, text)
        for doc in docs
        for meta, text in zip(doc.metas, doc.texts)
        if meta.get("kind") == "table"
    ]
    assert len(table_entries) >= 2
    for doc, meta, text in table_entries:
        assert doc.doc_id == "doc-xls-tariffs"
        assert meta.get("sheet_name") == "Тарифы"
        assert meta.get("pages") == [1]
        assert "table_id" not in text
        assert "matched_count" not in text
        assert "SQLite" not in text

    combined_table_text = "\n".join(text for _, _, text in table_entries)
    assert "Командировка РФ" in combined_table_text
    assert "=B2*1.0" in combined_table_text

    answer = (
        "### Краткий ответ\n"
        "Лимит для категории «Командировка РФ» составляет 4500.\n\n"
        "### Подробный ответ и цитаты\n"
        "> «Категория: Командировка РФ; Лимит руб: 4500; Итого: 4500 [формула: =B2*1.0]»\n"
        "> — Лимиты командировочных расходов, Приказ № 12-Т, Лист: Тарифы, стр. 1\n"
    )
    verified = verify_quotes(answer, docs)
    assert len(verified) == 1
    q = verified[0]
    assert q["verified"] is True
    assert q["matched_doc_id"] == "doc-xls-tariffs"
    assert q["sheet_name"] == "Тарифы"
    assert q["clause"] == "Лист: Тарифы"
    assert q["page"] == "1"
    assert "п. Лист:" not in q["source_label"]
    assert "Лист: Тарифы" in q["source_label"]

    agent_quotes = TableAwareRAGAgent._verify_answer_quotes(answer, sr)
    assert len(agent_quotes) == 1
    aq = agent_quotes[0]
    assert aq["verified"] is True
    assert aq["doc_id"] == "doc-xls-tariffs"
    assert aq["sheet_name"] == "Тарифы"
    assert aq["clause"] == "Лист: Тарифы"
    assert aq["page"] == "1"
    assert aq["page_num"] == 1
    assert "п. Лист:" not in aq["source_label"]

    # Also verify build_source_label directly does not emit "п. Лист:"
    lbl = build_source_label("Лимиты командировочных расходов", "Приказ № 12-Т", "Лист: Тарифы", "1")
    assert "п. Лист:" not in lbl
    assert "Лист: Тарифы" in lbl


def test_tool_trace_query_structured_table_enrichment_and_synthetic_source() -> None:
    passport = TablePassport(
        table_id="tbl-budget-7",
        doc_id="doc-xls-budget",
        file_name="budget_2025.xlsx",
        doc_number="№ 77-Б",
        doc_date="2025-03-01",
        file_format="xlsx",
        sheet_name="Свод",
        table_title="Бюджет подразделений",
        row_count=2,
        col_count=2,
        columns=[
            TableColumnSchema(name="Отдел", col_index=0),
            TableColumnSchema(name="Лимит", col_index=1),
        ],
    )
    tool_json = json.dumps(
        {
            "table_id": "tbl-budget-7",
            "table_title": "Бюджет подразделений",
            "columns": ["Отдел", "Лимит"],
            "rows": [{"Отдел": "ИТ", "Лимит": "500000"}],
            "matched_count": 1,
            "SQLite": "tables.db",
        },
        ensure_ascii=False,
    )
    sr = SearchResponse(
        query="Лимит ИТ",
        hits=[
            SearchHit(
                chunk_id="xls-b-1",
                doc_id="doc-xls-budget",
                file_name="budget_2025.xlsx",
                title="Сводный бюджет на 2025 год",
                doc_number="№ 77-Б",
                table_id="tbl-budget-7",
                page_numbers=[2],
                clean_text="Бюджет распределяется по подразделениям согласно таблице на листе Свод.",
                metadata={"sheet_name": "Свод"},
                score=0.87,
            )
        ],
        linked_tables=[passport],
    )

    # 1) Direct extra_texts JSON -> context_docs_from_search resolves real doc_id & strips raw JSON keys
    docs = context_docs_from_search(sr, extra_texts=[tool_json])
    tool_entries = [
        (doc, meta, text)
        for doc in docs
        for meta, text in zip(doc.metas, doc.texts)
        if meta.get("kind") == "tool"
    ]
    assert len(tool_entries) == 1
    tdoc, tmeta, ttext = tool_entries[0]
    assert tdoc.doc_id == "doc-xls-budget"
    assert tmeta.get("sheet_name") == "Свод"
    assert tmeta.get("pages") == [2]
    assert "table_id" not in ttext
    assert "matched_count" not in ttext
    assert "SQLite" not in ttext
    assert "Отдел: ИТ; Лимит: 500000" in ttext

    quote_answer = (
        "> «Отдел: ИТ; Лимит: 500000»\n"
        "> — Сводный бюджет на 2025 год, № 77-Б, Лист: Свод, стр. 2\n"
    )
    verified = verify_quotes(quote_answer, docs)
    assert len(verified) == 1
    assert verified[0]["verified"] is True
    assert verified[0]["matched_doc_id"] == "doc-xls-budget"
    assert verified[0]["sheet_name"] == "Свод"
    assert verified[0]["page"] == "2"
    assert "п. Лист:" not in verified[0]["source_label"]

    trace = ToolCallTrace(
        step=1,
        tool_name="query_structured_table",
        arguments={"table_id": "tbl-budget-7", "operation": "filter_rows"},
        matched_count=1,
        summary="Найдено 1 строк",
        result_preview="| Отдел | Лимит |\n| --- | --- |\n| ИТ | 500000 |",
    )
    direct_quotes = TableAwareRAGAgent._verify_answer_quotes(
        quote_answer,
        sr,
        messages=None,
        traces=[trace],
        linked_tables=[passport],
    )
    assert len(direct_quotes) == 1
    assert direct_quotes[0]["verified"] is True
    assert direct_quotes[0]["doc_id"] == "doc-xls-budget"
    assert direct_quotes[0]["sheet_name"] == "Свод"
    assert direct_quotes[0]["page_num"] == 2

    # 2) _verify_answer_quotes appends synthetic verified clickable source when query_structured_table
    # trace ran with matched_count > 0 but the LLM omitted a table blockquote for that table_id.
    prose_only_answer = (
        "### Краткий ответ\n"
        "Лимит для отдела ИТ составляет 500 000 руб.\n\n"
        "### Подробный ответ и цитаты\n"
        "> «Бюджет распределяется по подразделениям согласно таблице на листе Свод.»\n"
        "> — Сводный бюджет на 2025 год, № 77-Б, стр. 2\n"
    )
    quotes = TableAwareRAGAgent._verify_answer_quotes(
        prose_only_answer,
        sr,
        messages=None,
        traces=[trace],
        linked_tables=[passport],
    )
    synthetic = [q for q in quotes if q.get("clause") == "Лист: Свод"]
    assert len(synthetic) == 1
    assert synthetic[0]["verified"] is True
    assert synthetic[0]["doc_id"] == "doc-xls-budget"
    assert synthetic[0]["sheet_name"] == "Свод"
    assert synthetic[0]["page"] == "2"
    assert synthetic[0]["page_num"] == 2
    assert "table_id" not in synthetic[0]["source_label"]
    assert "п. Лист:" not in synthetic[0]["source_label"]

    # Also verify _hit_source_payload exposes sheet_name & table_id cleanly
    payload = TableAwareRAGAgent._hit_source_payload(sr.hits[0])
    assert payload["table_id"] == "tbl-budget-7"
    assert payload["sheet_name"] == "Свод"


def test_viewer_excel_clause_sheet_overrides_duplicate_quote_on_earlier_sheet(tmp_path: Path) -> None:
    store = StructuredTableStore(db_path=tmp_path / "viewer_tables.db")
    doc_id = "app_1_smeta_2019_xlsx"
    store.upsert_table(
        table_id="smeta::tbl_001",
        doc_id=doc_id,
        headers=["Наименование показателя", "Ставка", "Итого"],
        rows=[
            ["Резервы отпусков", "10%", "100000"],
            ["Прочие выплаты", "5%", "50000"],
        ],
        file_name="app_1_smeta_2019.xlsx",
        file_format="xlsx",
        sheet_name="смет_поквартально",
        table_title="Смета поквартально",
    )
    store.upsert_table(
        table_id="smeta::tbl_002",
        doc_id=doc_id,
        headers=["Наименование показателя", "Ставка", "Итого"],
        rows=[
            ["Резервы отпусков", "12%", "120000"],
            ["Командировки", "8%", "80000"],
        ],
        file_name="app_1_smeta_2019.xlsx",
        file_format="xlsx",
        sheet_name="смета_помесячно",
        table_title="Смета помесячно",
    )

    dummy_xlsx = tmp_path / "app_1_smeta_2019.xlsx"
    dummy_xlsx.write_bytes(b"PK\x03\x04")

    doc_meta = DocumentMeta(
        doc_id=doc_id,
        file_name="app_1_smeta_2019.xlsx",
        file_path=str(dummy_xlsx),
        file_hash="hash-smeta",
        file_format="xlsx",
        title="Приложение 1 Смета",
    )

    class FakeViewerPipeline:
        def __init__(self, t_store: StructuredTableStore, meta: DocumentMeta) -> None:
            self.table_store = t_store
            self._meta = meta

        def get_document(self, requested_doc_id: str) -> DocumentMeta | None:
            return self._meta if requested_doc_id == self._meta.doc_id else None

    pipeline = FakeViewerPipeline(store, doc_meta)

    # Duplicate quote appears on both sheets, but clause names sheet 2 -> must open sheet 2!
    resp = _build_document_view_sync(
        pipeline=pipeline,
        doc_id=doc_id,
        page=1,
        quote="Резервы отпусков",
        clause="Лист: смета_помесячно",
    )
    assert resp.total_pages == 2
    assert resp.target_page == 2
    assert resp.matched_quote_on_page is True
    assert resp.pages[1].label == "смета_помесячно"

    # Even when quote does not match sheet 2 text, explicit Лист clause pins target_page to sheet 2
    resp_no_quote_match = _build_document_view_sync(
        pipeline=pipeline,
        doc_id=doc_id,
        page=1,
        quote="Несуществующая строка таблицы",
        clause="Лист: смета_помесячно",
    )
    assert resp_no_quote_match.target_page == 2
    assert resp_no_quote_match.matched_quote_on_page is False
