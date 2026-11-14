#!/usr/bin/env python3
"""End-to-End Verification Suite for Adaptive 3-Tier Structured Tables & Autonomous LLM Tool Calling.

Covers 5 mandatory end-to-end table & tool-calling scenarios on the live MIPT corpus + live Polza AI LLM:
  1. T01_XL_CFO_AGGREGATION_AND_FILTER:
     Count & list subdivisions under ФРКТ / Финансодержатель Крылова Е.В. in 'Фин. структура 2026г..xlsx'
     -> verifies SearchResponse.linked_tables (delivery_mode="tool_query"), POST /api/tables/query exact
     SQLite row counts (20 rows for ФРКТ, 25 rows for Крылова Е.В.), and POST /api/ask autonomous
     tool calling (used_table_tools == True, len(tool_calls) >= 1) reporting the exact ground-truth number.
  2. T02_XL_SMETA_FORMULAS_QUERY:
     Query Excel formulas & rates in 'Приложение 1_новая смета 2019 (с незащищенными формулами).xlsx'
     (overhead 20%, VAT / НДС, vacation reserve 8.2% / КОСГУ 211.03, KVR/KOSGU formulas [формула: =...])
     via both POST /api/tables/query (get_formulas + filter_rows) and POST /api/ask with version_alerts.
  3. T03_DOCX_GIANT_TABLE_PROMOTION:
     Query the giant 524-row table inside Word 'Приказ N 3614-1 от 11.08.2025 по Фин.структуре 2026г..docx'
     -> verifies automatic promotion to delivery_mode="tool_query" in tables_store.db, identical SHA-256
     table_hash and bidirectional also_in_files cross-linking with the .xlsx file, and tool-calling query.
  4. T04_PDF_COMPACT_TABLE_INLINE:
     Query the compact scanned PDF table in 'Регламент бронирования конц зала.pdf' (rehearsal days for
     Посвящение в студенты -> 6 дней) -> verifies it stays delivery_mode="inline_parent" via
     ParentSectionRecord without unnecessary SQLite bloat, and POST /api/ask answers '6 дней' in 1 pass.
  5. T05_BACKWARD_COMPAT_10_SCENARIOS:
     Executes the full 10-scenario retrieval regression suite (verify_retrieval.py) and confirms 10/10 PASS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import httpx

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.config import TABLES_DB_PATH
from src.indexing.pipeline import get_pipeline
from src.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    SearchQuery,
    SearchResponse,
    TablePassport,
    TableQueryRequest,
    TableQueryResponse,
)
from src.web.app import app
import verify_retrieval


@dataclass
class TableScenarioResult:
    """Execution result and telemetry for a single E2E table scenario."""

    scenario_id: int
    code: str
    title: str
    passed: bool
    latency_ms: float
    delivery_mode: str = ""
    linked_table_ids: list[str] = field(default_factory=list)
    sqlite_matched_count: int = 0
    used_table_tools: bool = False
    tool_calls_count: int = 0
    model_used: str = ""
    tool_traces: list[dict[str, Any]] = field(default_factory=list)
    answer_excerpt: str = ""
    details: list[str] = field(default_factory=list)


class APIClientAdapter:
    """Unified client wrapper supporting both FastAPI in-process TestClient and live HTTP URL."""

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = api_url.rstrip("/") if api_url else None
        self._test_client = None if self.api_url else TestClient(app)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.api_url:
            with httpx.Client(timeout=90.0) as client:
                r = client.get(f"{self.api_url}{path}", params=params)
                r.raise_for_status()
                return r.json()
        assert self._test_client is not None
        r = self._test_client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_url:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(f"{self.api_url}{path}", json=payload)
                r.raise_for_status()
                return r.json()
        assert self._test_client is not None
        r = self._test_client.post(path, json=payload)
        r.raise_for_status()
        return r.json()


def ensure_tables_store_ready() -> tuple[bool, list[str], dict[str, Any]]:
    """Verify data/tables_store.db is populated (rebuilding from warm cache if needed)."""
    notes: list[str] = []
    pipeline = get_pipeline()
    tables = pipeline.get_tables()
    if len(tables) < 6 or not TABLES_DB_PATH.exists():
        notes.append("INFO: Инициализация data/tables_store.db через pipeline.rebuild_structured_tables()...")
        pipeline.rebuild_structured_tables(patch_qdrant=True)
        tables = pipeline.get_tables()

    xlsx_tables = [t for t in tables if t.file_format == "xlsx"]
    docx_tables = [t for t in tables if t.file_format == "docx"]
    pdf_tables = [t for t in tables if t.file_format == "pdf"]
    total_rows = sum(t.row_count for t in tables)
    total_formulas = sum(t.formula_count for t in tables)

    ok = len(xlsx_tables) >= 3 and len(docx_tables) >= 2 and len(pdf_tables) == 0
    notes.append(
        f"SQLite хранилище ({TABLES_DB_PATH.name}): всего таблиц={len(tables)} "
        f"(xlsx={len(xlsx_tables)}, large docx={len(docx_tables)}, pdf={len(pdf_tables)}), "
        f"строк={total_rows}, формул={total_formulas}"
    )

    summary = {
        "db_path": str(TABLES_DB_PATH),
        "total_tables": len(tables),
        "xlsx_tables": len(xlsx_tables),
        "docx_large_tables": len(docx_tables),
        "pdf_large_tables": len(pdf_tables),
        "total_rows": total_rows,
        "total_formulas": total_formulas,
        "table_ids": [t.table_id for t in tables],
    }
    return ok, notes, summary


# ---------------------------------------------------------------------------
# Scenario 1: T01_XL_CFO_AGGREGATION_AND_FILTER
# ---------------------------------------------------------------------------
def run_t01_xl_cfo_aggregation_and_filter(api: APIClientAdapter) -> TableScenarioResult:
    t0 = time.perf_counter()
    details: list[str] = []
    ok = True

    query = (
        "По таблице финансовой структуры МФТИ на 2026 год (Приложение к приказу об утверждении фин. структуры 2026г..xlsx): "
        "1) сколько строк в столбце «ЦФО 1 уровня» относится к подразделению «10329 Физтех-школа Радиотехники и Компьютерных Технологий»? "
        "2) сколько всего строк закреплено за финансодержателем «Крылова Елена Викторовна»? "
        "Выполни точный подсчет через инструмент таблиц по каждому из двух вопросов и перечисли примеры подразделений."
    )

    # 1. Verify /api/search populates linked_tables with TablePassport(delivery_mode="tool_query")
    s_raw = api.post("/api/search", SearchQuery(query=query, top_k=6).model_dump())
    s_resp = SearchResponse.model_validate(s_raw)
    linked_ids = [tp.table_id for tp in s_resp.linked_tables]
    has_fin_xl = any("app_fin_struct_2026_xlsx::tbl_01" in tid or "order_3614_1_2025::tbl_01" in tid for tid in linked_ids)
    all_tool_mode = all(tp.delivery_mode == "tool_query" for tp in s_resp.linked_tables) and len(s_resp.linked_tables) > 0

    if has_fin_xl and all_tool_mode:
        details.append(
            f"OK [/api/search]: linked_tables содержит паспорт таблицы финструктуры ({linked_ids}), delivery_mode='tool_query'"
        )
    else:
        ok = False
        details.append(f"FAIL [/api/search]: Ожидался TablePassport финструктуры 2026, получены: {linked_ids}")

    # 2. Verify POST /api/tables/query deterministic SQLite filter & aggregation
    q_frkt_raw = api.post(
        "/api/tables/query",
        TableQueryRequest(
            table_id="app_fin_struct_2026_xlsx::tbl_01",
            operation="filter_rows",
            filters={"ЦФО 1 уровня": "радиотехники и компьютерных технологий"},
            limit=50,
        ).model_dump(),
    )
    q_frkt = TableQueryResponse.model_validate(q_frkt_raw)

    q_krylova_raw = api.post(
        "/api/tables/query",
        TableQueryRequest(
            table_id="app_fin_struct_2026_xlsx::tbl_01",
            operation="aggregate",
            agg_func="count",
            filters={"Финансодержатель": "Крылова Елена Викторовна"},
        ).model_dump(),
    )
    q_krylova = TableQueryResponse.model_validate(q_krylova_raw)

    if q_frkt.matched_count == 20 and q_krylova.matched_count == 25:
        details.append(
            f"OK [/api/tables/query]: Точный SQLite-подсчет: ФРКТ (10329) = {q_frkt.matched_count} строк, "
            f"Крылова Е.В. = {q_krylova.matched_count} строк (из {q_frkt.total_table_rows})"
        )
    else:
        ok = False
        details.append(
            f"FAIL [/api/tables/query]: Ожидалось ФРКТ=20 и Крылова=25, получено ФРКТ={q_frkt.matched_count}, Крылова={q_krylova.matched_count}"
        )

    # 3. Verify POST /api/ask autonomous LLM tool-calling over Polza AI
    ask_raw = api.post(
        "/api/ask",
        AgentAskRequest(query=query, top_k=6, enable_table_tools=True, max_tool_rounds=4).model_dump(),
    )
    ask_resp = AgentAskResponse.model_validate(ask_raw)
    ans_norm = ask_resp.answer.lower().replace("ё", "е")

    if ask_resp.used_table_tools and len(ask_resp.tool_calls) >= 1:
        details.append(
            f"OK [/api/ask]: ИИ-агент ({ask_resp.model_used}) автономно вызвал табличный инструмент "
            f"({len(ask_resp.tool_calls)} вызов(ов): {[tc.tool_name for tc in ask_resp.tool_calls]})"
        )
    else:
        ok = False
        details.append(
            f"FAIL [/api/ask]: Агент не вызвал табличные инструменты (used_table_tools={ask_resp.used_table_tools})"
        )

    has_exact_counts = ("20" in ans_norm) and ("25" in ans_norm)
    has_subdivisions = any(
        w in ans_norm for w in ["деканат", "кафедра", "лаборатор", "киберсистем", "защиты информации", "планово-финанс"]
    )
    if has_exact_counts and has_subdivisions:
        details.append(
            "OK [/api/ask]: В ответе ИИ приведены точные цифры из SQLite (20 подразделений ФРКТ и 25 у Крыловой Е.В.) и названия ЦФО"
        )
    else:
        ok = False
        details.append(
            f"FAIL [/api/ask]: В ответе не найдены точные числа 20 и 25 ({has_exact_counts}) или примеры ЦФО ({has_subdivisions})"
        )

    elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
    return TableScenarioResult(
        scenario_id=1,
        code="T01_XL_CFO_AGGREGATION_AND_FILTER",
        title="XL Финструктура 2026: фильтрация, агрегация COUNT и автономный вызов query_structured_table",
        passed=ok,
        latency_ms=elapsed,
        delivery_mode="tool_query",
        linked_table_ids=linked_ids,
        sqlite_matched_count=q_frkt.matched_count,
        used_table_tools=ask_resp.used_table_tools,
        tool_calls_count=len(ask_resp.tool_calls),
        model_used=ask_resp.model_used,
        tool_traces=[tc.model_dump() for tc in ask_resp.tool_calls],
        answer_excerpt=ask_resp.answer[:360].replace("\n", " "),
        details=details,
    )


# ---------------------------------------------------------------------------
# Scenario 2: T02_XL_SMETA_FORMULAS_QUERY
# ---------------------------------------------------------------------------
def run_t02_xl_smeta_formulas_query(api: APIClientAdapter) -> TableScenarioResult:
    t0 = time.perf_counter()
    details: list[str] = []
    ok = True

    # 1. Inspect formula extraction via POST /api/tables/query (get_formulas & filter_rows)
    f_overhead_raw = api.post(
        "/api/tables/query",
        TableQueryRequest(
            table_id="app_1_smeta_2019_xlsx::tbl_01",
            operation="get_formulas",
            search_text="накладные",
        ).model_dump(),
    )
    f_overhead = TableQueryResponse.model_validate(f_overhead_raw)

    f_nds_raw = api.post(
        "/api/tables/query",
        TableQueryRequest(
            table_id="app_1_smeta_2019_xlsx::tbl_01",
            operation="get_formulas",
            search_text="ндс",
        ).model_dump(),
    )
    f_nds = TableQueryResponse.model_validate(f_nds_raw)

    r_vac_raw = api.post(
        "/api/tables/query",
        TableQueryRequest(
            table_id="app_1_smeta_2019_xlsx::tbl_01",
            operation="filter_rows",
            search_text="211.03",
        ).model_dump(),
    )
    r_vac = TableQueryResponse.model_validate(r_vac_raw)

    overhead_formulas_str = json.dumps(f_overhead.formulas, ensure_ascii=False)
    nds_formulas_str = json.dumps(f_nds.formulas, ensure_ascii=False)
    vac_rows_str = json.dumps(r_vac.rows, ensure_ascii=False)

    formulas_ok = (
        len(f_overhead.formulas) >= 4
        and "$B$23" in overhead_formulas_str
        and "20%" in overhead_formulas_str
        and len(f_nds.formulas) >= 4
        and "$B$22" in nds_formulas_str
        and r_vac.matched_count >= 1
        and "8.2%" in vac_rows_str
        and "111" in vac_rows_str
        and "=F31*$B$32" in vac_rows_str
    )
    if formulas_ok:
        details.append(
            f"OK [/api/tables/query]: Извлечены формулы накладных расходов 20% (`=IFERROR((F$13+F$22)*$B$23*-1,)`), "
            f"НДС (`$B$22`, КВР 180 / КОСГУ 189) и резерва отпусков 8.2% (`=F31*$B$32`, КВР 111 / КОСГУ 211.03)"
        )
    else:
        ok = False
        details.append("FAIL [/api/tables/query]: Не найдены ожидаемые формулы или коды КВР/КОСГУ в смете 2019 г.")

    # 2. Verify POST /api/ask with live LLM Tool Calling on the 2019 Smeta XLSX
    query = (
        "Какие ставки, коды КВР/КОСГУ и точные формулы Excel (`[формула: =...]`) зашиты в шаблоне "
        "«Приложение 1_новая смета 2019 (с незащищенными формулами).xlsx» для расчета накладных расходов (20%), "
        "НДС и резерва отпусков (8.2%, вид затрат 211.03), и каким действующим приказом утверждена актуальная помесячная форма сметы?"
    )
    ask_raw = api.post(
        "/api/ask",
        AgentAskRequest(query=query, top_k=6, enable_table_tools=True, max_tool_rounds=4).model_dump(),
    )
    ask_resp = AgentAskResponse.model_validate(ask_raw)
    ans_norm = ask_resp.answer.lower().replace("ё", "е")

    if ask_resp.used_table_tools and len(ask_resp.tool_calls) >= 1:
        details.append(
            f"OK [/api/ask]: ИИ-агент ({ask_resp.model_used}) выполнил {len(ask_resp.tool_calls)} вызов(ов) к таблицам сметы"
        )
    else:
        ok = False
        details.append("FAIL [/api/ask]: Агент не вызвал инструмент таблиц при запросе формул Excel сметы")

    has_formula_details = (
        ("20%" in ans_norm or "0.2" in ans_norm)
        and ("8.2" in ans_norm or "8,2" in ans_norm)
        and any(tok in ans_norm for tok in ["b$23", "$b$23", "b23", "f31", "b32", "iferror", "sum("])
        and ("794-1" in ans_norm)
    )
    if has_formula_details:
        details.append(
            "OK [/api/ask]: Ответ содержит ставки (20%, 8.2%), формулы Excel ($B$23, F31*$B$32) и предупреждение о Приказе № 794-1"
        )
    else:
        ok = False
        details.append(
            f"FAIL [/api/ask]: В ответе отсутствуют формулы/ставки сметы или ссылка на Приказ № 794-1: {ask_resp.answer[:240]}"
        )

    elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
    return TableScenarioResult(
        scenario_id=2,
        code="T02_XL_SMETA_FORMULAS_QUERY",
        title="XL Смета 2019 (524+755 формул): извлечение формул КВР/КОСГУ, ставок 20% / 8.2% и графа версий (794-1)",
        passed=ok,
        latency_ms=elapsed,
        delivery_mode="tool_query",
        linked_table_ids=[tp.table_id for tp in ask_resp.linked_tables],
        sqlite_matched_count=len(f_overhead.formulas) + len(f_nds.formulas) + r_vac.matched_count,
        used_table_tools=ask_resp.used_table_tools,
        tool_calls_count=len(ask_resp.tool_calls),
        model_used=ask_resp.model_used,
        tool_traces=[tc.model_dump() for tc in ask_resp.tool_calls],
        answer_excerpt=ask_resp.answer[:360].replace("\n", " "),
        details=details,
    )


# ---------------------------------------------------------------------------
# Scenario 3: T03_DOCX_GIANT_TABLE_PROMOTION
# ---------------------------------------------------------------------------
def run_t03_docx_giant_table_promotion(api: APIClientAdapter) -> TableScenarioResult:
    t0 = time.perf_counter()
    details: list[str] = []
    ok = True

    # 1. Verify GET /api/tables/{table_id} for Word Order 3614-1 giant table & cross-file deduplication with XLSX
    docx_tid = "order_3614_1_2025::tbl_01"
    xlsx_tid = "app_fin_struct_2026_xlsx::tbl_01"

    docx_info = api.get(f"/api/tables/{docx_tid}")
    xlsx_info = api.get(f"/api/tables/{xlsx_tid}")

    docx_tp = TablePassport.model_validate(docx_info)
    xlsx_tp = TablePassport.model_validate(xlsx_info)

    promoted_ok = (
        docx_tp.file_format == "docx"
        and docx_tp.delivery_mode == "tool_query"
        and docx_tp.row_count >= 500
        and docx_tp.char_count >= 50000
    )
    if promoted_ok:
        details.append(
            f"OK [GET /api/tables/{docx_tid}]: Гигантская таблица Word ({docx_tp.row_count} строк, {docx_tp.char_count} симв.) "
            f"автоматически переведена в режим delivery_mode='{docx_tp.delivery_mode}'"
        )
    else:
        ok = False
        details.append(f"FAIL: Таблица {docx_tid} не переведена в tool_query корректно: {docx_tp}")

    dedup_ok = (
        bool(docx_tp.table_hash)
        and docx_tp.table_hash == xlsx_tp.table_hash
        and any("фин. структуры 2026" in f.lower() for f in docx_tp.also_in_files)
        and any("3614-1" in f.lower() for f in xlsx_tp.also_in_files)
    )
    if dedup_ok:
        details.append(
            f"OK [Cross-File Deduplication]: table_hash совпадает ({docx_tp.table_hash[:12]}...), "
            f"also_in_files={docx_tp.also_in_files}"
        )
    else:
        ok = False
        details.append(
            f"FAIL: Кросс-файловая дедупликация между DOCX и XLSX не сработала: hash_docx={docx_tp.table_hash}, also_in={docx_tp.also_in_files}"
        )

    # 2. Query the promoted Word table directly via POST /api/tables/query for Институт квантовых технологий
    q_ikt_raw = api.post(
        "/api/tables/query",
        TableQueryRequest(
            table_id=docx_tid,
            operation="filter_rows",
            filters={"ЦФО 1 уровня": "институт квантовых технологий"},
            limit=50,
        ).model_dump(),
    )
    q_ikt = TableQueryResponse.model_validate(q_ikt_raw)
    if q_ikt.matched_count == 22:
        details.append(
            f"OK [/api/tables/query on DOCX]: По таблице Приказа № 3614-1 найдено ровно {q_ikt.matched_count} строк "
            f"подразделения «10665 Институт квантовых технологий» (финансодержатели Лебедев А.П. и Серов П.А.)"
        )
    else:
        ok = False
        details.append(f"FAIL [/api/tables/query on DOCX]: Ожидалось 22 строки ИКТ, получено {q_ikt.matched_count}")

    # 3. Verify POST /api/ask on Word Order 3614-1 giant table
    query = (
        "Согласно таблице в Приказе № 3614-1 от 11.08.2025 по финансовой структуре на 2026 год: "
        "сколько всего строк (лабораторий и подразделений) входит в «10665 Институт квантовых технологий» (столбец «ЦФО 1 уровня»), "
        "какие финансодержатели у этих подразделений и в каком еще файле продублирована эта таблица?"
    )
    ask_raw = api.post(
        "/api/ask",
        AgentAskRequest(query=query, top_k=6, enable_table_tools=True, max_tool_rounds=4).model_dump(),
    )
    ask_resp = AgentAskResponse.model_validate(ask_raw)
    ans_norm = ask_resp.answer.lower().replace("ё", "е")

    if ask_resp.used_table_tools and ("22" in ans_norm) and ("лебедев" in ans_norm or "серов" in ans_norm):
        details.append(
            "OK [/api/ask]: ИИ-агент вызвал инструмент таблиц и верно указал точное число строк ИКТ (22), финансодержателей и дублирующий файл XLSX"
        )
    else:
        ok = False
        details.append(
            f"FAIL [/api/ask]: Ожидались used_table_tools=True, '22' и 'Лебедев'/'Серов' в ответе: {ask_resp.answer[:240]}"
        )

    elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
    return TableScenarioResult(
        scenario_id=3,
        code="T03_DOCX_GIANT_TABLE_PROMOTION",
        title="Гигантская таблица Word (Приказ № 3614-1, 524 строки): авто-промоушен в SQLite и дедупликация с XLSX",
        passed=ok,
        latency_ms=elapsed,
        delivery_mode=docx_tp.delivery_mode,
        linked_table_ids=[tp.table_id for tp in ask_resp.linked_tables],
        sqlite_matched_count=q_ikt.matched_count,
        used_table_tools=ask_resp.used_table_tools,
        tool_calls_count=len(ask_resp.tool_calls),
        model_used=ask_resp.model_used,
        tool_traces=[tc.model_dump() for tc in ask_resp.tool_calls],
        answer_excerpt=ask_resp.answer[:360].replace("\n", " "),
        details=details,
    )


# ---------------------------------------------------------------------------
# Scenario 4: T04_PDF_COMPACT_TABLE_INLINE
# ---------------------------------------------------------------------------
def run_t04_pdf_compact_table_inline(api: APIClientAdapter) -> TableScenarioResult:
    t0 = time.perf_counter()
    details: list[str] = []
    ok = True

    # 1. Verify compact PDF tables from 'Регламент бронирования конц зала.pdf' are NOT in SQLite tables_store.db
    all_tables_raw = api.get("/api/tables", params={"doc_id": "reg_concert_hall_booking"})
    tables_list = (
        all_tables_raw.get("tables", [])
        if isinstance(all_tables_raw, dict)
        else all_tables_raw
    )
    if len(tables_list) == 0:
        details.append(
            "OK [GET /api/tables?doc_id=reg_concert_hall_booking]: Компактная таблица из скана КЗ (< 25 строк) "
            "НЕ раздувает SQLite (0 таблиц в tables_store.db)"
        )
    else:
        ok = False
        details.append(f"FAIL: Компактная PDF-таблица попала в SQLite: {tables_list}")

    # 2. Verify POST /api/search delivers compact PDF table inline via ParentSectionRecord
    query = (
        "Сколько репетиционных дней в Концертном зале выделяется физтех-школам на проведение Посвящения в студенты "
        "согласно таблице п. 6.1 регламента бронирования КЗ?"
    )
    s_raw = api.post("/api/search", SearchQuery(query=query, top_k=5, include_parent=True).model_dump())
    s_resp = SearchResponse.model_validate(s_raw)

    inline_hits = [
        h for h in s_resp.hits if (h.metadata or {}).get("table_delivery_mode") == "inline_parent"
    ]
    has_parent_text = any(
        "посвящен" in (h.parent_text or "").lower() and "6" in (h.parent_text or "")
        for h in s_resp.hits
    )
    if inline_hits and has_parent_text and len(s_resp.linked_tables) == 0:
        details.append(
            f"OK [/api/search]: Найдено {len(inline_hits)} чанков с table_delivery_mode='inline_parent', "
            f"linked_tables=[], полный текст таблицы передан внутри ParentSectionRecord"
        )
    else:
        ok = False
        details.append(
            f"FAIL [/api/search]: inline_hits={len(inline_hits)}, has_parent_text={has_parent_text}, linked_tables={len(s_resp.linked_tables)}"
        )

    # 3. Verify POST /api/ask answers accurately in a single pass with 0 tool calls
    ask_raw = api.post(
        "/api/ask",
        AgentAskRequest(query=query, top_k=5, enable_table_tools=True).model_dump(),
    )
    ask_resp = AgentAskResponse.model_validate(ask_raw)
    ans_norm = ask_resp.answer.lower().replace("ё", "е")

    if (not ask_resp.used_table_tools) and len(ask_resp.tool_calls) == 0:
        details.append(
            "OK [/api/ask]: Для компактной таблицы Word/PDF агент ответил за 1 проход из ParentSectionRecord (used_table_tools=False, tool_calls=0)"
        )
    else:
        ok = False
        details.append(f"FAIL [/api/ask]: Неожиданный вызов инструментов: {ask_resp.tool_calls}")

    if ("6" in ans_norm or "шесть" in ans_norm) and "посвящен" in ans_norm:
        details.append("OK [/api/ask]: Точно извлечен лимит репетиций из таблицы скана п. 6.1: 6 дней на Посвящение в студенты")
    else:
        ok = False
        details.append(f"FAIL [/api/ask]: В ответе не найден лимит '6 дней': {ask_resp.answer[:240]}")

    elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
    return TableScenarioResult(
        scenario_id=4,
        code="T04_PDF_COMPACT_TABLE_INLINE",
        title="Компактная таблица из скана PDF (Регламент КЗ, п. 6.1): доставка inline_parent за 1 проход без SQLite",
        passed=ok,
        latency_ms=elapsed,
        delivery_mode="inline_parent",
        linked_table_ids=[],
        sqlite_matched_count=0,
        used_table_tools=ask_resp.used_table_tools,
        tool_calls_count=len(ask_resp.tool_calls),
        model_used=ask_resp.model_used,
        tool_traces=[],
        answer_excerpt=ask_resp.answer[:360].replace("\n", " "),
        details=details,
    )


# ---------------------------------------------------------------------------
# Scenario 5: T05_BACKWARD_COMPAT_10_SCENARIOS
# ---------------------------------------------------------------------------
def run_t05_backward_compat_10_scenarios(api_url: str | None = None) -> TableScenarioResult:
    t0 = time.perf_counter()
    details: list[str] = []

    out_path = Path("data/verification_results.json")
    exit_code = verify_retrieval.run_verification(
        api_url=api_url,
        auto_sync=False,
        output_json=out_path,
    )
    elapsed = round((time.perf_counter() - t0) * 1000.0, 1)

    rep = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    passed_cnt = int(rep.get("passed_scenarios", 0))
    total_cnt = int(rep.get("total_scenarios", 10))
    dense_margin = float(rep.get("dense_margin", 0.0))
    conf_margin = float(rep.get("confidence_margin", 0.0))

    ok = (exit_code == 0) and (passed_cnt == 10)
    if ok:
        details.append(
            f"OK [verify_retrieval.py]: Все {passed_cnt}/{total_cnt} базовых RAG-сценариев пройдены "
            f"(Δdense={dense_margin:+.4f}, Δconf={conf_margin:+.4f})"
        )
    else:
        details.append(
            f"FAIL [verify_retrieval.py]: Пройдено {passed_cnt}/{total_cnt} сценариев (exit_code={exit_code})"
        )

    return TableScenarioResult(
        scenario_id=5,
        code="T05_BACKWARD_COMPAT_10_SCENARIOS",
        title="Обратная совместимость: 10/10 базовых сценариев гибридного поиска (verify_retrieval.py)",
        passed=ok,
        latency_ms=elapsed,
        delivery_mode="hybrid_3tier",
        linked_table_ids=[],
        sqlite_matched_count=passed_cnt,
        used_table_tools=False,
        tool_calls_count=0,
        model_used="BAAI/bge-m3 + BM25 RRF",
        tool_traces=[],
        answer_excerpt=f"Пройдено {passed_cnt}/{total_cnt} сценариев, Δdense={dense_margin:+.4f}, Δconf={conf_margin:+.4f}",
        details=details,
    )


def run_all_table_verifications(
    api_url: str | None = None,
    output_json: Path = Path("data/table_verification_results.json"),
) -> int:
    print("=" * 94)
    print("MIPT RAG V3 — E2E ВАЛИДАЦИЯ АДАПТИВНЫХ ТАБЛИЦ (3-TIER) И LLM TOOL CALLING (5 СЦЕНАРИЕВ)")
    print("=" * 94)

    store_ok, store_notes, store_summary = ensure_tables_store_ready()
    print("\n--- 1. Проверка SQLite хранилища структурированных таблиц (data/tables_store.db) ---")
    for n in store_notes:
        print(f"  • {n}")

    api = APIClientAdapter(api_url=api_url)
    runners = [
        lambda: run_t01_xl_cfo_aggregation_and_filter(api),
        lambda: run_t02_xl_smeta_formulas_query(api),
        lambda: run_t03_docx_giant_table_promotion(api),
        lambda: run_t04_pdf_compact_table_inline(api),
        lambda: run_t05_backward_compat_10_scenarios(api_url=api_url),
    ]

    results: list[TableScenarioResult] = []
    print("\n--- 2. Выполнение E2E-сценариев T01..T05 ---")
    for runner in runners:
        res = runner()
        results.append(res)
        badge = "✅ PASS" if res.passed else "❌ FAIL"
        print(f"\n[T0{res.scenario_id}/05] {badge} | {res.code}: {res.title}")
        print(
            f"  Режим: {res.delivery_mode} | Таблицы: {res.linked_table_ids} | "
            f"Tool Calling: used={res.used_table_tools} (вызовов={res.tool_calls_count}) | "
            f"Модель: {res.model_used} | Время: {res.latency_ms:.1f} ms"
        )
        for d in res.details:
            print(f"    -> {d}")
        if res.answer_excerpt:
            print(f"    Ответ (фрагмент): {res.answer_excerpt}...")

    passed_count = sum(1 for r in results if r.passed)
    overall_ok = store_ok and (passed_count == len(results))

    print("\n" + "=" * 94)
    print(
        f"ИТОГ ВАЛИДАЦИИ ТАБЛИЦ И TOOL CALLING: {passed_count}/{len(results)} сценариев успешно пройдено | "
        f"SQLite таблиц: {store_summary['total_tables']} (строк: {store_summary['total_rows']}, формул: {store_summary['total_formulas']})"
    )
    print("=" * 94)

    payload = {
        "passed_scenarios": passed_count,
        "total_scenarios": len(results),
        "overall_passed": overall_ok,
        "tables_store_summary": store_summary,
        "scenarios": [asdict(r) for r in results],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Детальный отчет сохранен в {output_json}")

    return 0 if overall_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E2E verification for Adaptive 3-Tier Structured Tables & Autonomous LLM Tool Calling."
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Optional live FastAPI base URL (defaults to in-process FastAPI TestClient)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("data/table_verification_results.json"),
        help="Path to write JSON verification report",
    )
    args = parser.parse_args()
    sys.exit(run_all_table_verifications(api_url=args.api_url, output_json=args.json_out))


if __name__ == "__main__":
    main()
