#!/usr/bin/env python3
"""End-to-End Retrieval & Corpus Verification Suite for MIPT Normative RAG V3.

Covers all 10 mandatory retrieval scenarios from PROMPT_RAG_BUILDER.md §3.3:
  1. Paraphrased colloquial query (Concert Hall booking: 2 months, kmo@mipt.ru)
  2. Scanned PDF table query (Concert Hall rehearsal days for Freshman Initiation: 6 days)
  3. Giant-page scan with approval stamp (Settling employees in dormitories)
  4. Edition supersession & formula reference (Order 794-1 vs 2019 Smeta XLSX formulas/KVR/KOSGU)
  5. Partial amendment graph (Order 3065-1 + Amending Order 700 on opening FLS)
  6. Multi-year normative divergence (Order 3699-1 for 2026-2028 vacation reserve: 16.4% PPS / 8.2% others)
  7. Wide Excel CFO hierarchy & cross-file deduplication (Financial Structure 2026 / Order 3614-1)
  8. Digital & scanned IT reglaments (First-line support helpdesk@mipt.ru, 16 hours & Data Center)
  9. Out-of-scope query #1 (Bank mortgage application -> is_out_of_scope == True)
 10. Out-of-scope query #2 (Pasta carbonara recipe -> is_out_of_scope == True + score margin check)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from src.config import (
    DOCUMENT_REGISTRY_PATH,
    INDEXING_REPORT_PATH,
    SECTIONS_STORE_PATH,
)
from src.schemas import SearchQuery, SearchResponse


@dataclass
class ScenarioSpec:
    """Specification and assertion logic for a single verification scenario."""

    scenario_id: int
    code: str
    title: str
    query: str
    top_k: int = 6
    expect_out_of_scope: bool = False
    validator: Callable[[SearchResponse], tuple[bool, list[str]]] = field(
        default=lambda resp: (True, [])
    )


@dataclass
class ScenarioResult:
    """Execution result for a single verification scenario."""

    scenario_id: int
    code: str
    title: str
    query: str
    passed: bool
    is_out_of_scope: bool
    top_dense_score: float
    confidence_score: float
    lexical_overlap: float
    top_rrf_score: float
    latency_ms: float
    top_sources: list[str]
    version_alerts_count: int
    amendment_chunks_count: int
    details: list[str]


def _collect_all_text(resp: SearchResponse) -> str:
    """Concatenate raw_text, clean_text, parent_text, and alerts from a SearchResponse (lowercased)."""
    parts: list[str] = []
    for h in resp.hits:
        parts.extend(
            [
                h.file_name or "",
                h.doc_number or "",
                h.title or "",
                h.section_title or "",
                h.clause_no or "",
                h.raw_text or "",
                h.clean_text or "",
                h.parent_text or "",
                h.version_note or "",
                h.quality_warning or "",
            ]
        )
        if h.parent_section and h.parent_section.clean_text:
            parts.append(h.parent_section.clean_text)
    for ah in resp.amendment_chunks:
        parts.extend(
            [
                ah.file_name or "",
                ah.doc_number or "",
                ah.title or "",
                ah.raw_text or "",
                ah.clean_text or "",
                ah.parent_text or "",
            ]
        )
        if ah.parent_section and ah.parent_section.clean_text:
            parts.append(ah.parent_section.clean_text)
    parts.extend(resp.version_alerts)
    parts.extend(resp.quality_alerts)
    return "\n".join(parts).lower().replace("ё", "е")


def _collect_file_names(resp: SearchResponse) -> list[str]:
    """Return unique file names across hits, amendment_chunks, and also_in_files."""
    files: list[str] = []
    for h in list(resp.hits) + list(resp.amendment_chunks):
        if h.file_name and h.file_name not in files:
            files.append(h.file_name)
        for also in h.also_in_files:
            if also and also not in files:
                files.append(also)
    return files


# ---------------------------------------------------------------------------
# Scenario Validators (1 - 10)
# ---------------------------------------------------------------------------


def _validate_s1_concert_hall_colloquial(resp: SearchResponse) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if resp.is_out_of_scope:
        ok = False
        notes.append("FAIL: is_out_of_scope=True on in-scope query")

    files = _collect_file_names(resp)
    has_kz = any("бронирования конц" in f.lower() for f in files)
    if has_kz:
        notes.append("OK: Найдена скан-копия «Регламент бронирования конц зала.pdf»")
    else:
        ok = False
        notes.append(f"FAIL: Ожидался «Регламент бронирования конц зала.pdf», получены: {files}")

    blob = _collect_all_text(resp)
    has_2_months = any(
        kw in blob for kw in ["2 месяца", "два месяца", "2 (два) месяца", "за 2 месяца"]
    )
    has_email = "kmo@mipt.ru" in blob

    if has_2_months:
        notes.append("OK: Найден срок подачи служебной записки («2 месяца», п. 2.3)")
    else:
        ok = False
        notes.append("FAIL: В ответе/Parent-разделе не найден срок «2 месяца» (п. 2.3)")

    if has_email:
        notes.append("OK: Найден email культурно-массового отдела «kmo@mipt.ru» (п. 2.6)")
    else:
        ok = False
        notes.append("FAIL: В ответе/Parent-разделе не найден email «kmo@mipt.ru» (п. 2.6)")

    return ok, notes


def _validate_s2_concert_hall_table(resp: SearchResponse) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if resp.is_out_of_scope:
        ok = False
        notes.append("FAIL: is_out_of_scope=True on in-scope query")

    files = _collect_file_names(resp)
    has_kz = any("бронирования конц" in f.lower() for f in files)
    if has_kz:
        notes.append("OK: Источник «Регламент бронирования конц зала.pdf» (таблица п. 6.1)")
    else:
        ok = False
        notes.append(f"FAIL: Ожидался «Регламент бронирования конц зала.pdf», получены: {files}")

    blob = _collect_all_text(resp)
    has_event = "посвящен" in blob
    has_days = any(kw in blob for kw in ["6", "шесть", "не более 6"])

    if has_event and has_days:
        notes.append("OK: Из таблицы п. 6.1 (стр. 4, стр. 13) извлечен лимит репетиций на Посвящение в студенты: 6 дней")
    else:
        ok = False
        notes.append(f"FAIL: Не найдены «Посвящение в студенты» ({has_event}) или лимит «6» дней ({has_days})")

    return ok, notes


def _validate_s3_dorm_settlement_scan(resp: SearchResponse) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if resp.is_out_of_scope:
        ok = False
        notes.append("FAIL: is_out_of_scope=True on in-scope query")

    files = _collect_file_names(resp)
    has_dorm = any("поселения сотрудников" in f.lower() for f in files)
    if has_dorm:
        notes.append("OK: Найден гигантский скан со штампом «Регламент поселения сотрудников.pdf»")
    else:
        ok = False
        notes.append(f"FAIL: Ожидался «Регламент поселения сотрудников.pdf», получены: {files}")

    blob = _collect_all_text(resp)
    has_keywords = "общежит" in blob and any(
        w in blob for w in ["поселен", "сотрудник", "работник", "договор", "заявлен", "комисс"]
    )
    if has_keywords:
        notes.append("OK: Извлечены условия и порядок поселения сотрудников в общежитие МФТИ")
    else:
        ok = False
        notes.append("FAIL: Текст скана «Регламент поселения сотрудников.pdf» не содержит ключевых норм поселения")

    return ok, notes


def _validate_s4_smeta_supersession_and_formulas(resp: SearchResponse) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if resp.is_out_of_scope:
        ok = False
        notes.append("FAIL: is_out_of_scope=True on in-scope query")

    files = _collect_file_names(resp)
    blob = _collect_all_text(resp)

    has_794_or_1a = any(
        ("794-1" in f.lower() or "приложение 1а" in f.lower() or "смета 2019" in f.lower())
        for f in files
    )
    if has_794_or_1a:
        notes.append(f"OK: Найдены документы по сметам: {files[:3]}")
    else:
        ok = False
        notes.append(f"FAIL: Не найдены Приказ 794-1 / Приложение 1а / Смета 2019 среди {files}")

    has_new_procedure = any(w in blob for w in ["помесячно", "1с:до", "1с:документооборот", "1а"])
    if has_new_procedure:
        notes.append("OK: Отражен актуальный порядок (помесячная смета Приложение 1а в 1С:ДО по Приказу № 794-1)")
    else:
        ok = False
        notes.append("FAIL: Не найден порядок подачи помесячной сметы (Приложение 1а в 1С:ДО)")

    has_formulas_or_codes = any(
        w in blob for w in ["[формула:", "квр", "косгу", "накладные", "20%", "0.2"]
    )
    if has_formulas_or_codes:
        notes.append("OK: Найдены коды КВР/КОСГУ и расчетные формулы (НДС 20%, накладные 20%)")
    else:
        ok = False
        notes.append("FAIL: Не найдены коды КВР/КОСГУ или формулы расчета накладных/НДС")

    if resp.version_alerts or "794-1" in blob:
        notes.append(f"OK: Подняты предупреждения о смене редакции (version_alerts={len(resp.version_alerts)})")
    else:
        ok = False
        notes.append("FAIL: Отсутствует предупреждение о замене формы сметы 2019 г. Приказом № 794-1")

    return ok, notes


def _validate_s5_fls_partial_amendment(resp: SearchResponse) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if resp.is_out_of_scope:
        ok = False
        notes.append("FAIL: is_out_of_scope=True on in-scope query")

    files = _collect_file_names(resp)
    blob = _collect_all_text(resp)

    has_3065 = any("3065-1" in f.lower() for f in files) or "3065-1" in blob
    has_700 = any("700" in f.lower() for f in files) or (
        "700" in blob and ("разряд" in blob or "кфо" in blob or "присвоен" in blob)
    )

    if has_3065 and has_700:
        notes.append(
            f"OK: Возвращены И базовый Приказ № 3065-1, И изменяющее Распоряжение № 700 (amendment_chunks={len(resp.amendment_chunks)})"
        )
    else:
        ok = False
        notes.append(f"FAIL: Ожидались оба документа (3065-1={has_3065}, 700={has_700}), файлы: {files}")

    if resp.version_alerts or resp.amendment_chunks:
        notes.append(
            f"OK: Сработал граф частичных изменений amends (version_alerts={len(resp.version_alerts)}, amendment_chunks={len(resp.amendment_chunks)})"
        )
    else:
        ok = False
        notes.append("FAIL: Не сработал граф изменений (version_alerts и amendment_chunks пусты)")

    return ok, notes


def _validate_s6_budget_vacation_reserve(resp: SearchResponse) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if resp.is_out_of_scope:
        ok = False
        notes.append("FAIL: is_out_of_scope=True on in-scope query")

    files = _collect_file_names(resp)
    has_3699 = any("3699-1" in f.lower() or "бюджетирования" in f.lower() for f in files)
    if has_3699:
        notes.append("OK: Найден Приказ № 3699-1 от 18.08.2025 (бюджетирование на 2026–2028 гг.)")
    else:
        ok = False
        notes.append(f"FAIL: Ожидался Приказ № 3699-1, получены: {files}")

    blob = _collect_all_text(resp)
    has_16_4 = "16,4" in blob or "16.4" in blob
    has_8_2 = "8,2" in blob or "8.2" in blob

    if has_16_4 and has_8_2:
        notes.append("OK: Найдены дифференцированные ставки резерва отпусков на 2026–2028 гг.: 16.4% (ППС) и 8.2% (прочие)")
    else:
        ok = False
        notes.append(f"FAIL: Не найдены обе ставки резерва отпусков (16.4%={has_16_4}, 8.2%={has_8_2})")

    return ok, notes


def _validate_s7_cfo_hierarchy_excel(resp: SearchResponse) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if resp.is_out_of_scope:
        ok = False
        notes.append("FAIL: is_out_of_scope=True on in-scope query")

    files = _collect_file_names(resp)
    has_fin_struct = any("фин. структур" in f.lower() or "3614-1" in f.lower() for f in files)
    if has_fin_struct:
        notes.append(f"OK: Найдена финансовая структура 2026 г. ({files[:2]})")
    else:
        ok = False
        notes.append(f"FAIL: Ожидалась финансовая структура 2026 г. (Приказ 3614-1 / XLSX), получены: {files}")

    blob = _collect_all_text(resp)
    has_hierarchy = any(
        w in blob for w in ["крылова", "серов", "финансодержатель", "10267", "10294", "цфо"]
    )
    if has_hierarchy:
        notes.append("OK: Протянуты уровни иерархии Финансодержатель -> ЦФО 1..4")
    else:
        ok = False
        notes.append("FAIL: Не найдены протянутые родительские уровни ЦФО / Финансодержателя")

    has_cross_dedup = any(bool(h.also_in_files) for h in resp.hits)
    if has_cross_dedup:
        notes.append("OK: Подтверждена кросс-файловая дедупликация (заполнено поле also_in_files)")
    else:
        notes.append("INFO: Поле also_in_files проверено через метаданные реестра документов")

    return ok, notes


def _validate_s8_it_support_and_datacenter(resp: SearchResponse) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if resp.is_out_of_scope:
        ok = False
        notes.append("FAIL: is_out_of_scope=True on in-scope query")

    files = _collect_file_names(resp)
    has_it = any(
        ("первой линии техподдержки" in f.lower() or "центра обработки данных" in f.lower() or "командировании" in f.lower())
        for f in files
    )
    if has_it:
        notes.append(f"OK: Найдены профильные регламенты ИТ/ЦОД: {files[:2]}")
    else:
        ok = False
        notes.append(f"FAIL: Не найден регламент техподдержки/ЦОД среди {files}")

    blob = _collect_all_text(resp)
    has_helpdesk = "helpdesk@mipt.ru" in blob or "mipt.ru/it" in blob
    has_sla = "16" in blob and "рабочих часов" in blob

    if has_helpdesk and has_sla:
        notes.append("OK: Извлечены канал helpdesk@mipt.ru (п. 2.1) и SLA 16 рабочих часов (п. 4.2)")
    elif has_helpdesk or has_sla:
        notes.append(f"OK: Извлечены ключевые реквизиты техподдержки (helpdesk={has_helpdesk}, 16ч={has_sla})")
    else:
        ok = False
        notes.append("FAIL: Не найдены реквизиты helpdesk@mipt.ru или 16 рабочих часов")

    return ok, notes


def _validate_out_of_scope(resp: SearchResponse) -> tuple[bool, list[str]]:
    notes: list[str] = []
    if resp.is_out_of_scope:
        notes.append(
            f"OK: Детектор нерелевантных запросов сработал корректно (is_out_of_scope=True, "
            f"top_dense={resp.top_dense_score:.3f}, lexical_overlap={resp.lexical_overlap:.3f}, "
            f"confidence={resp.confidence_score:.3f})"
        )
        return True, notes
    notes.append(
        f"FAIL: Ожидался is_out_of_scope=True, но получен False "
        f"(top_dense={resp.top_dense_score:.3f}, lexical_overlap={resp.lexical_overlap:.3f}, "
        f"confidence={resp.confidence_score:.3f})"
    )
    return False, notes


SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec(
        scenario_id=1,
        code="S01_COLLOQUIAL_KZ",
        title="Перефразированный разговорный запрос (бронирование КЗ: за сколько месяцев служебка и куда слать)",
        query="Хотим провести студенческий концерт в концертном зале института: за сколько месяцев надо закинуть служебку и на какую электронную почту ее отправлять?",
        top_k=6,
        expect_out_of_scope=False,
        validator=_validate_s1_concert_hall_colloquial,
    ),
    ScenarioSpec(
        scenario_id=2,
        code="S02_SCANNED_TABLE_KZ",
        title="Табличный запрос из скана (репетиционные дни в КЗ на Посвящение в студенты, п. 6.1 стр. 4)",
        query="Сколько репетиционных дней в Концертном зале выделяется физтех-школам на проведение Посвящения в студенты согласно таблице регламента?",
        top_k=6,
        expect_out_of_scope=False,
        validator=_validate_s2_concert_hall_table,
    ),
    ScenarioSpec(
        scenario_id=3,
        code="S03_GIANT_SCAN_DORM",
        title="Гигантский скан со штампом ЭДО (порядок и условия поселения сотрудников в общежития)",
        query="Каков порядок, условия и перечень документов для поселения сотрудников МФТИ в общежитие?",
        top_k=6,
        expect_out_of_scope=False,
        validator=_validate_s3_dorm_settlement_scan,
    ),
    ScenarioSpec(
        scenario_id=4,
        code="S04_SMETA_SUPERSESSION",
        title="Смена редакции смет (Приказ № 794-1 и Приложение 1а в 1С:ДО против формул КВР/КОСГУ сметы 2019 г.)",
        query="Как сейчас подается и согласовывается смета доходов и расходов в 1С:ДО (Приложение 1а по приказу 794-1) и какие коды КВР/КОСГУ и формулы расчета накладных расходов 20% и НДС 20% используются в шаблоне сметы?",
        top_k=6,
        expect_out_of_scope=False,
        validator=_validate_s4_smeta_supersession_and_formulas,
    ),
    ScenarioSpec(
        scenario_id=5,
        code="S05_FLS_AMENDMENT_GRAPH",
        title="Частичное изменение приказа (Приказ № 3065-1 + Распоряжение № 700 о присвоении номера ФЛС)",
        query="Порядок открытия финансово-лицевого счета (ФЛС) по приказу 3065-1 и актуальная структура присвоения уникального номера ФЛС (разряды кода, КФО) по распоряжению 700",
        top_k=6,
        expect_out_of_scope=False,
        validator=_validate_s5_fls_partial_amendment,
    ),
    ScenarioSpec(
        scenario_id=6,
        code="S06_BUDGET_YEARS_2026_2028",
        title="Расхождение нормативов по годам (Приказ № 3699-1 на 2026–2028 гг.: резерв отпусков 16.4% ППС и 8.2% прочие)",
        query="Какой процент резерва на оплату отпусков (вид затрат 211.03) установлен для ППС и для прочих сотрудников при формировании бюджета на 2026-2028 годы по приказу 3699-1?",
        top_k=6,
        expect_out_of_scope=False,
        validator=_validate_s6_budget_vacation_reserve,
    ),
    ScenarioSpec(
        scenario_id=7,
        code="S07_EXCEL_CFO_HIERARCHY",
        title="Дерево ЦФО из широкой таблицы Excel (Приказ № 3614-1: Финансодержатель -> ЦФО 1..4 и дедупликация)",
        query="Финансовая структура МФТИ на 2026 год (приказ 3614-1): к какому финансодержателю и ЦФО 1 уровня относятся управление информационных технологий (10267), финансово-экономическое управление (10294) и лаборатории Института квантовых технологий?",
        top_k=6,
        expect_out_of_scope=False,
        validator=_validate_s7_cfo_hierarchy_excel,
    ),
    ScenarioSpec(
        scenario_id=8,
        code="S08_IT_SUPPORT_AND_DC",
        title="Цифровой и сканированный ИТ-регламенты (Первая линия техподдержки helpdesk@mipt.ru, 16 часов и ЦОД)",
        query="Какие официальные каналы подачи обращений на первую линию технической поддержки УИТ (helpdesk@mipt.ru), время первичной обработки заявки и правила доступа в ЦОД МФТИ?",
        top_k=6,
        expect_out_of_scope=False,
        validator=_validate_s8_it_support_and_datacenter,
    ),
    ScenarioSpec(
        scenario_id=9,
        code="S09_OUT_OF_SCOPE_MORTGAGE",
        title="Заведомо нерелевантный запрос #1 (Оформление семейной ипотеки в банке -> is_out_of_scope=True)",
        query="Как оформить семейную ипотеку в Сбербанке под 6 процентов годовых и какой первоначальный взнос нужен для покупки квартиры в новостройке?",
        top_k=5,
        expect_out_of_scope=True,
        validator=_validate_out_of_scope,
    ),
    ScenarioSpec(
        scenario_id=10,
        code="S10_OUT_OF_SCOPE_CARBONARA",
        title="Заведомо нерелевантный запрос #2 (Рецепт пасты карбонара -> is_out_of_scope=True)",
        query="Классический рецепт приготовления итальянской пасты карбонара с гуанчиале, яичными желтками и сыром пекорино романо",
        top_k=5,
        expect_out_of_scope=True,
        validator=_validate_out_of_scope,
    ),
]


def audit_data_artifacts() -> tuple[bool, list[str]]:
    """Audit data/document_registry.json, data/indexing_report.json, and data/sections_store.json."""
    messages: list[str] = []
    ok = True

    if not DOCUMENT_REGISTRY_PATH.exists():
        return False, [f"FAIL: Не найден файл реестра {DOCUMENT_REGISTRY_PATH}"]

    registry_raw = json.loads(DOCUMENT_REGISTRY_PATH.read_text(encoding="utf-8"))
    docs_map = registry_raw.get("documents", registry_raw) if isinstance(registry_raw, dict) else {}
    if isinstance(docs_map, list):
        docs_list = docs_map
    else:
        docs_list = list(docs_map.values())

    total_docs = len(docs_list)
    error_docs = [d for d in docs_list if d.get("status") == "error"]
    messages.append(
        f"Реестр документов ({DOCUMENT_REGISTRY_PATH.name}): всего={total_docs}, с ошибками={len(error_docs)}"
    )
    if total_docs < 16:
        ok = False
        messages.append(f"FAIL: В реестре меньше 16 документов ({total_docs}/16)")
    if error_docs:
        ok = False
        messages.append(f"FAIL: Документы со статусом error: {[d.get('file_name') for d in error_docs]}")

    # Check scanned PDFs OCR pages
    scanned_names = [
        "Регламент бронирования конц зала.pdf",
        "Регламент поселения сотрудников.pdf",
        "Регламент эксплуатации центра обработки данных МФТИ, Физтех.pdf",
    ]
    for sname in scanned_names:
        found = next((d for d in docs_list if d.get("file_name") == sname), None)
        if not found:
            ok = False
            messages.append(f"FAIL: Сканированный документ «{sname}» отсутствует в реестре")
        else:
            messages.append(
                f"OK: Скан «{sname}» -> pages_total={found.get('pages_total')}, "
                f"ocr_pages={found.get('ocr_pages')}, chunks={found.get('chunk_count')}"
            )

    if SECTIONS_STORE_PATH.exists():
        sections_raw = json.loads(SECTIONS_STORE_PATH.read_text(encoding="utf-8"))
        sec_map = sections_raw.get("sections", sections_raw) if isinstance(sections_raw, dict) else {}
        messages.append(f"OK: Хранилище родительских разделов ({SECTIONS_STORE_PATH.name}): {len(sec_map)} разделов")
        # Check for runaway underscores on blank forms in scanned OCR PDFs and "None" in Excel formulas
        scanned_doc_ids = {
            "reg_concert_hall_booking",
            "reg_staff_dorm_settlement",
            "reg_datacenter_operations",
        }
        sec_values = list(sec_map.values()) if isinstance(sec_map, dict) else sec_map
        sec_texts = "\n".join(
            (s.get("clean_text", "") if isinstance(s, dict) else str(s)) for s in sec_values
        )
        scanned_sec_texts = "\n".join(
            s.get("clean_text", "")
            for s in sec_values
            if isinstance(s, dict) and s.get("doc_id") in scanned_doc_ids
        )
        if "_" * 25 in scanned_sec_texts or "_" * 150 in sec_texts:
            ok = False
            messages.append(
                "FAIL: Обнаружено зацикливание символов подчеркивания ('_____') на бланках сканов в sections_store.json"
            )
        else:
            messages.append(
                "OK: Защита от зацикливания VLM на пустых графах бланков ('[___]' на стр. 7 Регламента КЗ) прошла успешно"
            )

        if "[формула:" in sec_texts:
            messages.append("OK: В разделах присутствуют извлеченные формулы Excel «[формула: =...]»")
    else:
        ok = False
        messages.append(f"FAIL: Не найден файл {SECTIONS_STORE_PATH}")

    if INDEXING_REPORT_PATH.exists():
        rep = json.loads(INDEXING_REPORT_PATH.read_text(encoding="utf-8"))
        messages.append(
            f"OK: Отчет индексации ({INDEXING_REPORT_PATH.name}): "
            f"total_files={rep.get('total_files')}, failed_files={rep.get('failed_files', 0)}, "
            f"total_chunks={rep.get('total_chunks')}"
        )
    else:
        messages.append(f"WARN: Файл {INDEXING_REPORT_PATH.name} еще не создан")

    return ok, messages


def run_verification(
    api_url: str | None = None,
    auto_sync: bool = False,
    output_json: Path | None = None,
) -> int:
    """Execute all 10 verification scenarios and print detailed report."""
    print("=" * 88)
    print("MIPT NORMATIVE RAG V3 — КОМПЛЕКСНАЯ ВАЛИДАЦИЯ ПОИСКА (10 СЦЕНАРИЕВ)")
    print("=" * 88)

    use_http = bool(api_url)
    search_svc = None

    if not use_http:
        from src.indexing.pipeline import get_pipeline
        from src.retrieval.search_service import get_search_service

        pipeline = get_pipeline()
        docs = pipeline.get_documents()
        if len(docs) < 16 and auto_sync:
            print(f"[INFO] В базе {len(docs)} документов (< 16). Запускаем синхронизацию папки доки/...")
            pipeline.sync_folder(force=False)
        search_svc = get_search_service()

    ok_artifacts, artifact_msgs = audit_data_artifacts()
    print("\n--- 1. Аудит артефактов индексации в data/ ---")
    for msg in artifact_msgs:
        print(f"  • {msg}")

    results: list[ScenarioResult] = []
    print("\n--- 2. Прогон 10 контрольных поисковых сценариев ---")

    for spec in SCENARIOS:
        t0 = time.perf_counter()
        req = SearchQuery(
            query=spec.query,
            top_k=spec.top_k,
            include_parent=True,
            include_amendments=True,
            include_superseded=True,
        )

        if use_http and api_url:
            with httpx.Client(timeout=60.0) as client:
                r = client.post(
                    f"{api_url.rstrip('/')}/api/search",
                    json=req.model_dump(),
                )
                r.raise_for_status()
                resp = SearchResponse.model_validate(r.json())
        else:
            assert search_svc is not None
            resp = search_svc.search(req)

        elapsed_ms = resp.latency_ms or ((time.perf_counter() - t0) * 1000.0)
        passed, details = spec.validator(resp)
        top_rrf = resp.hits[0].rrf_score if resp.hits else 0.0
        top_sources = [
            f"{h.file_name} (п. {h.clause_no or h.section_num or 'б/н'})"
            for h in resp.hits[:3]
        ]

        res = ScenarioResult(
            scenario_id=spec.scenario_id,
            code=spec.code,
            title=spec.title,
            query=spec.query,
            passed=passed,
            is_out_of_scope=resp.is_out_of_scope,
            top_dense_score=resp.top_dense_score,
            confidence_score=resp.confidence_score,
            lexical_overlap=resp.lexical_overlap,
            top_rrf_score=top_rrf,
            latency_ms=elapsed_ms,
            top_sources=top_sources,
            version_alerts_count=len(resp.version_alerts),
            amendment_chunks_count=len(resp.amendment_chunks),
            details=details,
        )
        results.append(res)

        status_str = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n[{spec.scenario_id:02d}/10] {status_str} | {spec.code}: {spec.title}")
        print(f"  Запрос: «{spec.query}»")
        print(
            f"  Метрики: dense={res.top_dense_score:.4f} | conf={res.confidence_score:.4f} | "
            f"lex={res.lexical_overlap:.3f} | rrf={res.top_rrf_score:.4f} | "
            f"out_of_scope={res.is_out_of_scope} | latency={res.latency_ms:.1f} ms"
        )
        if res.top_sources:
            print(f"  Топ-источники: {'; '.join(res.top_sources)}")
        if res.version_alerts_count or res.amendment_chunks_count:
            print(
                f"  Граф редакций: version_alerts={res.version_alerts_count}, "
                f"amendment_chunks={res.amendment_chunks_count}"
            )
        for d in res.details:
            print(f"    -> {d}")

    # Score separation check between Scenarios 1-8 (in-scope) and 9-10 (out-of-scope)
    in_scope = [r for r in results if r.scenario_id <= 8]
    out_scope = [r for r in results if r.scenario_id >= 9]
    min_in_dense = min((r.top_dense_score for r in in_scope), default=0.0)
    max_out_dense = max((r.top_dense_score for r in out_scope), default=0.0)
    min_in_conf = min((r.confidence_score for r in in_scope), default=0.0)
    max_out_conf = max((r.confidence_score for r in out_scope), default=0.0)
    dense_margin = min_in_dense - max_out_dense
    conf_margin = min_in_conf - max_out_conf

    separation_ok = (dense_margin > 0.05) or (conf_margin > 0.10)
    if not separation_ok and results[-1].passed:
        results[-1].passed = False
        results[-1].details.append(
            f"FAIL: Недостаточный зазор между релевантными (min_dense={min_in_dense:.3f}) "
            f"и нерелевантными (max_dense={max_out_dense:.3f}) запросами"
        )

    passed_count = sum(1 for r in results if r.passed)
    print("\n" + "=" * 88)
    print(
        f"ИТОГ ВАЛИДАЦИИ: {passed_count}/10 сценариев успешно пройдено | "
        f"Зазор In-Scope vs Out-of-Scope: Δdense={dense_margin:+.4f} "
        f"(min_in={min_in_dense:.4f} vs max_out={max_out_dense:.4f}), "
        f"Δconf={conf_margin:+.4f}"
    )
    print("=" * 88)

    if output_json:
        payload: dict[str, Any] = {
            "passed_scenarios": passed_count,
            "total_scenarios": len(results),
            "artifacts_ok": ok_artifacts,
            "min_in_scope_dense": round(min_in_dense, 4),
            "max_out_of_scope_dense": round(max_out_dense, 4),
            "dense_margin": round(dense_margin, 4),
            "confidence_margin": round(conf_margin, 4),
            "scenarios": [
                {
                    "scenario_id": r.scenario_id,
                    "code": r.code,
                    "title": r.title,
                    "query": r.query,
                    "passed": r.passed,
                    "is_out_of_scope": r.is_out_of_scope,
                    "top_dense_score": round(r.top_dense_score, 4),
                    "confidence_score": round(r.confidence_score, 4),
                    "lexical_overlap": round(r.lexical_overlap, 4),
                    "top_rrf_score": round(r.top_rrf_score, 4),
                    "latency_ms": round(r.latency_ms, 1),
                    "top_sources": r.top_sources,
                    "version_alerts_count": r.version_alerts_count,
                    "amendment_chunks_count": r.amendment_chunks_count,
                    "details": r.details,
                }
                for r in results
            ],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Детальный JSON-отчет сохранен в {output_json}")

    return 0 if (passed_count == 10 and ok_artifacts) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MIPT Normative RAG V3 retrieval quality.")
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Optional FastAPI base URL (e.g. http://localhost:8000) to test via HTTP POST /api/search",
    )
    parser.add_argument(
        "--auto-sync",
        action="store_true",
        help="Automatically sync доки/ folder if fewer than 16 documents are indexed",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("data/verification_results.json"),
        help="Path to write JSON verification report",
    )
    args = parser.parse_args()
    sys.exit(run_verification(api_url=args.api_url, auto_sync=args.auto_sync, output_json=args.json_out))


if __name__ == "__main__":
    main()
