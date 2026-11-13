"""Autonomous Table-Aware RAG Agent (Модуль 4+) with Adaptive Context Routing and OpenAI Tool Calling."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

from src.config import settings
from src.indexing.table_store import OPENAI_TABLE_TOOLS, StructuredTableStore
from src.retrieval.search_service import SearchService, get_search_service
from src.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    SearchHit,
    SearchQuery,
    SearchResponse,
    TablePassport,
    TableQueryRequest,
    TableQueryResponse,
    ToolCallTrace,
)

logger = logging.getLogger(__name__)

# Verified OpenAI-compatible tool-capable models available on Polza AI in priority order
POLZA_TOOL_FALLBACK_MODELS: tuple[str, ...] = (
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "qwen/qwen3.6-flash",
    "deepseek/deepseek-v4.1-flash",
    "google/gemini-3.8-flash",
)

SYSTEM_PROMPT_RU = """Ты — экспертный ИИ-ассистент по нормативной базе, приказам, регламентам, сметам и финансовой структуре МФТИ (Физтех).
Отвечай строго на русском языке, опираясь ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты документов, граф редакций и результаты вызовов табличных инструментов.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА РАБОТЫ:
1. Точное цитирование источников:
   - Всегда указывай название документа, номер и дату приказа/распоряжения, номер пункта/раздела или название листа/таблицы.
   - Если в метаданных фрагмента или таблицы указано поле `also_in_files` (данные присутствуют сразу в нескольких файлах, например в отдельном `.xlsx` и приложении к приказу `.docx`), упомяни эту связь.
2. Учет актуальности и графа изменений (`Edition Graph`):
   - Внимательно проверяй блок предупреждений об актуальности (`version_alerts`) и изменяющие фрагменты (`amendment_chunks`).
   - Если базовый приказ частично изменен более поздним актом (например, Приказ № 3065-1 действует в редакции Распоряжения № 700), явно укажи актуальное правило с учетом изменений.
   - Если документ имеет статус `superseded` или `superseded_template_reference` (например, шаблон сметы 2019 г. с формулами при действующем Приказе № 794-1), обязательно предупреди об этом пользователя.
3. Адаптивная работа с таблицами (3-уровневая маршрутизация):
   - Компактные таблицы Word/PDF (`table_delivery_mode="inline_parent"`): их полный чистый раздел (`ParentSectionRecord`) УЖЕ передан в контексте. Отвечай по ним сразу, инструменты вызывать не нужно.
   - Крупные таблицы и книги Excel (`linked_tables`, `delivery_mode="tool_query"`): в контексте приведены только их компактные паспорта (`TablePassport` со списком столбцов, типами, примерами значений и числом строк) и короткие Child-фрагменты. Полные 200–600 строк НЕ вставлены в промпт.
   - Когда вопрос касается фильтрации строк, поиска по ФИО финансодержателя или подразделению/ЦФО, вывода полного списка позиций, точного подсчета количества (`count`), суммирования (`sum`), группировки (`group_by`), проверки ставок/КВР/КОСГУ или формул Excel (`get_formulas`) в таблице `tool_query`, **ОБЯЗАТЕЛЬНО вызови `query_structured_table` (и при необходимости `get_table_schema`)** перед финальным ответом!
   - Никогда не угадывай общее число строк, суммы или полный перечень подразделений по 1–2 текстовым чанкам, если в `linked_tables` есть паспорт соответствующей таблицы.
4. Эффективное использование `query_structured_table`:
   - Передавай точный `table_id` из блока паспортов таблиц.
   - Для поиска подстроки в конкретном столбце передавай `filters={"Точное имя столбца": "искомая подстрока"}` (поиск регистронезависимый; для чисел/процентов поддерживаются операторы `>`, `>=`, `<`, `<=`) либо используй `search_text` для сквозного поиска по всем ячейкам строки.
   - ВАЖНАЯ ОСОБЕННОСТЬ ТАБЛИЦЫ ФИНАНСОВОЙ СТРУКТУРЫ МФТИ: столбец `Финансодержатель` содержит ФИО ответственного лица (например, `Климов Роман Олегович`, `Крылова Елена Викторовна`), тогда как названия физтех-школ, институтов, управлений и лабораторий записаны ПОЛНОСТЬЮ (без аббревиатур) в столбцах `ЦФО 1 уровня`, `ЦФО 2 уровня`, `ЦФО 3 уровня`, `ЦФО 4 уровня`!
   - При поиске физтех-школ используй их полные названия или корни слов (например: `ФРКТ` -> `"радиотехники и компьютерных технологий"` в `ЦФО 1 уровня` или `search_text="радиотехники"`; `ФПМИ` -> `"прикладной математики и информатики"`; `ЛФИ` -> `"физики и исследований"`; `ФАКТ` -> `"аэрокосмических технологий"`; `ФЭФМ` -> `"электроники, фотоники"`; `ФБМФ` -> `"биологической и медицинской физики"`). Если вопрос звучит как «сколько подразделений подчиняется финансодержателю ФРКТ», сначала найди строки этой физтех-школы по `ЦФО 1 уровня` (узнав ФИО её финансодержателя и все подчиненные подразделения `ЦФО 2–4 уровня`), а также при необходимости проверь все строки найденного ФИО финансодержателя!
   - Если первый вызов `filters` вернул `matched_count == 0`, НИКОГДА не делай вывод об отсутствии данных сразу — обязательно повтори вызов с `search_text` по ключевому слову/корню или полному наименованию без узкого фильтра по одному столбцу!
   - Для подсчета количества или сумм используй `operation="aggregate"` (`agg_func="count"` или `"sum"`, при необходимости `group_by="..."`), либо `operation="filter_rows"` с достаточным `limit` (до 200), чтобы перечислить все найденные строки.
   - Для извлечения расчетных формул Excel используй `operation="get_formulas"` (можно совмещать с `search_text` или `filters`).
"""


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate long text cleanly with an ellipsis note."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n...[сокращено, всего {len(text)} симв.]"


def _parse_tool_arguments(raw_args: dict[str, Any] | str | None) -> dict[str, Any]:
    """Parse tool arguments from JSON string or dict without raising."""
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str):
        stripped = raw_args.strip()
        if not stripped:
            return {}
        try:
            loaded = json.loads(stripped)
            return loaded if isinstance(loaded, dict) else {"_raw": loaded}
        except json.JSONDecodeError:
            return {"_raw_invalid_json": stripped[:500]}
    return {}


class TableAwareRAGAgent:
    """
    End-to-end RAG Agent combining Hybrid RRF Search, Adaptive 3-Tier Table Context Routing,
    and Autonomous OpenAI Function-Calling over StructuredTableStore via Polza AI.
    """

    def __init__(
        self,
        search_service: SearchService | None = None,
        table_store: StructuredTableStore | None = None,
        llm_client: Any = None,
        client: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 75.0,
    ) -> None:
        self.search_service = search_service or get_search_service()
        self._table_store = table_store
        injected_client = llm_client if llm_client is not None else client
        self._client: Any = injected_client
        self._has_custom_client: bool = injected_client is not None
        self.api_key: str = settings.POLZA_API_KEY if api_key is None else api_key
        self.base_url: str = (base_url or settings.POLZA_BASE_URL).rstrip("/")
        self.default_model: str = (model or settings.POLZA_LLM_MODEL or "openai/gpt-4o-mini").strip()
        self.timeout: float = timeout
        self._verified_model: str | None = None

    @property
    def client(self) -> Any:
        """Return injected or lazily constructed OpenAI client pointed at Polza AI."""
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=2,
        )
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        self._client = value
        self._has_custom_client = value is not None

    @property
    def llm_client(self) -> Any:
        return self.client

    @llm_client.setter
    def llm_client(self, value: Any) -> None:
        self.client = value

    @property
    def table_store(self) -> StructuredTableStore:
        """Return bound StructuredTableStore or delegate to SearchService."""
        if self._table_store is not None:
            return self._table_store
        return self.search_service.table_store

    def _get_active_table_store(self) -> StructuredTableStore | None:
        """Return active StructuredTableStore without creating an empty DB file on read-only paths."""
        if self._table_store is not None:
            return self._table_store
        return self.search_service._get_active_table_store()

    def _candidate_models(self, requested_model: str | None) -> list[str]:
        """Build ordered list of candidate chat models with automatic Polza AI fallback."""
        explicit = (requested_model or "").strip()
        ordered: list[str] = []
        if explicit:
            ordered.append(explicit)
        elif self._verified_model:
            ordered.append(self._verified_model)

        if self.default_model and self.default_model not in ordered:
            ordered.append(self.default_model)

        if not self._has_custom_client:
            for m in POLZA_TOOL_FALLBACK_MODELS:
                if m not in ordered:
                    ordered.append(m)
        return ordered

    @staticmethod
    def _format_table_passport_for_prompt(tp: TablePassport) -> str:
        """Format a compact TablePassport (~600-1200 chars) for the LLM prompt instead of dumping 60k chars."""
        lines: list[str] = [
            f"- table_id: `{tp.table_id}`",
            f"  Название таблицы: {tp.table_title or tp.section_title or tp.file_name}",
            f"  Файл: {tp.file_name}"
            + (f" (Документ № {tp.doc_number} от {tp.doc_date})" if tp.doc_number else ""),
        ]
        if tp.sheet_name:
            lines.append(f"  Лист Excel: `{tp.sheet_name}`")
        if tp.also_in_files:
            lines.append(f"  Также содержится в файлах (also_in_files): {', '.join(tp.also_in_files)}")
        lines.append(
            f"  Размерность: {tp.row_count} строк × {tp.col_count} столбцов | "
            f"Формулы Excel: {'да (' + str(tp.formula_count) + ' ячеек)' if tp.has_formulas else 'нет'} | "
            f"Режим доставки: {tp.delivery_mode}"
        )
        col_descs: list[str] = []
        for col in tp.columns[:18]:
            samples = ", ".join(repr(s[:45]) for s in col.sample_values[:3])
            flags: list[str] = [col.inferred_type]
            if col.is_hierarchy:
                flags.append("иерархия")
            if col.has_formulas:
                flags.append("формулы")
            sample_part = f"; примеры: {samples}" if samples else ""
            col_descs.append(f"    * `{col.name}` ({', '.join(flags)}, заполнено {col.non_empty_count}{sample_part})")
        if col_descs:
            lines.append("  Столбцы:")
            lines.extend(col_descs)
        if tp.preview_rows:
            preview_json = json.dumps(tp.preview_rows[:2], ensure_ascii=False)
            lines.append(f"  Превью первых строк: {_truncate_text(preview_json, 450)}")
        return "\n".join(lines)

    @staticmethod
    def _format_hit_for_prompt(idx: int, hit: SearchHit, prefix: str = "Фрагмент") -> str:
        """Format a single Child chunk SearchHit with rich citation metadata."""
        delivery_mode = str((hit.metadata or {}).get("table_delivery_mode") or "normative_prose")
        header_parts = [f"[{prefix} #{idx}] {hit.title or hit.file_name}"]
        if hit.doc_number:
            header_parts.append(f"№ {hit.doc_number}" + (f" от {hit.doc_date}" if hit.doc_date else ""))
        if hit.edo_number:
            header_parts.append(
                f"Зарег. в ЭДО: № {hit.edo_number}" + (f" от {hit.edo_date}" if hit.edo_date else "")
            )
        header_parts.append(f"файл: {hit.file_name}")
        header_parts.append(f"статус: {hit.status}")
        if hit.clause_no:
            header_parts.append(f"п. {hit.clause_no}")
        elif hit.section_title:
            header_parts.append(f"раздел: {hit.section_title}")
        if hit.page_numbers:
            header_parts.append(f"стр. {', '.join(str(p) for p in hit.page_numbers)}")
        if hit.table_id:
            header_parts.append(f"table_id=`{hit.table_id}`")
        header_parts.append(f"mode={delivery_mode}")
        if hit.also_in_files:
            header_parts.append(f"также в: {', '.join(hit.also_in_files)}")

        body = _truncate_text(hit.clean_text.strip(), 2200)
        return " | ".join(header_parts) + "\n" + body

    def _build_messages(self, query: str, search_resp: SearchResponse) -> list[dict[str, Any]]:
        """
        Build grounded system + user prompt following Adaptive 3-Tier Table Context Routing:
        - Child chunks + amendment_chunks + version_alerts always included.
        - Full ParentSectionRecord included ONLY for normative prose & compact `inline_parent` tables.
        - `tool_query` tables represented via compact TablePassports (never dumping 60k chars of table text).
        """
        q_block = f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{query.strip()}"
        if len(search_resp.sub_queries) > 1:
            sq_lines = "\n".join(f"  {i}) {sq}" for i, sq in enumerate(search_resp.sub_queries, start=1))
            q_block += f"\n\nПОДВОПРОСЫ ДЛЯ ОБЯЗАТЕЛЬНОЙ ПРОВЕРКИ:\n{sq_lines}"
        sections_blocks: list[str] = [q_block]

        if search_resp.version_alerts or search_resp.quality_alerts:
            alert_lines: list[str] = []
            for va in search_resp.version_alerts:
                alert_lines.append(f"- [АКТУАЛЬНОСТЬ / РЕДАКЦИЯ] {va}")
            for qa in search_resp.quality_alerts:
                alert_lines.append(f"- [КАЧЕСТВО ИСТОЧНИКА] {qa}")
            sections_blocks.append("ВАЖНЫЕ ПРЕДУПРЕЖДЕНИЯ ПО СТАТУСУ И РЕДАКЦИЯМ ДОКУМЕНТОВ:\n" + "\n".join(alert_lines))

        if search_resp.linked_tables:
            passport_blocks = [
                self._format_table_passport_for_prompt(tp) for tp in search_resp.linked_tables
            ]
            sections_blocks.append(
                "ДОСТУПНЫЕ СТРУКТУРИРОВАННЫЕ ТАБЛИЦЫ В SQLite (`delivery_mode=tool_query` — "
                "для фильтрации, полного списка, подсчета, агрегации и формул вызывай `query_structured_table`):\n"
                + "\n\n".join(passport_blocks)
            )

        if search_resp.table_context:
            pre_tbl_blocks: list[str] = []
            for t_resp in search_resp.table_context[:6]:
                hdr = (
                    f"- Предвыборка `{t_resp.operation}` по `{t_resp.table_id}` "
                    f"(совпало {t_resp.matched_count} из {t_resp.total_table_rows} строк):"
                )
                if t_resp.formulas:
                    f_preview = "; ".join(
                        f"стр.{f.get('row_index')} `{f.get('column')}`={f.get('formula')}"
                        for f in t_resp.formulas[:10]
                    )
                    pre_tbl_blocks.append(f"{hdr}\n  Формулы: {f_preview}")
                elif t_resp.markdown_preview:
                    pre_tbl_blocks.append(f"{hdr}\n{_truncate_text(t_resp.markdown_preview, 2200)}")
            if pre_tbl_blocks:
                sections_blocks.append(
                    "ПРЕДВАРИТЕЛЬНАЯ ВЫБОРКА ИЗ СТРУКТУРИРОВАННЫХ ТАБЛИЦ "
                    "(при необходимости точной агрегации/фильтрации вызови `query_structured_table`):\n\n"
                    + "\n\n".join(pre_tbl_blocks)
                )

        if search_resp.hits:
            hit_blocks = [
                self._format_hit_for_prompt(i, h, prefix="Фрагмент")
                for i, h in enumerate(search_resp.hits, start=1)
            ]
            sections_blocks.append("НАЙДЕННЫЕ РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ (Child-чанки):\n\n" + "\n\n---\n\n".join(hit_blocks))

        if search_resp.amendment_chunks:
            amd_blocks = [
                self._format_hit_for_prompt(i, h, prefix="Изменение/Дополнение")
                for i, h in enumerate(search_resp.amendment_chunks, start=1)
            ]
            sections_blocks.append(
                "СВЯЗАННЫЕ ИЗМЕНЯЮЩИЕ ДОКУМЕНТЫ (Edition Graph / Amendments):\n\n"
                + "\n\n---\n\n".join(amd_blocks)
            )

        # Hydrate full ParentSectionRecord ONLY for normative prose and compact `inline_parent` tables
        inline_parents: list[str] = []
        seen_parent_ids: set[str] = set()
        all_hits = list(search_resp.hits) + list(search_resp.amendment_chunks)
        for h in all_hits:
            mode = str((h.metadata or {}).get("table_delivery_mode") or "normative_prose")
            if mode == "tool_query" or h.table_id:
                continue
            sec_id = h.section_id or (h.parent_section.section_id if h.parent_section else "")
            if not sec_id or sec_id in seen_parent_ids:
                continue
            parent_text = (h.parent_section.clean_text if h.parent_section else h.parent_text or "").strip()
            if not parent_text:
                continue
            seen_parent_ids.add(sec_id)
            sec_hits = [sh for sh in all_hits if sh.section_id == sec_id]
            sec_title = (
                (h.parent_section.section_title if h.parent_section else "")
                or h.section_title
                or h.unit_title
                or h.title
            )
            cpm_note = ""
            if h.parent_section is not None:
                cpm_note = SearchService._format_clause_page_map_annotation(h.parent_section, sec_hits)
            windowed_parent = SearchService._build_hit_centered_parent_window(
                parent_text=parent_text,
                matched_snippets=[sh.clean_text for sh in sec_hits],
                max_chars=9500,
            )
            header_line = f"[Полный раздел `{sec_id}` | {h.file_name} | {sec_title} | mode={mode}]"
            if cpm_note:
                inline_parents.append(f"{header_line}\n{cpm_note}\n{windowed_parent}")
            else:
                inline_parents.append(f"{header_line}\n{windowed_parent}")

        if inline_parents:
            sections_blocks.append(
                "ПОЛНЫЙ КОНТЕКСТ РАЗДЕЛОВ (`ParentSectionRecord` для нормативного текста и компактных таблиц `inline_parent`):\n\n"
                + "\n\n===\n\n".join(inline_parents)
            )

        user_prompt = "\n\n==============================\n\n".join(sections_blocks)
        return [
            {"role": "system", "content": SYSTEM_PROMPT_RU},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _build_tool_summary(
        tool_name: str,
        args: dict[str, Any],
        payload: dict[str, Any],
        q_resp: TableQueryResponse | None,
    ) -> str:
        """Create a concise human-readable Russian summary of a tool execution for UI & traces."""
        if "error" in payload:
            return f"Ошибка {tool_name}: {payload['error']}"
        tid = str(payload.get("table_id") or args.get("table_id") or "")
        if tool_name == "get_table_schema":
            cols_cnt = len(payload.get("columns") or [])
            rows_cnt = int(payload.get("row_count") or 0)
            return f"Схема таблицы `{tid}`: {rows_cnt} строк, {cols_cnt} столбцов"
        if q_resp is not None:
            op = q_resp.operation or str(args.get("operation") or "filter_rows")
            if op == "aggregate":
                agg_fn = str(args.get("agg_func") or "count")
                grp = str(args.get("group_by") or "")
                grp_note = f" по «{grp}»" if grp else ""
                return (
                    f"Агрегация ({agg_fn}{grp_note}) в `{tid}`: "
                    f"совпало {q_resp.matched_count} из {q_resp.total_table_rows} строк"
                )
            if op == "get_formulas":
                return (
                    f"Извлечено {len(q_resp.formulas)} формул (совпало {q_resp.matched_count} строк) "
                    f"из таблицы `{tid}`"
                )
            return (
                f"Найдено {q_resp.matched_count} из {q_resp.total_table_rows} строк "
                f"(возвращено {q_resp.returned_count}) в таблице `{tid}`"
            )
        return f"Выполнен вызов {tool_name}(`{tid}`)"

    @staticmethod
    def _build_result_preview(payload: dict[str, Any], q_resp: TableQueryResponse | None) -> str:
        """Build a readable preview snippet of tool output for ToolCallTrace."""
        if "error" in payload:
            return _truncate_text(str(payload["error"]), 600)
        if q_resp is not None:
            if q_resp.aggregation_result is not None:
                return _truncate_text(json.dumps(q_resp.aggregation_result, ensure_ascii=False), 700)
            if q_resp.formulas:
                return _truncate_text(json.dumps(q_resp.formulas[:8], ensure_ascii=False), 700)
            if q_resp.markdown_preview:
                return _truncate_text(q_resp.markdown_preview, 700)
        return _truncate_text(json.dumps(payload, ensure_ascii=False), 700)

    def _call_chat_completion(
        self,
        messages: list[dict[str, Any]],
        allow_tools: bool,
        requested_model: str | None = None,
    ) -> tuple[Any, str]:
        """
        Invoke OpenAI-compatible chat.completions.create on Polza AI with automatic model fallback
        if a model name returns 400/404/unsupported-tools or transient gateway failure.
        """
        llm = self.client
        if llm is None:
            raise RuntimeError("POLZA_API_KEY не задан и кастомный LLM-клиент не передан.")

        candidates = self._candidate_models(requested_model)
        last_exc: Exception | None = None

        for model_name in candidates:
            call_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.1,
            }
            if allow_tools:
                call_kwargs["tools"] = OPENAI_TABLE_TOOLS
                call_kwargs["tool_choice"] = "auto"

            try:
                resp = llm.chat.completions.create(**call_kwargs)
                resp_model = getattr(resp, "model", None)
                resolved_model = (
                    resp_model.strip()
                    if isinstance(resp_model, str) and resp_model.strip()
                    else model_name
                )
                if not requested_model and not self._has_custom_client:
                    self._verified_model = model_name
                return resp, resolved_model
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Chat completion call failed on model '%s' (allow_tools=%s): %s",
                    model_name,
                    allow_tools,
                    exc,
                )
                if self._has_custom_client:
                    break

        assert last_exc is not None
        raise last_exc

    def _synthesize_fallback_answer(
        self,
        query: str,
        search_resp: SearchResponse,
        active_store: StructuredTableStore | None,
        enable_table_tools: bool,
        traces: list[ToolCallTrace],
    ) -> str:
        """
        Deterministic Russian fallback synthesizer used only when Polza AI network/API is unreachable.
        Executes a deterministic StructuredTableStore lookup when linked_tables are present and no tool ran yet.
        """
        q_norm = query.lower().replace("ё", "е")
        parts: list[str] = []

        if search_resp.version_alerts:
            parts.append("**Предупреждения об актуальности и редакциях:**")
            for va in search_resp.version_alerts:
                parts.append(f"- {va}")

        if (
            enable_table_tools
            and search_resp.linked_tables
            and active_store is not None
            and not traces
        ):
            tp = search_resp.linked_tables[0]
            op = "get_formulas" if "формул" in q_norm else "filter_rows"
            stop_words = {
                "какие", "какой", "какая", "сколько", "список", "перечень", "таблице", "таблица",
                "покажи", "найди", "найти", "все", "всех", "для", "при", "или", "как", "кто",
                "строк", "строки", "данные", "значения", "мфти", "приказ", "файл", "файле",
            }
            tokens = [
                t for t in re.findall(r"[0-9a-zA-Zа-яА-ЯёЁ%-]{3,}", query)
                if t.lower().replace("ё", "е") not in stop_words
            ]
            search_term = tokens[0] if tokens else ""
            t_call = time.perf_counter()
            req_args: dict[str, Any] = {
                "table_id": tp.table_id,
                "operation": op,
                "search_text": search_term,
                "limit": 25,
            }
            payload, q_resp = active_store.execute_tool_call("query_structured_table", req_args)
            if q_resp is not None and q_resp.matched_count == 0 and search_term:
                req_args["search_text"] = ""
                payload, q_resp = active_store.execute_tool_call("query_structured_table", req_args)
            call_ms = round((time.perf_counter() - t_call) * 1000.0, 2)
            traces.append(
                ToolCallTrace(
                    step=1,
                    tool_name="query_structured_table",
                    arguments=req_args,
                    matched_count=q_resp.matched_count if q_resp else 0,
                    summary=self._build_tool_summary("query_structured_table", req_args, payload, q_resp),
                    result_preview=self._build_result_preview(payload, q_resp),
                    latency_ms=call_ms,
                )
            )

        if traces:
            parts.append("**Результаты запроса к структурированным таблицам:**")
            for tr in traces:
                parts.append(f"- {tr.summary}")
                if tr.result_preview:
                    parts.append(tr.result_preview)

        parts.append("**Релевантные положения нормативных документов МФТИ:**")
        for idx, hit in enumerate(search_resp.hits[:3], start=1):
            cite = f"{hit.title or hit.file_name}"
            if hit.doc_number:
                cite += f" (№ {hit.doc_number})"
            if hit.clause_no:
                cite += f", п. {hit.clause_no}"
            snippet = _truncate_text(
                hit.parent_text if (hit.metadata or {}).get("table_delivery_mode") == "inline_parent" and hit.parent_text else hit.clean_text,
                900,
            )
            parts.append(f"{idx}. **{cite}** [{hit.file_name}]:\n{snippet}")

        for amd in search_resp.amendment_chunks[:2]:
            parts.append(
                f"- **Изменяющий акт — {amd.title} (№ {amd.doc_number})**: {_truncate_text(amd.clean_text, 500)}"
            )

        return "\n\n".join(parts)

    def ask(
        self,
        request: AgentAskRequest | dict[str, Any] | str,
        **kwargs: Any,
    ) -> AgentAskResponse:
        """
        Execute end-to-end question answering with:
        1) Hybrid RRF Search + Out-of-Scope short-circuit
        2) Adaptive 3-Tier Table Context Routing (inline_parent vs tool_query TablePassports)
        3) Multi-round OpenAI Tool Calling against StructuredTableStore (up to max_tool_rounds)
        """
        t0 = time.perf_counter()
        if isinstance(request, str):
            req = AgentAskRequest(query=request, **kwargs)
        elif isinstance(request, AgentAskRequest):
            req = request
        elif isinstance(request, dict):
            req = AgentAskRequest.model_validate(request)
        else:
            raise TypeError(f"Unsupported AgentAskRequest type: {type(request)}")

        raw_query = (req.query or "").strip()
        search_resp = self.search_service.search(
            SearchQuery(
                query=raw_query,
                top_k=req.top_k,
                status_filter=req.status_filter,
                year_filter=req.year_filter,
                include_parent=True,
                include_amendments=True,
                include_superseded=True,
            )
        )

        # Step 1: Out-of-Scope short-circuit (0 tool calls wasted)
        if search_resp.is_out_of_scope or not raw_query or not search_resp.hits:
            conf_str = f"{search_resp.confidence_score:.2f}"
            refusal = (
                "К сожалению, ваш запрос выходит за рамки корпуса нормативных документов МФТИ "
                f"(оценка уверенности / релевантности корпусу: confidence = {conf_str}). "
                "Я могу отвечать только на вопросы по локальным нормативным актам, приказам, "
                "распоряжениям, регламентам, сметам и финансовой структуре МФТИ."
            )
            return AgentAskResponse(
                query=raw_query,
                answer=refusal,
                model_used="out_of_scope_guard",
                used_table_tools=False,
                tool_calls=[],
                search_response=search_resp,
                linked_tables=[],
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # Step 2: Build grounded messages with Adaptive 3-Tier Table Context Routing
        messages = self._build_messages(raw_query, search_resp)
        active_store = self._get_active_table_store()
        final_linked_tables: list[TablePassport] = list(search_resp.linked_tables)
        seen_linked_ids: set[str] = {tp.table_id for tp in final_linked_tables}

        tools_enabled = bool(
            req.enable_table_tools
            and req.max_tool_rounds > 0
            and final_linked_tables
            and active_store is not None
        )

        traces: list[ToolCallTrace] = []
        final_answer = ""
        model_used = ""
        step_counter = 0

        # Step 3: Autonomous LLM Tool Calling loop (with deterministic fallback if offline)
        try:
            max_rounds = req.max_tool_rounds if tools_enabled else 0
            pinned_model = req.model

            for _round_idx in range(max_rounds + 1):
                allow_tools_now = tools_enabled and (_round_idx < max_rounds)
                resp, resolved_model = self._call_chat_completion(
                    messages=messages,
                    allow_tools=allow_tools_now,
                    requested_model=pinned_model,
                )
                model_used = resolved_model
                if pinned_model is None and not self._has_custom_client:
                    pinned_model = resolved_model

                choices = list(getattr(resp, "choices", None) or [])
                if not choices:
                    break
                msg = choices[0].message
                msg_content = str(getattr(msg, "content", None) or "").strip()
                raw_tool_calls = getattr(msg, "tool_calls", None)
                tool_calls_list = list(raw_tool_calls) if raw_tool_calls else []

                if not tool_calls_list or not allow_tools_now:
                    final_answer = msg_content
                    break

                # Append assistant tool-call turn in clean OpenAI message format
                serialized_tcs: list[dict[str, Any]] = []
                for tc_idx, tc in enumerate(tool_calls_list, start=1):
                    tc_id = str(getattr(tc, "id", None) or f"call_{step_counter + tc_idx}")
                    fn_obj = getattr(tc, "function", None)
                    fn_name = str(getattr(fn_obj, "name", "") or "")
                    fn_args_raw = getattr(fn_obj, "arguments", "{}")
                    fn_args_str = (
                        fn_args_raw
                        if isinstance(fn_args_raw, str)
                        else json.dumps(fn_args_raw or {}, ensure_ascii=False)
                    )
                    serialized_tcs.append(
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": fn_args_str,
                            },
                        }
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": msg_content or None,
                        "tool_calls": serialized_tcs,
                    }
                )

                # Execute each requested tool call against StructuredTableStore
                assert active_store is not None
                for tc_dict in serialized_tcs:
                    step_counter += 1
                    t_call = time.perf_counter()
                    fn_name = tc_dict["function"]["name"]
                    fn_args_str = tc_dict["function"]["arguments"]
                    parsed_args = _parse_tool_arguments(fn_args_str)

                    try:
                        tool_payload, q_resp = active_store.execute_tool_call(fn_name, parsed_args)
                    except Exception as tool_exc:
                        logger.exception("Tool execution failed for %s: %s", fn_name, tool_exc)
                        tool_payload = {"error": f"{type(tool_exc).__name__}: {tool_exc}"}
                        q_resp = None

                    call_ms = round((time.perf_counter() - t_call) * 1000.0, 2)
                    matched_cnt = q_resp.matched_count if q_resp is not None else 0
                    summary = self._build_tool_summary(fn_name, parsed_args, tool_payload, q_resp)
                    preview = self._build_result_preview(tool_payload, q_resp)

                    traces.append(
                        ToolCallTrace(
                            step=step_counter,
                            tool_name=fn_name,
                            arguments=parsed_args,
                            matched_count=matched_cnt,
                            summary=summary,
                            result_preview=preview,
                            latency_ms=call_ms,
                        )
                    )

                    queried_tid = str(parsed_args.get("table_id") or "").strip()
                    if queried_tid and queried_tid not in seen_linked_ids:
                        extra_tp = active_store.get_table(queried_tid)
                        if extra_tp is not None:
                            seen_linked_ids.add(extra_tp.table_id)
                            final_linked_tables.append(extra_tp)

                    tool_json_str = _truncate_text(
                        json.dumps(tool_payload, ensure_ascii=False),
                        18000,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_dict["id"],
                            "content": tool_json_str,
                        }
                    )

            if not final_answer:
                final_answer = self._synthesize_fallback_answer(
                    query=raw_query,
                    search_resp=search_resp,
                    active_store=active_store,
                    enable_table_tools=req.enable_table_tools,
                    traces=traces,
                )
                if not model_used:
                    model_used = "deterministic-fallback"

        except Exception as llm_exc:
            logger.warning("Falling back to deterministic RAG synthesizer due to LLM error: %s", llm_exc)
            final_answer = self._synthesize_fallback_answer(
                query=raw_query,
                search_resp=search_resp,
                active_store=active_store,
                enable_table_tools=req.enable_table_tools,
                traces=traces,
            )
            model_used = "deterministic-fallback"

        return AgentAskResponse(
            query=raw_query,
            answer=final_answer,
            model_used=model_used,
            used_table_tools=bool(traces),
            tool_calls=traces,
            search_response=search_resp,
            linked_tables=final_linked_tables,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )


_RAG_AGENT_SINGLETON: TableAwareRAGAgent | None = None


def get_rag_agent(
    search_service: SearchService | None = None,
    table_store: StructuredTableStore | None = None,
) -> TableAwareRAGAgent:
    """Return process-wide TableAwareRAGAgent singleton."""
    global _RAG_AGENT_SINGLETON
    if _RAG_AGENT_SINGLETON is None:
        _RAG_AGENT_SINGLETON = TableAwareRAGAgent(
            search_service=search_service,
            table_store=table_store,
        )
    else:
        if search_service is not None and _RAG_AGENT_SINGLETON.search_service is not search_service:
            _RAG_AGENT_SINGLETON.search_service = search_service
        if table_store is not None and _RAG_AGENT_SINGLETON._table_store is not table_store:
            _RAG_AGENT_SINGLETON._table_store = table_store
    return _RAG_AGENT_SINGLETON
