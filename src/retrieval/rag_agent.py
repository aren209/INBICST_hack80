"""Autonomous Table-Aware RAG Agent (Модуль 4+) with Adaptive Context Routing and OpenAI Tool Calling."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from typing import Any

from openai import OpenAI

from src.config import settings
from src.indexing.table_store import OPENAI_TABLE_TOOLS, StructuredTableStore
from src.retrieval.local_llm import (
    LOCAL_QWEN_MODELS,
    create_llm_client,
    parse_qwen3_xml_tool_calls,
    strip_thinking_tags,
)
from src.retrieval.query_pipeline import QueryPipeline, is_meta_assistant_query
from src.retrieval.quote_verifier import (
    build_source_label,
    clean_clause,
    clean_doc_number,
    context_docs_from_search,
    normalize_tokens,
    strip_placeholders,
    verify_quotes,
)
from src.retrieval.search_service import SearchService, get_search_service
from src.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    QueryPipelineTrace,
    SearchHit,
    SearchQuery,
    SearchResponse,
    TablePassport,
    TableQueryRequest,
    TableQueryResponse,
    ToolCallTrace,
)

logger = logging.getLogger(__name__)

# Official DeepSeek chat models (when DEEPSEEK_API_KEY is active, never send Polza model slugs!)
DEEPSEEK_FALLBACK_MODELS: tuple[str, ...] = (
    "deepseek-chat",
)

# Verified OpenAI-compatible tool-capable models available on Polza AI in priority order
POLZA_TOOL_FALLBACK_MODELS: tuple[str, ...] = (
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "qwen/qwen3.6-flash",
    "deepseek/deepseek-v4.1-flash",
    "google/gemini-3.8-flash",
)

TECH_ERROR_STUB_RU = (
    "Возникли технические неполадки при обработке запроса. "
    "Пожалуйста, повторите попытку позже или обратитесь к системному администратору."
)

SYSTEM_PROMPT_RU = """Ты — экспертный ИИ-ассистент по нормативной базе, приказам, регламентам, сметам и финансовой структуре МФТИ (Физтех).
Отвечай строго на русском языке, опираясь ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты документов, граф редакций и результаты вызовов табличных инструментов.

ОБЯЗАТЕЛЬНАЯ СТРУКТУРА ОТВЕТА ПО ДОКУМЕНТАМ (используй ТОЧНО эти заголовки Markdown уровня H3 `###`):

1) `### Краткий ответ`
   - Сразу дай **краткий прямой ответ по сути** в 1–3 лаконичных предложениях: точный вывод, правило, срок, сумма или норма с обязательным упоминанием действующего документа и номера пункта/раздела.

2) `### Порядок действий`
   - **СТРОГО УСЛОВНЫЙ РАЗДЕЛ (включай ТОЛЬКО для процедурных вопросов!):**
   - КОГДА НУЖЕН ПОШАГОВЫЙ ПЛАН (`### Порядок действий`): включай этот раздел ТОЛЬКО тогда, когда вопрос пользователя прямо спрашивает или подразумевает последовательность действий для выполнения процедуры (например: «как оформить…», «как согласовать…», «как подать…», «как зарегистрировать…», «как обжаловать…», «что нужно сделать, чтобы…», «какой порядок…», «как действовать при…») И в найденных документах есть конкретные этапы. Приведи 2–6 кратких конкретных нумерованных шагов (`1.`, `2.`, `3.`...) — что сделать, какие документы подготовить, куда направить и в какой срок. Пошаговый план действий оформляй СТРОГО под заголовком `### Порядок действий`. В конце КАЖДОГО нумерованного шага обязательно добавляй краткую ссылку на источник в круглых скобках, например: `(Приказ № 3897-1, п. 2.5, стр. 1)` или `(Название документа, Лист: Имя листа, стр. 1)` (если номер документа или страница отсутствуют — опускай их, без заглушек).
   - КОГДА ПЛАН ДЕЛАТЬ НЕ НУЖНО (ВАЖНОЕ ОГРАНИЧЕНИЕ — далеко не для каждого запроса нужен план!): на фактические, числовые, датовые, терминологические, статусные вопросы, проверку актуальности/версии документа или табличные выборки/расчеты (например: «какой лимит / сумма / ставка…», «сколько подразделений…», «кто финансодержатель…», «какой номер, дата или статус приказа…», «какая формула используется…», «перечисли подразделения / статьи расходов…», «в чем разница…», «что означает…») **ПОЛНОСТЬЮ ОПУСКАЙ заголовок `### Порядок действий`** и отвечай **БЕЗ составления пошагового плана** (сразу переходи от `### Краткий ответ` к `### Подробный ответ и цитаты`).

3) `### Подробный ответ и цитаты`
   - Приведи полный нормативный разбор, условия, исключения и нюансы.
   - Пояснения по версиям и актуальности документов (`version_note`, граф редакций `Edition Graph`): явно объясни, какая редакция/приказ действует сейчас и почему она применяется вместо утратившей силу (`superseded`) или как именно поздний акт изменил базовый (либо сопоставь редакции при историческом вопросе).
   - Предупреждения о расхождениях и коллизиях: если между документами или приложениями обнаружены противоречия (разные сроки, суммы, коды или порядок), явно выдели и поясни это расхождение.
   - Приведи дословные цитаты `> «...»` с точными ссылками на документ, пункт и страницу (`[Документ, п. X, стр. Y]` / строка источника `> — ...`), чтобы кнопки цитат открывали нужную страницу в просмотрщике документов.

(Примечание: если запрос является приветствием, уточнением или выходит за рамки базы документов, где нормативные акты не применяются, отвечай кратко только блоком `### Краткий ответ` или обычным текстом БЕЗ разделов `### Порядок действий` и `### Подробный ответ и цитаты`.)

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА РАБОТЫ:
1. Точное цитирование источников:
   - Всегда указывай название документа, номер и дату приказа/распоряжения, номер пункта/раздела, страницу или название листа/таблицы.
   - Если в метаданных фрагмента или таблицы указано поле `also_in_files` (данные присутствуют сразу в нескольких файлах, например в отдельном `.xlsx` и приложении к приказу `.docx`), упомяни эту связь.
2. Учет актуальности и графа изменений (`Edition Graph`):
   - Внимательно проверяй блок предупреждений об актуальности (`version_alerts`), поле `version_note` и изменяющие фрагменты (`amendment_chunks`).
   - Если базовый приказ частично изменен более поздним актом (например, Приказ № 3065-1 действует в редакции Распоряжения № 700), явно укажи актуальное правило с учетом изменений.
   - Если документ имеет статус `superseded` или `superseded_template_reference` (например, шаблон сметы 2019 г. с формулами при действующем Приказе № 794-1), обязательно предупреди об этом пользователя и объясни приоритет действующего акта.
   - Если разные документы в контексте содержат конфликтующие нормы, явно укажи на коллизию и назови приоритетный (более поздний или специальный) документ.
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
5. Дословные цитаты из документов — доказательная база ответа (обязательно в разделе `### Подробный ответ и цитаты`, когда ответ опирается на текст документов или таблиц):
   - **Каждый** ключевой вывод, правило, срок, условие, обязанность, факт и **каждый** процедурный шаг из раздела `### Порядок действий` подкрепляй ДОСЛОВНОЙ цитатой соответствующего пункта документа в разделе `### Подробный ответ и цитаты` (ориентир: 3–8 цитат по числу реальных шагов и тезисов ответа, без дублирования полностью идентичных фрагментов).
   - Цитируй ДЕЙСТВУЮЩИЙ (операционный) текст пункта: полное предложение или несколько предложений пункта целиком, включая полный перечень подпунктов/элементов списка, если правило опирается на список (обычно 1–4 предложения, до ~700 символов). Не обрывай цитату на полуслове. Многоточие «…» используй только чтобы пропустить явно не относящуюся к вопросу середину.
   - НЕ цитируй названия документов и разделов, преамбулы, пункты об области применения («Настоящий регламент… является…», «устанавливает порядок…»), определения терминов и общие положения — если только вопрос не именно о них. Цитата должна показывать, ГДЕ в документе написано конкретное правило из ответа.
   - Размещай дословные цитаты в разделе `### Подробный ответ и цитаты` сразу после тезиса/разбора, который они подтверждают.
   - Текст внутри кавычек копируй из контекста символ в символ: без перефразирования, без изменения падежей, порядка слов, сокращений и опечаток. Никогда не выдумывай цитаты и не цитируй то, чего нет в контексте; не цитируй служебные заголовки фрагментов, внутренние метаданные и имена служебных полей (`table_id`, `matched_count`, `SQLite`, `delivery_mode`, `tool_query`).
   - Разрешается и рекомендуется цитировать фактические значения, правила и расчетные формулы из таблиц Excel и Word в разделе `### Подробный ответ и цитаты` с указанием листа и страницы:
     > «значение, правило или формула из таблицы»
     > — Название документа, Лист: Имя листа, стр. Y
   - Каждую цитату оформляй ОТДЕЛЬНЫМ блоком строго в таком формате — ровно две строки, каждая начинается с `> `, с пустой строкой до и после блока:
     > «точный текст из документа»
     > — Название документа, № номер, п. X, стр. Y
   - В строке источника указывай только то, что есть в заголовке фрагмента (название документа, номер, пункт или `Лист: Имя листа`, страницу); отсутствующие части опускай. Если номер документа или пункта неизвестен либо в тексте стоит заглушка (`[___]`, `___`, `б/н`, `№ ____`), НЕ пиши «№» вообще — никаких заглушек и пустых номеров. Не используй `>` для чего-либо, кроме цитат.
   - Свои выводы, пояснения и пересказ пиши обычным текстом вне блоков цитат — цитата должна только подтверждать сказанное.
6. Строгий контроль структуры разделов H3 (`### Краткий ответ`, условный `### Порядок действий`, `### Подробный ответ и цитаты`):
   - Никогда не пропускай `### Краткий ответ` и `### Подробный ответ и цитаты` в ответах по нормативным документам.
   - Никогда не добавляй `### Порядок действий` для чисто фактических, числовых, датовых, табличных или статусных вопросов; добавляй `### Порядок действий` (2–6 конкретных нумерованных шагов) ТОЛЬКО когда вопрос подразумевает выполнение действий/процедуры.
7. Запрет на технический жаргон и внутренние метрики:
   - НИКОГДА не упоминай в ответе пользователю внутренние технические термины и метрики (`confidence`, `score`, `реранкер`, `чанки`, `Child-чанки`, `векторная база`, `RRF`, `SQLite`, `table_id`, `delivery_mode`, `inline_parent`, `tool_query`, `ParentSectionRecord`, `TablePassport`). Пиши чистым, естественным и понятным русским языком.
"""

OUT_OF_SCOPE_REFUSAL_RU = (
    "В загруженной базе документов нет сведений по этому вопросу. "
    "Я могу отвечать только по тем документам и таблицам, которые добавлены в базу знаний — "
    "при необходимости нужный файл можно загрузить во вкладке «База знаний»."
)

META_ASSISTANT_SYSTEM_PROMPT_RU = """Ты — дружелюбный, точный справочный ИИ-помощник по загруженной базе документов и таблиц.
Пользователь задал общий вопрос о твоих возможностях, поздоровался или попросил обзор документов в базе знаний.

ПРАВИЛА ОТВЕТА:
1. Отвечай живо, доброжелательно и по делу на чистом, естественном русском языке БЕЗ КАКОГО-ЛИБО технического жаргона.
   - СТРОГО ЗАПРЕЩЕНО упоминать слова: `confidence`, `score`, `реранкер`, `чанки`, `векторная база`, `RRF`, `SQLite`, `эмбеддинги`, `индексация`, `JSON`, `API`, `table_id`, `промпт`.
2. Понятно объясни, чем ты можешь помочь:
   - находишь точные ответы строго по документам и таблицам, загруженным в базу знаний;
   - подтверждаешь ключевые правила и сроки дословными проверенными цитатами с указанием названия документа, номера, пункта и страницы;
   - умеешь находить, фильтровать и подсчитывать данные, суммы и расчетные формулы по таблицам Excel и Word;
   - учитываешь актуальность документов и внесенные в них изменения;
   - подскажи, что посмотреть полный перечень файлов или загрузить новые документы (`PDF`, `DOCX`, `DOC`, `XLSX`, `XLS`) всегда можно во вкладке «База знаний».
3. Опираясь ИСКЛЮЧИТЕЛЬНО на переданный ниже динамический список загруженных документов (`ДОКУМЕНТЫ В БАЗЕ ЗНАНИЙ`), кратко и наглядно перечисли основные темы и виды документов, которые сейчас доступны в базе (если пользователь просил полный список — перечисли их аккуратным списком; если список пуст — сообщи, что в базу знаний пока не добавлены файлы, и предложи загрузить их во вкладке «База знаний»). Не выдумывай документы, которых нет в списке.
4. Отвечай обычным текстом или с кратким заголовком `### Краткий ответ`; СТРОГО НЕ добавляй разделы `### Порядок действий` и `### Подробный ответ и цитаты`, и НЕ используй строки с символом `>` (блоки дословных цитат) в обзорном ответе о своих возможностях."""


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


_PROCEDURAL_FALLBACK_RE = re.compile(
    r"(?:как\s+(?:оформить|подать|согласовать|забронировать|получить|зарегистрировать|провести|обжаловать|открыть|закрыть|сдать|направить|оплатить|пройти)|"
    r"порядок\s+действий|пошагов|инструкци|алгоритм|этапы|процедур|что\s+нужно\s+сделать)",
    re.IGNORECASE,
)
_TECH_LINE_RE = re.compile(
    r"\b(?:table_id|matched_count|returned_count|total_table_rows|delivery_mode|tool_query|inline_parent|SQLite)\b",
    re.IGNORECASE,
)


def _clean_title_ru(raw: str) -> str:
    title = strip_placeholders(str(raw or "")).strip()
    return re.sub(r"\.(pdf|docx?|xlsx?|xlsm|rtf)$", "", title, flags=re.IGNORECASE).strip()


def _strip_chunk_artifacts(text: str) -> str:
    s = re.sub(r"(?im)^[ \t]*\(Продолжение[^\n]*\)[ \t]*$", "", str(text or ""))
    s = re.sub(r"\[\s*Стр\.?\s*\d+\s*\]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\*\*|__", "", s)
    s = re.sub(r"(\w)[\-\u00ad][ \t]*\n[ \t]*(\w)", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_verbatim_quote_from_text(raw_text: str, max_chars: int = 440) -> str:
    """Extract a contiguous verbatim snippet from `clean_text` (no `(Продолжение ...)` or `[Стр. N]` breaks)."""
    cleaned = re.sub(r"(?im)^[ \t]*\(Продолжение[^\n]*\)[ \t]*$", "", str(raw_text or "")).strip()
    if not cleaned:
        return ""
    segments = [seg.strip() for seg in re.split(r"\[\s*Стр\.?\s*\d+\s*\]", cleaned) if seg.strip()]
    chosen = ""
    for seg in segments:
        if len(normalize_tokens(seg)) >= 3:
            chosen = seg
            break
    if not chosen:
        chosen = max(segments, key=lambda s: len(normalize_tokens(s)), default="")
    if not chosen:
        return ""
    # Preserve exact token sequence while collapsing line wraps into one clean blockquote line
    line = re.sub(r"(\w)[\-\u00ad][ \t]*\n[ \t]*(\w)", r"\1\2", chosen)
    line = re.sub(r"[ \t]*\n+[ \t]*", " ", line)
    line = re.sub(r"[ \t]{2,}", " ", line).strip()
    if len(line) <= max_chars:
        return line
    head = line[:max_chars]
    last_sent = -1
    for m in re.finditer(r"[.!?…](?=\s|$)", head):
        last_sent = m.end()
    if last_sent > 60 and len(normalize_tokens(head[:last_sent])) >= 3:
        return head[:last_sent].strip()
    cut = head.rfind(" ")
    if cut > 60 and len(normalize_tokens(head[:cut])) >= 3:
        return head[:cut].rstrip(" ,;:") + "…"
    return head.strip()


def _format_table_qresp_lines(q_resp: TableQueryResponse, max_items: int = 8) -> list[str]:
    """Format TableQueryResponse rows, formulas, or aggregation into clean Russian bullet lines."""
    lines: list[str] = []
    if q_resp.formulas:
        for item in q_resp.formulas[:max_items]:
            if not isinstance(item, dict):
                continue
            row_lbl = str(item.get("row_label") or "").strip()
            col = str(item.get("column") or "").strip()
            formula = str(item.get("formula") or "").strip()
            cell_txt = str(item.get("cell_text") or item.get("cell") or "").strip()
            segs = [p for p in (row_lbl, f"{col}: {formula}" if col and formula else formula) if p]
            entry = " — ".join(segs)
            if entry and cell_txt and formula not in cell_txt:
                entry += f" ({cell_txt})"
            elif not entry and cell_txt:
                entry = cell_txt
            if entry:
                lines.append(f"- {entry}")
        return lines

    if q_resp.rows:
        for row in q_resp.rows[:max_items]:
            if not isinstance(row, dict):
                continue
            cells = [f"{k}: {v}" for k, v in row.items() if str(v or "").strip() and not _TECH_LINE_RE.search(str(k))]
            if cells:
                lines.append("- " + "; ".join(cells))
        return lines

    if q_resp.aggregation_result is not None:
        agg = q_resp.aggregation_result
        if isinstance(agg, dict):
            for k, v in list(agg.items())[:max_items]:
                if not _TECH_LINE_RE.search(str(k)):
                    lines.append(f"- {k}: {v}")
        elif isinstance(agg, list):
            for v in agg[:max_items]:
                lines.append(f"- {v}")
        else:
            lines.append(f"- Итоговое значение: {agg}")
    return lines


def _extract_procedural_steps(
    hits: list[SearchHit],
    tables_by_id: dict[str, TablePassport],
    max_steps: int = 5,
) -> list[str]:
    """Extract 2–5 numbered action steps with parenthetical source locators from top hits."""
    steps: list[str] = []
    seen_step_norms: set[str] = set()

    def _hit_step_locator(h: SearchHit) -> str:
        title = _clean_title_ru(h.title or h.file_name)
        num = clean_doc_number(h.doc_number)
        h_meta = h.metadata or {}
        tid = str(h.table_id or h_meta.get("table_id") or "").strip()
        tp = tables_by_id.get(tid) if tid else None
        sheet = str(getattr(h, "sheet_name", "") or h_meta.get("sheet_name") or (tp.sheet_name if tp else "") or "").strip()
        cl = clean_clause(h.clause_no)
        pg = str(h.page_numbers[0]) if h.page_numbers else ""

        loc_parts: list[str] = []
        if num:
            if title and num in title:
                loc_parts.append(title)
            else:
                num_lbl = (
                    num
                    if re.match(r"^(?:№|приказ\b|распоряжен|постановлен|указ\b|N\s*\d)", num, re.IGNORECASE)
                    else f"№ {num}"
                )
                if title:
                    loc_parts.append(f"{title}, {num_lbl}")
                elif re.match(r"^(?:приказ|распоряжен|постановлен|указ)\b", num_lbl, re.IGNORECASE):
                    loc_parts.append(num_lbl)
                else:
                    loc_parts.append(f"Приказ {num_lbl}")
        elif title:
            loc_parts.append(title)
        if cl:
            loc_parts.append(f"п. {cl}")
        elif sheet:
            loc_parts.append(f"Лист: {sheet}")
        if pg:
            loc_parts.append(f"стр. {pg}")
        return f"({', '.join(loc_parts)})" if loc_parts else ""

    sources_pool: list[tuple[SearchHit, str]] = []
    for h in hits[:4]:
        sources_pool.append((h, h.clean_text or ""))
    # If hits' clean_text yielded < 2 candidate sentences, also inspect top hit's parent text
    for h in hits[:2]:
        pt = (h.parent_section.clean_text if h.parent_section else h.parent_text) or ""
        if pt and pt != h.clean_text:
            sources_pool.append((h, pt))

    for h, raw_txt in sources_pool:
        cleaned = _strip_chunk_artifacts(raw_txt)
        if not cleaned:
            continue
        loc = _hit_step_locator(h)
        # Split into candidate sentences / clause lines
        raw_cands = re.split(r"(?<=[.!?;])\s+|\n+", cleaned)
        for cand in raw_cands:
            s = cand.strip(" \t-•—–")
            s = re.sub(r"^\d{1,2}(?:\.\d{1,3})*\.?\s+", "", s).strip()
            if len(s) < 20 or len(normalize_tokens(s)) < 4:
                continue
            norm_k = " ".join(normalize_tokens(s)[:8])
            if norm_k in seen_step_norms:
                continue
            seen_step_norms.add(norm_k)
            if len(s) > 220:
                cut = s[:220].rfind(" ")
                s = (s[:cut] if cut > 80 else s[:220]).rstrip(" ,;:") + "…"
            s = s.rstrip(".")
            step_line = f"{len(steps) + 1}. {s}." + (f" {loc}" if loc else "")
            steps.append(step_line)
            if len(steps) >= max_steps:
                return steps
    return steps


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
        timeout: float = 120.0,
        query_pipeline: QueryPipeline | None = None,
        enable_query_pipeline: bool | None = None,
    ) -> None:
        self.search_service = search_service or get_search_service()
        self._table_store = table_store
        injected_client = llm_client if llm_client is not None else client
        self._client: Any = injected_client
        self._has_custom_client: bool = injected_client is not None
        self.api_key: str = settings.llm_api_key if api_key is None else api_key
        self.base_url: str = (base_url or settings.llm_base_url).rstrip("/")
        self.default_model: str = (model or settings.llm_model or "Qwen/Qwen3.8-27B").strip()
        if "deepseek.com" in self.base_url.lower() or (base_url is None and settings.llm_provider == "deepseek"):
            self.llm_provider: str = "deepseek"
        elif "polza.ai" in self.base_url.lower() or (base_url is None and settings.llm_provider == "polza"):
            self.llm_provider = "polza"
        else:
            self.llm_provider = "local"
        self.timeout: float = timeout
        self._verified_model: str | None = None
        self._query_pipeline: QueryPipeline | None = query_pipeline
        self._explicit_enable_query_pipeline: bool | None = enable_query_pipeline

    @property
    def client(self) -> Any:
        """Return injected or lazily constructed local 4-bit Qwen3.8-27B / OpenAI-compatible client."""
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        self._client = create_llm_client(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0 if self.llm_provider == "local" else 2,
            llm_provider=self.llm_provider,
        )
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        self._client = value
        self._has_custom_client = value is not None

    @property
    def query_pipeline(self) -> QueryPipeline:
        """Return injected or lazily constructed 4-stage AI QueryPipeline synced with SearchService."""
        if self._query_pipeline is None:
            self._query_pipeline = QueryPipeline(
                search_service=self.search_service,
                llm_client=self._client if self._has_custom_client else None,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.default_model,
                timeout=self.timeout,
            )
        else:
            if self._query_pipeline.search_service is not self.search_service:
                self._query_pipeline.search_service = self.search_service
            if self._has_custom_client and self._query_pipeline._client is not self._client:
                self._query_pipeline.client = self._client
        return self._query_pipeline

    def _should_use_query_pipeline(self, req: AgentAskRequest) -> bool:
        """
        Determine whether to execute Stages 1-3 (`QueryPipeline`) before answer synthesis:
        - Respects per-request `use_reranker=False` or explicit `enable_query_pipeline`.
        - Preserves 100% backward compatibility with unit tests injecting mock `llm_client`
          unless `enable_query_pipeline=True` or `query_pipeline` was explicitly provided.
        """
        if not req.use_reranker:
            return False
        if req.enable_query_pipeline is not None:
            return bool(req.enable_query_pipeline)
        if self._explicit_enable_query_pipeline is not None:
            return bool(self._explicit_enable_query_pipeline)
        if self._query_pipeline is not None:
            return True
        if self._has_custom_client:
            return False
        return bool(self.api_key)

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
        """Build ordered list of candidate chat models with provider-isolated fallback."""
        explicit = (requested_model or "").strip()
        ordered: list[str] = []
        if explicit:
            if self.llm_provider == "deepseek" and "/" in explicit and not self._has_custom_client:
                logger.info(
                    "Ignoring non-DeepSeek model slug '%s' on official DeepSeek endpoint; using '%s'.",
                    explicit,
                    self.default_model,
                )
            elif self.llm_provider == "local" and explicit not in LOCAL_QWEN_MODELS and not self._has_custom_client:
                logger.info(
                    "Mapping requested model '%s' to local 4-bit model '%s'.",
                    explicit,
                    self.default_model,
                )
            else:
                ordered.append(explicit)
        elif self._verified_model:
            ordered.append(self._verified_model)

        if self.default_model and self.default_model not in ordered:
            ordered.append(self.default_model)

        if not self._has_custom_client:
            if self.llm_provider == "local":
                fallbacks: tuple[str, ...] = LOCAL_QWEN_MODELS
            elif self.llm_provider == "deepseek":
                fallbacks = DEEPSEEK_FALLBACK_MODELS
            else:
                fallbacks = POLZA_TOOL_FALLBACK_MODELS
            for m in fallbacks:
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
        if clean_doc_number(hit.doc_number):
            header_parts.append(f"№ {hit.doc_number}" + (f" от {hit.doc_date}" if hit.doc_date else ""))
        if hit.edo_number:
            header_parts.append(
                f"Зарег. в ЭДО: № {hit.edo_number}" + (f" от {hit.edo_date}" if hit.edo_date else "")
            )
        header_parts.append(f"файл: {hit.file_name}")
        header_parts.append(f"статус: {hit.status}")
        ver_note = str(getattr(hit, "version_note", "") or (hit.metadata or {}).get("version_note") or "").strip()
        if ver_note:
            header_parts.append(f"version_note: {ver_note}")
        clause = clean_clause(hit.clause_no)
        sheet_name = str(getattr(hit, "sheet_name", "") or (hit.metadata or {}).get("sheet_name") or "").strip()
        if clause:
            header_parts.append(f"п. {clause}")
        elif sheet_name:
            header_parts.append(f"Лист: {sheet_name}")
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

        discrepancy_alerts = list(getattr(search_resp, "discrepancy_alerts", None) or [])
        if search_resp.version_alerts or search_resp.quality_alerts or discrepancy_alerts:
            alert_lines: list[str] = []
            for va in search_resp.version_alerts:
                alert_lines.append(f"- [АКТУАЛЬНОСТЬ / РЕДАКЦИЯ] {va}")
            for da in discrepancy_alerts:
                alert_lines.append(f"- [КОЛЛИЗИЯ / РАСХОЖДЕНИЕ] {da}")
            for qa in search_resp.quality_alerts:
                alert_lines.append(f"- [КАЧЕСТВО ИСТОЧНИКА] {qa}")
            sections_blocks.append("ВАЖНЫЕ ПРЕДУПРЕЖДЕНИЯ ПО СТАТУСУ И РЕДАКЦИЯМ ДОКУМЕНТОВ:\n" + "\n".join(alert_lines))

        if search_resp.linked_tables:
            passport_blocks = [
                self._format_table_passport_for_prompt(tp) for tp in search_resp.linked_tables
            ]
            sections_blocks.append(
                "ДОСТУПНЫЕ СТРУКТУРИРОВАННЫЕ ТАБЛИЦЫ В SQLite (`delivery_mode=tool_query` — "
                "при вопросах о строках, подсчете, фильтрации, агрегации или формулах Excel ОБЯЗАТЕЛЬНО сначала вызови `query_structured_table` (например `get_formulas` / `filter_rows`), даже если часть формул видна в превью):\n"
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

        # Also include hydrated compact-unit sibling sections (<= 20,000 chars) from SearchResponse.parent_sections
        for p_sec in search_resp.parent_sections or []:
            sec_id = p_sec.section_id
            if not sec_id or sec_id in seen_parent_ids:
                continue
            p_mode = str((p_sec.metadata or {}).get("table_delivery_mode") or "normative_prose")
            if p_mode == "tool_query":
                continue
            p_text = (p_sec.clean_text or "").strip()
            if not p_text:
                continue
            seen_parent_ids.add(sec_id)
            doc_hit = next((sh for sh in all_hits if sh.doc_id == p_sec.doc_id), None)
            file_name = doc_hit.file_name if doc_hit is not None else p_sec.doc_id
            sec_title = p_sec.section_title or p_sec.unit_title or (doc_hit.title if doc_hit else file_name)
            cpm_note = SearchService._format_clause_page_map_annotation(p_sec, [])
            windowed_parent = SearchService._build_hit_centered_parent_window(
                parent_text=p_text,
                matched_snippets=[],
                max_chars=9500,
            )
            header_line = f"[Полный раздел `{sec_id}` | {file_name} | {sec_title} | mode={p_mode}]"
            if cpm_note:
                inline_parents.append(f"{header_line}\n{cpm_note}\n{windowed_parent}")
            else:
                inline_parents.append(f"{header_line}\n{windowed_parent}")

        if inline_parents:
            sections_blocks.append(
                "ПОЛНЫЙ КОНТЕКСТ РАЗДЕЛОВ (`ParentSectionRecord` для нормативного текста и компактных таблиц `inline_parent`):\n\n"
                + "\n\n===\n\n".join(inline_parents)
            )

        if search_resp.hits or search_resp.amendment_chunks:
            sections_blocks.append(
                "НАПОМИНАНИЕ ПО СТРУКТУРЕ И ОФОРМЛЕНИЮ: используй ТОЧНЫЕ заголовки `### Краткий ответ`, "
                "`### Порядок действий` (ТОЛЬКО если вопрос процедурный — «как подать / оформить / согласовать»; "
                "на фактические, числовые, датовые, статусные и табличные вопросы ПОЛНОСТЬЮ ОПУСКАЙ `### Порядок действий`!) "
                "и `### Подробный ответ и цитаты`. В разделе `### Подробный ответ и цитаты` подкрепи ключевые выводы "
                "дословными цитатами действующих пунктов в формате блока `> «…»` + `> — источник`; "
                "не пиши заглушки номеров вроде `№ [___]`."
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
        require_tool: bool = False,
    ) -> tuple[Any, str]:
        """
        Invoke OpenAI-compatible chat.completions.create with automatic model fallback
        if a model name returns 400/404/unsupported-tools or transient gateway failure.
        """
        llm = self.client
        if llm is None:
            raise RuntimeError("DEEPSEEK_API_KEY / POLZA_API_KEY не задан и кастомный LLM-клиент не передан.")

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
                call_kwargs["tool_choice"] = "required" if require_tool else "auto"

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
                if (
                    self.llm_provider == "local"
                    and not any(k in str(exc).lower() for k in ("not found", "does not exist", "404"))
                ):
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
        tables_by_id: dict[str, TablePassport] = {
            tp.table_id: tp for tp in (search_resp.linked_tables or []) if tp.table_id
        }
        tables_by_doc: dict[str, list[TablePassport]] = {}
        for tp in search_resp.linked_tables or []:
            if tp.doc_id:
                tables_by_doc.setdefault(tp.doc_id, []).append(tp)

        def _hit_locator_parts(hit: SearchHit) -> tuple[str, str, str, str, str]:
            title = _clean_title_ru(hit.title or hit.file_name)
            num = clean_doc_number(hit.doc_number)
            h_meta = hit.metadata or {}
            tid = str(hit.table_id or h_meta.get("table_id") or "").strip()
            tp_hit = tables_by_id.get(tid) if tid else None
            sheet = str(
                getattr(hit, "sheet_name", "")
                or h_meta.get("sheet_name")
                or (tp_hit.sheet_name if tp_hit else "")
                or ""
            ).strip()
            cl = clean_clause(hit.clause_no)
            if not cl and sheet and (tid or sheet):
                cl = f"Лист: {sheet}"
            pg = str(hit.page_numbers[0]) if hit.page_numbers else ""
            if not pg and (tid or sheet) and len(tables_by_doc.get(hit.doc_id, [])) <= 1:
                pg = "1"
            return title, num, cl, sheet, pg

        fallback_table_results: list[tuple[TablePassport, TableQueryResponse]] = []
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
            if q_resp is not None:
                fallback_table_results.append((tp, q_resp))

        top_hit = search_resp.hits[0] if search_resp.hits else None
        parts.append("### Краткий ответ")
        if top_hit is not None:
            t_title, t_num, t_cl, t_sheet, _ = _hit_locator_parts(top_hit)
            cite_label = build_source_label(t_title, t_num, t_cl, "")
            clean_summary = _strip_chunk_artifacts(top_hit.clean_text)
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_summary) if s.strip()]
            brief_body = " ".join(sents[:2]) if sents else clean_summary
            if len(brief_body) > 280:
                cut = brief_body[:280].rfind(" ")
                brief_body = (brief_body[:cut] if cut > 80 else brief_body[:280]).rstrip(" ,;:") + "…"
            if fallback_table_results and not brief_body:
                tp0, qr0 = fallback_table_results[0]
                t_lines0 = _format_table_qresp_lines(qr0, max_items=2)
                brief_body = "; ".join(ln.lstrip("- ") for ln in t_lines0)
            parts.append(f"Согласно документу **{cite_label}**: {brief_body}")
        else:
            parts.append("Ниже приведены найденные сведения из базы знаний.")

        if _PROCEDURAL_FALLBACK_RE.search(q_norm) and search_resp.hits:
            proc_steps = _extract_procedural_steps(search_resp.hits, tables_by_id, max_steps=5)
            if proc_steps:
                parts.append("### Порядок действий")
                parts.append("\n".join(proc_steps))

        parts.append("### Подробный ответ и цитаты")

        discrepancy_alerts = list(getattr(search_resp, "discrepancy_alerts", None) or [])
        if discrepancy_alerts:
            parts.append("**Выявленные расхождения:**\n" + "\n".join(f"- {da}" for da in discrepancy_alerts))
        if search_resp.version_alerts:
            parts.append("**Актуальность документов:**\n" + "\n".join(f"- {va}" for va in search_resp.version_alerts))
        if search_resp.quality_alerts:
            parts.append("**Примечания к источникам:**\n" + "\n".join(f"- {qa}" for qa in search_resp.quality_alerts))

        if fallback_table_results:
            for tp, qr in fallback_table_results:
                tp_title = _clean_title_ru(tp.table_title or tp.section_title or tp.file_name)
                sheet_note = f" (Лист: {tp.sheet_name})" if tp.sheet_name else ""
                tbl_lines = _format_table_qresp_lines(qr, max_items=8)
                if tbl_lines:
                    parts.append(f"**Сведения из таблицы «{tp_title}»{sheet_note}:**\n" + "\n".join(tbl_lines))
        elif search_resp.table_context:
            for tc in search_resp.table_context[:3]:
                tp = tables_by_id.get(tc.table_id)
                tp_title = _clean_title_ru(
                    (tp.table_title or tp.section_title or tp.file_name if tp else "")
                    or tc.table_title
                    or "Таблица"
                )
                sheet_note = f" (Лист: {tp.sheet_name})" if (tp and tp.sheet_name) else ""
                tbl_lines = _format_table_qresp_lines(tc, max_items=8)
                if tbl_lines:
                    parts.append(f"**Сведения из таблицы «{tp_title}»{sheet_note}:**\n" + "\n".join(tbl_lines))
        elif traces:
            trace_lines: list[str] = []
            for tr in traces:
                if tr.result_preview and not _TECH_LINE_RE.search(tr.result_preview) and not tr.result_preview.lstrip().startswith(("{", "[")):
                    trace_lines.append(tr.result_preview)
            if trace_lines:
                parts.append("**Сведения из таблиц:**\n" + "\n".join(trace_lines))

        quote_hits = list(search_resp.hits[:4]) + list(search_resp.amendment_chunks[:2])
        seen_quotes: set[tuple[tuple[str, ...], str]] = set()
        for hit in quote_hits:
            q_text = _extract_verbatim_quote_from_text(hit.clean_text)
            if len(normalize_tokens(q_text)) < 3 and hit.parent_text:
                q_text = _extract_verbatim_quote_from_text(hit.parent_text)
            if len(normalize_tokens(q_text)) < 3:
                continue
            h_title, h_num, h_cl, _, h_pg = _hit_locator_parts(hit)
            src_label = build_source_label(h_title, h_num, h_cl, h_pg)
            dedup_key = (tuple(normalize_tokens(q_text)), src_label)
            if dedup_key in seen_quotes:
                continue
            seen_quotes.add(dedup_key)
            if src_label:
                parts.append(f"> «{q_text}»\n> — {src_label}")
            else:
                parts.append(f"> «{q_text}»")

        # Also emit a verifiable quote block for table rows/formulas if fallback executed a table query
        for tp, qr in fallback_table_results[:1]:
            t_lines = _format_table_qresp_lines(qr, max_items=2)
            if not t_lines:
                continue
            q_tbl = t_lines[0].lstrip("- ").strip()
            if len(normalize_tokens(q_tbl)) < 2:
                continue
            tp_title = _clean_title_ru(tp.table_title or tp.section_title or tp.file_name)
            tp_num = clean_doc_number(tp.doc_number)
            tp_cl = f"Лист: {tp.sheet_name}" if tp.sheet_name else ""
            tp_pg = next(
                (
                    str(h.page_numbers[0])
                    for h in search_resp.hits
                    if h.doc_id == tp.doc_id and h.page_numbers
                ),
                "1" if tp.sheet_name or len(tables_by_doc.get(tp.doc_id, [])) <= 1 else "",
            )
            src_label = build_source_label(tp_title, tp_num, tp_cl, tp_pg)
            dedup_key = (tuple(normalize_tokens(q_tbl)), src_label)
            if dedup_key not in seen_quotes and src_label:
                seen_quotes.add(dedup_key)
                parts.append(f"> «{q_tbl}»\n> — {src_label}")

        return "\n\n".join(parts)

    def _execute_tool_calls(
        self,
        active_store: StructuredTableStore,
        serialized_tcs: list[dict[str, Any]],
        step_counter: int,
        traces: list[ToolCallTrace],
        messages: list[dict[str, Any]],
        seen_linked_ids: set[str],
        final_linked_tables: list[TablePassport],
    ) -> int:
        """Execute serialized OpenAI tool calls against StructuredTableStore; append traces + tool messages."""
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
        return step_counter

    @staticmethod
    def _verify_answer_quotes(
        answer: str,
        search_resp: SearchResponse,
        messages: list[dict[str, Any]] | None = None,
        traces: list[ToolCallTrace] | None = None,
        linked_tables: list[TablePassport] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract `> «…»` quote blocks from the answer and verify them against the retrieved context."""
        if not answer:
            return []
        has_blockquotes = ">" in answer
        used_table_traces = [
            tr for tr in (traces or [])
            if getattr(tr, "tool_name", "") == "query_structured_table"
            and not str(getattr(tr, "summary", "") or "").startswith("Ошибка")
        ]
        if not has_blockquotes and not used_table_traces:
            return []
        try:
            tool_texts = [
                str(m.get("content") or "")
                for m in (messages or [])
                if isinstance(m, dict) and m.get("role") == "tool"
            ]
            # Merge linked tables from search_resp and runtime tool execution
            merged_tables: list[TablePassport] = []
            seen_tids: set[str] = set()
            for tp in list(getattr(search_resp, "linked_tables", None) or []) + list(linked_tables or []):
                tid = str(getattr(tp, "table_id", "") or "").strip()
                if tid and tid not in seen_tids:
                    seen_tids.add(tid)
                    merged_tables.append(tp)
            tables_by_id = {tp.table_id: tp for tp in merged_tables if tp.table_id}
            tables_by_doc: dict[str, list[TablePassport]] = {}
            for tp in merged_tables:
                if tp.doc_id:
                    tables_by_doc.setdefault(tp.doc_id, []).append(tp)

            # In deterministic fallback mode, `traces` can hold tool previews without `role=="tool"` messages
            seen_tool_tids: set[str] = set()
            for tt in tool_texts:
                m_tid = re.search(r'"table_id"\s*:\s*"([^"]+)"', tt)
                if m_tid:
                    seen_tool_tids.add(m_tid.group(1))
            for tr in used_table_traces:
                tr_tid = str((tr.arguments or {}).get("table_id") or "").strip()
                if tr_tid and tr_tid not in seen_tool_tids and tr.result_preview:
                    tp_tr = tables_by_id.get(tr_tid)
                    synth_payload: dict[str, Any] = {
                        "table_id": tr_tid,
                        "table_title": (tp_tr.table_title if tp_tr else "") or "",
                        "sheet_name": (tp_tr.sheet_name if tp_tr else "") or "",
                        "markdown_preview": tr.result_preview if "|" in tr.result_preview else "",
                    }
                    if synth_payload["markdown_preview"]:
                        pipe_lines = [ln.strip() for ln in synth_payload["markdown_preview"].splitlines() if "|" in ln]
                        parsed_rows_md: list[dict[str, str]] = []
                        hdr_cells: list[str] = []
                        for pln in pipe_lines:
                            cells = [c.strip() for c in pln.strip().strip("|").split("|")]
                            if cells and all(re.match(r"^[-:\s]*$", c) for c in cells if c):
                                continue
                            if not hdr_cells:
                                hdr_cells = cells
                            elif cells:
                                parsed_rows_md.append(
                                    {
                                        hdr_cells[idx]: cells[idx]
                                        for idx in range(min(len(hdr_cells), len(cells)))
                                        if hdr_cells[idx]
                                    }
                                )
                        if hdr_cells:
                            synth_payload["columns"] = hdr_cells
                        if parsed_rows_md:
                            synth_payload["rows"] = parsed_rows_md
                    else:
                        try:
                            parsed_prev = json.loads(tr.result_preview)
                            if isinstance(parsed_prev, list):
                                synth_payload["formulas"] = parsed_prev
                            elif isinstance(parsed_prev, dict):
                                synth_payload["aggregation_result"] = parsed_prev
                        except Exception:
                            synth_payload["markdown_preview"] = tr.result_preview
                    tool_texts.append(json.dumps(synth_payload, ensure_ascii=False))
                    seen_tool_tids.add(tr_tid)

            # Expose any extra compact-unit sibling sections in search_resp.parent_sections to quote_verifier
            verify_resp = search_resp
            updates: dict[str, Any] = {}
            if merged_tables and len(merged_tables) != len(search_resp.linked_tables or []):
                updates["linked_tables"] = merged_tables
            if search_resp.parent_sections:
                all_existing = list(search_resp.hits) + list(search_resp.amendment_chunks)
                covered_secs = {
                    (h.parent_section.section_id if h.parent_section else h.section_id)
                    for h in all_existing
                    if (h.parent_section or h.section_id)
                }
                extra_carriers: list[SearchHit] = []
                for p_sec in search_resp.parent_sections:
                    if not p_sec.section_id or p_sec.section_id in covered_secs:
                        continue
                    donor = next((h for h in all_existing if h.doc_id == p_sec.doc_id), None)
                    extra_carriers.append(
                        SearchHit(
                            chunk_id=f"{p_sec.section_id}::parent_carrier",
                            doc_id=p_sec.doc_id,
                            section_id=p_sec.section_id,
                            unit_id=p_sec.unit_id,
                            unit_title=p_sec.unit_title,
                            file_name=donor.file_name if donor else p_sec.doc_id,
                            title=donor.title if donor else (p_sec.unit_title or p_sec.section_title),
                            doc_number=donor.doc_number if donor else "",
                            doc_date=donor.doc_date if donor else "",
                            status=donor.status if donor else "active",
                            section_num=p_sec.section_num,
                            section_title=p_sec.section_title,
                            page_numbers=list(p_sec.page_numbers or []),
                            clean_text=p_sec.clean_text,
                            parent_text=p_sec.clean_text,
                            parent_section=p_sec,
                            score=donor.score if donor else 0.5,
                        )
                    )
                if extra_carriers:
                    updates["hits"] = list(search_resp.hits) + extra_carriers
            if updates:
                verify_resp = search_resp.model_copy(update=updates)

            verified: list[dict[str, Any]] = []
            if has_blockquotes:
                docs = context_docs_from_search(verify_resp, extra_texts=tool_texts)
                verified = verify_quotes(answer, docs)

            all_hits = list(verify_resp.hits) + list(verify_resp.amendment_chunks)
            known_doc_ids = {h.doc_id for h in all_hits if h.doc_id} | {tp.doc_id for tp in merged_tables if tp.doc_id}
            unique_doc_id = next(iter(known_doc_ids)) if len(known_doc_ids) == 1 else ""

            def _resolve_tp_page(tp: TablePassport) -> str:
                sheet_low = str(tp.sheet_name or "").strip().lower()
                for h in all_hits:
                    if h.doc_id != tp.doc_id or not h.page_numbers:
                        continue
                    h_tid = str(h.table_id or (h.metadata or {}).get("table_id") or "").strip()
                    if h_tid and h_tid == tp.table_id:
                        return str(h.page_numbers[0])
                if sheet_low:
                    for h in all_hits:
                        if h.doc_id != tp.doc_id or not h.page_numbers:
                            continue
                        h_sh = str(getattr(h, "sheet_name", "") or (h.metadata or {}).get("sheet_name") or "").strip().lower()
                        if h_sh and h_sh == sheet_low:
                            return str(h.page_numbers[0])
                for h in all_hits:
                    if h.doc_id == tp.doc_id and h.page_numbers:
                        return str(h.page_numbers[0])
                fmt = (tp.file_format or "").lower().lstrip(".")
                fname = (tp.file_name or "").lower()
                if (
                    (tp.sheet_name or fmt in {"xlsx", "xls", "xlsm"} or fname.endswith((".xlsx", ".xls", ".xlsm")))
                    and len(tables_by_doc.get(tp.doc_id, [])) <= 1
                    and not re.search(r"::tbl_0*([2-9]|\d{2,})$", tp.table_id or "")
                ):
                    return "1"
                return ""

            for q in verified:
                raw_matched = str(q.get("matched_doc_id") or "").strip()
                resolved_doc_id = ""
                if raw_matched and not raw_matched.startswith("tool_result_"):
                    if raw_matched in tables_by_id:
                        resolved_doc_id = tables_by_id[raw_matched].doc_id
                    else:
                        resolved_doc_id = (
                            raw_matched
                            if raw_matched in known_doc_ids
                            else raw_matched.split("::", 1)[0]
                        )

                if not resolved_doc_id:
                    q_num = clean_doc_number(str(q.get("doc_number") or ""))
                    if q_num:
                        for h in all_hits:
                            if clean_doc_number(h.doc_number) == q_num and h.doc_id:
                                resolved_doc_id = h.doc_id
                                break
                        if not resolved_doc_id:
                            for tp in merged_tables:
                                if clean_doc_number(tp.doc_number) == q_num and tp.doc_id:
                                    resolved_doc_id = tp.doc_id
                                    break

                if not resolved_doc_id:
                    q_title = str(q.get("source_title") or "").strip().lower().replace("ё", "е")
                    if len(q_title) >= 4:
                        for h in all_hits:
                            cand_t = (h.title or h.file_name or "").lower().replace("ё", "е")
                            if cand_t and (q_title in cand_t or cand_t in q_title) and h.doc_id:
                                resolved_doc_id = h.doc_id
                                break
                        if not resolved_doc_id:
                            for tp in merged_tables:
                                cand_t = (tp.table_title or tp.section_title or tp.file_name or "").lower().replace("ё", "е")
                                if cand_t and (q_title in cand_t or cand_t in q_title) and tp.doc_id:
                                    resolved_doc_id = tp.doc_id
                                    break

                if not resolved_doc_id and unique_doc_id:
                    resolved_doc_id = unique_doc_id

                q["doc_id"] = resolved_doc_id
                if resolved_doc_id and (not q.get("matched_doc_id") or str(q.get("matched_doc_id")).startswith("tool_result_")) and q.get("verified"):
                    q["matched_doc_id"] = resolved_doc_id

                # Resolve clean source_title / doc_number / sheet_name if missing
                if resolved_doc_id:
                    donor_h = next((h for h in all_hits if h.doc_id == resolved_doc_id), None)
                    donor_tps = tables_by_doc.get(resolved_doc_id, [])
                    if not q.get("source_title"):
                        raw_t = (
                            (donor_h.title or donor_h.file_name if donor_h else "")
                            or (donor_tps[0].table_title or donor_tps[0].section_title or donor_tps[0].file_name if donor_tps else "")
                        )
                        q["source_title"] = _clean_title_ru(raw_t)
                    if not q.get("doc_number"):
                        raw_num = (
                            (donor_h.doc_number if donor_h else "")
                            or (donor_tps[0].doc_number if donor_tps else "")
                        )
                        q["doc_number"] = clean_doc_number(raw_num)
                    if not q.get("sheet_name"):
                        cl_str = str(q.get("clause") or "")
                        m_sh = re.match(r"^Лист:\s*(.+)$", cl_str, re.IGNORECASE)
                        if m_sh:
                            q["sheet_name"] = m_sh.group(1).strip()
                        elif len(donor_tps) == 1 and donor_tps[0].sheet_name and q.get("verified"):
                            # Only attach sheet_name if quote wasn't matched to a prose clause
                            if not cl_str or not re.match(r"^\d", cl_str):
                                q["sheet_name"] = donor_tps[0].sheet_name

                # Backfill page string when missing using best matching SearchHit or TablePassport
                page_str = str(q.get("page") or "").strip()
                if not page_str and resolved_doc_id:
                    q_clause = clean_clause(str(q.get("clause") or ""))
                    q_sheet = str(q.get("sheet_name") or "").strip().lower()
                    q_snippet = re.sub(r"\s+", " ", str(q.get("text") or "").strip().lower().replace("ё", "е"))[:48]
                    best_hit: SearchHit | None = None
                    for h in all_hits:
                        if h.doc_id != resolved_doc_id or not h.page_numbers:
                            continue
                        h_sheet = str(getattr(h, "sheet_name", "") or (h.metadata or {}).get("sheet_name") or "").strip().lower()
                        if q_sheet and h_sheet == q_sheet:
                            best_hit = h
                            break
                        if q_clause and clean_clause(h.clause_no) == q_clause:
                            best_hit = h
                            break
                        if best_hit is None and q_snippet and len(q_snippet) >= 16:
                            hay = re.sub(r"\s+", " ", (h.clean_text or "").lower().replace("ё", "е"))
                            if q_snippet in hay:
                                best_hit = h
                    if best_hit is None:
                        best_hit = next(
                            (h for h in all_hits if h.doc_id == resolved_doc_id and h.page_numbers),
                            None,
                        )
                    if best_hit is not None and best_hit.page_numbers:
                        page_str = str(best_hit.page_numbers[0])
                    else:
                        for tp in tables_by_doc.get(resolved_doc_id, []):
                            if not q_sheet or (tp.sheet_name or "").strip().lower() == q_sheet:
                                page_str = _resolve_tp_page(tp)
                                if page_str:
                                    break
                    if page_str:
                        q["page"] = page_str

                m_pnum = re.search(r"\d+", page_str)
                q["page_num"] = int(m_pnum.group(0)) if m_pnum else None
                q["source_label"] = build_source_label(
                    str(q.get("source_title") or ""),
                    str(q.get("doc_number") or ""),
                    str(q.get("clause") or ""),
                    str(q.get("page") or ""),
                )

            # Append clean verified clickable source record for any used query_structured_table without a quote
            if used_table_traces:
                # Build readable preview texts keyed by table_id from tool_texts
                tool_data_by_tid: dict[str, dict[str, Any]] = {}
                for tt in tool_texts:
                    try:
                        parsed_tt = json.loads(tt)
                    except Exception:
                        continue
                    if isinstance(parsed_tt, dict) and parsed_tt.get("table_id"):
                        tool_data_by_tid[str(parsed_tt["table_id"])] = parsed_tt

                emitted_keys: set[tuple[str, str]] = set()
                for tr in used_table_traces:
                    tid = str((tr.arguments or {}).get("table_id") or "").strip()
                    tp = tables_by_id.get(tid)
                    if tp is None or not tp.doc_id:
                        continue
                    sheet = str(tp.sheet_name or "").strip()
                    dedup_tp_key = (tp.doc_id, sheet.lower() or tid)
                    if dedup_tp_key in emitted_keys:
                        continue

                    already_covered = False
                    for vq in verified:
                        if vq.get("verified") is not True or vq.get("doc_id") != tp.doc_id:
                            continue
                        if sheet:
                            vq_sheet = str(vq.get("sheet_name") or "").strip().lower()
                            vq_cl = str(vq.get("clause") or "").strip().lower()
                            if vq_sheet == sheet.lower() or f"лист: {sheet.lower()}" in vq_cl:
                                already_covered = True
                                break
                        else:
                            # Word/PDF table without sheet_name: check if quote tokens appear in table rows/formulas/preview
                            td = tool_data_by_tid.get(tid, {})
                            preview_blob = " ".join(
                                [str(td.get("markdown_preview") or ""), str(tr.result_preview or "")]
                                + [str(r) for r in (td.get("rows") or tp.preview_rows or [])[:5]]
                                + [str(f) for f in (td.get("formulas") or [])[:5]]
                            )
                            q_toks = set(normalize_tokens(str(vq.get("text") or "")))
                            p_toks = set(normalize_tokens(preview_blob))
                            if q_toks and len(q_toks & p_toks) >= max(2, len(q_toks) // 2):
                                already_covered = True
                                break
                    if already_covered:
                        continue

                    emitted_keys.add(dedup_tp_key)
                    donor_h = next((h for h in all_hits if h.doc_id == tp.doc_id), None)
                    src_title = _clean_title_ru(
                        (donor_h.title or donor_h.file_name if donor_h else "")
                        or tp.table_title
                        or tp.section_title
                        or tp.file_name
                    )
                    doc_num = clean_doc_number(tp.doc_number or (donor_h.doc_number if donor_h else ""))
                    doc_date = str(tp.doc_date or (donor_h.doc_date if donor_h else "") or "")
                    clause_val = f"Лист: {sheet}" if sheet else str(tp.section_title or "").strip()
                    page_str = _resolve_tp_page(tp)
                    m_pnum = re.search(r"\d+", page_str)
                    page_num_val = int(m_pnum.group(0)) if m_pnum else None

                    # Build clean human-readable snippet & clause_text without technical field names
                    td = tool_data_by_tid.get(tid, {})
                    readable_lines: list[str] = []
                    for f_item in (td.get("formulas") or [])[:6]:
                        if isinstance(f_item, dict):
                            rl = str(f_item.get("row_label") or "").strip()
                            cl_n = str(f_item.get("column") or "").strip()
                            fm = str(f_item.get("formula") or "").strip()
                            seg = " — ".join(p for p in (rl, f"{cl_n}: {fm}" if cl_n and fm else fm) if p)
                            if seg:
                                readable_lines.append(seg)
                    if not readable_lines:
                        for row_item in (td.get("rows") or tp.preview_rows or [])[:5]:
                            if isinstance(row_item, dict):
                                cells = [
                                    f"{k}: {v}"
                                    for k, v in row_item.items()
                                    if str(v or "").strip() and not _TECH_LINE_RE.search(str(k))
                                ]
                                if cells:
                                    readable_lines.append("; ".join(cells))
                    if not readable_lines and tr.result_preview and not _TECH_LINE_RE.search(tr.result_preview) and not tr.result_preview.lstrip().startswith(("{", "[")):
                        readable_lines.append(tr.result_preview.strip()[:300])
                    if not readable_lines:
                        readable_lines.append(tp.table_title or tp.section_title or src_title)

                    snippet_txt = readable_lines[0][:280]
                    clause_txt = "\n".join(readable_lines)[:1200]
                    source_lbl = build_source_label(src_title, doc_num, clause_val, page_str)

                    rec: dict[str, Any] = {
                        "text": snippet_txt,
                        "source_raw": source_lbl,
                        "source_title": src_title,
                        "doc_number": doc_num,
                        "doc_date": doc_date,
                        "clause": clause_val,
                        "page": page_str,
                        "page_num": page_num_val,
                        "doc_id": tp.doc_id,
                        "matched_doc_id": tp.doc_id,
                        "verified": True,
                        "score": 1.0,
                        "clause_text": clause_txt,
                        "clause_highlights": [],
                        "metadata_source": "table_tool",
                        "source_label": source_lbl,
                    }
                    if sheet:
                        rec["sheet_name"] = sheet
                    verified.append(rec)

            return verified
        except Exception as exc:  # pragma: no cover - verification must never break answering
            logger.warning("Quote verification failed: %s", exc)
            return []

    def _collect_dynamic_corpus_catalog(self) -> list[str]:
        """
        Dynamically inspect `search_service.pipeline.registry` (100% corpus-agnostic, zero hardcoding)
        and return clean human-readable document descriptors currently loaded in the knowledge base.
        """
        pipeline = getattr(self.search_service, "pipeline", None)
        registry = getattr(pipeline, "registry", None)
        if not isinstance(registry, dict) or not registry:
            return []

        get_secs = getattr(self.search_service, "get_doc_parent_sections", None)
        entries: list[str] = []
        for doc_id, meta in registry.items():
            title = str(getattr(meta, "title", "") or getattr(meta, "file_name", "") or doc_id).strip()
            if not title:
                continue
            fmt = str(getattr(meta, "file_format", "") or "").upper().strip(".")
            tables_cnt = 0
            try:
                tables_cnt = int(getattr(meta, "tables_count", 0) or 0)
            except (TypeError, ValueError):
                tables_cnt = 0

            subject_hint = ""
            if callable(get_secs):
                try:
                    secs = get_secs(str(doc_id))
                except Exception:
                    secs = None
                if isinstance(secs, list) and secs:
                    first_txt = str(getattr(secs[0], "clean_text", "") or "")
                    m_subj = re.search(
                        r"(?:ПРИКАЗ|РАСПОРЯЖЕНИЕ)[^\n]*\n+\s*((?:О|Об)\s+[^\n]{8,140})",
                        first_txt,
                    )
                    if m_subj:
                        cand_subj = re.sub(r"\s+", " ", m_subj.group(1)).strip(" .,:;")
                        if "[" not in cand_subj and "_" not in cand_subj:
                            subject_hint = cand_subj
                    if not subject_hint:
                        for sec_obj in secs[:4]:
                            st = str(getattr(sec_obj, "section_title", "") or "").strip()
                            if (
                                st
                                and len(st) >= 8
                                and st.lower() not in {"основной текст", "метаданные", "утвержден", "регламент"}
                                and not st.lower().startswith(("приложение", "форма", "шапка", "лист"))
                                and "[" not in st
                                and "_" not in st
                                and st.lower() not in title.lower()
                            ):
                                subject_hint = st
                                break

            details: list[str] = []
            if fmt:
                details.append(fmt)
            if tables_cnt > 0:
                details.append(f"таблиц: {tables_cnt}")
            suffix = f" ({', '.join(details)})" if details else ""
            hint_part = (
                f" — {subject_hint}"
                if subject_hint and subject_hint.lower() not in title.lower()
                else ""
            )
            entries.append(f"- {title}{hint_part}{suffix}")
        return entries[:60]

    def _build_meta_assistant_messages(self, query: str, catalog: list[str]) -> list[dict[str, Any]]:
        """Build prompt for natural capability / greeting / corpus-overview answer."""
        if catalog:
            corpus_block = (
                f"ДОКУМЕНТЫ В БАЗЕ ЗНАНИЙ (всего загружено: {len(catalog)}):\n"
                + "\n".join(catalog)
            )
        else:
            corpus_block = "ДОКУМЕНТЫ В БАЗЕ ЗНАНИЙ: пока не загружено ни одного документа."
        user_prompt = f"ОБРАЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{query.strip()}\n\n{corpus_block}"
        return [
            {"role": "system", "content": META_ASSISTANT_SYSTEM_PROMPT_RU},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _synthesize_meta_assistant_fallback(catalog: list[str]) -> str:
        """Clean, friendly plain-Russian capability & dynamic corpus summary when LLM API is offline."""
        lines: list[str] = [
            "Здравствуйте! Я справочный помощник по загруженной базе документов и таблиц.",
            "",
            "Чем я могу помочь:",
            "- Нахожу ответы строго по документам и таблицам, загруженным в базу знаний.",
            "- Привожу дословные проверенные цитаты с указанием названия документа, номера, пункта и страницы.",
            "- Помогаю фильтровать, сопоставлять и подсчитывать данные, суммы и расчетные формулы по таблицам Excel и Word.",
            "- Учитываю актуальность документов и внесенные в них изменения.",
        ]
        if catalog:
            lines.append("")
            lines.append(f"Сейчас в базу знаний загружено документов: {len(catalog)}. Среди них:")
            lines.extend(catalog[:15])
            if len(catalog) > 15:
                lines.append(f"- …и еще {len(catalog) - 15} док. в разделе «База знаний».")
        else:
            lines.append("")
            lines.append("Сейчас в базу знаний пока не добавлены документы.")
        lines.append("")
        lines.append(
            "Посмотреть полный перечень файлов или добавить новые документы (PDF, DOCX, DOC, XLSX, XLS) "
            "можно во вкладке «База знаний»."
        )
        return "\n".join(lines)

    def _answer_meta_assistant(
        self,
        raw_query: str,
        requested_model: str | None = None,
    ) -> tuple[str, str]:
        """Generate natural plain-Russian response for greeting/capability/corpus-overview queries."""
        catalog = self._collect_dynamic_corpus_catalog()
        if self.client is not None:
            try:
                msgs = self._build_meta_assistant_messages(raw_query, catalog)
                resp, resolved_model = self._call_chat_completion(
                    messages=msgs,
                    allow_tools=False,
                    requested_model=requested_model,
                )
                choices = list(getattr(resp, "choices", None) or [])
                if choices:
                    txt = strip_thinking_tags(str(getattr(choices[0].message, "content", None) or ""))
                    txt = re.sub(r"(?m)^\s*>\s*", "", txt)
                    if txt:
                        return txt, resolved_model
            except Exception as exc:
                logger.warning("Meta-assistant LLM call failed, using natural fallback: %s", exc)
        return self._synthesize_meta_assistant_fallback(catalog), "meta_assistant"

    def ask(
        self,
        request: AgentAskRequest | dict[str, Any] | str,
        **kwargs: Any,
    ) -> AgentAskResponse:
        """
        Execute end-to-end question answering with:
        1) 4-Stage AI Retrieval Pipeline (Rewrite -> Hybrid Pool -> AI Rerank + Follow-up -> Verified Context)
           or deterministic Hybrid RRF Search when `use_reranker=False` / legacy mock client
        2) Natural Meta-Assistant capability/corpus overview OR polite Out-of-Scope short-circuit
        3) Adaptive 3-Tier Table Context Routing (inline_parent vs tool_query TablePassports)
        4) Multi-round OpenAI Tool Calling against StructuredTableStore (up to max_tool_rounds)
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
        pipeline_trace: QueryPipelineTrace | None = None
        if self._should_use_query_pipeline(req):
            run_res = self.query_pipeline.run(
                query=raw_query,
                top_k=req.top_k,
                status_filter=req.status_filter,
                year_filter=req.year_filter,
                requested_model=req.model,
            )
            search_resp = run_res.search_response
            pipeline_trace = run_res.trace
        elif is_meta_assistant_query(raw_query):
            search_resp = SearchResponse(
                query=raw_query,
                query_intent="meta_assistant",
                hits=[],
                is_out_of_scope=False,
                confidence_score=1.0,
            )
        else:
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

        # Step 1a: Capability / Greeting / Corpus-Overview intent (`meta_assistant`)
        if search_resp.query_intent == "meta_assistant" or is_meta_assistant_query(raw_query):
            meta_answer, meta_model = self._answer_meta_assistant(raw_query, req.model)
            clean_meta_resp = search_resp.model_copy(
                update={
                    "query_intent": "meta_assistant",
                    "hits": [],
                    "parent_sections": [],
                    "amendment_chunks": [],
                    "linked_tables": [],
                    "version_alerts": [],
                    "discrepancy_alerts": [],
                    "quality_alerts": [],
                    "is_out_of_scope": False,
                }
            )
            return AgentAskResponse(
                query=raw_query,
                answer=meta_answer,
                model_used=meta_model,
                used_table_tools=False,
                tool_calls=[],
                search_response=clean_meta_resp,
                linked_tables=[],
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                quotes=[],
                version_alerts=[],
                discrepancy_alerts=[],
                quality_alerts=[],
                pipeline_trace=pipeline_trace,
            )

        # Step 1b: Out-of-Scope short-circuit (0 tool calls wasted, zero bogus sources, zero technical jargon)
        if search_resp.is_out_of_scope or not raw_query or not search_resp.hits:
            clean_oos_resp = search_resp.model_copy(
                update={
                    "hits": [],
                    "parent_sections": [],
                    "amendment_chunks": [],
                    "linked_tables": [],
                    "version_alerts": [],
                    "discrepancy_alerts": [],
                    "quality_alerts": [],
                    "is_out_of_scope": True,
                }
            )
            return AgentAskResponse(
                query=raw_query,
                answer=OUT_OF_SCOPE_REFUSAL_RU,
                model_used="out_of_scope_guard",
                used_table_tools=False,
                tool_calls=[],
                search_response=clean_oos_resp,
                linked_tables=[],
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                quotes=[],
                version_alerts=[],
                discrepancy_alerts=[],
                quality_alerts=[],
                pipeline_trace=pipeline_trace,
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

        # Step 3: Autonomous LLM Tool Calling loop (offline mode uses deterministic synthesizer; upstream LLM crash returns sysadmin stub)
        if self.client is None:
            final_answer = self._synthesize_fallback_answer(
                query=raw_query,
                search_resp=search_resp,
                active_store=active_store,
                enable_table_tools=req.enable_table_tools,
                traces=traces,
            )
            model_used = "deterministic-fallback"
        else:
            try:
                max_rounds = req.max_tool_rounds if tools_enabled else 0
                pinned_model = req.model
                force_formula_tool_round0 = (
                    tools_enabled
                    and not self._has_custom_client
                    and "формул" in raw_query.lower().replace("ё", "е")
                    and any(
                        bool(getattr(tp, "has_formulas", False)) or int(getattr(tp, "formula_count", 0) or 0) > 0
                        for tp in (search_resp.linked_tables or [])
                    )
                )

                for _round_idx in range(max_rounds + 1):
                    allow_tools_now = tools_enabled and (_round_idx < max_rounds)
                    resp, resolved_model = self._call_chat_completion(
                        messages=messages,
                        allow_tools=allow_tools_now,
                        requested_model=pinned_model,
                        require_tool=(force_formula_tool_round0 and _round_idx == 0),
                    )
                    model_used = resolved_model
                    if pinned_model is None and not self._has_custom_client:
                        pinned_model = resolved_model

                    choices = list(getattr(resp, "choices", None) or [])
                    if not choices:
                        break
                    msg = choices[0].message
                    raw_msg_content = str(getattr(msg, "content", None) or "").strip()
                    raw_tool_calls = getattr(msg, "tool_calls", None)
                    tool_calls_list = list(raw_tool_calls) if raw_tool_calls else []
                    if not tool_calls_list and allow_tools_now and "<tool_call>" in raw_msg_content:
                        msg_content, tool_calls_list = parse_qwen3_xml_tool_calls(raw_msg_content)
                    else:
                        msg_content = strip_thinking_tags(raw_msg_content)

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
                    step_counter = self._execute_tool_calls(
                        active_store=active_store,
                        serialized_tcs=serialized_tcs,
                        step_counter=step_counter,
                        traces=traces,
                        messages=messages,
                        seen_linked_ids=seen_linked_ids,
                        final_linked_tables=final_linked_tables,
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
                logger.exception("Upstream LLM failure during RAG answer generation: %s", llm_exc)
                return AgentAskResponse(
                    query=raw_query,
                    answer=TECH_ERROR_STUB_RU,
                    model_used="technical_error_fallback",
                    used_table_tools=bool(traces),
                    tool_calls=traces,
                    search_response=search_resp,
                    linked_tables=final_linked_tables,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    quotes=[],
                    version_alerts=[],
                    discrepancy_alerts=[],
                    quality_alerts=[],
                    pipeline_trace=pipeline_trace,
                )

        return AgentAskResponse(
            query=raw_query,
            answer=final_answer,
            model_used=model_used,
            used_table_tools=bool(traces),
            tool_calls=traces,
            search_response=search_resp,
            linked_tables=final_linked_tables,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            quotes=self._verify_answer_quotes(
                final_answer,
                search_resp,
                messages,
                traces=traces,
                linked_tables=final_linked_tables,
            ),
            version_alerts=list(search_resp.version_alerts or []),
            discrepancy_alerts=list(getattr(search_resp, "discrepancy_alerts", None) or []),
            quality_alerts=list(search_resp.quality_alerts or []),
            pipeline_trace=pipeline_trace,
        )


    # ------------------------------------------------------------------
    # Streaming variant (token-by-token) for POST /api/ask/stream
    # ------------------------------------------------------------------
    @staticmethod
    def _hit_source_payload(hit: SearchHit) -> dict[str, Any]:
        """Lightweight hit projection for the UI «Источники» panel (no heavy text fields)."""
        h_meta = hit.metadata or {}
        sheet_name = str(getattr(hit, "sheet_name", "") or h_meta.get("sheet_name") or "").strip()
        table_id = str(getattr(hit, "table_id", "") or h_meta.get("table_id") or "").strip()
        return {
            "chunk_id": hit.chunk_id,
            "doc_id": hit.doc_id,
            "file_name": hit.file_name,
            "title": hit.title,
            "doc_number": hit.doc_number,
            "doc_date": hit.doc_date,
            "status": hit.status,
            "section_title": hit.section_title,
            "clause_no": hit.clause_no,
            "table_id": table_id,
            "sheet_name": sheet_name,
            "page_numbers": list(hit.page_numbers or []),
            "score": hit.score,
        }

    def _open_chat_stream(
        self,
        messages: list[dict[str, Any]],
        allow_tools: bool,
        requested_model: str | None = None,
        require_tool: bool = False,
    ) -> tuple[Any, str]:
        """Open a streamed chat completion (stream=True) with the same model fallback as _call_chat_completion."""
        llm = self.client
        if llm is None:
            raise RuntimeError("DEEPSEEK_API_KEY / POLZA_API_KEY не задан и кастомный LLM-клиент не передан.")

        last_exc: Exception | None = None
        for model_name in self._candidate_models(requested_model):
            call_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.1,
                "stream": True,
            }
            if allow_tools:
                call_kwargs["tools"] = OPENAI_TABLE_TOOLS
                call_kwargs["tool_choice"] = "required" if require_tool else "auto"
            try:
                stream = llm.chat.completions.create(**call_kwargs)
                if not requested_model and not self._has_custom_client:
                    self._verified_model = model_name
                return stream, model_name
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Streamed chat completion failed on model '%s' (allow_tools=%s): %s",
                    model_name,
                    allow_tools,
                    exc,
                )
                if self._has_custom_client:
                    break
                if (
                    self.llm_provider == "local"
                    and not any(k in str(exc).lower() for k in ("not found", "does not exist", "404"))
                ):
                    break
        assert last_exc is not None
        raise last_exc

    def ask_stream(
        self,
        request: AgentAskRequest | dict[str, Any] | str,
        should_stop: Callable[[], bool] | None = None,
        **kwargs: Any,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """
        Streaming counterpart of `ask()` — same retrieval, prompt and tool-calling loop, but yields
        (event, data) tuples as they happen:
          ("sources", {...})  — once, right after hybrid search
          ("status",  {...})  — optional progress (e.g. table tool calls)
          ("token",   {"text": delta}) — LLM answer deltas
          ("quotes",  {"quotes": [...]}) — verbatim quotes parsed from the answer + verification result
          ("done",    {...})  — latency / model / tool stats
          ("error",   {"message": ...}) — on unrecoverable failure (partial text is kept by the client)
        `should_stop()` is polled between chunks; when it returns True the upstream LLM stream is
        closed immediately (client disconnected / pressed «Стоп»).
        """
        t0 = time.perf_counter()
        stop = should_stop or (lambda: False)
        if isinstance(request, str):
            req = AgentAskRequest(query=request, **kwargs)
        elif isinstance(request, AgentAskRequest):
            req = request
        elif isinstance(request, dict):
            req = AgentAskRequest.model_validate(request)
        else:
            raise TypeError(f"Unsupported AgentAskRequest type: {type(request)}")

        raw_query = (req.query or "").strip()
        pipeline_trace: QueryPipelineTrace | None = None

        def _done(model_used: str, traces: list[ToolCallTrace], stopped: bool = False) -> tuple[str, dict[str, Any]]:
            return (
                "done",
                {
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "model_used": model_used,
                    "used_table_tools": bool(traces),
                    "tool_calls": len(traces),
                    "stopped": stopped,
                    "pipeline_trace": pipeline_trace.model_dump() if pipeline_trace is not None else None,
                },
            )

        def _error_event() -> tuple[str, dict[str, Any]]:
            return (
                "error",
                {
                    "error": TECH_ERROR_STUB_RU,
                    "message": TECH_ERROR_STUB_RU,
                    "error_type": "technical_error",
                },
            )

        try:
            if self._should_use_query_pipeline(req):
                run_res = None
                for ev_kind, payload in self.query_pipeline.run_stream(
                    query=raw_query,
                    top_k=req.top_k,
                    status_filter=req.status_filter,
                    year_filter=req.year_filter,
                    requested_model=req.model,
                    should_stop=stop,
                ):
                    if ev_kind == "status":
                        yield ("status", payload)
                    elif ev_kind == "result":
                        run_res = payload
                    if stop():
                        return
                assert run_res is not None
                search_resp = run_res.search_response
                pipeline_trace = run_res.trace
            elif is_meta_assistant_query(raw_query):
                search_resp = SearchResponse(
                    query=raw_query,
                    query_intent="meta_assistant",
                    hits=[],
                    is_out_of_scope=False,
                    confidence_score=1.0,
                )
            else:
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
        except Exception as retr_exc:
            logger.exception("Technical failure during streaming retrieval: %s", retr_exc)
            yield _error_event()
            return

        if stop():
            return

        # Capability / Greeting / Corpus-Overview intent (`meta_assistant`) — zero fake sources, smooth streaming
        if search_resp.query_intent == "meta_assistant" or is_meta_assistant_query(raw_query):
            yield (
                "sources",
                {
                    "hits": [],
                    "version_alerts": [],
                    "discrepancy_alerts": [],
                    "quality_alerts": [],
                    "is_out_of_scope": False,
                    "confidence_score": 1.0,
                },
            )
            if stop():
                return

            catalog = self._collect_dynamic_corpus_catalog()
            meta_msgs = self._build_meta_assistant_messages(raw_query, catalog)
            meta_emitted = False
            meta_model = "meta_assistant"
            if self.client is not None:
                try:
                    stream, meta_model = self._open_chat_stream(
                        messages=meta_msgs,
                        allow_tools=False,
                        requested_model=req.model,
                    )
                    try:
                        for chunk in stream:
                            if stop():
                                yield _done(meta_model, [], stopped=True)
                                return
                            chunk_model = getattr(chunk, "model", None)
                            if isinstance(chunk_model, str) and chunk_model.strip():
                                meta_model = chunk_model.strip()
                            choices = getattr(chunk, "choices", None) or []
                            if not choices:
                                continue
                            delta = getattr(choices[0], "delta", None)
                            text = getattr(delta, "content", None) if delta is not None else None
                            if text:
                                meta_emitted = True
                                yield ("token", {"text": text})
                    finally:
                        close_fn = getattr(stream, "close", None)
                        if callable(close_fn):
                            try:
                                close_fn()
                            except Exception:
                                pass
                except Exception as meta_exc:
                    logger.warning("Meta-assistant stream failed, using natural fallback: %s", meta_exc)

            if not meta_emitted:
                fallback_meta = self._synthesize_meta_assistant_fallback(catalog)
                yield ("token", {"text": fallback_meta})
                meta_model = "meta_assistant"

            yield ("quotes", {"quotes": []})
            yield _done(meta_model, [])
            return

        # Out-of-Scope short-circuit (0 bogus sources, 0 technical jargon)
        if search_resp.is_out_of_scope or not raw_query or not search_resp.hits:
            yield (
                "sources",
                {
                    "hits": [],
                    "version_alerts": [],
                    "discrepancy_alerts": [],
                    "quality_alerts": [],
                    "is_out_of_scope": True,
                    "confidence_score": search_resp.confidence_score,
                },
            )
            if stop():
                return
            yield ("token", {"text": OUT_OF_SCOPE_REFUSAL_RU})
            yield ("quotes", {"quotes": []})
            yield _done("out_of_scope_guard", [])
            return

        # Emit ONLY Stage-3 verified sources right after reranking
        yield (
            "sources",
            {
                "hits": [self._hit_source_payload(h) for h in search_resp.hits],
                "version_alerts": list(search_resp.version_alerts or []),
                "discrepancy_alerts": list(getattr(search_resp, "discrepancy_alerts", None) or []),
                "quality_alerts": list(search_resp.quality_alerts or []),
                "is_out_of_scope": False,
                "confidence_score": search_resp.confidence_score,
            },
        )
        if stop():
            return

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
        model_used = ""
        step_counter = 0
        emitted_any = False
        answer_parts: list[str] = []

        def _quotes_event(extra_text: str = "") -> tuple[str, dict[str, Any]]:
            full_answer = "".join(answer_parts) + extra_text
            return (
                "quotes",
                {
                    "quotes": self._verify_answer_quotes(
                        full_answer,
                        search_resp,
                        messages,
                        traces=traces,
                        linked_tables=final_linked_tables,
                    )
                },
            )

        if self.client is None:
            fallback = self._synthesize_fallback_answer(
                query=raw_query,
                search_resp=search_resp,
                active_store=active_store,
                enable_table_tools=req.enable_table_tools,
                traces=traces,
            )
            yield ("token", {"text": fallback})
            yield _quotes_event(fallback)
            yield _done("deterministic-fallback", traces)
            return

        try:
            max_rounds = req.max_tool_rounds if tools_enabled else 0
            pinned_model = req.model
            force_formula_tool_round0 = (
                tools_enabled
                and not self._has_custom_client
                and "формул" in raw_query.lower().replace("ё", "е")
                and any(
                    bool(getattr(tp, "has_formulas", False)) or int(getattr(tp, "formula_count", 0) or 0) > 0
                    for tp in (search_resp.linked_tables or [])
                )
            )

            for _round_idx in range(max_rounds + 1):
                allow_tools_now = tools_enabled and (_round_idx < max_rounds)
                stream, model_used = self._open_chat_stream(
                    messages=messages,
                    allow_tools=allow_tools_now,
                    requested_model=pinned_model,
                    require_tool=(force_formula_tool_round0 and _round_idx == 0),
                )
                if pinned_model is None and not self._has_custom_client:
                    pinned_model = model_used

                content_parts: list[str] = []
                tc_acc: dict[int, dict[str, Any]] = {}
                try:
                    for chunk in stream:
                        if stop():
                            yield _done(model_used, traces, stopped=True)
                            return
                        chunk_model = getattr(chunk, "model", None)
                        if isinstance(chunk_model, str) and chunk_model.strip():
                            model_used = chunk_model.strip()
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        delta = getattr(choices[0], "delta", None)
                        if delta is None:
                            continue
                        text = getattr(delta, "content", None)
                        if text:
                            content_parts.append(text)
                            answer_parts.append(text)
                            emitted_any = True
                            yield ("token", {"text": text})
                        for tcd in getattr(delta, "tool_calls", None) or []:
                            idx = int(getattr(tcd, "index", 0) or 0)
                            slot = tc_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                            if getattr(tcd, "id", None):
                                slot["id"] = tcd.id
                            fn = getattr(tcd, "function", None)
                            if fn is not None:
                                if getattr(fn, "name", None):
                                    slot["name"] += fn.name
                                if getattr(fn, "arguments", None):
                                    slot["arguments"] += fn.arguments
                finally:
                    close_fn = getattr(stream, "close", None)
                    if callable(close_fn):
                        try:
                            close_fn()
                        except Exception:  # pragma: no cover - best effort
                            pass

                if not tc_acc or not allow_tools_now:
                    break

                # Tool round: run the table tools (non-streamed), then loop for the next LLM turn
                serialized_tcs: list[dict[str, Any]] = []
                for tc_idx, idx in enumerate(sorted(tc_acc), start=1):
                    slot = tc_acc[idx]
                    serialized_tcs.append(
                        {
                            "id": slot["id"] or f"call_{step_counter + tc_idx}",
                            "type": "function",
                            "function": {"name": slot["name"], "arguments": slot["arguments"] or "{}"},
                        }
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(content_parts).strip() or None,
                        "tool_calls": serialized_tcs,
                    }
                )
                yield ("status", {"stage": "tools", "text": "Запрашиваю данные из таблиц…"})
                assert active_store is not None
                step_counter = self._execute_tool_calls(
                    active_store=active_store,
                    serialized_tcs=serialized_tcs,
                    step_counter=step_counter,
                    traces=traces,
                    messages=messages,
                    seen_linked_ids=seen_linked_ids,
                    final_linked_tables=final_linked_tables,
                )
                if stop():
                    yield _done(model_used, traces, stopped=True)
                    return

        except Exception as llm_exc:
            logger.exception("Upstream LLM stream failed: %s", llm_exc)
            yield _error_event()
            return

        if not emitted_any:
            fallback = self._synthesize_fallback_answer(
                query=raw_query,
                search_resp=search_resp,
                active_store=active_store,
                enable_table_tools=req.enable_table_tools,
                traces=traces,
            )
            yield ("token", {"text": fallback})
            answer_parts.append(fallback)
            model_used = model_used or "deterministic-fallback"

        yield _quotes_event()
        yield _done(model_used, traces)


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
