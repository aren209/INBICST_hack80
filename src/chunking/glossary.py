"""Bidirectional MIPT Domain Glossary Manager, Auto-Learner, Role/Topic Tagger, and Metadata/Relation Extractor."""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.config import (
    GLOSSARY_PATH,
    REGISTRY_OVERRIDES_PATH,
    STATUS_RU_MAP,
)
from src.schemas import (
    BlockType,
    ChunkRecord,
    DocumentMeta,
    DocumentStatus,
    ParsedBlock,
    ZoneType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default MIPT Domain Abbreviations (18+ mandatory + core MIPT abbreviations)
# ---------------------------------------------------------------------------
DEFAULT_ABBREVIATIONS: dict[str, str] = {
    "ФЛС": "финансово-лицевой счет",
    "ЦФО": "центр финансовой ответственности",
    "КФО": "код финансового обеспечения",
    "КВР": "код вида расходов",
    "КОСГУ": "классификация операций сектора государственного управления",
    "1С:ДО": "1С:Документооборот",
    "ПФО": "планово-финансовый отдел",
    "ФЭУ": "финансово-экономическое управление",
    "РИД": "результат интеллектуальной деятельности",
    "РЦНИ": "Российский центр научной информации",
    "КМО": "культурно-массовый отдел",
    "КЗ": "Концертный зал",
    "ЦОД": "центр обработки данных",
    "ЭДО": "электронный документооборот",
    "ППС": "профессорско-преподавательский состав",
    "ПУМС": "пульт управления механикой сцены",
    "ИОТ": "инструкция по охране труда",
    "ЭБ": "электронная библиотека / электробезопасность",
    "МФТИ": "Московский физико-технический институт (национальный исследовательский университет)",
    "УИТ": "управление информационных технологий",
    "УБУиФК": "управление бухгалтерского учета и финансового контроля",
    "ОТиТБ": "отдел охраны труда и техники безопасности",
    "ДНВР": "департамент по науке и внеучебной работе",
    "АХУ": "административно-хозяйственное управление",
    "НДС": "налог на добавленную стоимость",
    "ГПХ": "договор гражданско-правового характера",
    "СЗ": "служебная записка",
    "ТЗ": "техническое задание",
    "Физтех": "Московский физико-технический институт (МФТИ)",
    "ФШ": "физтех-школа",
    "МТО": "материально-техническое обеспечение",
    "РНФ": "Российский научный фонд",
    "РФФИ": "Российский фонд фундаментальных исследований",
    "ЭЦП": "электронно-цифровая подпись",
    "НИР": "научно-исследовательская работа",
    "ОКР": "опытно-конструкторская работа",
    "НИОКР": "научно-исследовательские и опытно-конструкторские работы",
}

# Inflected stem prefixes for reverse expansion -> abbreviation matching in Russian text
EXPANSION_STEM_MAP: dict[str, str] = {
    "финансово-лицев": "ФЛС",
    "финансового лицевого сч": "ФЛС",
    "финансовый лицевой сч": "ФЛС",
    "финансовому лицевому сч": "ФЛС",
    "центр финансовой ответственност": "ЦФО",
    "центра финансовой ответственност": "ЦФО",
    "центров финансовой ответственност": "ЦФО",
    "код финансового обеспечен": "КФО",
    "кода финансового обеспечен": "КФО",
    "код вида расход": "КВР",
    "коды вида расход": "КВР",
    "кодов вида расход": "КВР",
    "классификация операций сектора": "КОСГУ",
    "классификации операций сектора": "КОСГУ",
    "1с:документооборот": "1С:ДО",
    "планово-финансов": "ПФО",
    "финансово-экономическ": "ФЭУ",
    "результат интеллектуальной деятельност": "РИД",
    "результатов интеллектуальной деятельност": "РИД",
    "результаты интеллектуальной деятельност": "РИД",
    "российский центр научной информац": "РЦНИ",
    "российского центра научной информац": "РЦНИ",
    "культурно-массов": "КМО",
    "концертный зал": "КЗ",
    "концертного зала": "КЗ",
    "концертном зале": "КЗ",
    "концертному залу": "КЗ",
    "центр обработки данн": "ЦОД",
    "центра обработки данн": "ЦОД",
    "центре обработки данн": "ЦОД",
    "электронный документооборот": "ЭДО",
    "электронного документооборота": "ЭДО",
    "профессорско-преподавательск": "ППС",
    "пульт управления механикой сцен": "ПУМС",
    "пульта управления механикой сцен": "ПУМС",
    "инструкция по охране труд": "ИОТ",
    "инструкции по охране труд": "ИОТ",
    "электронная библиотек": "ЭБ",
    "электронный бюджет": "ЭБ",
    "электробезопасност": "ЭБ",
    "управление информационных технолог": "УИТ",
    "управления информационных технолог": "УИТ",
    "управлению информационных технолог": "УИТ",
    "гражданско-правового характер": "ГПХ",
    "материально-техническое обеспечен": "МТО",
    "материально-технического обеспечен": "МТО",
    "электронно-цифровой подпись": "ЭЦП",
    "электронно-цифровая подпись": "ЭЦП",
    "служебная записк": "СЗ",
    "служебную записк": "СЗ",
    "служебной записк": "СЗ",
}

# ---------------------------------------------------------------------------
# 12 Mandatory Role Patterns (ROLE_PATTERNS)
# ---------------------------------------------------------------------------
RAW_ROLE_PATTERNS: dict[str, str] = {
    "инициатор": r"(?i)\b(?:инициатор[а-яё]*|заявител[ьяюеми]+)\b",
    "организатор": r"(?i)\b(?:организатор[а-яё]*|ответственн[а-яё]+\s+за\s+(?:проведение|организацию|мероприятие))\b",
    "руководитель подразделения / ЦФО": (
        r"(?i)\b(?:руководител[ьяюеми]+\s+(?:структурного\s+)?(?:подразделения|ЦФО|проекта|темы|гранта|"
        r"физтех-школы|лаборатории|отдела|направления)|"
        r"заместител[ьяюеми]+\s+руководителя\s+проекта|"
        r"ответственн[а-яё]+\s+исполнител[ьяюеми]+|"
        r"директор[а-яё]*\s+(?:физтех-школ[а-яё]*|институт[а-яё]*|центр[а-яё]*))\b"
    ),
    "финансодержатель": r"(?i)\b(?:финансодержател[ьяюеми]+|держател[ьяюеми]+\s+ФЛС)\b",
    "начальник отдела (ПФО, КМО, УИТ и др.)": (
        r"(?i)\b(?:начальник[а-яё]*\s+(?:отдела|управления|ПФО|КМО|УИТ|ФЭУ|ОТиТБ|службы|"
        r"финансово-экономического\s+управления|управления\s+информационных\s+технологий|"
        r"управления\s+кампусом|культурно-массового\s+отдела)|"
        r"руководител[ьяюеми]+\s+(?:ПФО|КМО|УИТ|ФЭУ|Центра\s+трансфера\s+технологий)|"
        r"заведующ[а-яё]+\s+канцелярией)\b"
    ),
    "бухгалтерия": (
        r"(?i)\b(?:бухгалтери[яиейю]|главн[а-яё]+\s+бухгалтер[а-яё]*|УБУиФК|"
        r"финансово-экономическ[а-яё]+\s+управлени[а-яё]+|сотрудник[а-яё]*\s+ФЭУ)\b"
    ),
    "дирекция": (
        r"(?i)\b(?:дирекци[яиейю]|директор[а-яё]*\s+по\s+[а-яё,\s]{3,40}|"
        r"исполнительн[а-яё]+\s+дирекци[яиейю])\b"
    ),
    "проректор": r"(?i)\b(?:проректор[а-яё]*|перв[а-яё]+\s+проректор[а-яё]*|курирующ[а-яё]+\s+проректор[а-яё]*|ректор[а-яё]*)\b",
    "администратор / техспециалист": (
        r"(?i)\b(?:администратор[а-яё]*|техническ[а-яё]+\s+специалист[а-яё]*|"
        r"специалист[а-яё]*\s+(?:первой\s+линии|техподдержки|технической\s+поддержки|ЦОД|КЗ)|"
        r"оператор[а-яё]*|дежурн[а-яё]+\s+инженер[а-яё]*|звукорежиссер[а-яё]*|художник[а-яё]*\s+по\s+свету)\b"
    ),
    "комендант / управление кампусом": (
        r"(?i)\b(?:комендант[а-яё]*|заведующ[а-яё]+\s+общежити[а-яё]+|"
        r"управлени[еяю]\s+(?:кампусом|общежитиями|жилым\s+фондом|делами)|"
        r"жилищн[а-яё]+\s+комисси[яиейю]|паспортист[а-яё]*)\b"
    ),
    "кадровая служба / рекрутер": (
        r"(?i)\b(?:кадров[а-яё]+\s+служб[а-яё]+|отдел[а-яё]*\s+(?:кадров|подбора\s+персонала)|"
        r"управлени[еяюем]+\s+по\s+работе\s+с\s+персоналом|рекрутер[а-яё]*|специалист[а-яё]*\s+по\s+подбору)\b"
    ),
    "согласующее лицо": (
        r"(?i)\b(?:согласующ[а-яё]+\s+лиц[оауе]|лист\s+согласования|"
        r"визирующ[а-яё]+|эксперт[а-яё]*\s+согласования|инициатор\s+согласования)\b"
    ),
}

ROLE_PATTERNS: dict[str, re.Pattern[str]] = {
    role_name: re.compile(pattern_str) for role_name, pattern_str in RAW_ROLE_PATTERNS.items()
}

DEFAULT_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "финансово-лицевой счет (ФЛС)": [
        "флс",
        "финансово-лицевой",
        "финансового лицевого счета",
        "лицевой счет",
        "присвоение номера флс",
        "открытие флс",
        "3065-1",
        "распоряжение n 700",
    ],
    "бюджетирование и сметы": [
        "смета",
        "сметы",
        "бюджетирование",
        "бюджет",
        "квр",
        "косгу",
        "накладные расходы",
        "резерв на оплату отпусков",
        "резерв отпусков",
        "резерв отпускных",
        "794-1",
        "3699-1",
        "приложение 1а",
    ],
    "финансовая структура и ЦФО": [
        "финансовая структура",
        "цфо",
        "центр финансовой ответственности",
        "финансодержатель",
        "3614-1",
    ],
    "концертный зал и мероприятия": [
        "концертный зал",
        "кмо",
        "kmo@mipt.ru",
        "репетиц",
        "пумс",
        "посвящение в студенты",
        "бронирование",
        "сцена",
    ],
    "поселение и общежития": [
        "поселение",
        "общежити",
        "жилое помещение",
        "договор найма",
        "койко-место",
        "выселение",
    ],
    "командировки и суточные": [
        "командиров",
        "суточные",
        "авансовый отчет",
        "служебное задание",
        "проезд",
        "проживание",
    ],
    "подбор персонала и кадры": [
        "подбор персонала",
        "поиск и подбор",
        "ваканси",
        "кандидат",
        "собеседован",
        "резюме",
        "адаптац",
    ],
    "техподдержка и ИТ-сервисы": [
        "первая линия",
        "техподдержк",
        "технической поддержки",
        "инцидент",
        "обращение пользователя",
        "sla",
    ],
    "центр обработки данных (ЦОД)": [
        "центр обработки данных",
        "цод",
        "серверн",
        "телекоммуникацион",
        "стойк",
        "машинный зал",
    ],
    "документооборот (1С:ДО и ЭДО)": [
        "1с:до",
        "1с:документооборот",
        "эдо",
        "электронный документооборот",
        "служебная записка",
        "согласование",
    ],
}

# ---------------------------------------------------------------------------
# Compiled Regexes for Auto-Learning Definitions & Document Metadata
# ---------------------------------------------------------------------------

# Pattern 1: "... <expansion> (далее — «<term>»)" or "(далее - <АББР>)" or "(далее ─ ...)"
DALEE_DEFINITION_RE = re.compile(
    r"([А-ЯЁа-яё0-9][А-ЯЁа-яё0-9\s,\-«»\"']{4,140}?)\s*"
    r"\(\s*далее(?:\s+по\s+тексту)?(?:\s+именуем[а-яё]+)?\s*"
    r"[-–—─]+\s*[«\"']?([^»\"')\n]{2,45}?)[»\"']?\s*\)",
    re.IGNORECASE,
)

# Pattern 2: "<Expansion> (<АББР>)" where <АББР> is 2..10 uppercase Cyrillic letters
PAREN_ABBR_AFTER_EXPANSION_RE = re.compile(
    r"([А-ЯЁа-яё][а-яё\s\-]{8,110}?)\s*\(\s*([А-ЯЁ]{2,8})\s*\)"
)

# Pattern 3: "<АББР> (<Expansion>)" e.g. "ЭЦП (электронно-цифровой подписью)"
ABBR_BEFORE_PAREN_EXPANSION_RE = re.compile(
    r"\b([А-ЯЁ]{2,8})\s*\(\s*([а-яёА-ЯЁ][а-яё\s\-]{6,100}?)\s*\)"
)

# Leading boilerplate phrases to strip when extracting the noun phrase before "(далее — ...)"
LEADING_BOILERPLATE_RE = re.compile(
    r"^(?:.*?(?:определяет\s+порядок\s+открытия|порядок\s+открытия|предоставить\s+в|"
    r"направляет\s+в|формирует|в\s+соответствии\s+с|на\s+основании|в\s+рамках|"
    r"выплат\s+по|работ\s+по\s+созданию[,\s]+оформлению[,\s]+регистрации\s+и\s+трансфер[ау]|"
    r"выделяемых\s+на|утвердить|установить|заключает|оформляет|подает)\s+)+",
    re.IGNORECASE,
)

# Anti-garbage filters for auto-learned definitions
GARBAGE_TERM_RE = re.compile(
    r"(?:\b(?:руб|коп|тыс|млн|млрд|ндс|инн|кпп|огрн|бик|стр|табл|прил|пункт|раздел|глава|"
    r"приказ[ауом]?|распоряжени[еяю]|настоящ[а-яё]+|указанн[а-яё]+|вышеуказанн[а-яё]+|нижеследующ[а-яё]+|"
    r"стороны|договор\s*\(контракт)\b|"
    r"[|_=\t]{2,}|\bNone\b|\d{2}\.\d{2}\.\d{2,4}|\d+[,.]\d+\s*%?)",
    re.IGNORECASE,
)

RUSSIAN_MONTHS_MAP: dict[str, str] = {
    "января": "01",
    "февраля": "02",
    "марта": "03",
    "апреля": "04",
    "мая": "05",
    "июня": "06",
    "июля": "07",
    "августа": "08",
    "сентября": "09",
    "октября": "10",
    "ноября": "11",
    "декабря": "12",
}

# Requisite extraction regexes
HEADER_TABLE_REQUISITE_RE = re.compile(
    r"(?:от\s*)?(\d{2}\.\d{2}\.\d{4})\s*(?:г\.?)?\s*[|\t]+\s*(?:№|N)\s*([0-9][0-9А-ЯЁа-яёA-Za-z\-/]{0,15})|"
    r"(?:№|N)\s*([0-9][0-9А-ЯЁа-яёA-Za-z\-/]{0,15})\s*[|\t]+\s*(?:от\s*)?(\d{2}\.\d{2}\.\d{4})",
    re.IGNORECASE,
)

EDO_STAMP_RE = re.compile(
    r"\bЗарегистрировано\s+(?:в\s+(?:ЭДО|СЭД|реестре|журнале)(?:\s+[А-ЯЁA-Z0-9«»\"\-]+)?\s*)?"
    r"(?:от\s*)?(?:[«\"']?(\d{1,2})[»\"']?\s+([а-яё]+)\s+(\d{4})|(\d{2}\.\d{2}\.\d{4}))\s*(?:г\.?|года)?\s*"
    r"[^\n]{0,40}?(?:Рег(?:истрационный)?\.?\s*)?(?:№|N)\s*([0-9][0-9А-ЯЁа-яёA-Za-z\-/]{0,15})",
    re.IGNORECASE,
)

EDO_STAMP_REV_RE = re.compile(
    r"(?:\bЗарегистрировано\s+(?:в\s+(?:ЭДО|СЭД|реестре|журнале)(?:\s+[А-ЯЁA-Z0-9«»\"\-]+)?\s*)?[^\n]{0,40}?(?:№|N)\s*|"
    r"(?m:^\s*)(?:Регистрационный\s*(?:№|N|номер[:\s]*)|Рег\.\s*(?:№|N))\s*)"
    r"([0-9][0-9А-ЯЁа-яёA-Za-z\-/]{0,15})"
    r"(?:[^\n]{0,40}?(?:от|дата\s+регистрации[:\s]*)\s*"
    r"(?:[«\"']?(\d{1,2})[»\"']?\s+([а-яё]+)\s+(\d{4})|(\d{2}\.\d{2}\.\d{4})))?",
    re.IGNORECASE,
)

ORDER_HEADER_RE = re.compile(
    r"(?m)^\s*(?:ПРИКАЗ|РАСПОРЯЖЕНИЕ)\b"
    r"(?:[ \t]*\n?[ \t]*(?:г\.\s*[А-ЯЁа-яё\-]+|[А-ЯЁа-яё]{2,25}))?\s+"
    r"от\s*(?:[«\"']?(\d{1,2})[»\"']?\s+([а-яё]+)\s+(\d{4})|(\d{2}\.\d{2}\.\d{4}))\s*(?:г\.?|года)?\s*"
    r"(?:№|N)\s*([0-9][0-9А-ЯЁа-яёA-Za-z\-/]{0,15})",
    re.IGNORECASE,
)

ORDER_HEADER_REV_RE = re.compile(
    r"(?m)^\s*(?:ПРИКАЗ|РАСПОРЯЖЕНИЕ)\b"
    r"(?:[ \t]*\n?[ \t]*(?:г\.\s*[А-ЯЁа-яё\-]+|[А-ЯЁа-яё]{2,25}))?\s+"
    r"(?:№|N)\s*([0-9][0-9А-ЯЁа-яёA-Za-z\-/]{0,15})\s+"
    r"от\s*(?:[«\"']?(\d{1,2})[»\"']?\s+([а-яё]+)\s+(\d{4})|(\d{2}\.\d{2}\.\d{4}))",
    re.IGNORECASE,
)

HEADER_APPROVED_BY_RE = re.compile(
    r"(?m)^\s*(?:УТВЕРЖДЕН[ОЫА]?|Утвержден[оыа]?)\s+"
    r"(?:приказом|распоряжением)(?:\s+[^\n]{0,70}?)?\s+"
    r"от\s*(?:[«\"']?(\d{1,2})[»\"']?\s+([а-яё]+)\s+(\d{4})|(\d{2}\.\d{2}\.\d{4}))\s*(?:г\.?|года)?\s*"
    r"(?:№|N)\s*([0-9][0-9А-ЯЁа-яёA-Za-z\-/]{0,15})",
    re.IGNORECASE,
)

HEADER_APPROVED_BY_REV_RE = re.compile(
    r"(?m)^\s*(?:УТВЕРЖДЕН[ОЫА]?|Утвержден[оыа]?)\s+"
    r"(?:приказом|распоряжением)(?:\s+[^\n]{0,70}?)?\s+"
    r"(?:№|N)\s*([0-9][0-9А-ЯЁа-яёA-Za-z\-/]{0,15})\s+"
    r"от\s*(?:[«\"']?(\d{1,2})[»\"']?\s+([а-яё]+)\s+(\d{4})|(\d{2}\.\d{2}\.\d{4}))",
    re.IGNORECASE,
)

FILENAME_REQUISITES_RE = re.compile(
    r"(Приказ|Распоряжение|Регламент|Положение|Инструкция|Порядок|Приложение)?.*?"
    r"(?:№|N|No)\s*([0-9]{1,6}(?:[-/][0-9А-ЯЁа-яёA-Za-z]{1,6})?)"
    r"(?:.*?от\s*(\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2}))?",
    re.IGNORECASE,
)

FILENAME_REQUISITES_REV_RE = re.compile(
    r"(Приказ|Распоряжение|Регламент|Положение|Инструкция|Порядок|Приложение)?.*?"
    r"от\s*(\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2}).*?"
    r"(?:№|N|No)\s*([0-9]{1,6}(?:[-/][0-9А-ЯЁа-яёA-Za-z]{1,6})?)",
    re.IGNORECASE,
)

# Structural boundaries marking the end of the document's top header/title zone
_HEADER_BODY_STOP_RE = re.compile(
    r"^\s*(?:"
    r"(?:ПРИКАЗЫВАЮ|РАСПОРЯЖАЮСЬ|ОБЯЗЫВАЮ|ПОСТАНОВЛЯЮ)\s*:|"
    r"(?:Раздел|Глава)\s+(?:[0-9]{1,2}|[IVXLCDM]+)\b|"
    r"(?:1\.|I\.)\s+[А-ЯЁ]|"
    r"1\.1\.?\s+[А-ЯЁ]|"
    r"Общие\s+положения\s*$|"
    r"ЛИСТ\s+СОГЛАСОВАНИЯ\b"
    r")",
    re.IGNORECASE,
)

# Inline body/preamble citation phrases that reference OTHER acts rather than the document's own requisites
_EXTERNAL_ACT_CITATION_LINE_RE = re.compile(
    r"(?:в\s+соответствии\s+с|на\s+основании|во\s+исполнение|руководствуясь|"
    r"утративши[ми]+\s+силу|отменить\s+приказ|внести\s+изменения\s+в|"
    r"о\s+внесении\s+изменений\s+в)",
    re.IGNORECASE,
)

# Relation extraction regexes (supersedes / amends)
SUPERSEDES_PHRASE_RE = re.compile(
    r"(?:признать\s+утративши[ми]+\s+силу|отменить\s+действие|отменить\s+приказ|"
    r"считать\s+утративши[ми]+\s+силу|взамен\s+приказа)"
    r"[^.;\n]{0,140}?\b(?:приказ[ауом]?|распоряжени[еяюем]+|регламент[ауом]?|положени[еяюем]+|поряд[окауом]+|инструкци[юией]+)"
    r"[^.;\n]{0,80}?(?<!приложени[еяй]\s)(?:№|N)\s*(\d{1,5}(?:-[0-9А-Яа-яA-Za-z]+)?)",
    re.IGNORECASE,
)

# Unnumbered / title+date repeal clause (e.g. "Признать утратившим силу регламент поселения ... № б/н от 07.05.2024 г.")
UNNUMBERED_SUPERSEDES_PHRASE_RE = re.compile(
    r"(?:признать\s+утративши[ми]+\s+силу|считать\s+утративши[ми]+\s+силу|отменить\s+действие)"
    r"\s+((?:регламент[а-яё]*|положени[а-яё]*|поряд[а-яё]*|инструкци[а-яё]*|правил[а-яё]*)"
    r"[^.;\n]{12,180})",
    re.IGNORECASE,
)

AMENDS_PHRASE_RE = re.compile(
    r"(?:внести\s+изменения\s+в|во\s+изменение\s+приказа|изменить\s+редакцию|"
    r"дополнить\s+приказ|о\s+внесении\s+изменений\s+в)"
    r"[^.;\n]{0,140}?\b(?:приказ[ауом]?|распоряжени[еяюем]+|регламент[ауом]?|положени[еяюем]+|поряд[окауом]+|инструкци[юией]+)"
    r"[^.;\n]{0,80}?(?<!приложени[еяй]\s)(?:№|N)\s*(\d{1,5}(?:-[0-9А-Яа-яA-Za-z]+)?)",
    re.IGNORECASE,
)

# Edition / revision extraction regexes (Criterion K2)
EDITION_DATE_RE = re.compile(
    r"(?:в\s+ред(?:акции|\.)?(?:\s*(?:№|N|#)?\s*\d+(?:\.\d+)?)?(?:\s+(?:приказа|распоряжения|протокола|решения)[^\n,;()]{0,50}?)?\s+от|"
    r"(?:редакци[яиейю]|ред\.|верси[яиейю]|вер\.)(?:\s*(?:№|N|#)?\s*\d+(?:\.\d+)?)?\s+от|"
    r"с\s+изменени(?:ями|ем)(?:\s+и\s+дополнени(?:ями|ем))?(?:\s*,?\s*внесенн[а-яё]+[^\n,;()]{0,45}?)?\s+от|"
    r"в\s+новой\s+редакции\s+от)\s*"
    r"(?:[«\"']?(\d{1,2})[»\"']?\s+([а-яё]+)\s+(\d{4})|(\d{1,2}\.\d{1,2}\.\d{4}|\d{4}-\d{2}-\d{2}))",
    re.IGNORECASE,
)

EDITION_YEAR_RE = re.compile(
    r"(?:в\s+ред(?:акции|\.)|редакци[яиейю])\s+((?:19|20)\d{2})\s*(?:г(?:\.|ода)?)?\b",
    re.IGNORECASE,
)

EDITION_NUMBER_RE = re.compile(
    r"(?:редакци[яиейю]|ред\.|верси[яиейю]|вер\.|\bver\.?|\bv\.?)\s*(?:№|N|#)?\s*"
    r"(\d{1,3}(?:\.\d{1,2})?)(?!\.\d{2,4})\b",
    re.IGNORECASE,
)

FILENAME_VERSION_TAG_RE = re.compile(
    r"(?:^|[\s_.\-(])(?:v|ver|версия|редакция|ред)[\s_.\-]*(\d{1,2}(?:\.\d{1,2})?)(?=$|[\s_.\-)])",
    re.IGNORECASE,
)

NEW_EDITION_MARKER_RE = re.compile(
    r"(?:\b(?:новая\s+редакция|в\s+новой\s+редакции|новой\s+редакции|актуальная\s+редакция)\b|"
    r"\bнов(?:ый|ая|ое)\s*$)",
    re.IGNORECASE,
)

PARTIAL_AMENDMENT_TITLE_RE = re.compile(
    r"^\s*(?:(?:приказ|распоряжение)\b[^;\n]{0,75}?\s+)?(?:о\s+внесении\s+изменений\s+в|об\s+изменении|во\s+изменение)\b",
    re.IGNORECASE,
)

ANNUAL_TARGET_YEAR_RE = re.compile(
    r"\b(?:на\s+(?:плановый\s+период\s+)?|по\s+фин[.\s]*структуре\s+|бюджет[а-яё]*\s+на\s+)"
    r"((?:19|20)\d{2})(?:\s*[-–/и,]\s*((?:19|20)\d{2}))?\s*(?:учебн[а-яё]+\s+)?(?:гг?\.?|год[ауеыов]*)?",
    re.IGNORECASE,
)

# Applicable years extraction (1995-2050) with guards against order numbers and budget codes
_YEAR_CANDIDATE_RE = re.compile(r"(?<!\d)(199[5-9]|20[0-4]\d|2050)(?!\d)")
_YEAR_PREFIX_GUARD_RE = re.compile(
    r"(?:№\s*|(?:^|[^A-Za-zА-ЯЁа-яё0-9])(?:N(?:[oо°]\.?)?|КОСГУ|КВР|КФО|сч\.?)[\s:]*)$",
    re.IGNORECASE,
)
_YEAR_SUFFIX_ISO_DATE_RE = re.compile(r"^-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)")
_YEAR_SUFFIX_ORDER_INDEX_RE = re.compile(r"^[-–—]\d{1,3}(?:\b|(?=[а-яёa-z]\b))", re.IGNORECASE)


def _extract_applicable_years(year_probe: str) -> list[int]:
    """
    Extract applicable years in 1995..2050 from `year_probe`, ignoring 4-digit order numbers
    (e.g. `№ 2025-1`, `N 1998-1`, `№ 2026`, `2025-1`) and budget/account codes (`КОСГУ 2026`,
    `КВР 2020`, `КФО ...`, `сч. ...`) while preserving year ranges (`2025-2031`, `2026–2030`)
    and suffixed years (`2031г.`).
    """
    if not year_probe:
        return []
    found: list[int] = []
    for m in _YEAR_CANDIDATE_RE.finditer(year_probe):
        prefix = year_probe[: m.start()]
        suffix = year_probe[m.end() :]
        if _YEAR_PREFIX_GUARD_RE.search(prefix):
            continue
        if not _YEAR_SUFFIX_ISO_DATE_RE.match(suffix) and _YEAR_SUFFIX_ORDER_INDEX_RE.match(suffix):
            continue
        y_int = int(m.group(1))
        if 1995 <= y_int <= 2050 and y_int not in found:
            found.append(y_int)
    found.sort()
    return found

_ORDER_PREAMBLE_START_RE = re.compile(
    r"^\s*(?:В\s+целях|С\s+целью|На\s+основании|В\s+связи\s+с|Во\s+исполнение|В\s+соответствии\s+с|Руководствуясь|"
    r"ПРИКАЗЫВАЮ|РАСПОРЯЖАЮСЬ|ОБЯЗЫВАЮ|ПОСТАНОВЛЯЮ)\b",
    re.IGNORECASE,
)

_ACT_LEADING_NOUN_CANON_MAP: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^регламент[аеуом]*\b"), "регламент"),
    (re.compile(r"^положени[еяюийм]*\b"), "положение"),
    (re.compile(r"^поряд(?:ок|ка|ке|ку|ком)\b"), "порядок"),
    (re.compile(r"^инструкци[яиюейм]*\b"), "инструкция"),
    (re.compile(r"^правил[аоамх]*\b"), "правила"),
    (re.compile(r"^стандарт[аеуом]*\b"), "стандарт"),
    (re.compile(r"^методик[аиеуой]*\b"), "методика"),
    (re.compile(r"^политик[аиеуой]*\b"), "политика"),
    (re.compile(r"^переч(?:ень|ня|ню|нем)\b"), "перечень"),
    (re.compile(r"^тариф[аыове]*\b"), "тарифы"),
    (re.compile(r"^план[ауеом]*\b"), "план"),
)


def _normalize_lookup_key(text: str) -> str:
    """Normalize text for case-insensitive Russian lookup (lowercase, ё -> е, single spaces)."""
    norm = text.lower().replace("ё", "е")
    norm = re.sub(r"[«»\"'()]", "", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def _to_iso_date(date_str: str) -> str:
    """Convert DD.MM.YYYY, Russian text date ('18 августа 2025'), or YYYY-MM-DD to ISO YYYY-MM-DD."""
    s = (date_str or "").strip()
    if not s or "_" in s or "[" in s:
        return ""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        y, mo, d = int(s[:4]), int(s[5:7]), int(s[8:10])
        if 1990 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            return s
        return ""
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1990 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    m_ru = re.match(r"^[«\"']?(\d{1,2})[»\"']?\s+([а-яё]+)\s+(\d{4})", s, re.IGNORECASE)
    if m_ru:
        day, month_name, year = int(m_ru.group(1)), m_ru.group(2).lower(), int(m_ru.group(3))
        month_num = RUSSIAN_MONTHS_MAP.get(month_name, "")
        if month_num and 1990 <= year <= 2100 and 1 <= day <= 31:
            return f"{year:04d}-{month_num}-{day:02d}"
    return ""


def _parse_flex_date_groups(
    day_s: str | None,
    month_s: str | None,
    year_s: str | None,
    ddmmyyyy_s: str | None,
) -> str:
    """Convert regex capture groups `(day, month_ru, year, ddmmyyyy)` into ISO YYYY-MM-DD."""
    if ddmmyyyy_s:
        return _to_iso_date(ddmmyyyy_s)
    if day_s and month_s and year_s:
        month_num = RUSSIAN_MONTHS_MAP.get(month_s.strip().lower(), "")
        if month_num:
            day_i, year_i = int(day_s), int(year_s)
            if 1990 <= year_i <= 2100 and 1 <= day_i <= 31:
                return f"{year_i:04d}-{month_num}-{day_i:02d}"
    return ""


def _clean_valid_doc_number(raw: str | None) -> str:
    """
    Validate and normalize an extracted document/registration number.
    Rejects blank placeholders ('___', '[___]', '_____-1', '-1', 'б/н') and requires a leading digit.
    """
    if not raw:
        return ""
    cand = raw.strip().strip(".,;:)»\"'")
    if not cand or "_" in cand or "[" in cand or "]" in cand:
        return ""
    if cand.lower() in {"б/н", "б/д", "н/д", "none"}:
        return ""
    if not re.match(r"^\d{1,6}(?:[-/][0-9А-ЯЁа-яёA-Za-z]{1,6})?[А-ЯЁа-яёA-Za-z]?$", cand):
        return ""
    return cand


def _normalize_filename_for_requisites(file_name: str) -> str:
    """
    Normalize underscores and separators in uploaded filenames so patterns like
    `Приказ_№_2990_1_от_19_06_2026_Об_утверждении_регламента_поселения.pdf`
    become `Приказ № 2990-1 от 19.06.2026 Об утверждении регламента поселения`.
    """
    if not file_name:
        return ""
    stem = re.sub(r"\.(?:pdf|docx?|xlsx?|txt|md)$", "", Path(file_name).name, flags=re.IGNORECASE)
    # 1. Convert DD_MM_YYYY -> DD.MM.YYYY and YYYY_MM_DD -> YYYY-MM-DD
    s = re.sub(r"(?<!\d)(\d{2})_(\d{2})_((?:19|20)\d{2})(?!\d)", r"\1.\2.\3", stem)
    s = re.sub(r"(?<!\d)((?:19|20)\d{2})_(\d{2})_(\d{2})(?!\d)", r"\1-\2-\3", s)
    # 2. Convert underscore-separated order numbers after № / N / No (e.g. №_2990_1_от -> № 2990-1_от)
    s = re.sub(
        r"((?:№|N|No)[\s_]*)(\d{1,6})_(\d{1,4}[А-ЯЁа-яёA-Za-z]?)(?=[\s_]|$)",
        r"\1\2-\3",
        s,
        flags=re.IGNORECASE,
    )
    # 3. Replace remaining underscores with spaces and collapse whitespace
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_strict_header_zone(blocks_or_text: list[ParsedBlock] | str, max_chars: int = 1200) -> str:
    """
    Extract strictly the top document header/title zone from Page 1 BEFORE any body clauses,
    directives ('ПРИКАЗЫВАЮ:'), section headings ('1. Общие положения'), or data tables,
    filtering out lines that cite external/repealed/amended acts ('в соответствии с...', 'О внесении изменений в...').
    """
    raw_lines: list[str] = []
    if isinstance(blocks_or_text, list):
        non_empty = [b for b in blocks_or_text if (b.text and b.text.strip()) or b.table_rows]
        if not non_empty:
            return ""
        first_page = non_empty[0].page_num or 1
        for b in non_empty:
            if (b.page_num or 1) != first_page:
                break
            # Stop at multi-row data tables (1x2 requisite tables have 0 or 1 data row)
            if b.block_type == BlockType.TABLE.value and len(b.table_rows or []) >= 2:
                break
            if b.text:
                raw_lines.extend(b.text.split("\n"))
    else:
        raw_lines = str(blocks_or_text or "")[:2500].split("\n")

    kept_lines: list[str] = []
    total_len = 0
    for raw_ln in raw_lines:
        ln = raw_ln.strip()
        if not ln:
            continue
        if _HEADER_BODY_STOP_RE.match(ln):
            break
        if _EXTERNAL_ACT_CITATION_LINE_RE.search(ln):
            continue
        if total_len + len(ln) > max_chars:
            break
        kept_lines.append(ln)
        total_len += len(ln) + 1

    return "\n".join(kept_lines)


def _extract_edo_stamp(blocks_or_text: list[ParsedBlock] | str, full_text: str) -> tuple[str, str]:
    """
    Extract `(edo_number, edo_date)` from official ЭДО / registration stamps anywhere in the document.
    Prioritizes explicit `BlockType.STAMP` blocks first, then scans full document text.
    """
    probe_texts: list[str] = []
    if isinstance(blocks_or_text, list):
        stamp_texts = [
            b.text.strip()
            for b in blocks_or_text
            if b.text and b.block_type == BlockType.STAMP.value
        ]
        if stamp_texts:
            probe_texts.append("\n".join(stamp_texts))
    if full_text:
        probe_texts.append(full_text)

    for probe in probe_texts:
        for m_edo in EDO_STAMP_RE.finditer(probe):
            cand_num = _clean_valid_doc_number(m_edo.group(5))
            cand_date = _parse_flex_date_groups(
                m_edo.group(1), m_edo.group(2), m_edo.group(3), m_edo.group(4)
            )
            if cand_num:
                return cand_num, cand_date

        for m_rev in EDO_STAMP_REV_RE.finditer(probe):
            cand_num = _clean_valid_doc_number(m_rev.group(1))
            cand_date = _parse_flex_date_groups(
                m_rev.group(2), m_rev.group(3), m_rev.group(4), m_rev.group(5)
            )
            if cand_num:
                return cand_num, cand_date

    return "", ""


def _slugify_doc_id(file_name: str, doc_number: str = "", doc_date: str = "") -> str:
    """Generate a deterministic, clean ASCII/slug doc_id if none is provided by overrides."""
    if doc_number:
        clean_num = re.sub(r"[^0-9a-zA-Zа-яА-Я]+", "_", doc_number).strip("_").lower()
        year_part = doc_date[:4] if doc_date and len(doc_date) >= 4 else ""
        # Transliterate common Cyrillic letters in numbers like 1а -> 1a
        translit = {"а": "a", "б": "b", "в": "v", "г": "g", "к": "k", "з": "z", "п": "p", "о": "o", "с": "s", "ц": "ts", "д": "d"}
        clean_num_ascii = "".join(translit.get(ch, ch) for ch in clean_num)
        if re.match(r"^\d", clean_num_ascii):
            return f"doc_{clean_num_ascii}_{year_part}".rstrip("_")
    digest = hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:10]
    return f"doc_{digest}"


def edition_sort_tuple(edition_number: Any) -> tuple[int, int]:
    """Convert `edition_number` (int, '2.1', 'новая', or None) into a total-order tuple."""
    if edition_number is None or edition_number == "":
        return (0, 0)
    if isinstance(edition_number, bool):
        return (1, 0) if edition_number else (0, 0)
    if isinstance(edition_number, int):
        return (edition_number, 0)
    if isinstance(edition_number, float):
        major = int(edition_number)
        minor = int(round((edition_number - major) * 100))
        return (major, minor)
    s = str(edition_number).strip().lower().replace("ё", "е")
    if not s:
        return (0, 0)
    if s in {"новая", "актуальная", "новый", "new"}:
        # Beats unnumbered (0, 0) and edition 1 (1, 0), yields to explicit edition >= 2
        return (1, 50)
    m = re.match(r"^(\d{1,3})(?:\.(\d{1,3}))?$", s)
    if m:
        return (int(m.group(1)), int(m.group(2) or 0))
    return (0, 0)


def extract_edition_metadata(
    title: str = "",
    header_text: str = "",
    full_text: str = "",
    file_name: str = "",
) -> dict[str, Any]:
    """
    Extract explicit K2 edition/revision metadata from title, filename, and header/opening text:
    - `edition_date` (ISO `YYYY-MM-DD`, or `""`)
    - `edition_number` (`int` e.g. `2`, `str` e.g. `"2.1"` / `"новая"`, or `None`)
    Filters out external act citation lines (`в соответствии с ... в ред. от ...`) so cited federal
    laws never pollute the document's own edition date.
    """
    norm_fname = _normalize_filename_for_requisites(file_name) if file_name else ""
    safe_body_lines: list[str] = []
    for raw_ln in (full_text or "")[:2800].splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        if _HEADER_BODY_STOP_RE.match(ln):
            break
        if _EXTERNAL_ACT_CITATION_LINE_RE.search(ln):
            continue
        safe_body_lines.append(ln)

    probes = [p for p in (title, norm_fname, header_text, "\n".join(safe_body_lines)) if p]
    edition_date = ""
    edition_number: int | str | None = None
    has_new_marker = False

    for probe in probes:
        if not edition_date:
            m_ed_dt = EDITION_DATE_RE.search(probe)
            if m_ed_dt:
                cand_dt = _parse_flex_date_groups(
                    m_ed_dt.group(1), m_ed_dt.group(2), m_ed_dt.group(3), m_ed_dt.group(4)
                )
                if cand_dt:
                    edition_date = cand_dt

        if edition_number is None:
            for m_num in EDITION_NUMBER_RE.finditer(probe):
                raw_n = m_num.group(1)
                major_s = raw_n.split(".", 1)[0]
                if major_s.isdigit() and 1 <= int(major_s) < 1900:
                    edition_number = int(raw_n) if "." not in raw_n else raw_n
                    break

        if not has_new_marker and NEW_EDITION_MARKER_RE.search(probe):
            has_new_marker = True

    if edition_number is None and file_name:
        stem = re.sub(r"\.(?:pdf|docx?|xlsx?|txt|md)$", "", Path(file_name).name, flags=re.IGNORECASE)
        m_fv = FILENAME_VERSION_TAG_RE.search(stem)
        if m_fv:
            raw_v = m_fv.group(1)
            major_s = raw_v.split(".", 1)[0]
            if major_s.isdigit() and 1 <= int(major_s) < 100:
                edition_number = int(raw_v) if "." not in raw_v else raw_v

    if not edition_date:
        for probe in (title, norm_fname, header_text):
            if not probe:
                continue
            m_yr = EDITION_YEAR_RE.search(probe)
            if m_yr:
                edition_date = f"{int(m_yr.group(1)):04d}-01-01"
                break

    if edition_number is None and has_new_marker:
        edition_number = "новая"

    return {
        "edition_date": edition_date,
        "edition_number": edition_number,
        "is_new_edition": has_new_marker,
    }


def extract_target_plan_years(title: str = "", header_text: str = "") -> list[int]:
    """
    Extract explicit target planning years from phrases like 'на 2026 год', 'на 2026-2028 гг.',
    'по Фин.структуре 2026г.' so distinct annual plans with disjoint target years stay active.
    """
    years: list[int] = []
    for probe in (title, header_text):
        if not probe:
            continue
        for m in ANNUAL_TARGET_YEAR_RE.finditer(probe):
            y1 = int(m.group(1))
            if 2015 <= y1 <= 2035 and y1 not in years:
                years.append(y1)
            if m.group(2):
                y2 = int(m.group(2))
                if 2015 <= y2 <= 2035:
                    for y in range(min(y1, y2), max(y1, y2) + 1):
                        if y not in years:
                            years.append(y)
    years.sort()
    return years


def _extract_formal_header_subject(full_text: str) -> str:
    """
    Extract the formal subject line from the top of a Russian normative document:
    - For Orders/Dispositions: the 'О ...' / 'Об утверждении ...' heading between 'ПРИКАЗ № ...' and 'В целях ...'/'ПРИКАЗЫВАЮ:'
    - For standalone Regulations/Provisions/Instructions: the 'Регламент ...' / 'Положение ...' heading before '1. Общие положения'.
    """
    if not full_text:
        return ""
    lines = [ln.strip() for ln in full_text[:2600].splitlines() if ln.strip()]
    if not lines:
        return ""

    # 1. Check Order / Disposition formal subject after ПРИКАЗ / РАСПОРЯЖЕНИЕ + date/number
    for idx, ln in enumerate(lines[:18]):
        if re.match(r"^(?:ПРИКАЗ|РАСПОРЯЖЕНИЕ)\b", ln, re.IGNORECASE):
            subj_lines: list[str] = []
            inline_tail = re.sub(
                r"^(?:ПРИКАЗ|РАСПОРЯЖЕНИЕ)(?:\s+(?:от\s+)?(?:\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2}|[«\"]?\d{1,2}[»\"]?\s+[а-яё]+\s+\d{4})(?:\s*г\.?)?|\s+(?:№|N)\s*[0-9А-ЯЁа-яёA-Za-z\-/]+)+",
                "",
                ln,
                flags=re.IGNORECASE,
            ).strip(" -—:,.")
            if not inline_tail or inline_tail.lower() == ln.lower():
                inline_tail = re.sub(r"^(?:ПРИКАЗ|РАСПОРЯЖЕНИЕ)\b", "", ln, flags=re.IGNORECASE).strip(" -—:,.")
            if len(inline_tail) >= 12 and re.search(r"[а-яёА-ЯЁ]{4,}", inline_tail):
                subj_lines.append(ln)
            for sub_ln in lines[idx + 1 : idx + 10]:
                if _ORDER_PREAMBLE_START_RE.match(sub_ln) or _HEADER_BODY_STOP_RE.match(sub_ln):
                    break
                # Skip date / number / city lines right under ПРИКАЗ
                if re.match(
                    r"^(?:от\s+|№|N|\d{1,2}\.\d{1,2}\.\d{4}|[«\"]?\d{1,2}[»\"]?\s+[а-яё]+\s+\d{4}|г\.\s*[А-ЯЁа-яё\-]+$)",
                    sub_ln,
                    re.IGNORECASE,
                ):
                    if not subj_lines:
                        continue
                if re.match(r"^(?:№|N)\s*[0-9А-ЯЁа-яёA-Za-z\-/]+$", sub_ln, re.IGNORECASE) and not subj_lines:
                    continue
                subj_lines.append(sub_ln)
                if len(" ".join(subj_lines)) >= 160:
                    break
            if subj_lines:
                return " ".join(subj_lines)

    # 2. Check standalone Regulation / Provision / Instruction / Rules title before '1. Общие положения' / '1.1'
    act_start_re = re.compile(
        r"^(?:РЕГЛАМЕНТ|ПОЛОЖЕНИЕ|ПОРЯДОК|ИНСТРУКЦИЯ|ПРАВИЛА)\b",
        re.IGNORECASE,
    )
    for idx, ln in enumerate(lines[:28]):
        if ln.startswith("[") or re.match(r"^(?:СОГЛАСОВАНО|УТВЕРЖД|Приложение)\b", ln, re.IGNORECASE):
            continue
        if _HEADER_BODY_STOP_RE.match(ln):
            if idx + 1 < len(lines) and re.match(r"^\[Стр\.\s*1\]", lines[idx + 1]):
                continue
            break
        if act_start_re.match(ln):
            subj_lines = [ln]
            for sub_ln in lines[idx + 1 : idx + 6]:
                if _HEADER_BODY_STOP_RE.match(sub_ln) or _ORDER_PREAMBLE_START_RE.match(sub_ln):
                    break
                if sub_ln.startswith("[") or re.match(r"^\d+\.\d*", sub_ln):
                    break
                subj_lines.append(sub_ln)
                if len(" ".join(subj_lines)) >= 160:
                    break
            cand = " ".join(subj_lines).strip()
            if len(cand) > len("регламент") + 4:
                return cand

    return ""


def _normalize_subject_for_version_family(raw_subject: str) -> str:
    """
    Strip dates, document numbers, edition/version tags, institutional boilerplate, and approval wrappers
    ('Приказ об утверждении ...', 'О введении в действие ...'), canonicalizing the leading regulatory noun.
    Returns `""` if the cleaned subject is `< 18` chars or lacks substantive words.
    """
    if not raw_subject:
        return ""
    s = raw_subject.lower().replace("ё", "е")
    s = re.sub(r"\.(?:pdf|docx?|xlsx?|txt|md)\b", " ", s)
    s = re.sub(r"[_«»\"'“”„]", " ", s)

    # Strip parenthetical edition/revision/approval markers while keeping domain acronyms like (ПУМС), (ЦОД), (ФЛС)
    s = re.sub(
        r"\((?:в\s+ред|ред\.|редакц|верси|вер\.|v\.?\s*\d|с\s+изм|утв\.|утратил|действ|нов[а-я]+\s+ред|приказ|распоряжен|протокол)[^)]*\)",
        " ",
        s,
    )

    # Preserve code stem in ИОТ-056-2021 -> иот-056 while stripping the edition year
    s = re.sub(r"\b(иот\s*[-–]?\s*\d+)\s*[-–]\s*(?:19|20)\d{2}\b", r"\1", s)

    # Strip explicit edition/version phrases
    s = re.sub(
        r"\b(?:в\s+новой\s+редакции|новая\s+редакция|новой\s+редакции|актуальная\s+редакция|"
        r"старая\s+редакция|предыдущая\s+редакция)\b",
        " ",
        s,
    )
    s = re.sub(
        r"\b(?:в\s+ред(?:акции|\.)?|редакци[яиейю]|ред\.|верси[яиейю]|вер\.|ver\.?)\s*"
        r"(?:приказа\s+|распоряжения\s+)?(?:от\s*)?"
        r"(?:\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2}|(?:№|n|#)?\s*\d{1,3}(?:\.\d{1,2})?)?"
        r"(?:\s*г(?:\.|ода)?)?",
        " ",
        s,
    )
    s = re.sub(
        r"\bс\s+изменениями(?:\s+и\s+дополнениями)?(?:\s+от\s*(?:\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2}))?",
        " ",
        s,
    )
    s = re.sub(r"(?:^|\s)v\.?\s*\d{1,2}(?:\.\d{1,2})?\b", " ", s)

    # Strip document numbers, dates, and year qualifiers
    s = re.sub(r"(?:№|n|no)\s*(?:б/н|[0-9][0-9а-яa-z\-/]*)", " ", s)
    s = re.sub(
        r"\bот\s+(?:\d{1,2}\s+[а-я]+\s+\d{4}|\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2})(?:\s*г(?:\.|ода)?)?",
        " ",
        s,
    )
    s = re.sub(r"\b(?:\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2})(?:\s*г(?:\.|ода)?)?\b", " ", s)
    s = re.sub(
        r"\bна\s+(?:плановый\s+период\s+)?(?:19|20)\d{2}(?:\s*[-–/и,]\s*(?:19|20)\d{2})*\s*(?:учебн[а-я]+\s+)?(?:гг?\.?|год[ауеыов]*)\b",
        " ",
        s,
    )
    s = re.sub(r"\b(?:19|20)\d{2}(?:\s*[-–/]\s*(?:19|20)\d{2})?\s*(?:гг?\.?|год[ауеыов]*)?\b", " ", s)

    # Strip institutional boilerplate and bare edition adjectives
    s = re.sub(r"\bминистерств[а-я\s]+российской\s+федерации\b", " ", s)
    s = re.sub(
        r"\bфедеральн[а-я]+\s+государственн[а-я]+\s+автономн[а-я]+\s+образовательн[а-я]+\s+учреждени[а-я]+(?:\s+высшего\s+образования)?\b",
        " ",
        s,
    )
    s = re.sub(
        r"\bмосковск[а-я]+\s+физико[\s\-]*техническ[а-я]+\s+институт[а-я]*(?:\s*\(национальн[а-я\s]+университет\))?\b",
        " ",
        s,
    )
    s = re.sub(r"\bфгаоу\s+во\b", " ", s)
    s = re.sub(r"\b(?:в\s+|для\s+)?(?:мфти|физтех[аеу]?)\b", " ", s)
    s = re.sub(r"\b(?:нов(?:ый|ая|ое|ые)|стар(?:ый|ая|ое|ые)|утративш[а-я]+\s+силу)\b", " ", s)

    # Collapse punctuation (keep hyphens inside compound words like финансово-лицевой or иот-056)
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")

    # Iteratively strip leading order/approval wrappers so 'Приказ об утверждении Положения о Х' == 'Положение о Х'
    changed = True
    while changed and s:
        prev = s
        s = re.sub(r"^(?:приказ|распоряжение)(?:\s+и)?\s+", "", s).strip()
        s = re.sub(
            r"^(?:об\s+утверждении\s+и\s+введении\s+в\s+действие|о\s+введении\s+в\s+действие|"
            r"об\s+утверждении|об\s+актуализации|о\s+принятии)\s+",
            "",
            s,
        ).strip()
        s = re.sub(r"^(?:новой\s+редакции|редакции)\s+", "", s).strip()
        s = re.sub(r"^(?:о|об)\s+", "", s).strip()
        changed = s != prev

    # Canonicalize leading regulatory noun inflection (genitive/prepositional -> nominative)
    for pat, canon_noun in _ACT_LEADING_NOUN_CANON_MAP:
        if pat.search(s):
            s = pat.sub(canon_noun, s, count=1)
            break

    s = re.sub(r"\s+", " ", s).strip(" -")
    if len(s) < 18:
        return ""
    substantive_words = [
        w
        for w in s.split()
        if len(w) >= 3
        and w not in {"приказ", "распоряжение", "документ", "регламент", "положение", "порядок", "инструкция", "правила"}
    ]
    if len(substantive_words) < 2:
        return ""
    return s


def compute_version_family(
    doc_type: str = "",
    title: str = "",
    header_text: str = "",
    full_text: str = "",
    file_name: str = "",
) -> str:
    """
    Compute a canonical K2 `version_family` key for automatic multi-version reconciliation.
    - Returns `""` for overly short/generic titles (`< 18` chars normalized subject) so bare 'Приказ' orders never collide.
    - Separates standalone forms/templates (`form::...`) and partial amendments (`amendment::...`) from normative acts (`normative::...`).
    """
    norm_fname = _normalize_filename_for_requisites(file_name) if file_name else ""
    base_title = (title or norm_fname or "").strip()
    formal_subj = _extract_formal_header_subject(full_text or header_text)

    dt_lower = (doc_type or "").strip().lower().replace("ё", "е")
    probe_head = (formal_subj or base_title).strip().lower().replace("ё", "е")

    # Classify namespace so forms/annexes and partial-amendment orders never falsely supersede full regulations
    if any(k in dt_lower for k in ("анкета", "согласие", "смета", "приложение")) or probe_head.startswith("приложение"):
        namespace = "form"
    elif (
        (PARTIAL_AMENDMENT_TITLE_RE.match(base_title) or PARTIAL_AMENDMENT_TITLE_RE.match(formal_subj))
        and "новой редакции" not in probe_head
        and not SUPERSEDES_PHRASE_RE.search((full_text or "")[:6000])
        and not UNNUMBERED_SUPERSEDES_PHRASE_RE.search((full_text or "")[:6000])
    ):
        namespace = "amendment"
    else:
        namespace = "normative"

    subj_title = _normalize_subject_for_version_family(base_title)
    subj_header = _normalize_subject_for_version_family(formal_subj)

    if not subj_title:
        chosen = subj_header
    elif not subj_header:
        chosen = subj_title
    elif re.search(r"[а-я]", subj_header) and not re.search(r"[а-я]", subj_title):
        chosen = subj_header
    elif subj_header.startswith(subj_title) or subj_title in subj_header:
        chosen = subj_header
    else:
        chosen = subj_title

    if not chosen or len(chosen) < 18:
        return ""
    return f"{namespace}::{chosen}"


class GlossaryManager:
    """Bidirectional MIPT Domain Glossary Manager with Auto-Learning, Role/Topic Extraction, and Metadata Parser."""

    def __init__(
        self,
        glossary_path: str | Path | None = None,
        overrides_path: str | Path | None = None,
    ) -> None:
        self.glossary_path: Path = Path(glossary_path) if glossary_path else GLOSSARY_PATH
        self.overrides_path: Path = Path(overrides_path) if overrides_path else REGISTRY_OVERRIDES_PATH

        self.abbreviations: dict[str, str] = dict(DEFAULT_ABBREVIATIONS)
        self.query_synonyms: dict[str, list[str]] = {}
        self.role_patterns: dict[str, re.Pattern[str]] = dict(ROLE_PATTERNS)
        self.topic_keywords: dict[str, list[str]] = {
            k: list(v) for k, v in DEFAULT_TOPIC_KEYWORDS.items()
        }
        self.learned_definitions: dict[str, str] = {}

        # Lookup indexes
        self.abbr_upper_map: dict[str, str] = {}
        self.expansion_to_abbr: dict[str, str] = {}
        self._abbr_regex_map: dict[str, re.Pattern[str]] = {}

        # Registry overrides
        self.overrides_documents: dict[str, dict[str, Any]] = {}
        self.overrides_by_number: dict[str, dict[str, Any]] = {}

        self._load_glossary_yaml()
        self._load_overrides_yaml()
        self._rebuild_indexes()

    def _load_glossary_yaml(self) -> None:
        """Load abbreviations, query_synonyms, role_patterns, and topic_keywords from config/glossary.yaml."""
        if not self.glossary_path.exists():
            logger.warning("Glossary YAML not found at %s; using built-in MIPT defaults.", self.glossary_path)
            return
        try:
            with self.glossary_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data.get("abbreviations"), dict):
                for k, v in data["abbreviations"].items():
                    if k and v:
                        self.abbreviations[str(k).strip()] = str(v).strip()
            if isinstance(data.get("query_synonyms"), dict):
                for k, v in data["query_synonyms"].items():
                    if isinstance(v, list):
                        self.query_synonyms[ _normalize_lookup_key(str(k)) ] = [str(x).strip() for x in v if x]
            if isinstance(data.get("role_patterns"), dict):
                for role_name, pattern_str in data["role_patterns"].items():
                    if role_name and pattern_str:
                        try:
                            # Keep our enhanced pattern if default has broader coverage, or compile custom role
                            if role_name not in self.role_patterns:
                                self.role_patterns[str(role_name)] = re.compile(str(pattern_str))
                        except re.error as exc:
                            logger.warning("Invalid regex for role %s: %s", role_name, exc)
            if isinstance(data.get("topic_keywords"), dict):
                for topic_name, kw_list in data["topic_keywords"].items():
                    if isinstance(kw_list, list):
                        existing = self.topic_keywords.get(str(topic_name), [])
                        merged = list(dict.fromkeys(existing + [str(x).strip().lower().replace("ё", "е") for x in kw_list if x]))
                        self.topic_keywords[str(topic_name)] = merged
        except Exception as exc:
            logger.error("Failed to load glossary.yaml (%s): %s", self.glossary_path, exc)

    def _load_overrides_yaml(self) -> None:
        """Load document metadata & relation overrides from config/registry_overrides.yaml."""
        if not self.overrides_path.exists():
            logger.warning("Registry overrides YAML not found at %s.", self.overrides_path)
            return
        try:
            with self.overrides_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data.get("documents"), dict):
                self.overrides_documents = dict(data["documents"])
            if isinstance(data.get("by_doc_number"), dict):
                self.overrides_by_number = {
                    str(k).strip(): dict(v) for k, v in data["by_doc_number"].items() if isinstance(v, dict)
                }
        except Exception as exc:
            logger.error("Failed to load registry_overrides.yaml (%s): %s", self.overrides_path, exc)

    def _rebuild_indexes(self) -> None:
        """Rebuild fast bidirectional lookup tables and compiled word-boundary regexes."""
        self.abbr_upper_map.clear()
        self.expansion_to_abbr.clear()
        self._abbr_regex_map.clear()

        for abbr, expansion in self.abbreviations.items():
            norm_abbr = abbr.upper().replace("Ё", "Е")
            self.abbr_upper_map[norm_abbr] = abbr
            norm_exp = _normalize_lookup_key(expansion)
            if norm_exp:
                self.expansion_to_abbr[norm_exp] = abbr
                # If expansion has slash alternatives (e.g. "электронная библиотека / электробезопасность")
                for part in norm_exp.split("/"):
                    part_clean = part.strip()
                    if len(part_clean) >= 5:
                        self.expansion_to_abbr[part_clean] = abbr

            # Compile boundary regex for abbreviation detection in text
            escaped = re.escape(abbr)
            # Short 2-char abbreviations (КЗ, ЭБ, СЗ, ТЗ, ФШ) require case-sensitivity unless colon/digit present
            if len(abbr) <= 2 and abbr.isalpha():
                pattern = rf"(?<![А-ЯЁа-яёA-Za-z0-9]){escaped}(?![А-ЯЁа-яёA-Za-z0-9])"
                self._abbr_regex_map[abbr] = re.compile(pattern)
            else:
                pattern = rf"(?<![А-ЯЁа-яёA-Za-z0-9]){escaped}(?![А-ЯЁа-яёA-Za-z0-9])"
                self._abbr_regex_map[abbr] = re.compile(pattern, re.IGNORECASE)

    # -----------------------------------------------------------------------
    # Bidirectional Lookup API
    # -----------------------------------------------------------------------
    def get_expansion(self, abbr: str) -> str | None:
        """Return full expansion for a domain abbreviation (case-insensitive)."""
        if not abbr:
            return None
        key = abbr.strip()
        if key in self.abbreviations:
            return self.abbreviations[key]
        upper_key = key.upper().replace("Ё", "Е")
        canon = self.abbr_upper_map.get(upper_key)
        if canon:
            return self.abbreviations.get(canon)
        return None

    expand_abbr = get_expansion
    lookup_abbr = get_expansion

    def get_abbreviation(self, expansion: str) -> str | None:
        """Return canonical abbreviation for a full expansion or inflected phrase."""
        if not expansion:
            return None
        norm = _normalize_lookup_key(expansion)
        if norm in self.expansion_to_abbr:
            return self.expansion_to_abbr[norm]
        for stem, abbr in EXPANSION_STEM_MAP.items():
            if stem in norm:
                return abbr
        for known_exp, abbr in self.expansion_to_abbr.items():
            if len(known_exp) >= 8 and (known_exp in norm or norm in known_exp):
                return abbr
        return None

    find_abbr = get_abbreviation
    lookup_expansion = get_abbreviation

    def lookup(self, term: str) -> str | None:
        """Bidirectional lookup: returns expansion if given abbreviation, or abbreviation if given expansion."""
        exp = self.get_expansion(term)
        if exp is not None:
            return exp
        return self.get_abbreviation(term)

    # -----------------------------------------------------------------------
    # Auto-Learning Definitions with Anti-Garbage Protection
    # -----------------------------------------------------------------------
    @staticmethod
    def _clean_candidate_expansion(raw_expansion: str) -> str:
        """Strip leading clause numbers, verbs, and punctuation from a candidate expansion."""
        text = raw_expansion.strip()
        # Take only the segment after the last sentence/clause boundary or colon/semicolon
        parts = re.split(r"[.;:!?]\s+|\n+", text)
        text = parts[-1].strip()
        text = re.sub(r"^\s*(?:п\.\s*)?\d+(?:\.\d+)*\.?\s*", "", text)
        text = LEADING_BOILERPLATE_RE.sub("", text).strip(" ,–—-«»\"'")
        # If there is still a comma separating an introductory clause, keep the trailing noun phrase if >= 10 chars
        if "," in text:
            comma_tail = text.rsplit(",", 1)[-1].strip()
            if len(comma_tail) >= 12 and not comma_tail.startswith(("как ", "так ", "в том ", "при ")):
                text = comma_tail
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _is_valid_definition_pair(short_term: str, expansion: str) -> bool:
        """Strict anti-garbage filter rejecting numbers, currency amounts, broken table cells, and boilerplate."""
        s_term = short_term.strip(" «»\"'.,;:-–—─")
        s_exp = expansion.strip(" «»\"'.,;:-–—─")

        if not s_term or not s_exp:
            return False
        if len(s_term) < 2 or len(s_term) > 35:
            return False
        if len(s_exp) < 6 or len(s_exp) > 150:
            return False
        if len(s_exp) <= len(s_term):
            return False

        # Reject table pipes, broken underscores, ellipsis, or None
        if any(bad in s_term or bad in s_exp for bad in ("|", "\t", "___", "...", "…", "None")):
            return False

        # Reject if short_term or expansion matches currency/number/boilerplate garbage
        if GARBAGE_TERM_RE.search(s_term) or GARBAGE_TERM_RE.search(s_exp):
            return False

        # Reject if short_term starts/ends with digits (except 1С:...)
        if not s_term.upper().startswith("1С"):
            if re.search(r"\d", s_term):
                return False

        # Reject if expansion has > 12% digits
        digit_count = sum(ch.isdigit() for ch in s_exp)
        if digit_count > 0 and (digit_count / len(s_exp)) > 0.12:
            return False

        # Must contain Cyrillic letters
        if not re.search(r"[А-ЯЁа-яё]{2,}", s_term) or not re.search(r"[А-ЯЁа-яё]{4,}", s_exp):
            return False

        # Reject generic document self-references like "Регламент", "Порядок", "Приказ", "Договор", "Проект приказа"
        generic_terms = {
            "регламент",
            "порядок",
            "приказ",
            "проект приказа",
            "распоряжение",
            "положение",
            "инструкция",
            "договор",
            "контракт",
            "соглашение",
            "стороны",
            "организация",
            "университет",
            "институт",
            "работник",
            "сотрудник",
            "заказчик",
            "исполнитель",
        }
        if _normalize_lookup_key(s_term) in generic_terms:
            return False

        return True

    def learn_from_text(self, text: str) -> dict[str, str]:
        """Auto-learn new domain definitions from document text via '(далее — ...)' and '(АББР)' patterns."""
        if not text or len(text) < 15:
            return {}

        newly_learned: dict[str, str] = {}

        # 1. "(далее — «...»)" / "(далее - АББР)"
        for match in DALEE_DEFINITION_RE.finditer(text):
            raw_exp, raw_term = match.group(1), match.group(2)
            clean_term = raw_term.strip(" «»\"'.,;:-–—─")
            clean_exp = self._clean_candidate_expansion(raw_exp)
            if self._is_valid_definition_pair(clean_term, clean_exp):
                norm_upper = clean_term.upper().replace("Ё", "Е")
                if norm_upper not in self.abbr_upper_map:
                    self.abbreviations[clean_term] = clean_exp
                self.learned_definitions[clean_term] = self.abbreviations.get(
                    self.abbr_upper_map.get(norm_upper, clean_term), clean_exp
                )
                newly_learned[clean_term] = self.learned_definitions[clean_term]

        # 2. "<Expansion> (<АББР>)" where <АББР> is uppercase initials matching words in <Expansion>
        for match in PAREN_ABBR_AFTER_EXPANSION_RE.finditer(text):
            raw_exp, cand_abbr = match.group(1), match.group(2).strip()
            clean_exp = self._clean_candidate_expansion(raw_exp)
            if not self._is_valid_definition_pair(cand_abbr, clean_exp):
                continue
            # Verify acronym plausibility: at least 2 words in clean_exp start with letters in cand_abbr
            exp_words = [w for w in re.findall(r"[А-ЯЁа-яё]+", clean_exp) if len(w) > 2]
            if len(exp_words) >= len(cand_abbr):
                tail_words = exp_words[-len(cand_abbr) :]
                initials = "".join(w[0].upper().replace("Ё", "Е") for w in tail_words)
                if initials == cand_abbr.upper().replace("Ё", "Е"):
                     phrase = " ".join(tail_words)
                     if self._is_valid_definition_pair(cand_abbr, phrase):
                         norm_upper = cand_abbr.upper().replace("Ё", "Е")
                         if norm_upper not in self.abbr_upper_map:
                             self.abbreviations[cand_abbr] = phrase
                         self.learned_definitions[cand_abbr] = self.abbreviations.get(
                             self.abbr_upper_map.get(norm_upper, cand_abbr), phrase
                         )
                         newly_learned[cand_abbr] = self.learned_definitions[cand_abbr]

        # 3. "<АББР> (<Expansion>)" e.g. "ЭЦП (электронно-цифровой подписью)"
        for match in ABBR_BEFORE_PAREN_EXPANSION_RE.finditer(text):
            cand_abbr, raw_exp = match.group(1).strip(), match.group(2).strip()
            if self._is_valid_definition_pair(cand_abbr, raw_exp):
                norm_upper = cand_abbr.upper().replace("Ё", "Е")
                if norm_upper not in self.abbr_upper_map:
                    self.abbreviations[cand_abbr] = raw_exp
                self.learned_definitions[cand_abbr] = self.abbreviations.get(
                    self.abbr_upper_map.get(norm_upper, cand_abbr), raw_exp
                )
                newly_learned[cand_abbr] = self.learned_definitions[cand_abbr]

        if newly_learned:
            self._rebuild_indexes()

        return newly_learned

    auto_learn_definitions = learn_from_text

    # -----------------------------------------------------------------------
    # Chunk Tagging: Terms, Responsible Roles (12 patterns), Topics
    # -----------------------------------------------------------------------
    def extract_terms(self, text: str, auto_learn: bool = True) -> list[str]:
        """Extract domain abbreviations and expansions from text as ['АББР (расшифровка)', ...]."""
        if not text:
            return []

        if auto_learn:
            self.learn_from_text(text)

        matched_abbrs: list[str] = []
        seen: set[str] = set()

        # 1. Check direct abbreviation occurrences via word-boundary regexes
        for abbr, pattern in self._abbr_regex_map.items():
            if pattern.search(text):
                if abbr not in seen:
                    seen.add(abbr)
                    matched_abbrs.append(abbr)

        # 2. Check full expansions & Russian inflected stems in text
        norm_text = _normalize_lookup_key(text)
        for stem, abbr in EXPANSION_STEM_MAP.items():
            if stem in norm_text and abbr in self.abbreviations and abbr not in seen:
                seen.add(abbr)
                matched_abbrs.append(abbr)

        for exp_norm, abbr in self.expansion_to_abbr.items():
            if len(exp_norm) >= 8 and exp_norm in norm_text and abbr not in seen:
                seen.add(abbr)
                matched_abbrs.append(abbr)

        return [f"{abbr} ({self.abbreviations[abbr]})" for abbr in matched_abbrs if abbr in self.abbreviations]

    def extract_roles(self, text: str) -> list[str]:
        """Extract responsible roles from text using all 12 ROLE_PATTERNS."""
        if not text:
            return []
        roles_found: list[str] = []
        for role_name, pattern in self.role_patterns.items():
            if pattern.search(text):
                roles_found.append(role_name)
        return roles_found

    def extract_topics(self, text: str) -> list[str]:
        """Classify chunk text into MIPT normative topics based on keyword matching."""
        if not text:
            return []
        norm_text = text.lower().replace("ё", "е")
        topics_found: list[str] = []
        for topic_name, keywords in self.topic_keywords.items():
            if any(kw in norm_text for kw in keywords):
                topics_found.append(topic_name)
        return topics_found

    # -----------------------------------------------------------------------
    # Search Query Expansion (Bidirectional + Colloquial Synonyms)
    # -----------------------------------------------------------------------
    def expand_query(self, query: str) -> str:
        """Expand a search query with bidirectional glossary lookups and colloquial MIPT synonyms."""
        if not query or not query.strip():
            return ""

        base_query = query.strip()
        norm_query = _normalize_lookup_key(base_query)
        additions: list[str] = []
        seen_norm: set[str] = {norm_query}

        def _add_term(candidate: str) -> None:
            c = candidate.strip()
            if not c:
                return
            c_norm = _normalize_lookup_key(c)
            if c_norm and c_norm not in norm_query and c_norm not in seen_norm:
                seen_norm.add(c_norm)
                additions.append(c)

        # 1. Abbreviation -> Full expansion
        for abbr, pattern in self._abbr_regex_map.items():
            if pattern.search(base_query):
                _add_term(self.abbreviations[abbr])

        # 2. Full expansion / stem -> Abbreviation (and canonical expansion)
        for stem, abbr in EXPANSION_STEM_MAP.items():
            if stem in norm_query:
                _add_term(abbr)
                if abbr in self.abbreviations:
                    _add_term(self.abbreviations[abbr])

        for exp_norm, abbr in self.expansion_to_abbr.items():
            if len(exp_norm) >= 6 and exp_norm in norm_query:
                _add_term(abbr)

        # 3. Colloquial query synonyms from glossary.yaml
        for trigger_key, syn_list in self.query_synonyms.items():
            # Match either exact key or stem prefix (for words >= 5 chars)
            stem_key = trigger_key[:-1] if len(trigger_key) >= 6 else trigger_key
            if trigger_key in norm_query or (len(stem_key) >= 5 and stem_key in norm_query):
                for syn in syn_list:
                    _add_term(syn)

        if not additions:
            return base_query
        return f"{base_query} ({', '.join(additions)})"

    # -----------------------------------------------------------------------
    # Document Requisites & Edition Graph Relation Extractor
    # -----------------------------------------------------------------------
    extract_edition_metadata = staticmethod(extract_edition_metadata)
    compute_version_family = staticmethod(compute_version_family)
    extract_target_plan_years = staticmethod(extract_target_plan_years)
    edition_sort_tuple = staticmethod(edition_sort_tuple)

    def extract_relations_from_text(self, text: str, self_doc_number: str = "") -> dict[str, list[str]]:
        """Extract automatic 'supersedes', 'amends', and unnumbered repealed-act subjects from normative text."""
        if not text:
            return {"supersedes": [], "amends": [], "repealed_subjects": []}

        # Inspect only normative part before ЛИСТ СОГЛАСОВАНИЯ or first 15,000 chars
        head_text = text[:15000]
        supersedes: list[str] = []
        amends: list[str] = []
        repealed_subjects: list[str] = []

        for m in SUPERSEDES_PHRASE_RE.finditer(head_text):
            num = m.group(1).strip()
            if num and num != self_doc_number and num not in supersedes:
                supersedes.append(num)

        for m_un in UNNUMBERED_SUPERSEDES_PHRASE_RE.finditer(head_text):
            raw_clause = m_un.group(1).strip()
            norm_subj = _normalize_subject_for_version_family(raw_clause)
            if norm_subj and norm_subj not in repealed_subjects:
                repealed_subjects.append(norm_subj)

        for m in AMENDS_PHRASE_RE.finditer(head_text):
            num = m.group(1).strip()
            # If the document actually cancels/supersedes this order (like 794-1 canceling 1469-1),
            # keep it in supersedes rather than amends.
            if num and num != self_doc_number and num not in supersedes and num not in amends:
                amends.append(num)

        return {"supersedes": supersedes, "amends": amends, "repealed_subjects": repealed_subjects}

    def extract_document_meta(
        self,
        blocks_or_text: list[ParsedBlock] | str,
        file_name: str = "",
        file_path: str = "",
        sha256: str = "",
        overrides: dict[str, Any] | None = None,
    ) -> DocumentMeta:
        """Extract DocumentMeta (number, ISO date, doc_type, title, relations) and merge with registry_overrides.yaml."""
        if isinstance(blocks_or_text, list):
            full_text = "\n".join(b.text for b in blocks_or_text if b.text)
            pages_total = max((b.page_num for b in blocks_or_text if b.page_num), default=1) if blocks_or_text else 0
            ocr_pages = len({b.page_num for b in blocks_or_text if b.is_ocr})
        else:
            full_text = str(blocks_or_text or "")
            pages_total = 0
            ocr_pages = 0

        clean_fname = Path(file_name).name if file_name else (Path(file_path).name if file_path else "")
        norm_fname = _normalize_filename_for_requisites(clean_fname)
        header_zone = _extract_strict_header_zone(blocks_or_text, max_chars=1200)

        order_number = ""
        order_date = ""
        doc_type = "Документ"

        # 1. Check 1x2 header table requisites strictly inside header_zone ("от 28.03.2023 | № 794-1")
        m_tbl = HEADER_TABLE_REQUISITE_RE.search(header_zone)
        if m_tbl:
            raw_date = m_tbl.group(1) or m_tbl.group(4) or ""
            raw_num = m_tbl.group(2) or m_tbl.group(3) or ""
            cand_num = _clean_valid_doc_number(raw_num)
            cand_date = _to_iso_date(raw_date)
            if cand_num:
                order_number = cand_num
                order_date = cand_date

        # 2. Check standalone ORDER_HEADER_RE / ORDER_HEADER_REV_RE strictly inside header_zone
        if not order_number or not order_date:
            m_ord = ORDER_HEADER_RE.search(header_zone)
            if m_ord:
                cand_num = _clean_valid_doc_number(m_ord.group(5))
                cand_date = _parse_flex_date_groups(
                    m_ord.group(1), m_ord.group(2), m_ord.group(3), m_ord.group(4)
                )
                if cand_num:
                    order_number = order_number or cand_num
                    order_date = order_date or cand_date

        if not order_number or not order_date:
            m_rev = ORDER_HEADER_REV_RE.search(header_zone)
            if m_rev:
                cand_num = _clean_valid_doc_number(m_rev.group(1))
                cand_date = _parse_flex_date_groups(
                    m_rev.group(2), m_rev.group(3), m_rev.group(4), m_rev.group(5)
                )
                if cand_num:
                    order_number = order_number or cand_num
                    order_date = order_date or cand_date

        # 3. Check top-of-document approval stamp ("УТВЕРЖДЕН приказом ... от DD.MM.YYYY № ...") in header_zone
        if not order_number or not order_date:
            m_app = HEADER_APPROVED_BY_RE.search(header_zone)
            if m_app:
                cand_num = _clean_valid_doc_number(m_app.group(5))
                cand_date = _parse_flex_date_groups(
                    m_app.group(1), m_app.group(2), m_app.group(3), m_app.group(4)
                )
                if cand_num:
                    order_number = order_number or cand_num
                    order_date = order_date or cand_date

        if not order_number or not order_date:
            m_app_rev = HEADER_APPROVED_BY_REV_RE.search(header_zone)
            if m_app_rev:
                cand_num = _clean_valid_doc_number(m_app_rev.group(1))
                cand_date = _parse_flex_date_groups(
                    m_app_rev.group(2), m_app_rev.group(3), m_app_rev.group(4), m_app_rev.group(5)
                )
                if cand_num:
                    order_number = order_number or cand_num
                    order_date = order_date or cand_date

        # 4. Check normalized filename requisites ("Приказ_№_2990_1_от_19_06_2026_..." -> "2990-1", "2026-06-19")
        if norm_fname:
            m_fn = FILENAME_REQUISITES_RE.search(norm_fname)
            if m_fn:
                if m_fn.group(1) and doc_type == "Документ":
                    doc_type = m_fn.group(1).capitalize()
                cand_num = _clean_valid_doc_number(m_fn.group(2))
                cand_date = _to_iso_date(m_fn.group(3) or "")
                if cand_num:
                    order_number = order_number or cand_num
                    order_date = order_date or cand_date
            elif not order_number:
                m_fn_rev = FILENAME_REQUISITES_REV_RE.search(norm_fname)
                if m_fn_rev:
                    if m_fn_rev.group(1) and doc_type == "Документ":
                        doc_type = m_fn_rev.group(1).capitalize()
                    cand_num = _clean_valid_doc_number(m_fn_rev.group(3))
                    cand_date = _to_iso_date(m_fn_rev.group(2) or "")
                    if cand_num:
                        order_number = order_number or cand_num
                        order_date = order_date or cand_date

        # 5. Extract official ЭДО / registration stamp anywhere in the document
        edo_number, edo_date = _extract_edo_stamp(blocks_or_text, full_text)

        # Resolve primary doc_number / doc_date while preserving both order requisites & ЭДО stamp
        is_edo_stamp = False
        if order_number:
            doc_number = order_number
            doc_date = order_date
        elif edo_number:
            doc_number = edo_number
            doc_date = edo_date
            is_edo_stamp = True
        else:
            doc_number = ""
            doc_date = ""

        # 6. Classify doc_type from normalized filename + strict header zone
        upper_head = (norm_fname + "\n" + header_zone).upper()
        if doc_type == "Документ":
            if "РАСПОРЯЖЕНИЕ" in upper_head:
                doc_type = "Распоряжение"
            elif re.search(r"(?m)^\s*ПРИКАЗ\b", upper_head) or norm_fname.upper().startswith("ПРИКАЗ"):
                doc_type = "Приказ"
            elif "РЕГЛАМЕНТ" in upper_head:
                doc_type = "Регламент"
            elif "ПОЛОЖЕНИЕ" in upper_head:
                doc_type = "Положение"
            elif "ИНСТРУКЦИЯ" in upper_head:
                doc_type = "Инструкция"
            elif "СМЕТА" in upper_head or "БЛАНК СМЕТЫ" in upper_head:
                doc_type = "Смета"
            elif "АНКЕТА" in upper_head:
                doc_type = "Анкета"
            elif "СОГЛАСИЕ" in upper_head:
                doc_type = "Согласие"
            elif "ПРИЛОЖЕНИЕ" in upper_head:
                doc_type = "Приложение"

        # 7. Auto-extract relations from text
        auto_rels = self.extract_relations_from_text(full_text, self_doc_number=doc_number)
        supersedes = list(auto_rels["supersedes"])
        amends = list(auto_rels["amends"])

        # Extract applicable years strictly from normalized filename + header_zone + resolved doc_date/edo_date
        year_probe = f"{norm_fname}\n{header_zone}\n{doc_date[:4] if doc_date else ''}\n{edo_date[:4] if edo_date else ''}"
        found_years = _extract_applicable_years(year_probe)

        # 8. Merge with config/registry_overrides.yaml (by filename first, then by doc_number)
        yaml_override: dict[str, Any] = {}
        if clean_fname and clean_fname in self.overrides_documents:
            yaml_override = dict(self.overrides_documents[clean_fname])
        elif doc_number and doc_number in self.overrides_by_number:
            by_num_entry = self.overrides_by_number[doc_number]
            mapped_fname = by_num_entry.get("file_name", "")
            if mapped_fname and mapped_fname in self.overrides_documents:
                yaml_override = dict(self.overrides_documents[mapped_fname])
            else:
                yaml_override = dict(by_num_entry)

        # Combine with caller-supplied runtime overrides
        merged_ov: dict[str, Any] = dict(yaml_override)
        if overrides:
            for k, v in overrides.items():
                if v is not None and v != "" and v != []:
                    merged_ov[k] = v

        ov_num = _clean_valid_doc_number(str(merged_ov.get("doc_number") or ""))
        final_doc_number = ov_num or doc_number or ""
        if ov_num and ov_num != edo_number:
            is_edo_stamp = False

        raw_ov_date = str(merged_ov.get("doc_date") or "").strip()
        final_doc_date = (_to_iso_date(raw_ov_date) if raw_ov_date else "") or doc_date or ""

        final_doc_id = str(
            merged_ov.get("doc_id")
            or _slugify_doc_id(clean_fname or "document", final_doc_number, final_doc_date)
        )
        final_doc_type = str(merged_ov.get("doc_type") or doc_type or "Документ")
        default_clean_title = norm_fname or (clean_fname.rsplit(".", 1)[0] if clean_fname else "")
        final_title = str(
            merged_ov.get("title")
            or default_clean_title
            or (f"{final_doc_type} № {final_doc_number}".strip() if final_doc_number else final_doc_type)
        )
        final_status = str(merged_ov.get("status") or DocumentStatus.ACTIVE.value)

        ov_supersedes = merged_ov.get("supersedes", [])
        if isinstance(ov_supersedes, str):
            ov_supersedes = [x.strip() for x in ov_supersedes.split(",") if x.strip()]
        final_supersedes = list(dict.fromkeys((ov_supersedes or []) + supersedes))

        ov_superseded_by = merged_ov.get("superseded_by", [])
        if isinstance(ov_superseded_by, str):
            ov_superseded_by = [x.strip() for x in ov_superseded_by.split(",") if x.strip()]
        final_superseded_by = list(dict.fromkeys(ov_superseded_by or []))

        ov_amends = merged_ov.get("amends", [])
        if isinstance(ov_amends, str):
            ov_amends = [x.strip() for x in ov_amends.split(",") if x.strip()]
        # Do not keep an order in amends if it's in final_supersedes
        final_amends = [
            x for x in dict.fromkeys((ov_amends or []) + amends) if x not in final_supersedes
        ]

        ov_amended_by = merged_ov.get("amended_by", [])
        if isinstance(ov_amended_by, str):
            ov_amended_by = [x.strip() for x in ov_amended_by.split(",") if x.strip()]
        final_amended_by = list(dict.fromkeys(ov_amended_by or []))

        final_years = merged_ov.get("applicable_years") or found_years
        final_version_note = str(merged_ov.get("version_note") or "")
        final_quality_warning = str(merged_ov.get("quality_warning") or "")
        final_also_in_files = list(merged_ov.get("also_in_files") or [])

        # Extract K2 edition/revision metadata & canonical version_family
        edition_info = extract_edition_metadata(
            title=final_title,
            header_text=header_zone,
            full_text=full_text,
            file_name=clean_fname,
        )
        raw_ov_ed_date = str(merged_ov.get("edition_date") or "").strip()
        final_edition_date = (
            (_to_iso_date(raw_ov_ed_date) if raw_ov_ed_date else "")
            or str(edition_info.get("edition_date") or "")
        )
        final_edition_number = (
            merged_ov.get("edition_number")
            if merged_ov.get("edition_number") not in (None, "")
            else edition_info.get("edition_number")
        )
        final_version_family = str(
            merged_ov.get("version_family")
            or compute_version_family(
                doc_type=final_doc_type,
                title=final_title,
                header_text=header_zone,
                full_text=full_text,
                file_name=clean_fname,
            )
            or ""
        )
        target_plan_years = extract_target_plan_years(
            title=final_title,
            header_text=_extract_formal_header_subject(full_text) or header_zone,
        )
        repealed_subjects = list(auto_rels.get("repealed_subjects") or [])

        # Auto-populate version_note for newly uploaded documents with auto-detected relations
        if not final_version_note:
            notes: list[str] = []
            if final_supersedes:
                notes.append(f"Отменяет документ(ы): № {', '.join(final_supersedes)}.")
            if final_superseded_by:
                notes.append(f"Отменен документом: № {', '.join(final_superseded_by)}.")
            if final_amends:
                notes.append(f"Вносит изменения в документ(ы): № {', '.join(final_amends)}.")
            if final_amended_by:
                notes.append(f"Действует с учетом изменений, внесенных: № {', '.join(final_amended_by)}.")
            final_version_note = " ".join(notes)

        doc_metadata: dict[str, Any] = dict(merged_ov.get("metadata") or {})
        doc_metadata["edo_number"] = edo_number
        doc_metadata["edo_date"] = edo_date
        doc_metadata["is_edo_stamp"] = is_edo_stamp
        if order_number:
            doc_metadata["order_number"] = order_number
            doc_metadata["order_date"] = order_date
        doc_metadata["edition_date"] = final_edition_date
        doc_metadata["edition_number"] = final_edition_number
        doc_metadata["version_family"] = final_version_family
        if target_plan_years:
            doc_metadata["target_plan_years"] = target_plan_years
        if repealed_subjects:
            doc_metadata["repealed_subjects"] = repealed_subjects

        doc_meta_obj = DocumentMeta(
            doc_id=final_doc_id,
            file_name=clean_fname,
            file_path=str(file_path or ""),
            sha256=str(sha256 or ""),
            doc_number=final_doc_number,
            doc_date=final_doc_date,
            edition_date=final_edition_date,
            edition_number=final_edition_number,
            version_family=final_version_family,
            doc_type=final_doc_type,
            title=final_title,
            status=final_status,
            applicable_years=final_years,
            supersedes=final_supersedes,
            superseded_by=final_superseded_by,
            amends=final_amends,
            amended_by=final_amended_by,
            version_note=final_version_note,
            quality_warning=final_quality_warning,
            pages_total=pages_total,
            ocr_pages=ocr_pages,
            also_in_files=final_also_in_files,
            metadata=doc_metadata,
        )
        return doc_meta_obj

    def build_u00_meta_chunk(self, doc_meta: DocumentMeta) -> ChunkRecord | None:
        """Build synthetic u00_meta passport ChunkRecord (zone='meta', in NON_ANSWER_ZONES) when alerts/relations exist."""
        edo_number = str((doc_meta.metadata or {}).get("edo_number") or getattr(doc_meta, "edo_number", "") or "").strip()
        edo_date = str((doc_meta.metadata or {}).get("edo_date") or getattr(doc_meta, "edo_date", "") or "").strip()
        is_edo_stamp = bool((doc_meta.metadata or {}).get("is_edo_stamp", getattr(doc_meta, "is_edo_stamp", False)))
        edition_date = str((doc_meta.metadata or {}).get("edition_date") or getattr(doc_meta, "edition_date", "") or "").strip()
        edition_number = (doc_meta.metadata or {}).get("edition_number")
        if edition_number is None:
            edition_number = getattr(doc_meta, "edition_number", None)
        version_family = str((doc_meta.metadata or {}).get("version_family") or getattr(doc_meta, "version_family", "") or "").strip()

        has_meta_Trigger = any(
            [
                doc_meta.version_note,
                doc_meta.quality_warning,
                doc_meta.superseded_by,
                doc_meta.amended_by,
                doc_meta.supersedes,
                doc_meta.amends,
                edition_date,
                edition_number not in (None, ""),
                doc_meta.status in {
                    DocumentStatus.SUPERSEDED.value,
                    DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value,
                },
            ]
        )
        if not has_meta_Trigger:
            return None

        status_ru = STATUS_RU_MAP.get(doc_meta.status, doc_meta.status)
        lines: list[str] = [
            f"ПАСПОРТ РЕДАКЦИИ ДОКУМЕНТА: {doc_meta.title}",
            f"Реквизиты: {doc_meta.doc_type} № {doc_meta.doc_number or 'б/н'} от {doc_meta.doc_date or 'б/д'} | Статус: {status_ru}",
        ]
        if edition_date or edition_number not in (None, ""):
            ed_parts: list[str] = []
            if edition_number not in (None, ""):
                ed_parts.append(f"№ {edition_number}" if str(edition_number) != "новая" else "новая редакция")
            if edition_date:
                ed_parts.append(f"от {edition_date}")
            lines.append(f"Сведения о редакции: {' '.join(ed_parts)}")
        if edo_number:
            lines.append(f"Регистрация ЭДО: Зарег. в ЭДО {edo_date or 'б/д'} № {edo_number}")
        if doc_meta.applicable_years:
            lines.append(f"Применимые годы: {', '.join(str(y) for y in doc_meta.applicable_years)}")
        if doc_meta.supersedes:
            lines.append(f"Отменяет акты: {', '.join(doc_meta.supersedes)}")
        if doc_meta.superseded_by:
            lines.append(f"Отменен / заменен актами: {', '.join(doc_meta.superseded_by)}")
        if doc_meta.amends:
            lines.append(f"Частично изменяет акты: {', '.join(doc_meta.amends)}")
        if doc_meta.amended_by:
            lines.append(f"Изменен актами (действует с поправками): {', '.join(doc_meta.amended_by)}")
        if doc_meta.version_note:
            lines.append(f"Примечание к редакции (version_note): {doc_meta.version_note}")
        if doc_meta.quality_warning:
            lines.append(f"Предупреждение о качестве исходника (quality_warning): {doc_meta.quality_warning}")
        if doc_meta.also_in_files:
            lines.append(f"Дублируется в файлах: {', '.join(doc_meta.also_in_files)}")

        clean_text = "\n".join(lines).strip()
        terms = self.extract_terms(clean_text, auto_learn=False)
        roles = self.extract_roles(clean_text)
        topics = self.extract_topics(clean_text)

        chunk_id = f"{doc_meta.doc_id}::u00_meta::c000"
        years_str = ", ".join(str(y) for y in doc_meta.applicable_years) if doc_meta.applicable_years else "не ограничено"
        roles_str = ", ".join(roles) if roles else "общие"
        topics_str = ", ".join(topics) if topics else "метаданные и редакции"
        terms_str = "; ".join(terms) if terms else "нет"
        num_str = doc_meta.doc_number or "б/н"
        date_str = doc_meta.doc_date or "б/д"
        edo_hdr_part = f"Зарег. в ЭДО {edo_date or 'б/д'} № {edo_number} | " if edo_number else ""

        embed_text = (
            f"[Документ: {doc_meta.title} (№ {num_str} от {date_str}) | "
            f"{edo_hdr_part}"
            f"Редакция/Статус: {status_ru} | Годы: {years_str} | "
            f"Часть: Паспорт редакции и метаданные | Раздел 0: Паспорт редакции | "
            f"Пункт: META | Роли: {roles_str} | Темы: {topics_str}]\n"
            f"[Термины: {terms_str}]\n"
            f"{clean_text}"
        )

        return ChunkRecord(
            chunk_id=chunk_id,
            doc_id=doc_meta.doc_id,
            unit_id="u00_meta",
            unit_title="Паспорт редакции и метаданные",
            section_id=f"{doc_meta.doc_id}::u00_meta::s00",
            section_num="0",
            section_title="Паспорт редакции",
            clause_no="META",
            zone=ZoneType.META.value,
            block_type=BlockType.META.value,
            clean_text=clean_text,
            raw_text=clean_text,
            embed_text=embed_text,
            prev_id=None,
            next_id=None,
            roles=roles,
            topics=topics,
            terms=terms,
            also_in_files=list(doc_meta.also_in_files),
            page_numbers=[1],
            metadata={
                "file_name": doc_meta.file_name,
                "doc_number": doc_meta.doc_number,
                "doc_date": doc_meta.doc_date,
                "edition_date": edition_date,
                "edition_number": edition_number,
                "version_family": version_family,
                "edo_number": edo_number,
                "edo_date": edo_date,
                "is_edo_stamp": is_edo_stamp,
                "doc_type": doc_meta.doc_type,
                "title": doc_meta.title,
                "status": doc_meta.status,
                "applicable_years": list(doc_meta.applicable_years),
                "supersedes": list(doc_meta.supersedes),
                "superseded_by": list(doc_meta.superseded_by),
                "amends": list(doc_meta.amends),
                "amended_by": list(doc_meta.amended_by),
                "version_note": doc_meta.version_note,
                "quality_warning": doc_meta.quality_warning,
                "is_non_answer_zone": True,
            },
        )


@lru_cache(maxsize=1)
def get_glossary_manager() -> GlossaryManager:
    """Return a cached singleton GlossaryManager instance."""
    return GlossaryManager()
