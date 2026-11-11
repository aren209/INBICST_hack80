from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import GLOSSARY_PATH

DEFAULT_MIPT_GLOSSARY: dict[str, str] = {
    "ФЛС": "финансовый лицевой счет",
    "КФО": "код финансового обеспечения",
    "ЦФО": "центр финансовой ответственности",
    "РИД": "результат интеллектуальной деятельности",
    "НИР": "научно-исследовательская работа",
    "ОКР": "опытно-конструкторская работа",
    "НИОКТР": "научно-исследовательские, опытно-конструкторские и технологические работы",
    "ЕГИСУ": "единая государственная информационная система учета НИОКТР",
    "ПФО": "планово-финансовый отдел",
    "ФЭУ": "финансово-экономическое управление",
    "УИТ": "управление информационных технологий",
    "ЦОД": "центр обработки данных",
    "ЭБ": "электробезопасность",
    "РНФ": "Российский научный фонд",
    "РЦНИ": "Российский центр научной информации (РФФИ)",
    "1С:ДО": "система электронного документооборота 1С:Документооборот",
    "СЭД": "система электронного документооборота 1С:Документооборот",
    "ФОТ": "фонд оплаты труда",
    "ТЗ": "техническое задание",
    "КП": "карточка проекта",
    "НМЦК": "начальная максимальная цена контракта",
    "ДГП": "договор гражданско-правового характера (ГПХ)",
    "ГПХ": "договор гражданско-правового характера",
}


class GlossaryManager:
    """Manages MIPT abbreviations and expands text for BM25 and search queries."""

    ABBR_DEF_PATTERN = re.compile(
        r"\b([А-ЯЁ]{2,8}(?::[А-ЯЁ]{2,4})?)\s*[—–-]\s*([а-яА-ЯёЁ0-9\s,()-]{6,100})"
    )

    def __init__(self, glossary_path: Path = GLOSSARY_PATH) -> None:
        self.glossary_path = glossary_path
        self.abbr_to_full: dict[str, str] = dict(DEFAULT_MIPT_GLOSSARY)
        if self.glossary_path.exists():
            try:
                loaded = json.loads(self.glossary_path.read_text(encoding="utf-8"))
                self.abbr_to_full.update(loaded)
            except Exception:
                pass

    def learn_from_text(self, text: str) -> None:
        for m in self.ABBR_DEF_PATTERN.finditer(text):
            abbr = m.group(1).strip().upper()
            definition = re.sub(r"\s+", " ", m.group(2)).strip(" .;,")
            if len(abbr) >= 2 and len(definition) >= 6 and abbr not in self.abbr_to_full:
                self.abbr_to_full[abbr] = definition

    def save(self) -> None:
        self.glossary_path.parent.mkdir(parents=True, exist_ok=True)
        self.glossary_path.write_text(
            json.dumps(self.abbr_to_full, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def expand_text(self, text: str) -> str:
        """Append bidirectional glossary expansions for abbreviations/terms found in text."""
        upper_text = text.upper()
        lower_text = text.lower()
        expansions: list[str] = []

        for abbr, full_term in self.abbr_to_full.items():
            if re.search(rf"(?<![А-ЯЁA-Z]){re.escape(abbr)}(?![А-ЯЁA-Z])", upper_text):
                if full_term.lower() not in lower_text:
                    expansions.append(f"{abbr} ({full_term})")
            elif full_term.lower() in lower_text:
                expansions.append(f"{abbr} ({full_term})")

        if not expansions:
            return text
        return f"{text}\n[Термины: {'; '.join(expansions)}]"
