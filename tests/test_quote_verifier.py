"""Tests for verbatim quote extraction / verification (src/retrieval/quote_verifier.py)."""

from src.retrieval.quote_verifier import ContextDoc, extract_quotes, normalize_tokens, verify_quotes

CONTEXT = ContextDoc(
    doc_id="doc1",
    title="Положение о стипендиальном обеспечении",
    doc_number="123-1",
    texts=[
        "4.2. Государственная академическая стипендия назначается обучающимся, "
        "имеющим оценки «отлично» и «хорошо», по-\nсле прохождения промежуточной аттестации. "
        "Размер стипендии устанавливается приказом ректора."
    ],
)

ANSWER = """Стипендия назначается по итогам сессии.

> «Государственная академическая стипендия назначается обучающимся, имеющим оценки «отлично» и «хорошо», после прохождения промежуточной аттестации»
> — Положение о стипендиальном обеспечении, № 123-1, п. 4.2, стр. 5

Размер определяется отдельно:

> «Размер стипендии устанавливается приказом проректора»
> — Положение о стипендиальном обеспечении, № 123-1

> «Студенты получают бесплатное питание в столовой»
> — Положение о стипендиальном обеспечении

> Обычная цитата без кавычек — не документ.
"""


def test_extract_quotes_parses_text_and_source():
    qs = extract_quotes(ANSWER)
    assert len(qs) == 3
    q = qs[0]
    assert q.text.startswith("Государственная академическая")
    assert q.text.endswith("аттестации")
    assert q.doc_number == "123-1"
    assert q.clause == "4.2"
    assert q.page == "5"
    assert q.source_title == "Положение о стипендиальном обеспечении, № 123-1"


def test_normalization_handles_hyphenation_yo_and_quotes():
    assert normalize_tokens("по-\nсле «Ёлка»") == ["после", "елка"]


def test_verify_quotes_exact_fuzzy_and_invented():
    res = verify_quotes(ANSWER, [CONTEXT])
    assert [r["verified"] for r in res] == [True, False, False]
    assert res[0]["matched_doc_id"] == "doc1"
    assert res[1]["score"] < 0.9  # one substituted word out of 5 tokens


def test_verify_quotes_with_ellipsis_and_small_typo():
    answer = (
        "> «Государственная академическая стипендия назначается … после прохождения промежуточной аттестации»\n"
        "> — Положение, п. 4.2\n\n"
        "> «Государственная академическая стипендия назначается обучающимся имеющим оценки отлично и хорошо "
        "после прохождения промежуточной аттестации размер стипендии устанавливается приказом ректора университета»\n"
        "> — Положение"
    )
    res = verify_quotes(answer, [CONTEXT])
    assert res[0]["verified"] is True
    assert res[1]["verified"] is True  # 1 extra token out of ~22 → fuzzy ≥ 0.9


def test_no_quotes_returns_empty():
    assert verify_quotes("Просто ответ без цитат.", [CONTEXT]) == []


# ---------------------------------------------------------------------------------------------
# Enrichment (full clause text + chunk metadata) and placeholder stripping
# ---------------------------------------------------------------------------------------------
from src.retrieval.quote_verifier import (  # noqa: E402
    MAX_CLAUSE_CHARS,
    clean_clause,
    clean_doc_number,
    context_docs_from_search,
    extract_quotes as _extract,
    strip_placeholders,
)
from src.schemas import ParentSectionRecord, SearchHit, SearchResponse  # noqa: E402

PARENT = (
    "[Стр. 1]\nПриложение к приказу № [___]\n**Регламент пользования залом**\n---\n**1. Общие положения**\n"
    "1.1. Настоящий регламент является локальным нормативным актом, устанавливающим порядок бронирования.\n"
    "1.2. Зал находится в управлении отдела.\n**2. Порядок бронирования**\n"
    "2.1. Для бронирования зала необходимо:\n- уточнить у начальника отдела свободные даты;\n"
    "- согласовать с начальником отдела наличие техни-\nческих сотрудников;\n"
    "2.1.1. Служебная записка подаётся не позднее чем за 10 рабочих дней.\n"
    "[Стр. 2]\n2.2. Бронирование подтверждается письмом отдела.\n3. Заключительные положения\n"
    "3.1. Регламент вступает в силу с даты подписания."
)


def _search_resp() -> SearchResponse:
    ps = ParentSectionRecord(section_id="s1", doc_id="reg_hall", clean_text=PARENT, page_numbers=[1, 2])
    hit = SearchHit(
        chunk_id="c1", doc_id="reg_hall", title="Регламент пользования залом", doc_number="123-4",
        clause_no="2.1", page_numbers=[1], clean_text="2.1. Для бронирования зала необходимо:", parent_section=ps,
    )
    return SearchResponse(hits=[hit])


def test_placeholder_numbers_are_stripped():
    for bad in ("[___]", "___", "б/н", "Б/Н", "[ _ _ ]", "—", ""):
        assert clean_doc_number(bad) == ""
    assert clean_doc_number("3897-1") == "3897-1"
    assert clean_clause("п. б/н (ч. 1)") == ""
    assert clean_clause("п. 2.5") == "2.5"
    assert strip_placeholders("Регламент бронирования, № [___], п. 1.1, стр. 1") == "Регламент бронирования, п. 1.1, стр. 1"
    q = _extract("> «текст пункта для проверки»\n> — Регламент, № [___], п. 1.1, стр. 1")[0]
    assert q.doc_number == "" and "[___]" not in q.source_raw and q.clause == "1.1"


def test_enrichment_takes_full_clause_and_chunk_metadata():
    answer = (
        "> «уточнить у начальника отдела свободные даты»\n"
        "> — Регламент бронирования и пользования залом, № [___], п. 1.1, стр. 7\n"
    )
    res = verify_quotes(answer, context_docs_from_search(_search_resp()))
    q = res[0]
    assert q["verified"] is True
    assert q["doc_number"] == "123-4"  # from chunk metadata, not the placeholder
    assert q["source_title"] == "Регламент пользования залом"
    assert q["clause"] == "2.1"  # detected from the context, not the model's "1.1"
    assert q["page"] == "1"  # from the [Стр. N] marker, not the model's "7"
    ct = q["clause_text"]
    assert ct.startswith("2.1. Для бронирования зала необходимо:")
    assert "технических сотрудников" in ct  # hyphenated line break glued, list items kept
    assert "2.1.1." in ct  # sub-clauses belong to the clause
    assert "2.2." not in ct and "1.2." not in ct and "[Стр" not in ct
    (a, b), = q["clause_highlights"]
    assert ct[a:b] == "уточнить у начальника отдела свободные даты"
    assert "[___]" not in q["source_label"] and q["source_label"].startswith("Регламент пользования залом, № 123-4, п. 2.1")


def test_enrichment_subclause_page_marker_and_multi_clause_quote():
    docs = context_docs_from_search(_search_resp())
    q = verify_quotes("> «Бронирование подтверждается письмом отдела»\n> — Регламент", docs)[0]
    assert q["clause"] == "2.2" and q["page"] == "2" and q["clause_text"] == "2.2. Бронирование подтверждается письмом отдела."
    q = verify_quotes("> «Служебная записка подаётся не позднее чем за 10 рабочих дней»\n> — Регламент", docs)[0]
    assert q["clause"] == "2.1.1" and q["clause_text"].startswith("2.1.1.")


def test_enrichment_prose_paragraph_and_bounds():
    long_par = " ".join(f"Предложение номер {i} описывает правило работы зала." for i in range(80))
    text = "Общие сведения\nЗал расположен в корпусе.\n" + long_par
    doc = ContextDoc(doc_id="d", title="Тех. регламент", doc_number="б/н", texts=[text])
    q = verify_quotes("> «Предложение номер 40 описывает правило работы зала»\n> — Тех. регламент, № б/н", [doc])[0]
    assert q["verified"] and q["doc_number"] == "" and q["clause"] == ""
    assert len(q["clause_text"]) <= MAX_CLAUSE_CHARS + 4
    assert "номер 40 описывает" in q["clause_text"]
    assert q["clause_text"].startswith("… Предложение") and q["clause_text"].endswith(" …")


def test_unverified_quote_keeps_cleaned_model_source():
    q = verify_quotes("> «Такого текста в документе нет совсем»\n> — Регламент, № [___], п. 3", [CONTEXT])[0]
    assert q["verified"] is False and q["clause_text"] == "" and q["doc_number"] == ""
    assert q["source_label"] == "Регламент, п. 3"


def test_chunk_continuation_artifacts_do_not_break_matching():
    parent = (
        "65. Работник обязан представить отчёт.\nК Отчету прилагаются копии Решения о\nкомандировании и\n\n"
        "(Продолжение п. 65: 65. Работник обязан представить...)\nмаршрутный лист.\n66. Следующий пункт."
    )
    ps = ParentSectionRecord(section_id="s", doc_id="d", clean_text=parent)
    resp = SearchResponse(hits=[SearchHit(chunk_id="c", doc_id="d", title="Регламент", clean_text="x", parent_section=ps)])
    q = verify_quotes("> «К Отчету прилагаются копии Решения о командировании и маршрутный лист.»\n> — Регламент",
                      context_docs_from_search(resp))[0]
    assert q["verified"] and q["clause"] == "65" and "Продолжение" not in q["clause_text"]
