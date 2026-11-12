from __future__ import annotations

from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parent.parent / "data" / "registry_overrides.yaml"


def load_docs() -> dict[str, dict]:
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return data["documents"]


def test_eb_document_is_electronic_budget_not_electrical_safety() -> None:
    docs = load_docs()
    eb = docs["Приказ и регламент по ЭБ.pdf"]
    assert "Электронный бюджет" in eb["doc_title"]
    assert "электробезопас" not in eb["doc_title"].lower()
    assert eb["doc_number"] == "1027-1"
    assert eb["doc_date"] == "2019-06-25"
    assert any("электронный бюджет" in tag.lower() for tag in eb["topic_tags"])


def test_tech_support_order_number_is_2993_1() -> None:
    docs = load_docs()
    tech = docs["Приказ и Регламент работы первой линии техподдержки.pdf"]
    assert tech["doc_id"] == "order_2993_1_2026"
    assert tech["doc_number"] == "2993-1"
    assert tech["doc_date"] == "2026-06-19"
    assert "2993-1" in tech["doc_title"]
    assert all("2429" not in str(v) for v in tech.values() if isinstance(v, str))


def test_hr_regulation_is_unsigned_draft() -> None:
    docs = load_docs()
    hr = docs["Регламент поиска и подбора персонала.pdf"]
    assert hr["doc_id"] == "reg_podbor_personala_2025"
    assert hr["status"] == "draft"
    assert hr["doc_number"].startswith("б/н")
    assert hr["doc_date"] == ""


def test_794_amends_2019_order_not_2022() -> None:
    docs = load_docs()
    order = docs["Приказ N 794-1 от 28.03.2023.docx"]
    assert "19.09.2019" in order["doc_title"]
    assert "order_1469_1_2019" in order["supersedes"]
    assert "order_1469_1_2022" not in order["supersedes"]


def test_460_title_matches_scan() -> None:
    docs = load_docs()
    order = docs["460-1 Открытие ФЛС_приказ об изменении в приказе на открытие ФЛС.pdf"]
    assert "О внесении изменений" in order["doc_title"]


def test_komandirovanie_is_draft_and_has_no_cycle() -> None:
    docs = load_docs()
    kom = docs["Регламент о командировании.pdf"]
    assert kom["status"] == "draft"
    assert kom["supersedes"] == []
    assert kom["superseded_by"] == []


def test_version_graph_has_no_cycles() -> None:
    docs = load_docs()
    ids = {doc["doc_id"] for doc in docs.values()}
    edges = {
        doc_id: {x for x in (doc.get("superseded_by") or []) if x in ids}
        for doc_id, doc in ((d["doc_id"], d) for d in docs.values())
    }

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise AssertionError(f"Цикл в графе версий через {node}")
        if node in visited:
            return
        visiting.add(node)
        for nxt in edges.get(node, ()):
            visit(nxt)
        visiting.remove(node)
        visited.add(node)

    for doc_id in ids:
        visit(doc_id)
