from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.config import (
    CHUNKS_DUMP_PATH,
    OCR_CACHE_DIR,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    SECTIONS_STORE_PATH,
)
from src.retrieval.hybrid_search import HybridRetriever


def test_chunks_and_sections_exist() -> None:
    assert CHUNKS_DUMP_PATH.exists(), "chunks_dump.json must be created by ingestion"
    assert SECTIONS_STORE_PATH.exists(), "sections_store.json must be created by ingestion"

    chunks = json.loads(CHUNKS_DUMP_PATH.read_text(encoding="utf-8"))
    sections = json.loads(SECTIONS_STORE_PATH.read_text(encoding="utf-8"))

    assert len(chunks) >= 600, f"Expected >= 600 chunks across 20 docs, got {len(chunks)}"
    assert len(sections) >= 40, f"Expected >= 40 parent sections, got {len(sections)}"

    # Check all files are represented (>= 20: новый документ не должен ломать тест)
    file_names = {c["file_name"] for c in chunks}
    assert len(file_names) >= 20, f"Expected all 20 files represented, got {len(file_names)}"

    # Check zero approval sheets leaked into chunks
    approval_chunks = [c for c in chunks if c["zone"] == "approval"]
    assert len(approval_chunks) == 0, "Approval sheets (ЛИСТ СОГЛАСОВАНИЯ) must be excluded"

    # Check minimum chunk size and bilateral overlap on non-meta chunks
    non_meta = [c for c in chunks if c["zone"] != "meta"]
    min_len = min(len(c["raw_text"]) for c in non_meta)
    assert min_len >= 160, f"All non-meta chunks must be >= 160 chars, got min={min_len}"

    # Interior chunks (having both prev_id and next_id) must contain both leading and trailing overlap
    interior = [c for c in non_meta if c.get("prev_id") and c.get("next_id")]
    assert len(interior) >= 400
    assert all(
        c["raw_text"].startswith("...") and c["raw_text"].endswith("...")
        for c in interior
    ), "Interior chunks must have bilateral overlap (...prev and next...)"

    # Check that form_smeta_2019 is parsed as form blocks, not CFO hierarchy
    smeta_2019 = [c for c in non_meta if c["doc_id"] == "form_smeta_2019"]
    assert smeta_2019 and all(c["zone"] == "form" for c in smeta_2019)


def test_duplicate_clauses_in_3065_preserved() -> None:
    chunks = json.loads(CHUNKS_DUMP_PATH.read_text(encoding="utf-8"))
    c_3065 = [c for c in chunks if c["doc_id"] == "order_3065_1_2021"]
    clauses_2_6 = [c for c in c_3065 if c["clause_no"].startswith("2.6")]
    clauses_2_7 = [c for c in c_3065 if c["clause_no"].startswith("2.7")]

    # Приказ 3065-1 has duplicate 2.6 and 2.7 in the regulation text
    assert len(clauses_2_6) >= 2, f"Both duplicate 2.6 clauses must be preserved, got {len(clauses_2_6)}"
    assert len(clauses_2_7) >= 2, f"Both duplicate 2.7 clauses must be preserved, got {len(clauses_2_7)}"
    assert len({c["chunk_id"] for c in clauses_2_6}) == len(clauses_2_6)


def test_footnote_glued_numbers_cleaned() -> None:
    chunks = json.loads(CHUNKS_DUMP_PATH.read_text(encoding="utf-8"))
    kom_chunks = [c for c in chunks if c["doc_id"] == "reg_komandirovanie_2025"]
    full_text = "\n".join(c["raw_text"] for c in kom_chunks)
    assert "8123" not in full_text, "Glued footnote '№ 8123' must be cleaned to '№ 812'"
    assert "812" in full_text


def test_ocr_scanned_docs_populated() -> None:
    chunks = json.loads(CHUNKS_DUMP_PATH.read_text(encoding="utf-8"))
    scanned_doc_ids = [
        "order_460_1_2017",
        "order_3897_1_2025",
        "order_1027_1_2019",
        "order_1292_1_2025",
        "reg_poselenie_2024",
    ]
    for did in scanned_doc_ids:
        doc_chunks = [c for c in chunks if c["doc_id"] == did and c["zone"] != "meta"]
        assert len(doc_chunks) >= 2, f"Scanned doc {did} must have OCR body chunks, got {len(doc_chunks)}"

    cached_pages = list(OCR_CACHE_DIR.glob("*.json"))
    assert len(cached_pages) >= 40, f"Expected >= 40 cached OCR pages, got {len(cached_pages)}"


def test_qdrant_collection_and_hybrid_search() -> None:
    client = QdrantClient(url=QDRANT_URL)
    assert client.collection_exists(QDRANT_COLLECTION_NAME)
    count_res = client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True)
    assert count_res.count >= 600

    retriever = HybridRetriever()

    # 1. Opening FLS (should retrieve order_3065_1_2021 and surface version alert about 460-1)
    r_fls = retriever.search("Каков порядок открытия финансового лицевого счета (ФЛС) в 1С:Документооборот?", top_k=6)
    assert not r_fls.abstain_recommended
    doc_ids = {h.doc_id for h in r_fls.hits}
    assert "order_3065_1_2021" in doc_ids
    assert len(r_fls.version_alerts) >= 1
    assert all(h.zone != "meta" for h in r_fls.hits), "meta-паспорт не должен попадать в ответ"

    # 2. Per diem (суточные) collision between 3699-1 and Регламент о командировании
    r_sut = retriever.search("Какой размер суточных при командировках по России для работников и студентов?", top_k=6)
    assert not r_sut.abstain_recommended
    sut_docs = {h.doc_id for h in r_sut.hits}
    assert "order_3699_1_2025" in sut_docs or "reg_komandirovanie_2025" in sut_docs
    assert len(r_sut.version_alerts) >= 1

    # 3. Housing regulation (Регламент поселения сотрудников) + missing clauses 2.1-2.5 quality alert
    r_pos = retriever.search("Какие правила подачи заявки на поселение сотрудников в общежитие?", top_k=6)
    assert not r_pos.abstain_recommended
    assert any(h.doc_id == "reg_poselenie_2024" for h in r_pos.hits)
    assert len(r_pos.quality_alerts) >= 1

    # 4. Out-of-domain questions -> should recommend abstain, even if BM25 находит общие слова
    for query in (
        "Как приготовить пиццу Маргарита в духовке и сколько грамм сыра моцарелла нужно?",
        "Как оформить загранпаспорт?",
        "Сколько стоит билет на поезд Москва-Казань?",
    ):
        r_ood = retriever.search(query, top_k=4)
        assert r_ood.abstain_recommended, (
            f"Expected abstain on out-of-domain query {query!r}, "
            f"got max_dense={r_ood.max_dense_score}"
        )
