"""Unit and integration tests for Item #1 (Duplicate Detection & Prevention) and Item #6 (Criterion K2 Multi-Version Actuality)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import docx
import pymupdf as fitz
import pytest
from qdrant_client import QdrantClient, models

from src.chunking.glossary import (
    compute_version_family,
    extract_edition_metadata,
    get_glossary_manager,
)
from src.config import DOCUMENT_REGISTRY_PATH, SECTIONS_STORE_PATH
from src.indexing.pipeline import IndexingPipeline, _normalize_text_for_dedup
from src.indexing.qdrant_store import QdrantHybridStore
from src.indexing.table_store import StructuredTableStore
from src.retrieval.query_pipeline import QueryPipeline
from src.retrieval.search_service import K2_SUPERSEDED_SCORE_PENALTY, SearchService
from src.schemas import (
    ChunkRecord,
    DocumentMeta,
    DocumentStatus,
    SearchHit,
)


# -----------------------------------------------------------------------------
# Helpers: Fast deterministic fake embedder + in-memory Qdrant store for tests
# -----------------------------------------------------------------------------


class _DeterministicEmbedder:
    """Fast local embedder for isolated pipeline/retrieval tests without external GPU/API calls."""

    class _Dense:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            vecs: list[list[float]] = []
            for t in texts:
                vec = [0.0] * 1024
                vec[0] = 0.85
                low = (t or "").lower()
                if "командиров" in low:
                    vec[1] = 0.50
                if "общежит" in low or "поселен" in low:
                    vec[2] = 0.50
                if "закуп" in low:
                    vec[3] = 0.50
                vecs.append(vec)
            return vecs

    class _Sparse:
        def encode_with_tokens(self, text: str) -> tuple[models.SparseVector, list[str]]:
            toks = [w for w in (text or "").lower().split() if len(w) >= 3][:20]
            indices = [abs(hash(w)) % 100_000 + 1 for w in dict.fromkeys(toks)]
            values = [1.0] * len(indices)
            return models.SparseVector(indices=indices, values=values), toks

    def __init__(self) -> None:
        self.dense = self._Dense()
        self.sparse = self._Sparse()

    def embed_chunks(self, texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        d_vecs = self.dense.embed_texts(texts)
        s_vecs = [self.sparse.encode_with_tokens(t)[0] for t in texts]
        return d_vecs, s_vecs

    def embed_texts(self, dense_texts: list[str], sparse_texts: list[str] | None = None) -> tuple[list[list[float]], list[models.SparseVector]]:
        s_in = sparse_texts if sparse_texts is not None else dense_texts
        d_vecs = self.dense.embed_texts(dense_texts)
        s_vecs = [self.sparse.encode_with_tokens(t)[0] for t in s_in]
        return d_vecs, s_vecs

    def embed_query(self, query_text: str, sparse_query_text: str | None = None) -> tuple[list[float], models.SparseVector, list[str]]:
        d_vec = self.dense.embed_texts([query_text])[0]
        s_vec, toks = self.sparse.encode_with_tokens(sparse_query_text or query_text)
        return d_vec, s_vec, toks


def _build_isolated_pipeline(tmp_path: Path, overrides_yaml: str = "version: '3.1'\ndocuments: {}\nby_doc_number: {}\n") -> IndexingPipeline:
    reg_path = tmp_path / "document_registry.json"
    sec_path = tmp_path / "sections_store.json"
    tbl_path = tmp_path / "tables_store.db"
    ovr_path = tmp_path / "registry_overrides.yaml"
    ovr_path.write_text(overrides_yaml, encoding="utf-8")

    qdrant = QdrantHybridStore(in_memory=True, collection_name="test_dedup_k2")
    qdrant.ensure_collection()
    table_store = StructuredTableStore(db_path=tbl_path)
    embedder = _DeterministicEmbedder()
    return IndexingPipeline(
        embedder=embedder,  # type: ignore[arg-type]
        qdrant_store=qdrant,
        table_store=table_store,
        registry_path=reg_path,
        sections_path=sec_path,
        overrides_path=ovr_path,
    )


def _write_docx(path: Path, paragraphs: list[str]) -> Path:
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))
    return path


def _write_pdf(path: Path, lines: list[str]) -> Path:
    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)
    text = "\n".join(lines)
    # Insert Unicode Russian text via built-in CJK/Unicode fallback font or htmlbox
    rect = fitz.Rect(40, 40, 555, 800)
    page.insert_htmlbox(rect, "<p style='font-family: sans-serif; font-size: 11pt;'>" + "<br/>".join(lines) + "</p>")
    pdf.save(str(path))
    pdf.close()
    return path


# -----------------------------------------------------------------------------
# 1. Live Database Cleanliness & Live K2 Verification
# -----------------------------------------------------------------------------


class TestLiveDatabaseDedupAndK2:
    def test_live_registry_has_no_duplicate_sha256_or_redundant_copy(self) -> None:
        reg_path = DOCUMENT_REGISTRY_PATH
        sec_path = SECTIONS_STORE_PATH
        assert reg_path.exists(), "Live document_registry.json must exist"
        assert sec_path.exists(), "Live sections_store.json must exist"

        registry: dict[str, dict[str, Any]] = json.loads(reg_path.read_text(encoding="utf-8"))
        sections: dict[str, dict[str, Any]] = json.loads(sec_path.read_text(encoding="utf-8"))

        # Redundant duplicate doc_2990_1_2026 must be removed; canonical doc_9380b426a8 retained
        assert "doc_2990_1_2026" not in registry
        assert "doc_9380b426a8" in registry

        # Zero duplicate SHA-256 across all registered documents
        seen_sha: dict[str, str] = {}
        for doc_id, meta in registry.items():
            sha = meta.get("sha256", "")
            assert sha, f"Document {doc_id} missing sha256"
            assert sha not in seen_sha, f"Duplicate sha256 between {doc_id} and {seen_sha[sha]}"
            seen_sha[sha] = doc_id

        # Zero orphan sections belonging to deleted duplicate
        for sec_id, sec in sections.items():
            assert sec.get("doc_id") in registry, f"Orphan section {sec_id} in sections_store.json"

        # Zero duplicate full normalized text across registered documents
        norm_texts_seen: dict[str, str] = {}
        by_doc_secs: dict[str, list[tuple[str, str]]] = {}
        for sec_id, sec in sections.items():
            by_doc_secs.setdefault(sec["doc_id"], []).append((sec_id, sec.get("clean_text", "")))
        for doc_id, items in by_doc_secs.items():
            items.sort(key=lambda kv: kv[0])
            norm_full = _normalize_text_for_dedup("\n".join(t for _, t in items))
            if len(norm_full) >= 200:
                assert norm_full not in norm_texts_seen, (
                    f"Exact normalized full-text duplicate in live DB between {doc_id} and {norm_texts_seen[norm_full]}"
                )
                norm_texts_seen[norm_full] = doc_id

    def test_live_k2_dorm_settlement_editions_reconciled_in_registry_and_qdrant(self) -> None:
        registry: dict[str, dict[str, Any]] = json.loads(DOCUMENT_REGISTRY_PATH.read_text(encoding="utf-8"))
        old_reg = registry["reg_staff_dorm_settlement"]
        new_ord = registry["doc_9380b426a8"]

        assert old_reg["status"] == DocumentStatus.SUPERSEDED.value
        assert any("2990-1" in ref or "doc_9380b426a8" in ref for ref in old_reg.get("superseded_by", []))
        assert old_reg.get("version_note", ""), "Superseded regulation must have Russian version_note"

        assert new_ord["status"] == DocumentStatus.ACTIVE.value
        assert "reg_staff_dorm_settlement" in new_ord.get("supersedes", []) or any(
            "регламент" in s.lower() or "reg_staff_dorm_settlement" in s for s in new_ord.get("supersedes", [])
        )

        # Verify Qdrant collection has zero points for removed duplicate doc_2990_1_2026 and updated status on reg_staff_dorm_settlement
        q_store = QdrantHybridStore()
        dup_chunks = q_store.get_document_chunks("doc_2990_1_2026")
        assert dup_chunks == []

        old_chunks = q_store.get_document_chunks("reg_staff_dorm_settlement")
        assert len(old_chunks) > 0
        assert all(c.status == DocumentStatus.SUPERSEDED.value for c in old_chunks)


# -----------------------------------------------------------------------------
# 2. Item #1 — Exact SHA-256 & Cross-Format Content Duplicate Rejection
# -----------------------------------------------------------------------------


class TestDuplicatePrevention:
    def test_exact_sha256_duplicate_rejected_before_indexing(self, tmp_path: Path) -> None:
        pipeline = _build_isolated_pipeline(tmp_path)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        paragraphs = [
            "Приказ № 412-1 от 14.04.2025 Об утверждении Порядка выдачи пропусков сотрудникам",
            "1. Общие положения",
            "1.1. Настоящий Порядок выдачи пропусков сотрудникам регулирует оформление электронных пропусков на территорию кампуса.",
            "1.2. Заявка на выдачу постоянного пропуска подается через личный кабинет работника за 3 рабочих дня.",
            "2. Сроки оформления",
            "2.1. Бюро пропусков изготавливает электронную карту в течение двух рабочих дней после согласования руководителем.",
        ]
        f1 = _write_docx(docs_dir / "Prikaz_412_1_propuski.docx", paragraphs)
        res1 = pipeline.index_file(f1, force_reindex=False)
        assert res1["status"] == "success"
        orig_id = res1["doc_id"]
        orig_qdrant_count = pipeline.qdrant.count_points()
        assert orig_qdrant_count > 0

        # Exact binary copy under a different filename, even with upload-style metadata_overrides
        f2 = docs_dir / "Копия Приказ 412-1 пропуски.docx"
        shutil.copyfile(f1, f2)
        res2 = pipeline.index_file(
            f2,
            metadata_overrides={"status": "active"},
            force_reindex=False,
        )

        # Dict & attribute compatibility checks
        assert res2.status == "duplicate"
        assert res2["status"] == "duplicate"
        assert res2["doc_id"] == orig_id
        assert res2["duplicate_of"] == orig_id
        assert res2["reason"] == "exact_sha256"
        assert res2["existing_file_name"] == "Prikaz_412_1_propuski.docx"
        assert "Документ уже присутствует в базе" in res2["message"]

        # Qdrant & registry were NOT polluted
        assert len(pipeline.registry) == 1
        assert pipeline.qdrant.count_points() == orig_qdrant_count

    def test_content_duplicate_cross_format_pdf_and_resaved_docx_rejected(self, tmp_path: Path) -> None:
        from src.indexing.pipeline import compute_file_sha256

        pipeline = _build_isolated_pipeline(tmp_path)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        clauses = [
            "Регламент бронирования переговорных комнат научно-образовательных подразделений",
            "1. Общие правила бронирования",
            "1.1. Бронирование переговорных комнат осуществляется ответственными сотрудниками лабораторий через единый календарь.",
            "1.2. Максимальная продолжительность одной брони составляет четыре астрономических часа подряд.",
            "2. Отмена и перенос заявки",
            "2.1. При отмене мероприятия заявитель обязан снять бронь не позднее чем за два часа до начала встречи.",
            "2.2. Повторное нарушение правил бронирования влечет приостановку доступа к системе резервирования на месяц.",
        ]
        f_docx = _write_docx(docs_dir / "reglament_peregovornyh.docx", clauses)
        res_orig = pipeline.index_file(f_docx, force_reindex=False)
        assert res_orig["status"] == "success"
        orig_id = res_orig["doc_id"]
        points_before = pipeline.qdrant.count_points()

        # Case 1: Re-saved .docx with different binary SHA-256 (extra whitespace & punctuation variations)
        resaved_clauses = [c.replace("  ", " ") + "   " for c in clauses]
        f_resaved = _write_docx(docs_dir / "reglament_peregovornyh_resaved.docx", resaved_clauses)
        assert compute_file_sha256(f_docx) != compute_file_sha256(f_resaved)

        res_dup_docx = pipeline.index_file(f_resaved, force_reindex=False)
        assert res_dup_docx["status"] == "duplicate"
        assert res_dup_docx["reason"] == "content_duplicate"
        assert res_dup_docx["duplicate_of"] == orig_id
        assert len(pipeline.registry) == 1
        assert pipeline.qdrant.count_points() == points_before

        # Case 2: Cross-format .pdf with identical semantic text
        f_pdf = _write_pdf(docs_dir / "reglament_peregovornyh_scan_export.pdf", clauses)
        assert compute_file_sha256(f_docx) != compute_file_sha256(f_pdf)

        res_dup_pdf = pipeline.index_file(f_pdf, force_reindex=False)
        assert res_dup_pdf["status"] == "duplicate"
        assert res_dup_pdf["reason"] == "content_duplicate"
        assert res_dup_pdf["duplicate_of"] == orig_id
        assert len(pipeline.registry) == 1
        assert pipeline.qdrant.count_points() == points_before

    def test_near_identical_chunk_overlap_rejected_and_batch_plus_force_reindex_work(self, tmp_path: Path) -> None:
        pipeline = _build_isolated_pipeline(tmp_path)
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()

        # 11 clauses -> >= 90% chunk overlap when 10 of 11 clauses match verbatim
        base_lines = [
            "Положение о порядке хранения лабораторных реактивов и прекурсоров в корпусах",
        ] + [
            f"{i}. Ответственный сотрудник лаборатории ежедневно проверяет температурный журнал шкафа хранения № {i} и фиксирует показания датчиков влажности."
            for i in range(1, 12)
        ]
        f_base = _write_docx(batch_dir / "01_polozhenie_reaktivy.docx", base_lines)
        r1 = pipeline.index_file(f_base, force_reindex=False)
        assert r1["status"] == "success"
        canon_id = r1["doc_id"]

        # Modify 1 of 11 numbered clauses (10/11 = 90.9% >= 90% chunk hash match, no new edition/date/number)
        tweaked_lines = list(base_lines)
        tweaked_lines[-1] = "11. Ответственный сотрудник лаборатории еженедельно сверяет описи шкафа хранения № 11."
        f_near_dup = _write_docx(batch_dir / "02_polozhenie_reaktivy_copy2.docx", tweaked_lines)
        r_near = pipeline.index_file(f_near_dup, force_reindex=False)
        assert r_near["status"] == "duplicate"
        assert r_near["reason"] == "content_duplicate"
        assert r_near["duplicate_of"] == canon_id

        # Add a genuinely distinct document to batch_dir and run index_directory(force_reindex=False)
        distinct_lines = [
            "Приказ № 881-1 от 10.09.2025 Об утверждении Регламента выдачи спецодежды работникам служб эксплуатации",
            "1. Спецодежда выдается работникам инженерных служб согласно нормам охраны труда один раз в год.",
            "2. Учет выданных комплектов ведется заведующим складом материально-технического снабжения.",
        ]
        _write_docx(batch_dir / "03_prikaz_881_specodezhda.docx", distinct_lines)

        batch_resp = pipeline.index_directory(batch_dir, force_reindex=False)
        assert batch_resp.failed_files == 0
        assert batch_resp.errors == []
        assert batch_resp.indexed_files == 1  # Only 03_prikaz_881_specodezhda.docx is newly indexed
        assert batch_resp.skipped_files >= 2  # f_base (duplicate/skipped) + f_near_dup (duplicate)
        assert len(pipeline.registry) == 2

        # force_reindex=True reindexes canonical doc_id in place without creating a second doc_id
        r_force = pipeline.index_file(f_base, force_reindex=True)
        assert r_force["status"] == "success"
        assert r_force["doc_id"] == canon_id
        assert len(pipeline.registry) == 2


# -----------------------------------------------------------------------------
# 3. Item #6 — K2 Edition Metadata, Short-Title Guard & Automatic Reconciliation
# -----------------------------------------------------------------------------


class TestK2EditionMetadataAndReconciliation:
    def test_edition_metadata_extraction_and_version_family_guard(self) -> None:
        ed1 = extract_edition_metadata("Положение об оплате труда работников (в ред. от 15.03.2025)", "")
        assert ed1["edition_date"] == "2025-03-15"

        ed2 = extract_edition_metadata("Регламент закупок товаров (редакция № 3 от 01.09.2024)", "")
        assert ed2["edition_date"] == "2024-09-01"
        assert ed2["edition_number"] == 3

        ed3 = extract_edition_metadata("Порядок премирования сотрудников", "Утверждена версия 2.1 с изменениями от 20.11.2025")
        assert ed3["edition_date"] == "2025-11-20"
        assert ed3["edition_number"] == "2.1"

        # Same normative subject across an Order wrapper and standalone Regulation with edition markers
        vf_a = compute_version_family(
            title="Приказ № 105-1 от 12.02.2021 Об утверждении Положения о служебных командировках работников",
            doc_type="Приказ",
        )
        vf_b = compute_version_family(
            title="Положение о служебных командировках работников (в ред. от 18.06.2025, редакция № 2)",
            doc_type="Положение",
        )
        assert vf_a != ""
        assert vf_a == vf_b

        # Short / generic titles MUST return "" so unrelated orders never collide
        assert compute_version_family(title="Приказ", doc_type="Приказ") == ""
        assert compute_version_family(title="Приказ № 15-1 от 01.02.2024", doc_type="Приказ") == ""
        assert compute_version_family(title="Распоряжение № 44 от 10.05.2025", doc_type="Распоряжение") == ""
        assert compute_version_family(title="Положение", doc_type="Положение") == ""

        # Partial amendment gets separate namespace so it does not auto-supersede base act
        vf_amend = compute_version_family(
            title="Приказ № 210-1 от 01.04.2025 О внесении изменений в Положение о служебных командировках работников",
            doc_type="Приказ",
        )
        assert vf_amend.startswith("amendment::")
        assert vf_amend != vf_a

    def test_automatic_multi_version_reconciliation_by_date_and_edition_without_yaml(self, tmp_path: Path) -> None:
        pipeline = _build_isolated_pipeline(tmp_path)
        docs_dir = tmp_path / "editions"
        docs_dir.mkdir()

        # Upload OUT OF ORDER: v2 (2023) -> v3 (2025, newest) -> v1 (2020, oldest), with NO explicit repeal clause!
        f_v2 = _write_docx(
            docs_dir / "polozhenie_komandirovki_2023.docx",
            [
                "Положение о служебных командировках работников института (редакция № 2 от 10.04.2023)",
                "1. Суточные при командировках по России составляют 700 рублей в сутки.",
                "2. Авансовый отчет представляется в бухгалтерию в течение 3 рабочих дней.",
            ],
        )
        f_v3 = _write_docx(
            docs_dir / "prikaz_550_1_komandirovki_2025.docx",
            [
                "Приказ № 550-1 от 15.09.2025 Об утверждении Положения о служебных командировках работников института (редакция № 3)",
                "1. Суточные при командировках по России составляют 1200 рублей в сутки.",
                "2. Авансовый отчет представляется через ЭДО в течение 5 рабочих дней.",
            ],
        )
        f_v1 = _write_docx(
            docs_dir / "polozhenie_komandirovki_2020.docx",
            [
                "Положение о служебных командировках работников института (редакция № 1 от 01.03.2020)",
                "1. Суточные при командировках по России составляют 500 рублей в сутки.",
                "2. Авансовый отчет сдается на бумажном носителе в течение 3 рабочих дней.",
            ],
        )

        r2 = pipeline.index_file(f_v2)
        r3 = pipeline.index_file(f_v3)
        r1 = pipeline.index_file(f_v1)
        assert r2["status"] == "success" and r3["status"] == "success" and r1["status"] == "success"

        m1 = pipeline.get_document(r1["doc_id"])
        m2 = pipeline.get_document(r2["doc_id"])
        m3 = pipeline.get_document(r3["doc_id"])
        assert m1 is not None and m2 is not None and m3 is not None

        assert m3.status == DocumentStatus.ACTIVE.value
        assert m2.status == DocumentStatus.SUPERSEDED.value
        assert m1.status == DocumentStatus.SUPERSEDED.value

        assert any("550-1" in ref or r3["doc_id"] in ref for ref in m1.superseded_by)
        assert any("550-1" in ref or r3["doc_id"] in ref for ref in m2.superseded_by)
        assert len(m3.supersedes) >= 2
        assert "Утратил актуальность: действует более новая редакция" in m1.version_note
        assert "Утратил актуальность: действует более новая редакция" in m2.version_note
        assert "заменяет предыдущ" in m3.version_note.lower()

        # Verify persisted in document_registry.json AND Qdrant payloads!
        saved_reg = json.loads((tmp_path / "document_registry.json").read_text(encoding="utf-8"))
        assert saved_reg[r1["doc_id"]]["status"] == "superseded"
        assert saved_reg[r2["doc_id"]]["status"] == "superseded"
        assert saved_reg[r3["doc_id"]]["status"] == "active"

        q_v1_chunks = pipeline.qdrant.get_document_chunks(r1["doc_id"])
        q_v3_chunks = pipeline.qdrant.get_document_chunks(r3["doc_id"])
        assert all(c.status == "superseded" and "Утратил актуальность" in c.version_note for c in q_v1_chunks)
        assert all(c.status == "active" for c in q_v3_chunks)

        # Also verify disjoint annual plans do NOT supersede each other
        f_plan_2025 = _write_docx(
            docs_dir / "plan_zakupok_2025.docx",
            [
                "Приказ № 12-1 от 15.12.2024 Об утверждении Плана закупок лабораторного оборудования на 2025 год",
                "1. Утвердить перечень закупаемого оборудования для учебных лабораторий на 2025 год.",
            ],
        )
        f_plan_2026 = _write_docx(
            docs_dir / "plan_zakupok_2026.docx",
            [
                "Приказ № 19-1 от 18.12.2025 Об утверждении Плана закупок лабораторного оборудования на 2026 год",
                "1. Утвердить перечень закупаемого оборудования для учебных лабораторий на 2026 год.",
            ],
        )
        rp25 = pipeline.index_file(f_plan_2025)
        rp26 = pipeline.index_file(f_plan_2026)
        assert pipeline.get_document(rp25["doc_id"]).status == DocumentStatus.ACTIVE.value  # type: ignore[union-attr]
        assert pipeline.get_document(rp26["doc_id"]).status == DocumentStatus.ACTIVE.value  # type: ignore[union-attr]


# -----------------------------------------------------------------------------
# 4. Item #6 — Current vs Historical Retrieval Ranking & Reranker Digest
# -----------------------------------------------------------------------------


class TestK2RetrievalAndReranker:
    def test_current_vs_historical_retrieval_ranking_and_reranker_digest(self, tmp_path: Path) -> None:
        pipeline = _build_isolated_pipeline(tmp_path)
        docs_dir = tmp_path / "retr"
        docs_dir.mkdir()

        f_old = _write_docx(
            docs_dir / "komandirovki_2020.docx",
            [
                "Положение о служебных командировках работников университета (редакция № 1 от 10.03.2020)",
                "1. Размер суточных при служебных командировках составляет 500 рублей за каждый день нахождения в командировке.",
            ],
        )
        f_new = _write_docx(
            docs_dir / "komandirovki_2025.docx",
            [
                "Приказ № 900-1 от 01.10.2025 Об утверждении Положения о служебных командировках работников университета (редакция № 2)",
                "1. Размер суточных при служебных командировках составляет 1500 рублей за каждый день нахождения в командировке.",
            ],
        )
        r_old = pipeline.index_file(f_old)
        r_new = pipeline.index_file(f_new)
        old_id = r_old["doc_id"]
        new_id = r_new["doc_id"]

        svc = SearchService(
            pipeline=pipeline,
            table_store=pipeline.table_store,
            glossary=get_glossary_manager(),
        )

        # 1. Temporal intent detection checks
        intent_cur = svc.detect_temporal_edition_intent("Какой размер суточных при служебных командировках?")
        assert intent_cur["is_historical"] is False

        intent_old_phrase = svc.detect_temporal_edition_intent(
            "Какой размер суточных был в старой редакции положения о служебных командировках?"
        )
        assert intent_old_phrase["is_historical"] is True
        assert intent_old_phrase["explicit_old_phrase"] is True

        intent_before_repeal = svc.detect_temporal_edition_intent(
            "Какие суточные действовали до отмены предыдущей редакции по командировкам?"
        )
        assert intent_before_repeal["is_historical"] is True

        intent_hist_year = svc.detect_temporal_edition_intent(
            "Какой размер суточных при служебных командировках был в 2020 году?"
        )
        assert intent_hist_year["is_historical"] is True
        assert old_id in intent_hist_year["matched_superseded_doc_ids"]

        intent_active_year = svc.detect_temporal_edition_intent(
            "Какой размер суточных при служебных командировках действует в 2025 году?"
        )
        assert intent_active_year["is_historical"] is False

        # 2. Current actuality query -> active 2025 edition ranks #1 ahead of superseded 2020 edition (0.65 penalty)
        resp_cur = svc.search("Какой размер суточных при служебных командировках работников?", top_k=5)
        assert len(resp_cur.hits) >= 1
        assert resp_cur.hits[0].doc_id == new_id
        assert resp_cur.hits[0].status == DocumentStatus.ACTIVE.value
        old_hits_cur = [h for h in resp_cur.hits if h.doc_id == old_id]
        if old_hits_cur:
            assert resp_cur.hits[0].score > old_hits_cur[0].score
            assert K2_SUPERSEDED_SCORE_PENALTY == pytest.approx(0.65)

        # 3. Historical queries ("в старой редакции" & "в 2020 году") -> superseded 2020 edition lifted & ranked #1!
        resp_hist_phrase = svc.search(
            "Какой размер суточных был в старой редакции положения о служебных командировках?",
            top_k=5,
        )
        assert any(h.doc_id == old_id for h in resp_hist_phrase.hits)
        assert resp_hist_phrase.hits[0].doc_id == old_id

        resp_hist_year = svc.search(
            "Какой размер суточных при служебных командировках работников был в 2020 году?",
            top_k=5,
        )
        assert any(h.doc_id == old_id for h in resp_hist_year.hits)
        assert resp_hist_year.hits[0].doc_id == old_id

        # 4. Date-recency tie-breaker among two ACTIVE documents with identical raw similarity
        pt_2022 = {
            "point_uuid": "u-2022",
            "rrf_score": 0.50,
            "dense_score": 0.80,
            "sparse_score": 4.0,
            "payload": {
                "chunk_id": "act_2022::c01",
                "doc_id": "act_2022",
                "title": "Порядок закупа расходных материалов",
                "doc_date": "2022-02-01",
                "edition_date": "2022-02-01",
                "status": "active",
                "clean_text": "Закупка расходных материалов согласуется отделом снабжения.",
            },
        }
        pt_2026 = {
            "point_uuid": "u-2026",
            "rrf_score": 0.50,
            "dense_score": 0.80,
            "sparse_score": 4.0,
            "payload": {
                "chunk_id": "act_2026::c01",
                "doc_id": "act_2026",
                "title": "Порядок закупа расходных материалов",
                "doc_date": "2026-02-01",
                "edition_date": "2026-02-01",
                "status": "active",
                "clean_text": "Закупка расходных материалов согласуется отделом снабжения.",
            },
        }
        h_2022 = svc._raw_point_to_hit(pt_2022, query_norm="закупка расходных материалов", query_tokens_set=set(), include_parent=False)
        h_2026 = svc._raw_point_to_hit(pt_2026, query_norm="закупка расходных материалов", query_tokens_set=set(), include_parent=False)
        assert h_2026.score > h_2022.score

        # 5. Stage 3 Reranker candidate digest formatting includes status, doc_date, edition_date, and version_note
        digest = QueryPipeline._format_candidate_digest(resp_hist_phrase.hits, table_by_id={})
        assert "status=superseded" in digest
        assert "status=active" in digest
        assert "doc_date=2020-03-10" in digest
        assert "version_note=Утратил актуальность" in digest
