from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from src.config import (
    DENSE_ABSTAIN_THRESHOLD,
    NON_ANSWER_ZONES,
    PARENT_SECTION_MAX_CHARS,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    SECTIONS_STORE_PATH,
)
from src.indexing.embedder import LocalDenseEmbedder, LocalSparseBM25


@dataclass
class RetrievedChunk:
    chunk_id: str
    rrf_score: float
    dense_score: float
    doc_id: str
    doc_title: str
    doc_number: str
    doc_date: str
    status: str
    unit_title: str
    section_no: str
    section_title: str
    clause_no: str
    zone: str
    roles: list[str]
    page_from: int
    page_to: int
    raw_text: str
    parent_section_text: str = ""
    version_note: str = ""
    quality_warning: str = ""
    supersedes: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)


@dataclass
class SearchResponse:
    query: str
    abstain_recommended: bool
    abstain_reason: str
    max_dense_score: float
    lexical_hits: int
    hits: list[RetrievedChunk]
    version_alerts: list[str]
    quality_alerts: list[str]


class HybridRetriever:
    """
    Hybrid Search over Qdrant (Dense + Lemmatized Sparse BM25 with exact=True)
    + V3 Parent-Child Section Expansion + Version-Aware Collision Detection.
    """

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION_NAME,
        dense_embedder: LocalDenseEmbedder | None = None,
        sparse_bm25: LocalSparseBM25 | None = None,
        sections_path: Path = SECTIONS_STORE_PATH,
        dense_abstain_threshold: float = DENSE_ABSTAIN_THRESHOLD,
    ) -> None:
        self.client = QdrantClient(url=qdrant_url, timeout=30)
        self.collection_name = collection_name
        self.dense_embedder = dense_embedder or LocalDenseEmbedder()
        self.sparse_bm25 = sparse_bm25 or LocalSparseBM25()
        self.dense_abstain_threshold = dense_abstain_threshold
        self.sections_store: dict[str, dict[str, Any]] = {}
        if sections_path.exists():
            try:
                self.sections_store = json.loads(
                    sections_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

    def search(
        self,
        query: str,
        top_k: int = 8,
        prefetch_limit: int = 30,
        expand_parent_section: bool = True,
    ) -> SearchResponse:
        q_dense = self.dense_embedder.embed_query(query)
        q_sparse = self.sparse_bm25.encode_query(query)

        # Exclude approval sheets from normative search
        base_filter = models.Filter(
            must_not=[
                models.FieldCondition(
                    key="zone",
                    match=models.MatchValue(value="approval"),
                )
            ]
        )

        # 1. Exact Dense search to get calibrated cosine similarity for abstain detection
        dense_res = self.client.query_points(
            collection_name=self.collection_name,
            query=q_dense,
            using="dense",
            query_filter=base_filter,
            search_params=models.SearchParams(exact=True),
            limit=max(prefetch_limit * 2, 60),
            with_payload=True,
        )
        dense_scores: dict[str, float] = {}
        for idx, pt in enumerate(dense_res.points):
            cid = (pt.payload or {}).get("chunk_id", str(pt.id))
            dense_scores[cid] = float(pt.score)
        max_dense_score = max(
            (float(pt.score) for pt in dense_res.points[:prefetch_limit]),
            default=0.0,
        )

        # 2. Sparse BM25 search to check lexical overlap
        sparse_hits_count = 0
        if q_sparse.indices:
            sparse_res = self.client.query_points(
                collection_name=self.collection_name,
                query=q_sparse,
                using="bm25",
                query_filter=base_filter,
                limit=prefetch_limit,
                with_payload=False,
            )
            sparse_hits_count = len(sparse_res.points)

        # 3. Hybrid RRF query in Qdrant
        prefetches: list[models.Prefetch] = [
            models.Prefetch(
                query=q_dense,
                using="dense",
                filter=base_filter,
                params=models.SearchParams(exact=True),
                limit=prefetch_limit,
            )
        ]
        if q_sparse.indices:
            prefetches.append(
                models.Prefetch(
                    query=q_sparse,
                    using="bm25",
                    filter=base_filter,
                    limit=prefetch_limit,
                )
            )

        rrf_res = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetches,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=max(top_k * 3, 24),
            with_payload=True,
        )

        # 4. Version-aware soft boosting + semantic verification gate
        # Active docs get +12% RRF boost, draft docs get +4%, superseded get 0.94.
        # Chunks that missed the dense semantic pool (or fall far below threshold)
        # are penalized so a single-word BM25 match from an unrelated regulation
        # never outranks semantically relevant clauses.
        scored_points: list[tuple[float, models.ScoredPoint]] = []
        for pt in rrf_res.points:
            pl = pt.payload or {}
            cid = pl.get("chunk_id", str(pt.id))
            status = pl.get("status", "active")
            if status == "active":
                boost = 1.12
            elif status == "draft":
                boost = 1.04
            else:
                boost = 0.94

            d_score = dense_scores.get(cid, 0.0)
            if d_score == 0.0 or d_score < (self.dense_abstain_threshold - 0.015):
                boost *= 0.25

            scored_points.append((float(pt.score) * boost, pt))

        scored_points.sort(key=lambda x: x[0], reverse=True)

        # Служебные «паспорта» документов (zone=meta) и листы согласования
        # оставляем в кандидатах для алертов, но не отдаём как ответные чанки:
        # иначе короткий паспорт с названием/версией вытесняет конкретный пункт.
        answer_points = [
            (score, pt)
            for score, pt in scored_points
            if (pt.payload or {}).get("zone", "body") not in NON_ANSWER_ZONES
        ]
        selected = answer_points[:top_k]

        # Собираем множества doc_id, попавших в итоговый ответ, и связанных с ними
        # версий/приложений, чтобы не показывать ложные алерты от случайных кандидатов.
        relevant_alert_doc_ids: set[str] = set()
        for _, pt in selected:
            pl = pt.payload or {}
            did = pl.get("doc_id", "")
            if did:
                relevant_alert_doc_ids.add(did)
            for ref_key in ("supersedes", "superseded_by", "related_docs"):
                for ref_id in pl.get(ref_key) or []:
                    if ref_id:
                        relevant_alert_doc_ids.add(str(ref_id))

        alert_points = [
            (score, pt)
            for score, pt in scored_points[: max(top_k * 3, 24)]
            if (pt.payload or {}).get("doc_id", "") in relevant_alert_doc_ids
        ]

        hits: list[RetrievedChunk] = []
        version_alerts: list[str] = []
        quality_alerts: list[str] = []
        seen_version_notes: set[str] = set()
        seen_quality_notes: set[str] = set()

        # Алерты собираем по широкому пулу кандидатов, чтобы предупреждение
        # о версии/дефекте не терялось, даже если паспорт не попал в ответ.
        for _, pt in alert_points:
            pl = pt.payload or {}
            v_note = pl.get("version_note", "")
            if v_note and v_note not in seen_version_notes:
                seen_version_notes.add(v_note)
                version_alerts.append(f"[{pl.get('doc_title')}]: {v_note}")

            q_warn = pl.get("quality_warning", "")
            if q_warn and q_warn not in seen_quality_notes:
                seen_quality_notes.add(q_warn)
                quality_alerts.append(f"[{pl.get('doc_title')}]: {q_warn}")

        for adj_score, pt in selected:
            pl = pt.payload or {}
            cid = pl.get("chunk_id", str(pt.id))
            sec_id = pl.get("section_id", "")
            parent_text = ""
            if expand_parent_section and sec_id in self.sections_store:
                parent_text = self.sections_store[sec_id].get("text", "")[
                    :PARENT_SECTION_MAX_CHARS
                ]

            v_note = pl.get("version_note", "")
            q_warn = pl.get("quality_warning", "")

            hits.append(
                RetrievedChunk(
                    chunk_id=cid,
                    rrf_score=round(adj_score, 4),
                    dense_score=round(dense_scores.get(cid, 0.0), 4),
                    doc_id=pl.get("doc_id", ""),
                    doc_title=pl.get("doc_title", ""),
                    doc_number=pl.get("doc_number", ""),
                    doc_date=pl.get("doc_date", ""),
                    status=pl.get("status", "active"),
                    unit_title=pl.get("unit_title", ""),
                    section_no=pl.get("section_no", ""),
                    section_title=pl.get("section_title", ""),
                    clause_no=pl.get("clause_no", ""),
                    zone=pl.get("zone", "body"),
                    roles=pl.get("roles", []),
                    page_from=int(pl.get("page_from", 1)),
                    page_to=int(pl.get("page_to", 1)),
                    raw_text=pl.get("raw_text", ""),
                    parent_section_text=parent_text,
                    version_note=v_note,
                    quality_warning=q_warn,
                    supersedes=pl.get("supersedes", []),
                    superseded_by=pl.get("superseded_by", []),
                )
            )

        # 5. Abstain ("нет ответа в базе") check.
        # Порог — основной сигнал: e5-large даёт 0.79–0.80 даже на внекорпусных
        # вопросах, поэтому одного lexical_hits недостаточно (BM25 всегда
        # возвращает top-N при любом общем слове).
        abstain = max_dense_score < self.dense_abstain_threshold
        abstain_reason = ""
        if abstain:
            abstain_reason = (
                f"Семантическая близость ниже порога достоверности "
                f"({max_dense_score:.3f} < {self.dense_abstain_threshold}). "
                "В предоставленной базе регламентов нет релевантного документа по данной теме."
            )

        return SearchResponse(
            query=query,
            abstain_recommended=abstain,
            abstain_reason=abstain_reason,
            max_dense_score=round(max_dense_score, 4),
            lexical_hits=sparse_hits_count,
            hits=hits,
            version_alerts=version_alerts,
            quality_alerts=quality_alerts,
        )
