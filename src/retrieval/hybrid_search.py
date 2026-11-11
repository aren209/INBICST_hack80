from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from src.config import (
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
        dense_abstain_threshold: float = 0.785,
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
            limit=prefetch_limit,
            with_payload=True,
        )
        dense_scores: dict[str, float] = {}
        for pt in dense_res.points:
            cid = (pt.payload or {}).get("chunk_id", str(pt.id))
            dense_scores[cid] = float(pt.score)
        max_dense_score = max(dense_scores.values(), default=0.0)

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
            limit=top_k * 2,
            with_payload=True,
        )

        # 4. Version-aware soft boosting (active docs get +12% RRF boost, while keeping superseded visible)
        scored_points: list[tuple[float, models.ScoredPoint]] = []
        for pt in rrf_res.points:
            pl = pt.payload or {}
            status = pl.get("status", "active")
            boost = 1.12 if status == "active" else 0.96
            scored_points.append((float(pt.score) * boost, pt))

        scored_points.sort(key=lambda x: x[0], reverse=True)
        selected = scored_points[:top_k]

        hits: list[RetrievedChunk] = []
        version_alerts: list[str] = []
        quality_alerts: list[str] = []
        seen_version_notes: set[str] = set()
        seen_quality_notes: set[str] = set()

        for adj_score, pt in selected:
            pl = pt.payload or {}
            cid = pl.get("chunk_id", str(pt.id))
            sec_id = pl.get("section_id", "")
            parent_text = ""
            if expand_parent_section and sec_id in self.sections_store:
                parent_text = self.sections_store[sec_id].get("text", "")[:3500]

            v_note = pl.get("version_note", "")
            if v_note and v_note not in seen_version_notes:
                seen_version_notes.add(v_note)
                version_alerts.append(
                    f"[{pl.get('doc_title')}]: {v_note}"
                )

            q_warn = pl.get("quality_warning", "")
            if q_warn and q_warn not in seen_quality_notes:
                seen_quality_notes.add(q_warn)
                quality_alerts.append(
                    f"[{pl.get('doc_title')}]: {q_warn}"
                )

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

        # 5. Abstain ("нет ответа в базе") check
        abstain = False
        abstain_reason = ""
        if max_dense_score < self.dense_abstain_threshold and sparse_hits_count == 0:
            abstain = True
            abstain_reason = (
                f"Низкая семантическая близость ({max_dense_score:.3f} < {self.dense_abstain_threshold}) "
                "и отсутствие ключевых терминов запроса во внутренних регламентах МФТИ."
            )
        elif max_dense_score < (self.dense_abstain_threshold - 0.025):
            abstain = True
            abstain_reason = (
                f"Семантическая близость ниже порога достоверности ({max_dense_score:.3f}). "
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
