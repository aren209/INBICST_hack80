"""Qdrant Hybrid Vector Store (Dense 1024d Cosine + Sparse BM25 CRC32 IDF + RRF Fusion)."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import (
    DENSE_VECTOR_NAME,
    EMBEDDING_DIM,
    NON_ANSWER_ZONES,
    SPARSE_VECTOR_NAME,
    settings,
)
from src.indexing.embedder import cosine_similarity
from src.schemas import ChunkRecord, DocumentMeta, make_point_uuid

logger = logging.getLogger(__name__)


class QdrantHybridStore:
    """
    Manages the Qdrant hybrid collection with:
    - Named dense vector ('dense', 1024d, Cosine)
    - Named sparse vector ('bm25', Modifier.IDF)
    - Payload indexes on doc_id, doc_number, zone, status, unit_id, section_id, content_hash
    - Idempotent delete_by_doc_id, batched upsert_chunks, update_document_payload, and RRF query_points
    """

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
        embedding_dim: int = EMBEDDING_DIM,
        in_memory: bool = False,
        client: QdrantClient | None = None,
    ) -> None:
        self.url = url or settings.QDRANT_URL
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.embedding_dim = embedding_dim
        self.in_memory = in_memory
        self._client: QdrantClient | None = client
        self._collection_ready: bool = False

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            if self.in_memory:
                self._client = QdrantClient(":memory:")
            else:
                self._client = QdrantClient(url=self.url, timeout=30.0)
        return self._client

    def ensure_collection(self) -> bool:
        """Create the hybrid Qdrant collection and payload indexes if they do not exist."""
        try:
            exists = self.client.collection_exists(self.collection_name)
            if not exists:
                logger.info("Creating Qdrant hybrid collection '%s'...", self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=self.embedding_dim,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        )
                    },
                )
                self._create_payload_indexes()
            elif not self._collection_ready:
                self._create_payload_indexes()
            self._collection_ready = True
            return True
        except Exception as exc:
            logger.error("Failed to ensure Qdrant collection '%s': %s", self.collection_name, exc)
            return False

    def _create_payload_indexes(self) -> None:
        """Create keyword and integer payload indexes for fast filtered retrieval."""
        if self.in_memory:
            return
        keyword_fields = (
            "doc_id",
            "doc_number",
            "zone",
            "status",
            "unit_id",
            "section_id",
            "content_hash",
        )
        for field_name in keyword_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as idx_exc:
                logger.debug("Keyword index check for %s: %s", field_name, idx_exc)

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="applicable_years",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
        except Exception as idx_exc:
            logger.debug("Integer index check for applicable_years: %s", idx_exc)

    @staticmethod
    def _coerce_sparse_vector(sparse: models.SparseVector | dict[str, Any]) -> models.SparseVector:
        if isinstance(sparse, models.SparseVector):
            return sparse
        if isinstance(sparse, dict):
            return models.SparseVector(
                indices=[int(i) for i in sparse.get("indices", [])],
                values=[float(v) for v in sparse.get("values", [])],
            )
        return models.SparseVector(indices=[], values=[])

    @staticmethod
    def build_chunk_payload(
        chunk: ChunkRecord,
        doc_meta: DocumentMeta | None = None,
    ) -> dict[str, Any]:
        """Build complete Qdrant point payload combining ChunkRecord and DocumentMeta fields."""
        payload = chunk.model_dump()
        if doc_meta is not None:
            payload["file_name"] = doc_meta.file_name
            payload["doc_number"] = doc_meta.doc_number
            payload["doc_date"] = doc_meta.doc_date
            payload["doc_type"] = doc_meta.doc_type
            payload["title"] = doc_meta.title
            payload["status"] = doc_meta.status
            payload["applicable_years"] = list(doc_meta.applicable_years)
            payload["supersedes"] = list(doc_meta.supersedes)
            payload["superseded_by"] = list(doc_meta.superseded_by)
            payload["amends"] = list(doc_meta.amends)
            payload["amended_by"] = list(doc_meta.amended_by)
            payload["version_note"] = doc_meta.version_note
            payload["quality_warning"] = doc_meta.quality_warning
            # Merge document-level also_in_files with chunk-level also_in_files
            combined_also = list(dict.fromkeys(list(chunk.also_in_files) + list(doc_meta.also_in_files)))
            if doc_meta.file_name in combined_also:
                combined_also.remove(doc_meta.file_name)
            payload["also_in_files"] = combined_also
        else:
            meta = chunk.metadata or {}
            for key in (
                "file_name",
                "doc_number",
                "doc_date",
                "doc_type",
                "title",
                "status",
                "applicable_years",
                "supersedes",
                "superseded_by",
                "amends",
                "amended_by",
                "version_note",
                "quality_warning",
            ):
                if key not in payload and key in meta:
                    payload[key] = meta[key]
        return payload

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Idempotently delete all points belonging to doc_id from Qdrant."""
        if not doc_id:
            return 0
        self.ensure_collection()
        try:
            before = self.count_points(doc_id=doc_id)
            doc_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=doc_filter),
                wait=True,
            )
            return before
        except Exception as exc:
            logger.error("Failed to delete points for doc_id='%s': %s", doc_id, exc)
            return 0

    def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        dense_vecs: list[list[float]],
        sparse_vecs: list[models.SparseVector | dict[str, Any]],
        doc_meta: DocumentMeta | None = None,
        batch_size: int = 64,
    ) -> int:
        """Upsert ChunkRecords with their named dense and sparse vectors in batches."""
        if not chunks:
            return 0
        if len(chunks) != len(dense_vecs) or len(chunks) != len(sparse_vecs):
            raise ValueError(
                f"Length mismatch in upsert_chunks: chunks={len(chunks)}, "
                f"dense={len(dense_vecs)}, sparse={len(sparse_vecs)}"
            )

        self.ensure_collection()
        points: list[models.PointStruct] = []
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vecs, sparse_vecs):
            point_id = chunk.point_uuid or make_point_uuid(chunk.chunk_id)
            sp = self._coerce_sparse_vector(sparse_vec)
            payload = self.build_chunk_payload(chunk, doc_meta=doc_meta)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: sp,
                    },
                    payload=payload,
                )
            )

        upserted = 0
        for start in range(0, len(points), max(1, batch_size)):
            batch = points[start : start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )
            upserted += len(batch)

        return upserted

    def update_document_payload(self, doc_id: str, patch_dict: dict[str, Any]) -> int:
        """Update payload fields across all points belonging to doc_id."""
        if not doc_id or not patch_dict:
            return 0
        self.ensure_collection()
        try:
            doc_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=patch_dict,
                points=doc_filter,
                wait=True,
            )
            return self.count_points(doc_id=doc_id)
        except Exception as exc:
            logger.error("Failed to update payload for doc_id='%s': %s", doc_id, exc)
            return 0

    def update_chunk_payload_by_uuid(self, point_uuid: str, patch_dict: dict[str, Any]) -> bool:
        """Update payload fields on a specific point UUID (e.g. cross-file deduplication also_in_files)."""
        if not point_uuid or not patch_dict:
            return False
        self.ensure_collection()
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=patch_dict,
                points=[point_uuid],
                wait=True,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to update chunk payload %s: %s", point_uuid, exc)
            return False

    def get_document_chunks(self, doc_id: str) -> list[ChunkRecord]:
        """Scroll and return all ChunkRecords stored in Qdrant for a given doc_id."""
        if not doc_id:
            return []
        if not self.ensure_collection():
            return []
        try:
            doc_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
            records: list[ChunkRecord] = []
            next_offset: Any = None
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=doc_filter,
                    limit=256,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for pt in points:
                    if pt.payload:
                        try:
                            records.append(ChunkRecord.model_validate(pt.payload))
                        except Exception as val_err:
                            logger.warning("Skipping malformed chunk payload in scroll: %s", val_err)
                if next_offset is None:
                    break
            records.sort(key=lambda c: (c.unit_id, c.chunk_id))
            return records
        except Exception as exc:
            logger.error("Failed to scroll chunks for doc_id='%s': %s", doc_id, exc)
            return []

    def count_points(self, doc_id: str | None = None) -> int:
        """Count total points in collection or points for a specific doc_id."""
        if not self.ensure_collection():
            return 0
        try:
            count_filter = None
            if doc_id:
                count_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            res = self.client.count(
                collection_name=self.collection_name,
                count_filter=count_filter,
                exact=True,
            )
            return int(res.count)
        except Exception as exc:
            logger.warning("Failed to count points in Qdrant: %s", exc)
            return 0

    def check_health(self) -> dict[str, Any]:
        """Inspect Qdrant connectivity, collection status, and point count."""
        try:
            exists = self.client.collection_exists(self.collection_name)
            if not exists:
                self.ensure_collection()
            count = self.count_points()
            return {
                "connected": True,
                "url": ":memory:" if self.in_memory else self.url,
                "collection": self.collection_name,
                "points_count": count,
                "status": "ok",
            }
        except Exception as exc:
            return {
                "connected": False,
                "url": ":memory:" if self.in_memory else self.url,
                "collection": self.collection_name,
                "points_count": 0,
                "status": f"error: {exc}",
            }

    @staticmethod
    def build_search_filter(
        status_filter: list[str] | str | None = None,
        year_filter: int | None = None,
        doc_id_filter: list[str] | str | None = None,
        doc_number_filter: list[str] | str | None = None,
        unit_id_filter: str | None = None,
        exclude_zones: set[str] | None = None,
    ) -> models.Filter | None:
        """Construct a Qdrant Filter for hybrid search."""
        must: list[models.Condition] = []
        must_not: list[models.Condition] = []

        zones_to_exclude = NON_ANSWER_ZONES if exclude_zones is None else exclude_zones
        for zone_val in sorted(zones_to_exclude):
            must_not.append(
                models.FieldCondition(
                    key="zone",
                    match=models.MatchValue(value=zone_val),
                )
            )

        if status_filter:
            if isinstance(status_filter, str):
                statuses = [s.strip() for s in status_filter.split(",") if s.strip()]
            else:
                statuses = [str(s).strip() for s in status_filter if str(s).strip()]
            if len(statuses) == 1:
                must.append(
                    models.FieldCondition(
                        key="status",
                        match=models.MatchValue(value=statuses[0]),
                    )
                )
            elif len(statuses) > 1:
                must.append(
                    models.FieldCondition(
                        key="status",
                        match=models.MatchAny(any=statuses),
                    )
                )

        if year_filter is not None:
            must.append(
                models.FieldCondition(
                    key="applicable_years",
                    match=models.MatchValue(value=int(year_filter)),
                )
            )

        if doc_id_filter:
            if isinstance(doc_id_filter, str):
                must.append(
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id_filter),
                    )
                )
            elif len(doc_id_filter) == 1:
                must.append(
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id_filter[0]),
                    )
                )
            elif len(doc_id_filter) > 1:
                must.append(
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchAny(any=list(doc_id_filter)),
                    )
                )

        if doc_number_filter:
            if isinstance(doc_number_filter, str):
                must.append(
                    models.FieldCondition(
                        key="doc_number",
                        match=models.MatchValue(value=doc_number_filter),
                    )
                )
            elif len(doc_number_filter) == 1:
                must.append(
                    models.FieldCondition(
                        key="doc_number",
                        match=models.MatchValue(value=doc_number_filter[0]),
                    )
                )
            elif len(doc_number_filter) > 1:
                must.append(
                    models.FieldCondition(
                        key="doc_number",
                        match=models.MatchAny(any=list(doc_number_filter)),
                    )
                )

        if unit_id_filter:
            must.append(
                models.FieldCondition(
                    key="unit_id",
                    match=models.MatchValue(value=unit_id_filter),
                )
            )

        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)

    @staticmethod
    def _compute_sparse_overlap_score(
        query_sparse: models.SparseVector,
        point_sparse: Any,
    ) -> float:
        """Compute raw lexical TF overlap score between query sparse vector and point sparse vector."""
        if not query_sparse or not query_sparse.indices or point_sparse is None:
            return 0.0
        q_map = dict(zip(query_sparse.indices, query_sparse.values))
        p_indices = getattr(point_sparse, "indices", None)
        p_values = getattr(point_sparse, "values", None)
        if p_indices is None and isinstance(point_sparse, dict):
            p_indices = point_sparse.get("indices", [])
            p_values = point_sparse.get("values", [])
        if not p_indices or not p_values:
            return 0.0
        score = 0.0
        for idx, val in zip(p_indices, p_values):
            if idx in q_map:
                score += float(q_map[idx]) * float(val)
        return float(score)

    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: models.SparseVector | dict[str, Any],
        top_k: int = 15,
        prefetch_limit: int = 40,
        query_filter: models.Filter | None = None,
        status_filter: list[str] | str | None = None,
        year_filter: int | None = None,
        doc_id_filter: list[str] | str | None = None,
        doc_number_filter: list[str] | str | None = None,
        unit_id_filter: str | None = None,
        exclude_zones: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute Hybrid RRF search via Qdrant query_points with Dense + Sparse BM25 Prefetch.
        Returns a list of hit dicts containing payload, rrf_score, exact dense_score (cosine), and sparse_score.
        """
        if not self.ensure_collection():
            return []

        sp_vec = self._coerce_sparse_vector(query_sparse)
        q_filter = query_filter
        if q_filter is None:
            q_filter = self.build_search_filter(
                status_filter=status_filter,
                year_filter=year_filter,
                doc_id_filter=doc_id_filter,
                doc_number_filter=doc_number_filter,
                unit_id_filter=unit_id_filter,
                exclude_zones=exclude_zones,
            )

        eff_prefetch_limit = max(top_k * 3, prefetch_limit)
        prefetches: list[models.Prefetch] = [
            models.Prefetch(
                query=query_dense,
                using=DENSE_VECTOR_NAME,
                limit=eff_prefetch_limit,
                filter=q_filter,
            )
        ]
        if sp_vec.indices:
            prefetches.append(
                models.Prefetch(
                    query=sp_vec,
                    using=SPARSE_VECTOR_NAME,
                    limit=eff_prefetch_limit,
                    filter=q_filter,
                )
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetches,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )

        # Also fetch top sparse-only scores when sp_vec is non-empty so sparse_score is IDF-weighted
        sparse_scores_by_id: dict[str, float] = {}
        if sp_vec.indices:
            try:
                sp_res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=sp_vec,
                    using=SPARSE_VECTOR_NAME,
                    query_filter=q_filter,
                    limit=eff_prefetch_limit,
                    with_payload=False,
                    with_vectors=False,
                )
                for pt in sp_res.points:
                    sparse_scores_by_id[str(pt.id)] = float(pt.score)
            except Exception as sp_exc:
                logger.debug("Optional sparse score lookup skipped: %s", sp_exc)

        results: list[dict[str, Any]] = []
        for pt in response.points:
            payload = dict(pt.payload or {})
            rrf_score = float(pt.score)

            # Extract stored dense vector from point to compute exact cosine similarity
            dense_score = 0.0
            point_sp_vec = None
            if isinstance(pt.vector, dict):
                pt_dense = pt.vector.get(DENSE_VECTOR_NAME)
                if isinstance(pt_dense, list) and pt_dense:
                    dense_score = cosine_similarity(query_dense, [float(x) for x in pt_dense])
                point_sp_vec = pt.vector.get(SPARSE_VECTOR_NAME)

            pt_id_str = str(pt.id)
            sparse_score = sparse_scores_by_id.get(
                pt_id_str,
                self._compute_sparse_overlap_score(sp_vec, point_sp_vec),
            )

            results.append(
                {
                    "point_uuid": pt_id_str,
                    "rrf_score": rrf_score,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "payload": payload,
                }
            )

        return results


# Alias for external callers
QdrantStore = QdrantHybridStore
