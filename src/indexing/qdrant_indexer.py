from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.chunking.hierarchical_chunker import ChunkRecord, ParentSectionRecord
from src.config import (
    CHUNKS_DUMP_PATH,
    DENSE_BATCH_SIZE,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    SECTIONS_STORE_PATH,
)
from src.indexing.embedder import LocalDenseEmbedder, LocalSparseBM25


class QdrantIndexer:
    """Indexes hierarchical chunks (Dense + Sparse BM25) into Qdrant via Docker."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION_NAME,
        dense_embedder: LocalDenseEmbedder | None = None,
        sparse_bm25: LocalSparseBM25 | None = None,
    ) -> None:
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.client = QdrantClient(url=self.qdrant_url, timeout=60)
        self.dense_embedder = dense_embedder or LocalDenseEmbedder()
        self.sparse_bm25 = sparse_bm25 or LocalSparseBM25()

    @staticmethod
    def chunk_id_to_uuid(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def recreate_collection(self, dense_dim: int = 1024) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(),
            },
        )

        # Create payload indexes for fast metadata filtering
        for field_name in (
            "doc_id",
            "unit_id",
            "section_id",
            "clause_no",
            "status",
            "zone",
            "roles",
            "topic_tags",
        ):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def index_all(
        self,
        chunks: list[ChunkRecord],
        sections: list[ParentSectionRecord],
        sections_path: Path = SECTIONS_STORE_PATH,
        chunks_dump_path: Path = CHUNKS_DUMP_PATH,
        batch_size: int = 64,
    ) -> int:
        if not chunks:
            return 0

        embed_texts = [c.embed_text for c in chunks]

        # 1. Fit & encode lemmatized Sparse BM25 over all chunks
        tokenized_corpus = self.sparse_bm25.fit(embed_texts)
        sparse_vectors = [
            self.sparse_bm25.encode_document_lemmas(lemmas)
            for lemmas in tokenized_corpus
        ]

        # 2. Compute local Dense embeddings (multilingual-e5-large)
        dense_vectors = self.dense_embedder.embed_passages(
            embed_texts, batch_size=DENSE_BATCH_SIZE
        )
        dense_dim = len(dense_vectors[0])

        # 3. Recreate collection & upsert points in batches
        self.recreate_collection(dense_dim=dense_dim)

        points: list[models.PointStruct] = []
        for chunk, d_vec, s_vec in zip(chunks, dense_vectors, sparse_vectors):
            points.append(
                models.PointStruct(
                    id=self.chunk_id_to_uuid(chunk.chunk_id),
                    vector={
                        "dense": d_vec,
                        "bm25": s_vec,
                    },
                    payload=chunk.to_dict(),
                )
            )

        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

        # 4. Save parent sections store & chunks dump for Parent-Child expansion and auditing
        sections_path.parent.mkdir(parents=True, exist_ok=True)
        sec_map = {s.section_id: asdict(s) for s in sections}
        sections_path.write_text(
            json.dumps(sec_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        chunks_dump_path.write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return len(points)
