from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.glossary import GlossaryManager
from src.chunking.hierarchical_chunker import (
    ChunkRecord,
    HierarchicalChunker,
    ParentSectionRecord,
)
from src.config import DATA_DIR, DENSE_MODEL_NAME, QDRANT_COLLECTION_NAME, QDRANT_URL
from src.indexing.embedder import LocalDenseEmbedder, LocalSparseBM25
from src.indexing.qdrant_indexer import QdrantIndexer
from src.ingestion.ocr_engine import OcrEngine
from src.ingestion.parsers import DocumentIngestionRouter
from src.ingestion.registry import DocumentRegistry


def run_pipeline(data_dir: Path = DATA_DIR) -> dict[str, int]:
    t0 = time.time()
    registry = DocumentRegistry()
    ocr_engine = OcrEngine()
    router = DocumentIngestionRouter(registry=registry, ocr_engine=ocr_engine)
    glossary = GlossaryManager()
    chunker = HierarchicalChunker(glossary=glossary)

    files = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".pdf", ".docx", ".xlsx", ".xls")
    )
    print(f"=== Старт индексации корпуса: найдено {len(files)} файлов в {data_dir} ===")

    all_chunks: list[ChunkRecord] = []
    all_sections: list[ParentSectionRecord] = []
    per_doc_counts: dict[str, int] = {}

    for idx, fpath in enumerate(files, start=1):
        ft0 = time.time()
        meta, blocks = router.parse_file(fpath)
        doc_chunks, doc_sections = chunker.chunk_document(meta, blocks)
        all_chunks.extend(doc_chunks)
        all_sections.extend(doc_sections)
        per_doc_counts[fpath.name] = len(doc_chunks)
        print(
            f"[{idx:02d}/{len(files):02d}] {fpath.name}\n"
            f"       doc_id={meta.doc_id} | status={meta.status} | "
            f"blocks={len(blocks)} -> chunks={len(doc_chunks)} | sections={len(doc_sections)} "
            f"({time.time() - ft0:.1f}s)"
        )

    glossary.save()
    print(
        f"\nВсего сформировано: {len(all_chunks)} чанков и {len(all_sections)} родительских разделов."
    )
    if ocr_engine.api_errors:
        print(f"Предупреждения OCR API ({len(ocr_engine.api_errors)}):")
        for err in ocr_engine.api_errors[:10]:
            print(f"  - {err}")
    else:
        print("OCR.space API: 0 ошибок (все сканированные страницы успешно распознаны и закэшированы).")

    print(f"\nЗагрузка локальной модели эмбеддингов ({DENSE_MODEL_NAME}) и расчёт BM25...")
    dense_embedder = LocalDenseEmbedder(model_name=DENSE_MODEL_NAME)
    sparse_bm25 = LocalSparseBM25(glossary=glossary)
    indexer = QdrantIndexer(
        qdrant_url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION_NAME,
        dense_embedder=dense_embedder,
        sparse_bm25=sparse_bm25,
    )

    indexed_count = indexer.index_all(all_chunks, all_sections)
    elapsed = time.time() - t0
    print(
        f"\n=== Индексация в Qdrant ({QDRANT_URL}, коллекция '{QDRANT_COLLECTION_NAME}') завершена ==="
    )
    print(f"Загружено точек: {indexed_count} | Общее время: {elapsed:.1f} сек.")
    return per_doc_counts


if __name__ == "__main__":
    run_pipeline()
