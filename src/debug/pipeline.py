from __future__ import annotations

"""Разбор загруженного файла для отладочной формы.

Модуль намеренно не зависит от Streamlit: его можно вызывать из тестов,
CLI или другого интерфейса.
"""

import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from src.chunking.glossary import GlossaryManager
from src.chunking.hierarchical_chunker import HierarchicalChunker
from src.config import CHUNK_OVERLAP_RATIO, DEBUG_TMP_DIR, MIN_CLAUSE_CHARS
from src.ingestion.ocr_engine import OcrEngine
from src.ingestion.parsers import DocumentIngestionRouter
from src.ingestion.registry import DocumentRegistry


def process_uploaded_document(
    file_name: str,
    file_bytes: bytes,
    *,
    registry: DocumentRegistry | None = None,
    ocr_engine: OcrEngine | None = None,
    glossary: GlossaryManager | None = None,
    chunker: HierarchicalChunker | None = None,
) -> dict:
    """Разобрать загруженный PDF/DOCX/XLSX в блоки, чанки и разделы.

    Возвращает JSON-сериализуемый словарь, пригодный для показа в UI и
    скачивания. OCR-ошибки не скрываются: они попадают в `ocr_errors`.
    """
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".docx", ".xlsx"}:
        raise ValueError(
            f"Неподдерживаемый формат {suffix!r}. Ожидается PDF, DOCX или XLSX."
        )

    safe_name = Path(file_name).name or f"upload{suffix}"
    registry = registry or DocumentRegistry()
    ocr_engine = ocr_engine or OcrEngine()
    glossary = glossary or GlossaryManager()
    chunker = chunker or HierarchicalChunker(glossary=glossary)

    DEBUG_TMP_DIR.mkdir(parents=True, exist_ok=True)
    # Не используем tempfile.mkdtemp: в некоторых песочницах Windows созданный
    # им каталог получает ACL, запрещающий запись. Обычный подкаталог работает.
    tmp_root = DEBUG_TMP_DIR / f"run_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=False)
    tmp_path = tmp_root / safe_name
    tmp_path.write_bytes(file_bytes)

    try:
        router = DocumentIngestionRouter(registry=registry, ocr_engine=ocr_engine)
        meta, blocks = router.parse_file(tmp_path)
        chunks, sections = chunker.chunk_document(meta, blocks)
    finally:
        # Файл нужен только на время разбора; метаданные уже содержат имя.
        shutil.rmtree(tmp_root, ignore_errors=True)

    non_meta_lengths = [
        len(c.raw_text) for c in chunks if c.zone != "meta"
    ] or [len(c.raw_text) for c in chunks]

    return {
        "file_name": safe_name,
        "metadata": asdict(meta),
        "blocks": [asdict(b) for b in blocks],
        "chunks": [c.to_dict() for c in chunks],
        "sections": [asdict(s) for s in sections],
        "stats": {
            "blocks": len(blocks),
            "chunks": len(chunks),
            "sections": len(sections),
            "min_chunk_chars": min(non_meta_lengths, default=0),
            "max_chunk_chars": max(non_meta_lengths, default=0),
            "min_clause_chars_target": chunker.min_clause_chars,
            "overlap_ratio": chunker.overlap_ratio,
            "ocr_errors": list(ocr_engine.api_errors),
        },
    }


def process_uploaded_path(file_path: Path) -> dict:
    """Удобная обёртка для CLI/тестов: прочитать файл и разобрать."""
    return process_uploaded_document(
        file_name=file_path.name,
        file_bytes=file_path.read_bytes(),
    )
