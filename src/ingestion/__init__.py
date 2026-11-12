"""Module 1: Document Ingestion & Hybrid OCR Engine (`parsers.py` & `ocr_engine.py`)."""

from __future__ import annotations

from src.ingestion.ocr_engine import (
    HybridOCREngine,
    get_ocr_engine,
    sanitize_ocr_text,
)
from src.ingestion.parsers import (
    DocumentParser,
    WordNumberingResolver,
    parse_document,
    parse_docx,
    parse_file,
    parse_pdf,
    parse_xlsx,
)

__all__ = [
    "DocumentParser",
    "HybridOCREngine",
    "WordNumberingResolver",
    "get_ocr_engine",
    "parse_document",
    "parse_docx",
    "parse_file",
    "parse_pdf",
    "parse_xlsx",
    "sanitize_ocr_text",
]
