from __future__ import annotations

import io

from docx import Document

from src.debug.pipeline import process_uploaded_document


def _docx_bytes() -> bytes:
    doc = Document()
    doc.add_paragraph("РЕГЛАМЕНТ тестового документа")
    doc.add_paragraph("1. Общие положения")
    doc.add_paragraph("1.1. Настоящий тестовый пункт описывает порядок действий.")
    doc.add_paragraph("2. Порядок работы")
    doc.add_paragraph("2.1. Сотрудник подаёт заявку в подразделение.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_debug_pipeline_returns_ocr_blocks_and_chunks() -> None:
    result = process_uploaded_document("test_debug.docx", _docx_bytes())

    assert result["file_name"] == "test_debug.docx"
    assert result["stats"]["blocks"] > 0
    assert result["stats"]["chunks"] > 0
    assert result["stats"]["sections"] >= 1
    assert result["stats"]["ocr_errors"] == []

    first = result["chunks"][0]
    assert first["raw_text"]
    assert first["embed_text"]
    assert "doc_id" in first

    assert any(b["block_type"] == "paragraph" for b in result["blocks"])
    assert any(s["text"] for s in result["sections"])
