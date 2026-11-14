"""Comprehensive test suite for API duplicate protection (#1), DOCX/DOC visual A4 rendering (#3),
structured 3-section H3 RAG prompt (#5), and unified sysadmin error stub (#7).
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import docx
from fastapi.testclient import TestClient
import httpx
import pymupdf as fitz
import pytest

from src.config import DOCS_DIR
from src.retrieval.docx_preview import (
    compute_bytes_sha256,
    ensure_word_preview_pdf,
    render_preview_pdf_page_png,
)
from src.retrieval.rag_agent import (
    META_ASSISTANT_SYSTEM_PROMPT_RU,
    OUT_OF_SCOPE_REFUSAL_RU,
    SYSTEM_PROMPT_RU,
    TECH_ERROR_STUB_RU,
    TableAwareRAGAgent,
)
from src.schemas import (
    DocumentMeta,
    DocumentStatus,
    IndexingFileStatus,
    SearchHit,
    SearchResponse,
)
from tests.test_doc_xls_and_viewer import build_genuine_biff8_xls, build_genuine_word97_ole2_doc

web_app = importlib.import_module("src.web.app")


EXPECTED_TECH_STUB = (
    "Возникли технические неполадки при обработке запроса. "
    "Пожалуйста, повторите попытку позже или обратитесь к системному администратору."
)


def _build_rich_official_docx_bytes() -> bytes:
    """Create a multi-page official MIPT-style .docx with 1x2 requisite header, bold/italic, data table, and stamp."""
    d = docx.Document()
    p_hdr = d.add_paragraph("Министерство науки и высшего образования Российской Федерации")
    p_hdr.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER

    p_uni = d.add_paragraph()
    p_uni.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    r_uni = p_uni.add_run("«Московский физико-технический институт (национальный исследовательский университет)»")
    r_uni.bold = True

    p_kind = d.add_paragraph()
    p_kind.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    r_kind = p_kind.add_run("ПРИКАЗ")
    r_kind.bold = True

    # 1x2 borderless requisite header table
    req_tbl = d.add_table(rows=1, cols=2)
    req_tbl.rows[0].cells[0].text = "от 14.11.2026"
    req_tbl.rows[0].cells[1].text = "№ 8810-1"

    p_subj = d.add_paragraph()
    r_subj = p_subj.add_run("Об утверждении порядка тестовой проверки визуального предпросмотра документов")
    r_subj.bold = True

    p_c1 = d.add_paragraph(
        "1. Установить срок подачи служебной записки через систему 1С:Документооборот "
        "не менее чем за 14 календарных дней до начала мероприятия."
    )
    r_ital = p_c1.add_run(" (ответственное подразделение — ФЭУ).")
    r_ital.italic = True

    # Data table with multi-row grid
    data_tbl = d.add_table(rows=3, cols=3)
    headers = ["Код ЦФО", "Подразделение", "Норматив, руб."]
    for i, h in enumerate(headers):
        data_tbl.rows[0].cells[i].text = h
    data_tbl.rows[1].cells[0].text = "ЦФО-101"
    data_tbl.rows[1].cells[1].text = "Лаборатория прикладной фотоники"
    data_tbl.rows[1].cells[2].text = "350 000"
    data_tbl.rows[2].cells[0].text = "ЦФО-204"
    data_tbl.rows[2].cells[1].text = "Центр квантовых вычислений"
    data_tbl.rows[2].cells[2].text = "520 000"

    d.add_page_break()

    d.add_paragraph(
        "2. Специальная проверочная фраза на второй странице документа: "
        "оригиналы отчетов передаются в комнату 312 Главного корпуса в течение трех рабочих дней."
    )
    d.add_paragraph("Зарегистрировано в ЭДО № ЭДО-9941 от 14.11.2026")

    sig_tbl = d.add_table(rows=1, cols=2)
    sig_tbl.rows[0].cells[0].text = "Ректор"
    sig_tbl.rows[0].cells[1].text = "Д.В. Ливанов"

    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _build_rtf_doc_bytes() -> bytes:
    """Create a valid RTF `.doc` with Cyrillic CP1251 escapes (`\\'xx`)."""
    def _rtf_esc(text: str) -> str:
        out: list[str] = []
        for ch in text:
            code = ord(ch)
            if code < 128 and ch not in "\\{}":
                out.append(ch)
            else:
                b = ch.encode("cp1251", errors="replace")
                out.extend(f"\\'{byte:02x}" for byte in b)
        return "".join(out)

    p1 = _rtf_esc("ПРИКАЗ от 20.05.2026 № 712-1")
    p2 = _rtf_esc("1. Утвердить регламент хранения архивных экземпляров договоров в отделе документационного обеспечения.")
    rtf = f"{{\\rtf1\\ansi\\ansicpg1251 {p1}\\par {p2}\\par}}"
    return rtf.encode("ascii")


# =============================================================================
# 1. Item #1 — Duplicate Upload Rejection (HTTP 409) & Content-Dedup Task State
# =============================================================================


class TestDuplicateUploadProtection:
    """Verifies SHA-256 409 Conflict on upload & upload-url, `force=true` bypass, and content-dedup task cleanup."""

    def test_upload_sha256_duplicate_returns_409_without_saving_file_unless_forced(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        uploads_dir = tmp_path / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(web_app, "UPLOADS_DIR", uploads_dir)

        docx_bytes = _build_rich_official_docx_bytes()
        sha = compute_bytes_sha256(docx_bytes)

        existing_meta = DocumentMeta(
            doc_id="order_8810_1",
            file_name="Приказ N 8810-1 от 14.11.2026.docx",
            file_path=str(tmp_path / "Приказ N 8810-1 от 14.11.2026.docx"),
            file_format="docx",
            sha256=sha,
            title="Приказ № 8810-1 от 14.11.2026",
            status=DocumentStatus.ACTIVE,
            chunk_count=6,
        )

        fake_pipeline = MagicMock()
        fake_pipeline.registry = {"order_8810_1": existing_meta}
        fake_pipeline.index_file.return_value = IndexingFileStatus(
            file_name="Копия_приказа.docx",
            doc_id="order_8810_1",
            sha256=sha,
            status="success",
            chunks_created=6,
        )
        monkeypatch.setattr(web_app, "get_pipeline", lambda: fake_pipeline)

        client = TestClient(web_app.app)

        # 1) Exact SHA-256 duplicate without force -> HTTP 409 Conflict, file NOT written to UPLOADS_DIR
        resp_dup = client.post(
            "/api/documents/upload",
            files=[
                (
                    "files",
                    (
                        "Копия_приказа.docx",
                        docx_bytes,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                )
            ],
            data={"async_task": "false"},
        )
        assert resp_dup.status_code == 409
        body = resp_dup.json()
        assert body["status"] == "duplicate"
        assert body["duplicate_of"] == "order_8810_1"
        assert body["existing_title"] == "Приказ № 8810-1 от 14.11.2026"
        assert body["existing_file_name"] == "Приказ N 8810-1 от 14.11.2026.docx"
        assert (
            body["detail"]
            == "Дубликат документа: в базе уже загружен «Приказ № 8810-1 от 14.11.2026» (Приказ N 8810-1 от 14.11.2026.docx)"
        )
        assert list(uploads_dir.iterdir()) == []
        assert fake_pipeline.index_file.call_count == 0

        # 2) Explicit query param `?force=true` -> allowed, saved, and indexed with force=True
        resp_forced = client.post(
            "/api/documents/upload?force=true",
            files=[
                (
                    "files",
                    (
                        "Копия_приказа.docx",
                        docx_bytes,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                )
            ],
            data={"async_task": "false"},
        )
        assert resp_forced.status_code == 200
        assert resp_forced.json()["status"] == "completed"
        assert (uploads_dir / "Копия_приказа.docx").is_file()
        assert fake_pipeline.index_file.call_count == 1
        assert fake_pipeline.index_file.call_args.kwargs.get("force") is True

    def test_upload_url_sha256_duplicate_returns_409_without_saving_file_unless_forced(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        uploads_dir = tmp_path / "uploads_url"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(web_app, "UPLOADS_DIR", uploads_dir)
        monkeypatch.setattr(web_app, "_validate_public_http_url", lambda url: None)

        docx_bytes = _build_rich_official_docx_bytes()
        sha = compute_bytes_sha256(docx_bytes)

        existing_meta = DocumentMeta(
            doc_id="order_8810_1",
            file_name="Приказ N 8810-1 от 14.11.2026.docx",
            file_format="docx",
            sha256=sha,
            title="Приказ № 8810-1 от 14.11.2026",
            status=DocumentStatus.ACTIVE,
            chunk_count=5,
        )
        fake_pipeline = MagicMock()
        fake_pipeline.registry = {"order_8810_1": existing_meta}
        fake_pipeline.index_file.return_value = IndexingFileStatus(
            file_name="remote_order.docx",
            doc_id="order_8810_1",
            sha256=sha,
            status="success",
            chunks_created=5,
        )
        monkeypatch.setattr(web_app, "get_pipeline", lambda: fake_pipeline)

        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                content=docx_bytes,
                headers={
                    "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                },
            )
        )
        monkeypatch.setattr(web_app, "_create_url_download_client", lambda: httpx.Client(transport=transport))

        client = TestClient(web_app.app)

        # 1) Duplicate URL download -> HTTP 409 Conflict, nothing saved to UPLOADS_DIR
        r_dup = client.post(
            "/api/documents/upload-url",
            json={"url": "https://docs.example.org/files/remote_order.docx", "async_task": False},
        )
        assert r_dup.status_code == 409
        payload = r_dup.json()
        assert payload["status"] == "duplicate"
        assert payload["duplicate_of"] == "order_8810_1"
        assert payload["existing_title"] == "Приказ № 8810-1 от 14.11.2026"
        assert payload["existing_file_name"] == "Приказ N 8810-1 от 14.11.2026.docx"
        assert "Дубликат документа: в базе уже загружен" in payload["detail"]
        assert list(uploads_dir.iterdir()) == []
        assert fake_pipeline.index_file.call_count == 0

        # 2) `?force=true` bypasses 409
        r_ok = client.post(
            "/api/documents/upload-url?force=true",
            json={"url": "https://docs.example.org/files/remote_order.docx", "async_task": False},
        )
        assert r_ok.status_code == 200
        assert r_ok.json()["status"] == "completed"
        assert (uploads_dir / "remote_order.docx").is_file()

    @pytest.mark.parametrize("return_as_dict", [True, False])
    def test_content_duplicate_in_background_task_unlinks_upload_and_sets_duplicate_task_status(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        return_as_dict: bool,
    ) -> None:
        uploads_dir = tmp_path / f"uploads_cd_{return_as_dict}"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(web_app, "UPLOADS_DIR", uploads_dir)

        # Different binary bytes (so SHA-256 differs from existing doc), triggering content-overlap duplicate in index_file
        re_saved_bytes = _build_rich_official_docx_bytes() + b"\x00"
        fake_pipeline = MagicMock()
        fake_pipeline.registry = {}

        dup_msg = "Дубликат по тексту (96% совпадения): в базе уже есть «Приказ № 3065-1»"
        if return_as_dict:
            fake_pipeline.index_file.return_value = {
                "status": "duplicate",
                "duplicate_of": "order_3065_1",
                "existing_title": "Приказ № 3065-1",
                "existing_file_name": "Приказ N 3065-1 от 25.11.2021.docx",
                "message": dup_msg,
            }
        else:
            fake_pipeline.index_file.return_value = IndexingFileStatus(
                file_name="Пересохраненный_3065.docx",
                status="duplicate",
                duplicate_of="order_3065_1",
                existing_title="Приказ № 3065-1",
                existing_file_name="Приказ N 3065-1 от 25.11.2021.docx",
                message=dup_msg,
            )
        monkeypatch.setattr(web_app, "get_pipeline", lambda: fake_pipeline)

        client = TestClient(web_app.app)
        resp = client.post(
            "/api/documents/upload",
            files=[
                (
                    "files",
                    (
                        "Пересохраненный_3065.docx",
                        re_saved_bytes,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                )
            ],
            data={"async_task": "false"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "duplicate"
        assert data["duplicate_of"] == "order_3065_1"
        assert data["existing_title"] == "Приказ № 3065-1"
        assert data["existing_file_name"] == "Приказ N 3065-1 от 25.11.2021.docx"
        assert data["message"] == dup_msg

        # Newly uploaded duplicate file MUST be unlinked from UPLOADS_DIR
        assert not (uploads_dir / "Пересохраненный_3065.docx").exists()

        # Task snapshot endpoint also exposes `status == "duplicate"` and metadata
        task_snap = client.get(f"/api/tasks/{data['task_id']}").json()
        assert task_snap["status"] == "duplicate"
        assert task_snap["duplicate_of"] == "order_3065_1"
        assert task_snap["existing_title"] == "Приказ № 3065-1"
        assert task_snap["message"] == dup_msg


# =============================================================================
# 2. Item #3 — Visual A4 Rendering for DOCX & DOC (`/preview.pdf`, `/pages/{n}.png`, `/view`)
# =============================================================================


class TestWordVisualDocumentPreviewAndViewer:
    """Verifies high-fidelity A4 PDF & PNG page rendering for real and synthetic `.docx` and `.doc` files."""

    def test_real_mipt_docx_files_render_clean_a4_pdf_and_crisp_png_pages(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "preview_cache"
        candidates = [
            DOCS_DIR / "orders" / "Приказ N 3065-1 от 25.11.2021.docx",
            DOCS_DIR / "orders" / "Приказ N 3699-1 от 18.08.2025.docx",
            DOCS_DIR / "appendices_and_forms" / "Приложение 1а Бланк сметы.docx",
        ]
        existing_real = [p for p in candidates if p.is_file()]
        assert existing_real, "Expected real .docx files in documents/"

        for docx_path in existing_real:
            pdf_path = ensure_word_preview_pdf(docx_path, cache_dir=cache_dir)
            assert pdf_path.is_file()
            assert pdf_path.read_bytes()[:5] == b"%PDF-"

            # Second call hits cache without rebuilding
            mtime_before = pdf_path.stat().st_mtime_ns
            pdf_path_cached = ensure_word_preview_pdf(docx_path, cache_dir=cache_dir)
            assert pdf_path_cached == pdf_path
            assert pdf_path_cached.stat().st_mtime_ns == mtime_before

            with fitz.open(str(pdf_path)) as doc:
                assert len(doc) >= 2
                page1_text = doc[0].get_text("text")
                assert len(page1_text.strip()) >= 80
                assert "\ufffd" not in page1_text  # zero tofu / replacement glyphs

            png_bytes = render_preview_pdf_page_png(pdf_path, page_number=1, dpi=150)
            assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
            pix = fitz.Pixmap(png_bytes)
            assert pix.width > 700
            assert pix.height > 700
            # Verify non-blank visual ink pixels exist on rendered A4 page
            samples = pix.samples
            non_white = sum(1 for i in range(0, len(samples), 3 * 17) if samples[i] < 220 or samples[i + 1] < 220)
            assert non_white >= 50

    def test_viewer_endpoints_support_docx_and_ole2_and_rtf_doc_with_highlights(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        preview_cache = tmp_path / "preview_cache"
        monkeypatch.setattr("src.retrieval.docx_preview.PREVIEW_CACHE_DIR", preview_cache)

        docx_file = tmp_path / "Приказ_8810.docx"
        docx_file.write_bytes(_build_rich_official_docx_bytes())

        ole_doc_file = tmp_path / "Приказ_4512_OLE2.doc"
        ole_doc_file.write_bytes(build_genuine_word97_ole2_doc())

        rtf_doc_file = tmp_path / "Приказ_712_RTF.doc"
        rtf_doc_file.write_bytes(_build_rtf_doc_bytes())

        xls_file = tmp_path / "Смета_тест.xls"
        xls_file.write_bytes(build_genuine_biff8_xls())

        registry_docs = {
            "doc_docx": DocumentMeta(
                doc_id="doc_docx",
                file_name=docx_file.name,
                file_path=str(docx_file),
                file_format="docx",
                sha256=hashlib.sha256(docx_file.read_bytes()).hexdigest(),
                title="Приказ № 8810-1",
                doc_number="8810-1",
                doc_date="2026-11-14",
            ),
            "doc_ole": DocumentMeta(
                doc_id="doc_ole",
                file_name=ole_doc_file.name,
                file_path=str(ole_doc_file),
                file_format="doc",
                sha256=hashlib.sha256(ole_doc_file.read_bytes()).hexdigest(),
                title="Приказ № 4512-1",
                doc_number="4512-1",
                doc_date="2026-04-15",
            ),
            "doc_rtf": DocumentMeta(
                doc_id="doc_rtf",
                file_name=rtf_doc_file.name,
                file_path=str(rtf_doc_file),
                file_format="doc",
                sha256=hashlib.sha256(rtf_doc_file.read_bytes()).hexdigest(),
                title="Приказ № 712-1",
                doc_number="712-1",
                doc_date="2026-05-20",
            ),
            "doc_xls": DocumentMeta(
                doc_id="doc_xls",
                file_name=xls_file.name,
                file_path=str(xls_file),
                file_format="xls",
                sha256=hashlib.sha256(xls_file.read_bytes()).hexdigest(),
                title="Смета тест",
            ),
        }

        fake_pipeline = MagicMock()
        fake_pipeline.registry = registry_docs
        fake_pipeline.sections_store = {}
        fake_pipeline.get_document.side_effect = lambda d_id: registry_docs.get(d_id)
        fake_pipeline.get_tables.return_value = []
        monkeypatch.setattr(web_app, "get_pipeline", lambda: fake_pipeline)

        client = TestClient(web_app.app)

        for doc_id, expected_fmt in (("doc_docx", "docx"), ("doc_ole", "doc"), ("doc_rtf", "doc")):
            # 1) GET /preview.pdf serves inline PDF
            r_pdf = client.get(f"/api/documents/{doc_id}/preview.pdf")
            assert r_pdf.status_code == 200
            assert r_pdf.headers["content-type"].startswith("application/pdf")
            assert "inline" in r_pdf.headers.get("content-disposition", "")
            assert r_pdf.content.startswith(b"%PDF-")

            # 2) GET /pages/1.png renders crisp A4 PNG at dpi=150
            r_png = client.get(f"/api/documents/{doc_id}/pages/1.png")
            assert r_png.status_code == 200
            assert r_png.headers["content-type"] == "image/png"
            pix = fitz.Pixmap(r_png.content)
            assert pix.width > 700
            assert pix.height > 900

            # Out-of-range page -> 404
            assert client.get(f"/api/documents/{doc_id}/pages/999.png").status_code == 404

            # 3) GET /view exposes visual A4 page metadata
            r_view = client.get(f"/api/documents/{doc_id}/view")
            assert r_view.status_code == 200
            v = r_view.json()
            assert v["format"] == expected_fmt
            assert v["file_format"] == expected_fmt
            assert v["has_page_images"] is True
            assert v["preview_pdf_url"] == f"/api/documents/{doc_id}/preview.pdf"
            assert v["preview_pages_total"] >= 1
            assert v["total_pages"] == v["preview_pages_total"]
            assert v["pages"][0]["image_url"] == f"/api/documents/{doc_id}/pages/1.png"
            assert v["pages"][0]["width"] > 500
            assert v["pages"][0]["height"] > 700

        # 4) Quote lookup on page 2 of multi-page .docx resolves target_page == 2 AND normalized highlight boxes!
        r_quote = client.get(
            "/api/documents/doc_docx/view",
            params={
                "page": 99,  # out-of-bounds page hint is safely clamped and overridden by quote location
                "quote": "оригиналы отчетов передаются в комнату 312 Главного корпуса в течение трех рабочих дней",
            },
        )
        assert r_quote.status_code == 200
        vq = r_quote.json()
        assert vq["target_page"] == 2
        assert vq["matched_quote_on_page"] is True
        assert len(vq["pages"][1]["highlights"]) >= 1
        hl0 = vq["pages"][1]["highlights"][0]
        assert 0.0 <= hl0["x"] <= 1.0 and 0.0 <= hl0["y"] <= 1.0
        assert hl0["w"] > 0.01 and hl0["h"] > 0.005

        # 5) Excel (.xls) keeps `has_page_images == False` and returns 404 on `/preview.pdf` & `/pages/1.png`
        assert client.get("/api/documents/doc_xls/preview.pdf").status_code == 404
        assert client.get("/api/documents/doc_xls/pages/1.png").status_code == 404
        v_xls = client.get("/api/documents/doc_xls/view").json()
        assert v_xls["has_page_images"] is False
        assert v_xls["preview_pdf_url"] is None
        assert v_xls["preview_pages_total"] == 0


# =============================================================================
# 3. Item #5 — Structured Normative H3 Prompt (`src/retrieval/rag_agent.py`)
# =============================================================================


class TestStructuredH3PromptContract:
    """Verifies exact H3 headings (`### Краткий ответ`, conditional `### Порядок действий`, `### Подробный ответ и цитаты`)."""

    def test_system_prompt_enforces_three_h3_sections_and_strict_conditional_actions(self) -> None:
        # Exact H3 headings matching frontend parser
        assert "### Краткий ответ" in SYSTEM_PROMPT_RU
        assert "### Порядок действий" in SYSTEM_PROMPT_RU
        assert "### Подробный ответ и цитаты" in SYSTEM_PROMPT_RU

        # Conditional inclusion & explicit exclusion of `### Порядок действий`
        assert "СТРОГО УСЛОВНЫЙ РАЗДЕЛ" in SYSTEM_PROMPT_RU
        assert "ПОЛНОСТЬЮ ОПУСКАЙ заголовок `### Порядок действий`" in SYSTEM_PROMPT_RU
        for procedural_kw in ("как оформить", "как согласовать", "как подать", "какой порядок"):
            assert procedural_kw in SYSTEM_PROMPT_RU
        for non_proc_kw in ("сколько подразделений", "кто финансодержатель", "какая формула"):
            assert non_proc_kw in SYSTEM_PROMPT_RU

        # K2 version/actuality (`version_note`, `Edition Graph`) + K4 conflict warnings + `> «...»` quotes
        assert "version_note" in SYSTEM_PROMPT_RU
        assert "Edition Graph" in SYSTEM_PROMPT_RU
        assert "противоречия" in SYSTEM_PROMPT_RU or "коллизи" in SYSTEM_PROMPT_RU
        assert "> «точный текст из документа»" in SYSTEM_PROMPT_RU

        # Meta-assistant / greeting prompt omits `### Порядок действий` and `### Подробный ответ и цитаты`
        assert "НЕ добавляй разделы `### Порядок действий` и `### Подробный ответ и цитаты`" in META_ASSISTANT_SYSTEM_PROMPT_RU

    def test_build_messages_surfaces_version_note_and_h3_structure_reminder(self) -> None:
        agent = TableAwareRAGAgent(api_key="sk-test", base_url="https://api.deepseek.com", model="deepseek-chat")
        hit = SearchHit(
            chunk_id="ord_3065::c01",
            doc_id="ord_3065",
            file_name="Приказ N 3065-1.docx",
            title="Приказ № 3065-1",
            doc_number="3065-1",
            clause_no="1.2",
            page_numbers=[1],
            clean_text="1.2. Открытие ФЛС осуществляется на основании служебной записки.",
            metadata={"version_note": "Действует в редакции Распоряжения № 700 от 28.12.2024"},
            score=0.92,
        )
        sr = SearchResponse(
            query="Какой порядок открытия ФЛС?",
            hits=[hit],
            version_alerts=["Приказ № 3065-1 изменен Распоряжением № 700."],
            confidence_score=0.92,
        )
        msgs = agent._build_messages("Какой порядок открытия ФЛС?", sr)
        assert msgs[0]["content"] == SYSTEM_PROMPT_RU
        user_msg = msgs[1]["content"]
        assert "Действует в редакции Распоряжения № 700" in user_msg
        assert "### Краткий ответ" in user_msg
        assert "### Порядок действий" in user_msg
        assert "### Подробный ответ и цитаты" in user_msg


# =============================================================================
# 4. Item #7 — Unified Sysadmin Fallback Stub on Technical Errors
# =============================================================================


class TestUnifiedSysadminTechnicalErrorStub:
    """Verifies TECH_ERROR_STUB_RU across HTTP 5xx, SSE `/api/ask/stream`, LLM failures, and global middleware."""

    def test_tech_error_stub_constant_exact_text(self) -> None:
        assert TECH_ERROR_STUB_RU == EXPECTED_TECH_STUB

    def test_api_ask_qdrant_and_llm_crashes_return_only_sysadmin_stub_and_log_traceback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret_diag = "RuntimeError: Qdrant cluster token=SUPER_SECRET_TOKEN_991 connection refused at 10.0.4.12:6333"

        # Case A: Unhandled Qdrant / Search crash inside `/api/ask` -> HTTP 500 with TECH_ERROR_STUB_RU
        crashing_agent = MagicMock()
        crashing_agent.ask.side_effect = RuntimeError(secret_diag)
        monkeypatch.setattr(web_app, "_resolve_rag_agent", lambda: crashing_agent)

        client = TestClient(web_app.app)
        with caplog.at_level(logging.ERROR):
            r_500 = client.post("/api/ask", json={"query": "Какой срок подачи заявки?"})
        assert r_500.status_code == 500
        j_500 = r_500.json()
        assert j_500 == {"detail": EXPECTED_TECH_STUB, "error_type": "technical_error"}
        assert "SUPER_SECRET_TOKEN_991" not in r_500.text
        assert "RuntimeError" not in r_500.text
        assert "Traceback" not in r_500.text
        assert any("SUPER_SECRET_TOKEN_991" in rec.getMessage() or rec.exc_info for rec in caplog.records)

        # Case B: `TableAwareRAGAgent.ask()` catches upstream LLM API timeout/quota crash -> graceful AskResponse stub
        mock_search = MagicMock()
        mock_search.search.return_value = SearchResponse(
            query="Какой лимит суточных?",
            hits=[
                SearchHit(
                    chunk_id="c1",
                    doc_id="d1",
                    file_name="Приказ 3699-1.docx",
                    title="Приказ № 3699-1",
                    clean_text="Норма суточных составляет 100 рублей.",
                    score=0.91,
                )
            ],
            confidence_score=0.91,
        )
        mock_search._get_active_table_store.return_value = None

        failing_llm = MagicMock()
        failing_llm.chat.completions.create.side_effect = TimeoutError("Upstream DeepSeek 504 Gateway Timeout sk-live-secret")

        real_agent = TableAwareRAGAgent(search_service=mock_search, llm_client=failing_llm)
        monkeypatch.setattr(web_app, "_resolve_rag_agent", lambda: real_agent)

        r_graceful = client.post("/api/ask", json={"query": "Какой лимит суточных?"})
        assert r_graceful.status_code == 200
        jg = r_graceful.json()
        assert jg["answer"] == EXPECTED_TECH_STUB
        assert jg["quotes"] == []
        assert "sk-live-secret" not in r_graceful.text
        assert "TimeoutError" not in r_graceful.text

    def test_api_ask_stream_sse_emits_sanitized_error_event_on_llm_or_qdrant_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_search = MagicMock()
        mock_search.search.return_value = SearchResponse(
            query="Как оформить командировку?",
            hits=[
                SearchHit(
                    chunk_id="c1",
                    doc_id="d1",
                    file_name="Регламент.pdf",
                    title="Регламент командировок",
                    clean_text="Служебная записка подается за 5 дней.",
                    score=0.95,
                )
            ],
            confidence_score=0.95,
        )
        mock_search._get_active_table_store.return_value = None

        # Simulate mid-stream LLM crash after emitting 1 delta chunk
        def _broken_stream_gen():
            chunk1 = MagicMock()
            chunk1.model = "deepseek-chat"
            delta1 = MagicMock()
            delta1.content = "### Краткий ответ\nСлужебная записка "
            delta1.tool_calls = None
            choice1 = MagicMock()
            choice1.delta = delta1
            chunk1.choices = [choice1]
            yield chunk1
            raise ConnectionError("Polza upstream socket reset peer=10.9.0.5 key=pza_secret_key")

        failing_stream_llm = MagicMock()
        failing_stream_llm.chat.completions.create.side_effect = lambda **kw: _broken_stream_gen()

        agent = TableAwareRAGAgent(search_service=mock_search, llm_client=failing_stream_llm)
        monkeypatch.setattr(web_app, "_resolve_rag_agent", lambda: agent)

        client = TestClient(web_app.app)
        with client.stream("POST", "/api/ask/stream", json={"query": "Как оформить командировку?"}) as resp:
            assert resp.status_code == 200
            raw_sse = "".join(resp.iter_text())

        assert "event: error" in raw_sse
        assert "ConnectionError" not in raw_sse
        assert "pza_secret_key" not in raw_sse

        # Parse the `event: error` JSON payload
        error_payloads: list[dict[str, Any]] = []
        blocks = [b.strip() for b in raw_sse.split("\n\n") if b.strip()]
        for blk in blocks:
            if blk.startswith("event: error"):
                data_line = next(ln for ln in blk.splitlines() if ln.startswith("data: "))
                error_payloads.append(json.loads(data_line[len("data: ") :]))
        assert len(error_payloads) == 1
        assert error_payloads[0]["error"] == EXPECTED_TECH_STUB
        assert error_payloads[0]["message"] == EXPECTED_TECH_STUB
        assert error_payloads[0]["error_type"] == "technical_error"

    def test_search_and_global_unhandled_exceptions_sanitized_while_4xx_validation_preserved(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 1) `/api/search` Qdrant crash -> 500 with TECH_ERROR_STUB_RU
        broken_search = MagicMock()
        broken_search.search.side_effect = RuntimeError("Qdrant internal gRPC crash")
        monkeypatch.setattr(web_app, "get_search_service", lambda: broken_search)

        # 2) Unhandled exception inside `/api/documents` (via `get_pipeline` raising unexpected ZeroDivisionError)
        client = TestClient(web_app.app)
        r_search_500 = client.post("/api/search", json={"query": "смета расходов"})
        assert r_search_500.status_code == 500
        assert r_search_500.json() == {"detail": EXPECTED_TECH_STUB, "error_type": "technical_error"}
        assert "gRPC" not in r_search_500.text

        monkeypatch.setattr(web_app, "get_pipeline", lambda: (_ for _ in ()).throw(ZeroDivisionError("secret math crash")))
        r_global_500 = client.get("/api/documents")
        assert r_global_500.status_code == 500
        assert r_global_500.json() == {"detail": EXPECTED_TECH_STUB, "error_type": "technical_error"}
        assert "ZeroDivisionError" not in r_global_500.text

        # 3) Specific 4xx user validation errors remain intact!
        healthy_pipeline = MagicMock()
        healthy_pipeline.registry = {}
        healthy_pipeline.get_document.return_value = None
        monkeypatch.setattr(web_app, "get_pipeline", lambda: healthy_pipeline)

        # 400 empty query on /api/ask, /api/ask/stream, /api/search
        for ep in ("/api/ask", "/api/ask/stream", "/api/search"):
            r_empty = client.post(ep, json={"query": "   "})
            assert r_empty.status_code == 400
            assert "Пустой поисковый запрос" in r_empty.json()["detail"]
            assert r_empty.json()["detail"] != EXPECTED_TECH_STUB

        # 400 unsupported file format on /api/documents/upload
        r_bad_ext = client.post(
            "/api/documents/upload",
            files=[("files", ("malware.exe", b"MZ\x90\x00", "application/octet-stream"))],
        )
        assert r_bad_ext.status_code == 400
        assert "Неподдерживаемый формат файла" in r_bad_ext.json()["detail"]

        # 404 document not found on /api/documents/{doc_id}/view
        r_404 = client.get("/api/documents/nonexistent_doc_999/view")
        assert r_404.status_code == 404
        assert "не найден" in r_404.json()["detail"]
