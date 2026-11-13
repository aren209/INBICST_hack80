"""Comprehensive unit and integration tests for URL document upload (`/api/documents/upload-url`) and conditional step-by-step plan prompt."""

from __future__ import annotations

import io
import socket
import sys
import time
from pathlib import Path
from typing import Any

import docx
from fastapi import HTTPException
from fastapi.testclient import TestClient
import httpx
import openpyxl
import pymupdf as fitz
import pytest

from src.indexing.embedder import HybridEmbedder, SparseBM25Encoder
from src.indexing.pipeline import IndexingPipeline
from src.indexing.qdrant_store import QdrantHybridStore
from src.indexing.table_store import StructuredTableStore
from src.retrieval.rag_agent import SYSTEM_PROMPT_RU, TableAwareRAGAgent
from src.schemas import SearchHit, SearchResponse
from src.web.app import (
    _download_document_from_url,
    _normalize_document_download_url,
    _sniff_document_extension,
    _validate_public_http_url,
    app,
)
from tests.test_doc_xls_and_viewer import (
    _DummyDenseEmbedder,
    build_genuine_biff8_xls,
    build_genuine_word97_ole2_doc,
)

web_app_mod = sys.modules["src.web.app"]


# -----------------------------------------------------------------------------
# Helpers for Generating Real In-Memory Documents (PDF, DOCX, XLSX) & Mock DNS
# -----------------------------------------------------------------------------


def _make_pdf_bytes(title: str = "ПРИКАЗ № 991-1 от 14.04.2026", clause: str = "1.1. Утвердить порядок подачи заявок за 5 рабочих дней.") -> bytes:
    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=595, height=842)
    page.insert_text((72, 100), title, fontsize=13)
    page.insert_text((72, 140), clause, fontsize=11)
    buf = pdf_doc.tobytes()
    pdf_doc.close()
    return buf


def _make_docx_bytes(title: str = "ПРИКАЗ № 991 от 15.04.2026", clause: str = "2.1. Служебная записка согласуется руководителем подразделения.") -> bytes:
    d = docx.Document()
    d.add_heading(title, level=1)
    d.add_paragraph(clause)
    bio = io.BytesIO()
    d.save(bio)
    return bio.getvalue()


def _make_xlsx_bytes() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Лимиты 2026"
    ws.append(["Подразделение", "Лимит, руб."])
    ws.append(["ФРКТ", 450000])
    ws.append(["ФПМИ", 520000])
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


def _install_mock_public_dns(monkeypatch: pytest.MonkeyPatch, extra_private_hosts: dict[str, str] | None = None) -> None:
    """
    Resolve synthetic public domains (e.g. `docs.example.org`, `cloud-api.yandex.net`) to public IP `93.184.216.34`
    without real network calls, while preserving real SSRF rejection on private/loopback host mappings.
    """
    private_map = dict(extra_private_hosts or {})

    def _fake_getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
        h_low = (host or "").strip().lower().rstrip(".")
        if h_low in private_map:
            target_ip = private_map[h_low]
        elif h_low in {"localhost", "metadata.google.internal"}:
            target_ip = "127.0.0.1"
        elif h_low == "nonexistent.invalid":
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        else:
            target_ip = "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (target_ip, int(port or 443)))]

    monkeypatch.setattr(web_app_mod.socket, "getaddrinfo", _fake_getaddrinfo)


def _build_isolated_pipeline(tmp_path: Path, col_name: str = "test_url_upload_col") -> IndexingPipeline:
    qdrant_mem = QdrantHybridStore(in_memory=True, collection_name=col_name)
    table_store = StructuredTableStore(db_path=tmp_path / f"{col_name}_tables.db")
    embedder = HybridEmbedder(dense_embedder=_DummyDenseEmbedder(), sparse_encoder=SparseBM25Encoder())
    return IndexingPipeline(
        qdrant_store=qdrant_mem,
        embedder=embedder,
        table_store=table_store,
        registry_path=tmp_path / f"{col_name}_registry.json",
        sections_path=tmp_path / f"{col_name}_sections.json",
        report_path=tmp_path / f"{col_name}_report.json",
    )


# -----------------------------------------------------------------------------
# 1. Direct PDF & Multi-URL Upload End-to-End Integration Tests
# -----------------------------------------------------------------------------


class TestUrlUploadEndpointIntegration:
    def test_direct_pdf_url_upload_sync_indexes_and_appears_in_documents(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_mock_public_dns(monkeypatch)
        iso_pipeline = _build_isolated_pipeline(tmp_path, "col_pdf_sync")
        uploads_dir = tmp_path / "uploads"
        monkeypatch.setattr(web_app_mod, "get_pipeline", lambda: iso_pipeline)
        monkeypatch.setattr(web_app_mod, "UPLOADS_DIR", uploads_dir)

        pdf_bytes = _make_pdf_bytes(
            title="Order N 774-1 of 10.04.2026",
            clause="1.1. Submit travel request 7 business days prior to departure.",
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == "https://docs.example.org/normative/Prikaz_774-1.pdf":
                return httpx.Response(
                    200,
                    headers={"Content-Type": "application/pdf"},
                    content=pdf_bytes,
                )
            return httpx.Response(404, text="Not found")

        monkeypatch.setattr(
            web_app_mod,
            "_create_url_download_client",
            lambda: httpx.Client(transport=httpx.MockTransport(_handler), follow_redirects=False),
        )

        client = TestClient(app)
        resp = client.post(
            "/api/documents/upload-url",
            json={
                "url": "  https://docs.example.org/normative/Prikaz_774-1.pdf  ",
                "ocr_backend": "local_rapidocr",
                "force_reindex": True,
                "async_task": False,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] == "completed"
        assert payload["files"] == ["Prikaz_774-1.pdf"]
        assert payload["error"] is None
        assert payload["result"]["total_files"] == 1
        assert payload["result"]["files"][0]["status"] == "success"
        assert (uploads_dir / "Prikaz_774-1.pdf").is_file()

        # Verify document is listed in GET /api/documents
        docs_resp = client.get("/api/documents")
        assert docs_resp.status_code == 200
        docs_data = docs_resp.json()
        assert docs_data["total_documents"] == 1
        assert docs_data["documents"][0]["file_name"] == "Prikaz_774-1.pdf"
        assert docs_data["documents"][0]["file_format"] == "pdf"

    def test_empty_url_request_returns_400_russian_message(self) -> None:
        client = TestClient(app)
        r = client.post("/api/documents/upload-url", json={"url": "   ", "urls": []})
        assert r.status_code == 400
        assert "Не указана ссылка на документ" in r.json()["detail"]

    def test_async_url_upload_and_task_polling(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_mock_public_dns(monkeypatch)
        iso_pipeline = _build_isolated_pipeline(tmp_path, "col_docx_async")
        uploads_dir = tmp_path / "uploads_async"
        monkeypatch.setattr(web_app_mod, "get_pipeline", lambda: iso_pipeline)
        monkeypatch.setattr(web_app_mod, "UPLOADS_DIR", uploads_dir)

        docx_bytes = _make_docx_bytes()

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                },
                content=docx_bytes,
            )

        monkeypatch.setattr(
            web_app_mod,
            "_create_url_download_client",
            lambda: httpx.Client(transport=httpx.MockTransport(_handler), follow_redirects=False),
        )

        client = TestClient(app)
        resp = client.post(
            "/api/documents/upload-url",
            json={
                "url": "https://docs.example.org/files/Reglament_2026.docx",
                "async_task": True,
                "force_reindex": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["files"] == ["Reglament_2026.docx"]
        assert "фоновом режиме" in body["message"]

        # Poll /api/tasks/{task_id} until completed
        task_id = body["task_id"]
        final_task = None
        for _ in range(50):
            t_resp = client.get(f"/api/tasks/{task_id}")
            assert t_resp.status_code == 200
            t_json = t_resp.json()
            if t_json["status"] in ("completed", "failed"):
                final_task = t_json
                break
            time.sleep(0.05)

        assert final_task is not None
        assert final_task["status"] == "completed"
        assert final_task["percent"] == 100


# -----------------------------------------------------------------------------
# 2. Extensionless URLs, RFC 5987 Content-Disposition & Magic-Byte Sniffing
# -----------------------------------------------------------------------------


class TestFilenameAndMagicByteDetection:
    def test_extensionless_url_with_rfc5987_content_disposition_uploads_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_mock_public_dns(monkeypatch)
        iso_pipeline = _build_isolated_pipeline(tmp_path, "col_cd_utf8")
        uploads_dir = tmp_path / "uploads_cd"
        monkeypatch.setattr(web_app_mod, "get_pipeline", lambda: iso_pipeline)
        monkeypatch.setattr(web_app_mod, "UPLOADS_DIR", uploads_dir)

        docx_bytes = _make_docx_bytes(
            title="ПРИКАЗ № 991 от 15.04.2026",
            clause="3.2. Заявка на закупку оборудования подается через систему 1С.",
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "https://portal.example.org/download?id=991"
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": "attachment; filename*=UTF-8''%D0%9F%D1%80%D0%B8%D0%BA%D0%B0%D0%B7%20991.docx",
                },
                content=docx_bytes,
            )

        monkeypatch.setattr(
            web_app_mod,
            "_create_url_download_client",
            lambda: httpx.Client(transport=httpx.MockTransport(_handler), follow_redirects=False),
        )

        client = TestClient(app)
        resp = client.post(
            "/api/documents/upload-url",
            json={
                "url": "https://portal.example.org/download?id=991",
                "force_reindex": True,
                "async_task": False,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "completed"
        assert data["files"] == ["Приказ 991.docx"]
        assert (uploads_dir / "Приказ 991.docx").is_file()

        docs = client.get("/api/documents").json()["documents"]
        assert any(d["file_name"] == "Приказ 991.docx" and d["file_format"] == "docx" for d in docs)

    def test_magic_byte_sniffing_for_pdf_docx_xlsx_doc_xls_and_rtf(self) -> None:
        assert _sniff_document_extension(_make_pdf_bytes(), "application/octet-stream") == ".pdf"
        assert _sniff_document_extension(_make_docx_bytes(), "application/octet-stream") == ".docx"
        assert _sniff_document_extension(_make_xlsx_bytes(), "application/octet-stream") == ".xlsx"
        assert _sniff_document_extension(build_genuine_word97_ole2_doc(), "application/octet-stream") == ".doc"
        assert _sniff_document_extension(build_genuine_biff8_xls(), "application/octet-stream") == ".xls"
        assert _sniff_document_extension(b"{\\rtf1\\ansi Hello}", "application/octet-stream") == ".doc"

        # Non-office ZIP archive must NOT be misclassified as docx/xlsx
        plain_zip_bio = io.BytesIO()
        import zipfile

        with zipfile.ZipFile(plain_zip_bio, "w") as zf:
            zf.writestr("readme.txt", "just a zip archive")
        assert _sniff_document_extension(plain_zip_bio.getvalue(), "application/zip") is None

    def test_extensionless_urls_sniff_all_five_formats_and_preserve_explicit_date_filename(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_mock_public_dns(monkeypatch)
        payloads: dict[str, bytes] = {
            "/dl/pdf": _make_pdf_bytes(),
            "/dl/docx": _make_docx_bytes(),
            "/dl/xlsx": _make_xlsx_bytes(),
            "/dl/doc": build_genuine_word97_ole2_doc(),
            "/dl/xls": build_genuine_biff8_xls(),
            "/dl/exe": b"MZ\x90\x00\x03\x00\x00\x00",
        }

        def _handler(req: httpx.Request) -> httpx.Response:
            body = payloads.get(req.url.path)
            if body is None:
                return httpx.Response(404)
            return httpx.Response(200, headers={"Content-Type": "application/octet-stream"}, content=body)

        mock_client = httpx.Client(transport=httpx.MockTransport(_handler), follow_redirects=False)

        fn_pdf, _ = _download_document_from_url(
            "https://files.example.org/dl/pdf",
            explicit_filename="Приказ № 794-1 от 28.03.2023",
            client=mock_client,
        )
        assert fn_pdf == "Приказ № 794-1 от 28.03.2023.pdf"

        fn_docx, _ = _download_document_from_url("https://files.example.org/download?type=docx_test", client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=payloads["/dl/docx"])),
            follow_redirects=False,
        ))
        assert fn_docx.startswith("doc_") and fn_docx.endswith(".docx")

        fn_xlsx, _ = _download_document_from_url("https://files.example.org/dl/xlsx", client=mock_client)
        assert fn_xlsx == "xlsx.xlsx"

        fn_doc, _ = _download_document_from_url("https://files.example.org/dl/doc", client=mock_client)
        assert fn_doc == "doc.doc"

        fn_xls, _ = _download_document_from_url("https://files.example.org/dl/xls", client=mock_client)
        assert fn_xls == "xls.xls"

        with pytest.raises(HTTPException) as exc_info:
            _download_document_from_url("https://files.example.org/dl/exe", client=mock_client)
        assert exc_info.value.status_code == 400
        for ext_label in ("PDF", "DOCX", "DOC", "XLSX", "XLS"):
            assert ext_label in exc_info.value.detail


# -----------------------------------------------------------------------------
# 3. Smart Cloud Sharing Link Normalization Tests
# -----------------------------------------------------------------------------


class TestCloudUrlNormalization:
    def test_google_docs_sheets_drive_and_github_url_rewrites(self) -> None:
        # Google Docs -> export docx
        assert (
            _normalize_document_download_url(
                "https://docs.google.com/document/d/1AbCdEfGhIjK_99/edit?usp=sharing"
            )
            == "https://docs.google.com/document/d/1AbCdEfGhIjK_99/export?format=docx"
        )

        # Google Sheets -> export xlsx
        assert (
            _normalize_document_download_url(
                "https://docs.google.com/spreadsheets/d/1SheetId_2026/edit#gid=0"
            )
            == "https://docs.google.com/spreadsheets/d/1SheetId_2026/export?format=xlsx"
        )

        # Google Drive file view & open?id= -> uc?export=download
        assert (
            _normalize_document_download_url(
                "https://drive.google.com/file/d/1DriveFileXYZ/view?usp=drive_link"
            )
            == "https://drive.google.com/uc?export=download&id=1DriveFileXYZ"
        )
        assert (
            _normalize_document_download_url("https://drive.google.com/open?id=1DriveOpenABC")
            == "https://drive.google.com/uc?export=download&id=1DriveOpenABC"
        )

        # GitHub blob -> raw.githubusercontent.com
        assert (
            _normalize_document_download_url(
                "https://github.com/mipt-corp/normative-acts/blob/main/orders/prikaz_794.pdf"
            )
            == "https://raw.githubusercontent.com/mipt-corp/normative-acts/main/orders/prikaz_794.pdf"
        )

    def test_yandex_disk_public_key_resolution_and_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_mock_public_dns(monkeypatch)
        pdf_bytes = _make_pdf_bytes()
        direct_href = "https://downloader.disk.yandex.ru/disk/abcdef123/67000/Prikaz_YDisk.pdf?uid=0"

        def _ydisk_handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "cloud-api.yandex.net":
                assert request.url.path == "/v1/disk/public/resources/download"
                assert "public_key=" in str(request.url)
                return httpx.Response(200, json={"href": direct_href, "method": "GET"})
            if str(request.url) == direct_href:
                return httpx.Response(
                    200,
                    headers={"Content-Type": "application/pdf"},
                    content=pdf_bytes,
                )
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(_ydisk_handler), follow_redirects=False)

        resolved = _normalize_document_download_url("https://disk.yandex.ru/i/PublicKey987", client=mock_client)
        assert resolved == direct_href

        resolved_short = _normalize_document_download_url("https://yadi.sk/d/ShortKey321", client=mock_client)
        assert resolved_short == direct_href

        fname, content = _download_document_from_url("https://disk.yandex.ru/i/PublicKey987", client=mock_client)
        assert fname == "Prikaz_YDisk.pdf"
        assert content == pdf_bytes


# -----------------------------------------------------------------------------
# 4. HTML Webpage Trap Guard, Size Cap & HTTP Error Mapping Tests
# -----------------------------------------------------------------------------


class TestHtmlWebpageRejectionAndLimits:
    def test_html_login_webpage_rejected_with_400_even_on_pdf_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_mock_public_dns(monkeypatch)
        iso_pipeline = _build_isolated_pipeline(tmp_path, "col_html_trap")
        monkeypatch.setattr(web_app_mod, "get_pipeline", lambda: iso_pipeline)
        monkeypatch.setattr(web_app_mod, "UPLOADS_DIR", tmp_path / "uploads_html")

        def _html_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                content=b"<!DOCTYPE html><html><body><h1>Sign in to continue</h1></body></html>",
            )

        monkeypatch.setattr(
            web_app_mod,
            "_create_url_download_client",
            lambda: httpx.Client(transport=httpx.MockTransport(_html_handler), follow_redirects=False),
        )

        client = TestClient(app)
        resp = client.post(
            "/api/documents/upload-url",
            json={"url": "https://drive.example.org/private/secret_order.pdf", "async_task": False},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "веб-страница (HTML)" in detail
        assert "PDF, DOCX, DOC, XLSX, XLS" in detail
        assert len(iso_pipeline.get_documents()) == 0

    def test_html_bytes_with_octet_stream_rejected_unless_explicit_word_doc(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_mock_public_dns(monkeypatch)
        html_body = b"  <html><head><title>Login</title></head><body>Please log in</body></html>"

        # 1) Generic octet-stream with HTML body -> rejected as HTML webpage
        c_trap = httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, headers={"Content-Type": "application/octet-stream"}, content=html_body)
            ),
            follow_redirects=False,
        )
        with pytest.raises(HTTPException) as exc_trap:
            _download_document_from_url("https://portal.example.org/file/123", client=c_trap)
        assert exc_trap.value.status_code == 400
        assert "веб-страница (HTML)" in exc_trap.value.detail

        # 2) Explicit Word-exported HTML with Content-Disposition `.doc` -> permitted as `.doc`
        c_word_html = httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    headers={
                        "Content-Type": "text/html; charset=utf-8",
                        "Content-Disposition": 'attachment; filename="WordExport.doc"',
                    },
                    content=html_body,
                )
            ),
            follow_redirects=False,
        )
        fname, data = _download_document_from_url("https://portal.example.org/export/123", client=c_word_html)
        assert fname == "WordExport.doc"
        assert data == html_body

    def test_http_404_and_size_limit_exceeded_return_clean_russian_400(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_mock_public_dns(monkeypatch)
        monkeypatch.setattr(web_app_mod, "MAX_URL_UPLOAD_BYTES", 1024)  # 1 KB cap for fast test

        def _handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/missing.pdf":
                return httpx.Response(404, text="Not Found")
            if req.url.path == "/huge.pdf":
                return httpx.Response(
                    200,
                    headers={"Content-Type": "application/pdf"},
                    content=b"%PDF-1.7\n" + (b"0" * 2048),
                )
            return httpx.Response(500)

        monkeypatch.setattr(
            web_app_mod,
            "_create_url_download_client",
            lambda: httpx.Client(transport=httpx.MockTransport(_handler), follow_redirects=False),
        )

        client = TestClient(app)
        r_404 = client.post(
            "/api/documents/upload-url",
            json={"url": "https://docs.example.org/missing.pdf", "async_task": False},
        )
        assert r_404.status_code == 400
        assert "HTTP 404" in r_404.json()["detail"]

        r_big = client.post(
            "/api/documents/upload-url",
            json={"url": "https://docs.example.org/huge.pdf", "async_task": False},
        )
        assert r_big.status_code == 400
        assert "превышает допустимый лимит" in r_big.json()["detail"]


# -----------------------------------------------------------------------------
# 5. SSRF Protection Tests (Direct Private/Loopback/Metadata & Redirect Hops)
# -----------------------------------------------------------------------------


class TestSsrfProtection:
    @pytest.mark.parametrize(
        "blocked_url",
        [
            "http://127.0.0.1:8000/api/health",
            "http://localhost/secret.pdf",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.12/internal.pdf",
            "http://192.168.1.10/admin.pdf",
            "http://[::1]:8000/local.pdf",
            "http://metadata.google.internal/computeMetadata/v1/",
            "ftp://docs.example.org/file.pdf",
            "file:///etc/passwd",
        ],
    )
    def test_ssrf_blocks_loopback_private_metadata_and_non_http_schemes(self, blocked_url: str) -> None:
        client = TestClient(app)
        resp = client.post(
            "/api/documents/upload-url",
            json={"url": blocked_url, "async_task": False},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        if blocked_url.startswith(("ftp:", "file:")):
            assert "http:// или https://" in detail
        else:
            assert "локальных и внутренних адресов запрещена" in detail

    def test_ssrf_blocks_open_redirect_to_loopback_and_dns_rebinding_to_private_ip(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_mock_public_dns(monkeypatch, extra_private_hosts={"intranet.corp.example": "10.25.0.8"})
        redirect_followed = {"called_loopback": False}

        def _redirect_handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == "https://public.example.org/jump":
                return httpx.Response(302, headers={"Location": "http://127.0.0.1:8000/secret.pdf"})
            if request.url.host == "127.0.0.1":
                redirect_followed["called_loopback"] = True
                return httpx.Response(200, content=_make_pdf_bytes())
            return httpx.Response(404)

        monkeypatch.setattr(
            web_app_mod,
            "_create_url_download_client",
            lambda: httpx.Client(transport=httpx.MockTransport(_redirect_handler), follow_redirects=False),
        )

        client = TestClient(app)
        # 1) Redirect from public URL -> 127.0.0.1 must be blocked BEFORE making request to 127.0.0.1
        r_redir = client.post(
            "/api/documents/upload-url",
            json={"url": "https://public.example.org/jump", "async_task": False},
        )
        assert r_redir.status_code == 400
        assert "локальных и внутренних адресов запрещена" in r_redir.json()["detail"]
        assert redirect_followed["called_loopback"] is False

        # 2) Hostname resolving to private RFC1918 IP (10.25.0.8) must be blocked by _validate_public_http_url
        with pytest.raises(HTTPException) as exc_priv:
            _validate_public_http_url("https://intranet.corp.example/budget.xlsx")
        assert exc_priv.value.status_code == 400
        assert "локальных и внутренних адресов запрещена" in exc_priv.value.detail


# -----------------------------------------------------------------------------
# 6. Conditional Step-by-Step Action Plan Prompt Verification Tests
# -----------------------------------------------------------------------------


class TestConditionalActionPlanPrompt:
    def test_system_prompt_contains_procedural_plan_rule_and_non_procedural_guardrails(self) -> None:
        # 1) Procedural "how-to" instructions: brief direct answer first + numbered step-by-step plan
        assert "КОГДА НУЖЕН ПОШАГОВЫЙ ПЛАН" in SYSTEM_PROMPT_RU
        assert "краткий прямой ответ по сути" in SYSTEM_PROMPT_RU
        assert "Пошаговый план действий" in SYSTEM_PROMPT_RU
        assert "как оформить" in SYSTEM_PROMPT_RU
        assert "как согласовать" in SYSTEM_PROMPT_RU
        assert "какой порядок" in SYSTEM_PROMPT_RU

        # 2) Explicit non-procedural guardrails ("не для каждого запроса" + factual/numeric/table exclusions)
        assert "КОГДА ПЛАН ДЕЛАТЬ НЕ НУЖНО" in SYSTEM_PROMPT_RU
        assert "далеко не для каждого запроса нужен план" in SYSTEM_PROMPT_RU.lower()
        assert "БЕЗ составления пошагового плана" in SYSTEM_PROMPT_RU
        assert "сколько подразделений" in SYSTEM_PROMPT_RU
        assert "кто финансодержатель" in SYSTEM_PROMPT_RU
        assert "какая формула" in SYSTEM_PROMPT_RU

        # 3) Core existing RAG rules remain intact (Edition Graph, Table Tool Calling, Verbatim Quotes, Jargon Ban)
        assert "Edition Graph" in SYSTEM_PROMPT_RU
        assert "query_structured_table" in SYSTEM_PROMPT_RU
        assert "Дословные цитаты из документов" in SYSTEM_PROMPT_RU
        assert "> «точный текст из документа»" in SYSTEM_PROMPT_RU
        assert "7. Запрет на технический жаргон и внутренние метрики:" in SYSTEM_PROMPT_RU

    def test_rag_agent_build_messages_injects_updated_system_prompt(self) -> None:
        agent = TableAwareRAGAgent(api_key="sk-test", base_url="https://api.deepseek.com", model="deepseek-chat")
        hit = SearchHit(
            chunk_id="reg_kz::c01",
            doc_id="reg_kz",
            file_name="Регламент КЗ.pdf",
            title="Регламент бронирования КЗ",
            doc_number="3897-1",
            clause_no="2.3",
            page_numbers=[2],
            clean_text="2.3. Для бронирования концертного зала подается служебная записка за 2 месяца.",
            score=0.91,
        )
        sr = SearchResponse(query="Как забронировать концертный зал?", hits=[hit], confidence_score=0.91)
        msgs = agent._build_messages("Как забронировать концертный зал?", sr)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SYSTEM_PROMPT_RU
        assert "Пошаговый план действий" in msgs[0]["content"]
        assert "БЕЗ составления пошагового плана" in msgs[0]["content"]
