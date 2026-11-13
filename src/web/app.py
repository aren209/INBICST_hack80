"""FastAPI Application for MIPT Normative RAG V3 (Adaptive 3-Tier Tables & Tool Calling).

Provides REST API and SSE endpoints for:
- System health & statistics (/api/health)
- Document upload and incremental sync (/api/documents/upload, /api/documents/sync-folder)
- Background task tracking and real-time SSE streaming (/api/tasks/{task_id}, /api/tasks/{task_id}/stream)
- Document registry, edition graph, and chunk inspection (/api/documents, /api/documents/{doc_id}/chunks)
- Metadata editing and document deletion (/api/documents/{doc_id}/metadata, /api/documents/{doc_id})
- Structured SQLite Tables Explorer & Query Engine (/api/tables, /api/tables/{table_id}, /api/tables/query)
- Hybrid RRF search with Parent-Child hydration & out-of-scope detection (/api/search)
- Autonomous Table-Aware LLM RAG Agent with Tool Calling (/api/ask)
- Static SPA web interface at http://localhost:8000/
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Optional
import urllib.parse
import uuid
import warnings

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

import pymupdf as fitz

from src.config import CLAUSE_NUMBER_RE, DOCS_DIR, OCR_CACHE_DIR, SUPPORTED_EXTENSIONS, UPLOADS_DIR, settings
from src.indexing.pipeline import IndexingPipeline, get_pipeline
from src.indexing.table_store import get_table_store
from src.ingestion.parsers import parse_document
from src.retrieval.quote_verifier import clean_clause
from src.retrieval.search_service import get_search_service
from src.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    ChatbotContextRequest,
    ChatbotContextResponse,
    ChunkRecord,
    DocumentMeta,
    DocumentStatus,
    DocumentViewerBlock,
    DocumentViewerHighlight,
    DocumentViewerPage,
    DocumentViewerResponse,
    IndexingFileStatus,
    IndexingReport,
    ParentSectionRecord,
    SearchQuery,
    SearchResponse,
    TablePassport,
    TableQueryRequest,
    TableQueryResponse,
    ToolCallTrace,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Structured Tables & RAG Agent Delegation Helpers
# -----------------------------------------------------------------------------

def _list_structured_tables(
    pipeline: IndexingPipeline,
    doc_id: Optional[str] = None,
) -> list[TablePassport]:
    """Retrieve TablePassports from pipeline.get_tables(...) or StructuredTableStore."""
    clean_doc_id = doc_id.strip() if doc_id and doc_id.strip() else None
    get_tables_fn = getattr(pipeline, "get_tables", None)
    if callable(get_tables_fn):
        sig = inspect.signature(get_tables_fn)
        if "doc_id" in sig.parameters:
            tables = list(get_tables_fn(doc_id=clean_doc_id) or [])
        else:
            all_tables = list(get_tables_fn() or [])
            tables = (
                [t for t in all_tables if getattr(t, "doc_id", "") == clean_doc_id]
                if clean_doc_id
                else all_tables
            )
        return tables

    return get_table_store().list_tables(doc_id=clean_doc_id)


def _get_structured_table(
    pipeline: IndexingPipeline,
    table_id: str,
) -> Optional[TablePassport]:
    """Retrieve a single TablePassport by table_id from pipeline or StructuredTableStore."""
    get_table_fn = getattr(pipeline, "get_table", None)
    if callable(get_table_fn):
        return get_table_fn(table_id)
    return get_table_store().get_table(table_id)


def _query_structured_table(
    pipeline: IndexingPipeline,
    req: TableQueryRequest,
) -> TableQueryResponse:
    """Execute a deterministic TableQueryRequest against pipeline or StructuredTableStore."""
    query_fn = getattr(pipeline, "query_table", None)
    if callable(query_fn):
        return query_fn(req)
    return get_table_store().query_table(req)


def _resolve_rag_agent() -> Any:
    """Resolve the TableAwareRAGAgent singleton from src.retrieval.rag_agent."""
    from src.retrieval.rag_agent import get_rag_agent

    return get_rag_agent()


# -----------------------------------------------------------------------------
# Background Task State Manager
# -----------------------------------------------------------------------------

class TaskState:
    """Thread-safe background task state with SSE broadcast queue."""

    def __init__(self, task_id: str, task_type: str):
        self.task_id = task_id
        self.task_type = task_type
        self.status = "pending"  # pending, running, completed, failed
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.stage = "init"
        self.stage_index = 0
        self.total_stages = 6
        self.stage_label = "Инициализация"
        self.percent = 0
        self.message = "Задача создана"
        self.file_name = ""
        self.doc_id = ""
        self.parent_count = 0
        self.chunk_count = 0
        self.tables_count = 0
        self.ocr_pages = 0
        self.events: list[dict[str, Any]] = []
        self.result: Any = None
        self.error: Optional[str] = None
        self.listeners: list[asyncio.Queue] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def add_listener(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.listeners.append(q)
        return q

    def remove_listener(self, q: asyncio.Queue) -> None:
        if q in self.listeners:
            self.listeners.remove(q)

    def emit_event(self, ev: dict[str, Any]) -> None:
        """Update properties and broadcast event thread-safely."""
        for k in (
            "stage",
            "stage_index",
            "total_stages",
            "stage_label",
            "percent",
            "message",
            "file_name",
            "doc_id",
            "parent_count",
            "chunk_count",
            "tables_count",
            "ocr_pages",
        ):
            if k in ev:
                setattr(self, k, ev[k])

        if not ev.get("timestamp"):
            ev["timestamp"] = datetime.now(timezone.utc).isoformat()

        self.events.append(ev)
        if len(self.events) > 300:
            self.events = self.events[-300:]

        for q in list(self.listeners):
            try:
                if self.loop and self.loop.is_running():
                    self.loop.call_soon_threadsafe(q.put_nowait, ev)
                else:
                    q.put_nowait(ev)
            except Exception as broadcast_err:
                logger.debug("Broadcast listener error: %s", broadcast_err)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
            "stage_label": self.stage_label,
            "percent": self.percent,
            "message": self.message,
            "file_name": self.file_name,
            "doc_id": self.doc_id,
            "parent_count": self.parent_count,
            "chunk_count": self.chunk_count,
            "tables_count": self.tables_count,
            "ocr_pages": self.ocr_pages,
            "events": self.events[-50:],
            "result": self.result,
            "error": self.error,
        }


_TASKS: dict[str, TaskState] = {}


def _get_or_create_task(task_type: str) -> TaskState:
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    task = TaskState(task_id, task_type)
    try:
        task.loop = asyncio.get_running_loop()
    except RuntimeError:
        task.loop = None
    _TASKS[task_id] = task
    return task


# -----------------------------------------------------------------------------
# Request & Response DTOs
# -----------------------------------------------------------------------------

class SyncFolderRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    folder_path: Optional[str] = None
    force: bool = False
    force_reindex: Optional[bool] = None
    ocr_backend: Optional[str] = None
    async_task: bool = True


class MetadataPatchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: Optional[str] = None
    doc_number: Optional[str] = None
    doc_date: Optional[str] = None
    doc_type: Optional[str] = None
    status: Optional[str] = None
    supersedes: Optional[Any] = None
    amends: Optional[Any] = None
    applicable_years: Optional[Any] = None
    version_note: Optional[str] = None
    quality_warning: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------

app = FastAPI(
    title="MIPT Normative RAG V3",
    description=(
        "База знаний, иерархический чанкер V3 (Parent-Child + 12% Bilateral Overlap), "
        "3-уровневое хранилище структурированных таблиц SQLite и автономный ИИ-агент с Tool Calling"
    ),
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Static SPA Routing
# -----------------------------------------------------------------------------

@app.get("/", response_class=FileResponse, summary="Serve Russian Web SPA")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found in static dir")
    return FileResponse(index_file, media_type="text/html")


# -----------------------------------------------------------------------------
# System Health & Statistics
# -----------------------------------------------------------------------------

@app.get("/api/health", summary="Get Qdrant, Polza AI, SQLite Table Store & Knowledge Base Health")
async def get_health():
    pipeline = get_pipeline()
    health_info = dict(pipeline.get_health() or {})
    if "structured_tables_count" not in health_info:
        try:
            health_info["structured_tables_count"] = len(_list_structured_tables(pipeline))
        except Exception as exc:
            logger.debug("Could not count structured tables in /api/health: %s", exc)
            health_info["structured_tables_count"] = 0

    health_info["tables_count"] = int(health_info.get("structured_tables_count") or 0)
    health_info.setdefault(
        "llm",
        {
            "provider": settings.llm_provider,
            "configured": bool(settings.llm_api_key),
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
        },
    )
    health_info["llm_provider"] = settings.llm_provider
    health_info["llm_model"] = settings.llm_model
    health_info.setdefault("supported_extensions", sorted(SUPPORTED_EXTENSIONS))
    if isinstance(health_info.get("polza_api"), dict):
        health_info["polza_api"]["llm_model"] = settings.llm_model
        health_info["polza_api"]["llm_provider"] = settings.llm_provider
    health_info["timestamp"] = datetime.now(timezone.utc).isoformat()
    health_info["docs_dir"] = str(DOCS_DIR)
    health_info["uploads_dir"] = str(UPLOADS_DIR)
    return health_info


# -----------------------------------------------------------------------------
# Document Upload Endpoint
# -----------------------------------------------------------------------------

@app.post("/api/documents/upload", summary="Upload documents and index into knowledge base")
async def upload_documents(
    files: list[UploadFile] = File(...),
    status: Optional[str] = Form("active"),
    supersedes: Optional[str] = Form(""),
    amends: Optional[str] = Form(""),
    applicable_years: Optional[str] = Form(""),
    version_note: Optional[str] = Form(""),
    ocr_backend: Optional[str] = Form(None),
    force_reindex: bool = Form(False),
    async_task: bool = Form(True),
):
    if not files:
        raise HTTPException(status_code=400, detail="Не передано ни одного файла для загрузки")

    saved_paths: list[Path] = []
    for upload in files:
        if not upload.filename:
            continue
        safe_filename = Path(upload.filename).name
        ext = Path(safe_filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            allowed_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла '{safe_filename}'. Допустимые форматы: {allowed_str}",
            )
        dest_path = UPLOADS_DIR / safe_filename
        content = await upload.read()
        dest_path.write_bytes(content)
        saved_paths.append(dest_path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="Имена файлов некорректны")

    metadata_overrides: dict[str, Any] = {}
    if status and status.strip():
        metadata_overrides["status"] = status.strip()
    if supersedes and supersedes.strip():
        metadata_overrides["supersedes"] = supersedes.strip()
    if amends and amends.strip():
        metadata_overrides["amends"] = amends.strip()
    if applicable_years and applicable_years.strip():
        metadata_overrides["applicable_years"] = applicable_years.strip()
    if version_note and version_note.strip():
        metadata_overrides["version_note"] = version_note.strip()

    clean_ocr_backend = ocr_backend.strip() if ocr_backend and ocr_backend.strip() else None

    task = _get_or_create_task(task_type="upload")
    pipeline = get_pipeline()

    def _run_upload_indexing():
        task.status = "running"
        task.started_at = datetime.now(timezone.utc).isoformat()
        total_files = len(saved_paths)
        file_statuses: list[dict[str, Any]] = []

        try:
            for idx, file_path in enumerate(saved_paths, start=1):
                task.file_name = file_path.name

                def _file_progress(ev: dict[str, Any], _idx: int = idx, _tot: int = total_files):
                    overall_pct = int(((_idx - 1) + (ev.get("percent", 0) / 100.0)) / max(1, _tot) * 100)
                    patched = dict(ev)
                    patched["percent"] = max(1, min(99 if ev.get("stage") != "done" or _idx < _tot else 100, overall_pct))
                    patched["message"] = f"[{_idx}/{_tot}] {ev.get('message', '')}"
                    task.emit_event(patched)

                st = pipeline.index_file(
                    file_path=file_path,
                    metadata_overrides=metadata_overrides if metadata_overrides else None,
                    ocr_backend=clean_ocr_backend,
                    force=force_reindex,
                    progress_callback=_file_progress,
                )
                file_statuses.append(st.model_dump())

            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.percent = 100
            task.result = {"files": file_statuses, "total_files": total_files}
            task.emit_event({
                "stage": "done",
                "stage_index": 6,
                "total_stages": 6,
                "stage_label": "Завершено",
                "percent": 100,
                "message": f"Успешно обработано файлов: {total_files}",
                "file_name": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.exception("Upload indexing failed: %s", exc)
            task.status = "failed"
            task.error = str(exc)
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.emit_event({
                "stage": "error",
                "stage_index": task.stage_index,
                "total_stages": 6,
                "stage_label": "Ошибка",
                "percent": task.percent,
                "message": f"Ошибка обработки: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    if async_task:
        asyncio.create_task(asyncio.to_thread(_run_upload_indexing))
        return {
            "task_id": task.task_id,
            "status": "running",
            "files": [p.name for p in saved_paths],
            "message": "Индексация загруженных файлов запущена в фоновом режиме",
        }
    else:
        await asyncio.to_thread(_run_upload_indexing)
        return {
            "task_id": task.task_id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
        }


# -----------------------------------------------------------------------------
# Incremental Sync Folder Endpoint
# -----------------------------------------------------------------------------

@app.post("/api/documents/sync-folder", summary="Incrementally sync docs directory by SHA-256")
async def sync_folder(
    req: Optional[SyncFolderRequest] = Body(default=None),
    force: Optional[bool] = None,
    force_reindex: Optional[bool] = None,
    ocr_backend: Optional[str] = None,
    async_task: Optional[bool] = None,
):
    effective_req = req or SyncFolderRequest()
    effective_force = bool(
        force
        if force is not None
        else (
            force_reindex
            if force_reindex is not None
            else (
                effective_req.force_reindex
                if effective_req.force_reindex is not None
                else effective_req.force
            )
        )
    )
    raw_ocr = ocr_backend if ocr_backend is not None else effective_req.ocr_backend
    clean_ocr_backend = raw_ocr.strip() if raw_ocr and raw_ocr.strip() else None
    effective_async = async_task if async_task is not None else effective_req.async_task

    task = _get_or_create_task(task_type="sync_folder")
    pipeline = get_pipeline()
    target_dir = Path(effective_req.folder_path).resolve() if effective_req.folder_path else DOCS_DIR.resolve()

    def _run_sync():
        task.status = "running"
        task.started_at = datetime.now(timezone.utc).isoformat()
        try:
            report = pipeline.sync_folder(
                folder_path=target_dir,
                force=effective_force,
                ocr_backend=clean_ocr_backend,
                progress_callback=task.emit_event,
            )
            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.percent = 100
            task.result = report.model_dump()
            task.emit_event({
                "stage": "done",
                "stage_index": 6,
                "total_stages": 6,
                "stage_label": "Завершено",
                "percent": 100,
                "message": (
                    f"Синхронизация завершена. Обработано файлов: {report.indexed_files}, "
                    f"пропущено: {report.skipped_files}, ошибок: {report.failed_files}"
                ),
                "parent_count": report.total_parent_sections,
                "chunk_count": report.total_chunks,
                "tables_count": getattr(report, "tables_count", 0),
                "ocr_pages": report.total_ocr_pages,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.exception("Sync folder failed: %s", exc)
            task.status = "failed"
            task.error = str(exc)
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.emit_event({
                "stage": "error",
                "stage_index": task.stage_index,
                "total_stages": 6,
                "stage_label": "Ошибка",
                "percent": task.percent,
                "message": f"Ошибка синхронизации: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    if effective_async:
        asyncio.create_task(asyncio.to_thread(_run_sync))
        return {
            "task_id": task.task_id,
            "status": "running",
            "message": f"Синхронизация папки '{target_dir.name}' запущена в фоновом режиме",
        }
    else:
        await asyncio.to_thread(_run_sync)
        return {
            "task_id": task.task_id,
            "status": task.status,
            "report": task.result,
            "result": task.result,
            "error": task.error,
        }


# -----------------------------------------------------------------------------
# Background Task Status & SSE Streaming
# -----------------------------------------------------------------------------

@app.get("/api/tasks/{task_id}", summary="Get task progress snapshot")
async def get_task_status(task_id: str):
    task = _TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Задача '{task_id}' не найдена")
    return task.to_dict()


@app.get("/api/tasks/{task_id}/stream", summary="Live Server-Sent Events stream for task progress")
async def stream_task_events(task_id: str):
    task = _TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Задача '{task_id}' не найдена")

    async def event_generator():
        q = task.add_listener()
        try:
            initial_payload = task.to_dict()
            yield f"data: {json.dumps(initial_payload, ensure_ascii=False)}\n\n"

            if task.status in ("completed", "failed"):
                return

            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if ev.get("stage") in ("done", "error") or task.status in ("completed", "failed"):
                        break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            task.remove_listener(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# -----------------------------------------------------------------------------
# Document Registry & Inspection Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/documents", summary="List all indexed documents, edition links, table counts, and report summary")
async def get_documents():
    pipeline = get_pipeline()
    docs = pipeline.get_documents()
    report = pipeline.get_indexing_report()

    # Cross-check live SQLite table counts per doc_id so UI & API always surface accurate tables_count
    table_counts_by_doc: dict[str, int] = {}
    total_structured_tables = 0
    try:
        all_tables = _list_structured_tables(pipeline)
        total_structured_tables = len(all_tables)
        for tp in all_tables:
            d_id = getattr(tp, "doc_id", "")
            if d_id:
                table_counts_by_doc[d_id] = table_counts_by_doc.get(d_id, 0) + 1
    except Exception as exc:
        logger.debug("Could not inspect structured tables in /api/documents: %s", exc)

    doc_dicts: list[dict[str, Any]] = []
    for d in docs:
        d_dump = d.model_dump()
        live_cnt = table_counts_by_doc.get(d.doc_id, 0)
        d_dump["tables_count"] = max(int(d_dump.get("tables_count") or 0), live_cnt)
        doc_dicts.append(d_dump)

    report_tables_cnt = max(
        int(getattr(report, "tables_count", 0) or 0),
        total_structured_tables,
    )

    return {
        "documents": doc_dicts,
        "total_documents": len(doc_dicts),
        "structured_tables_count": total_structured_tables,
        "tables_count": total_structured_tables,
        "report_summary": {
            "run_id": report.run_id,
            "started_at": report.started_at,
            "finished_at": report.finished_at,
            "total_files": report.total_files,
            "indexed_files": report.indexed_files,
            "skipped_files": report.skipped_files,
            "failed_files": report.failed_files,
            "total_parent_sections": report.total_parent_sections,
            "total_chunks": report.total_chunks,
            "tables_count": report_tables_cnt,
            "structured_tables_count": total_structured_tables,
            "total_ocr_pages": report.total_ocr_pages,
            "total_duration_sec": report.total_duration_sec,
        },
        "latest_report": report.model_dump(),
    }


@app.get("/api/documents/{doc_id}/chunks", summary="Get ParentSectionRecord and ChunkRecord tree for document inspector")
async def get_document_chunks(doc_id: str):
    pipeline = get_pipeline()
    try:
        data = pipeline.get_document_chunks(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Документ '{doc_id}' не найден в реестре")

    doc_meta = data["document"]
    sections = data["sections"]
    chunks = data["chunks"]

    return {
        "document": doc_meta.model_dump() if hasattr(doc_meta, "model_dump") else doc_meta,
        "sections": [s.model_dump() if hasattr(s, "model_dump") else s for s in sections],
        "chunks": [c.model_dump() if hasattr(c, "model_dump") else c for c in chunks],
    }


# -----------------------------------------------------------------------------
# Side-by-Side Document Viewer Endpoints (/file, /pages/{n}.png, /view)
# -----------------------------------------------------------------------------

_VIEWER_MAX_EXCEL_ROWS: int = 250

_DOCUMENT_MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".xls": "application/vnd.ms-excel",
}


def _resolve_document_file(
    pipeline: IndexingPipeline,
    doc_id: str,
    require_disk_file: bool = True,
) -> tuple[DocumentMeta, Optional[Path]]:
    """Locate `DocumentMeta` and its physical file on disk across `file_path`, `DOCS_DIR`, and `UPLOADS_DIR`."""
    doc_meta = pipeline.get_document(doc_id)
    if doc_meta is None:
        raise HTTPException(status_code=404, detail=f"Документ '{doc_id}' не найден в реестре")

    candidates: list[Path] = []
    if doc_meta.file_path:
        candidates.append(Path(doc_meta.file_path))
    if doc_meta.file_name:
        candidates.append(DOCS_DIR / doc_meta.file_name)
        candidates.append(UPLOADS_DIR / doc_meta.file_name)

    target_path: Optional[Path] = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            target_path = cand.resolve()
            break

    if require_disk_file and target_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Исходный файл для документа '{doc_id}' ({doc_meta.file_name}) не найден на диске",
        )
    return doc_meta, target_path


def _norm_viewer_text(text: str) -> str:
    """Normalize Russian text (`ё -> е`, lowercase, single spaces) for tolerant viewer quote matching."""
    if not text:
        return ""
    s = text.lower().replace("ё", "е").replace("\u00a0", " ")
    s = re.sub(r"[«»\"'„“”`]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _tokenize_viewer_words(text: str) -> list[str]:
    """Extract normalized alphanumeric words (`ё -> е`, lowercase) for word-sequence alignment."""
    if not text:
        return []
    return re.findall(r"[0-9a-zа-я]+", text.lower().replace("ё", "е"))


def _split_quote_segments(quote: str) -> list[str]:
    """Split a quote containing `…` or `...` ellipsis into clean contiguous segments."""
    raw = (quote or "").strip().strip("«»\"'„“” ")
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"(?:…|\.{3,})", raw) if p and p.strip()]
    return [p for p in parts if len(p) >= 6] or ([raw] if len(raw) >= 3 else [])


def _priority_page_order(total_pages: int, requested_page: Optional[int]) -> list[int]:
    """Order pages starting with `requested_page`, then immediate neighbors `±1, ±2`, then remaining pages."""
    if total_pages <= 0:
        return [1]
    all_pages = list(range(1, total_pages + 1))
    if requested_page is None or not (1 <= requested_page <= total_pages):
        return all_pages
    ordered: list[int] = [requested_page]
    for delta in (1, 2, 3):
        for cand in (requested_page - delta, requested_page + delta):
            if 1 <= cand <= total_pages and cand not in ordered:
                ordered.append(cand)
    for p in all_pages:
        if p not in ordered:
            ordered.append(p)
    return ordered


def _merge_line_rects_normalized(
    raw_rects: list[tuple[float, float, float, float]],
    page_w: float,
    page_h: float,
) -> list[DocumentViewerHighlight]:
    """Merge horizontal word/anchor bboxes on the same line into crisp normalized `[0..1]` bands."""
    if not raw_rects or page_w <= 0 or page_h <= 0:
        return []

    sorted_rects = sorted(raw_rects, key=lambda r: (round(r[1], 1), r[0]))
    merged: list[list[float]] = []
    for x0, y0, x1, y1 in sorted_rects:
        if x1 <= x0 or y1 <= y0:
            continue
        if merged:
            prev = merged[-1]
            prev_mid_y = 0.5 * (prev[1] + prev[3])
            cur_mid_y = 0.5 * (y0 + y1)
            max_h = max(prev[3] - prev[1], y1 - y0, 1.0)
            # Same visual text line if vertical centers align within 55% of line height
            if abs(cur_mid_y - prev_mid_y) <= 0.55 * max_h and x0 <= prev[2] + 28.0:
                prev[0] = min(prev[0], x0)
                prev[1] = min(prev[1], y0)
                prev[2] = max(prev[2], x1)
                prev[3] = max(prev[3], y1)
                continue
        merged.append([x0, y0, x1, y1])

    out: list[DocumentViewerHighlight] = []
    for x0, y0, x1, y1 in merged[:40]:
        nx = max(0.0, min(1.0, x0 / page_w))
        ny = max(0.0, min(1.0, y0 / page_h))
        nw = max(0.005, min(1.0 - nx, (x1 - x0) / page_w))
        nh = max(0.005, min(1.0 - ny, (y1 - y0) / page_h))
        out.append(
            DocumentViewerHighlight(
                x=round(nx, 5),
                y=round(ny, 5),
                w=round(nw, 5),
                h=round(nh, 5),
            )
        )
    return out


def _find_pdf_quote_highlights_on_page(
    fitz_page: Any,
    quote_segments: list[str],
) -> list[DocumentViewerHighlight]:
    """
    Locate normalized `[0..1]` highlight rectangles for `quote_segments` on `fitz_page`:
    1) Exact word-sequence alignment over `fitz_page.get_text("words")` (handling hyphenated line wraps!).
    2) Multi-anchor fallback via `fitz_page.search_for(...)` (4-7 word anchors from start/middle/end).
    """
    rect = fitz_page.rect
    page_w = float(rect.width or 0.0)
    page_h = float(rect.height or 0.0)
    if page_w <= 0 or page_h <= 0 or not quote_segments:
        return []

    raw_words = fitz_page.get_text("words") or []
    if raw_words:
        # Build flat normalized word tokens; join hyphenated line-wrap pairs into virtual combined tokens
        tokens: list[tuple[str, list[tuple[float, float, float, float]]]] = []
        i = 0
        n_w = len(raw_words)
        while i < n_w:
            w_item = raw_words[i]
            x0, y0, x1, y1, w_str = float(w_item[0]), float(w_item[1]), float(w_item[2]), float(w_item[3]), str(w_item[4])
            blk_no = int(w_item[5]) if len(w_item) > 5 else 0
            ln_no = int(w_item[6]) if len(w_item) > 6 else 0

            # Detect hyphenated wrap at end of line (e.g., "служеб-" + next line "ную")
            if (
                w_str.endswith(("-", "\xad"))
                and len(w_str) >= 3
                and i + 1 < n_w
                and (int(raw_words[i + 1][6]) != ln_no or int(raw_words[i + 1][5]) != blk_no)
            ):
                nxt = raw_words[i + 1]
                nxt_str = str(nxt[4])
                nx0, ny0, nx1, ny1 = float(nxt[0]), float(nxt[1]), float(nxt[2]), float(nxt[3])
                joined_toks = _tokenize_viewer_words(w_str[:-1] + nxt_str)
                if len(joined_toks) == 1:
                    tokens.append((joined_toks[0], [(x0, y0, x1, y1), (nx0, ny0, nx1, ny1)]))
                    i += 2
                    continue

            sub_toks = _tokenize_viewer_words(w_str)
            for st in sub_toks:
                tokens.append((st, [(x0, y0, x1, y1)]))
            i += 1

        if tokens:
            page_tok_strs = [t[0] for t in tokens]
            matched_boxes: list[tuple[float, float, float, float]] = []

            def _find_subseq(needle: list[str]) -> list[int]:
                k = len(needle)
                if k == 0 or k > len(page_tok_strs):
                    return []
                first_w = needle[0]
                limit = len(page_tok_strs) - k + 1
                for start_idx in range(limit):
                    if page_tok_strs[start_idx] == first_w and page_tok_strs[start_idx : start_idx + k] == needle:
                        return list(range(start_idx, start_idx + k))
                return []

            for seg in quote_segments:
                seg_toks = _tokenize_viewer_words(seg)
                if not seg_toks:
                    continue
                hit_indices = _find_subseq(seg_toks)
                if not hit_indices and len(seg_toks) >= 6:
                    # Try 5-word start, middle, and end anchors so minor middle punctuation/OCR diffs still highlight
                    win = min(6, max(4, len(seg_toks) // 2))
                    head_idx = _find_subseq(seg_toks[:win])
                    tail_idx = _find_subseq(seg_toks[-win:])
                    if head_idx and tail_idx and tail_idx[-1] >= head_idx[0] and (tail_idx[-1] - head_idx[0]) <= len(seg_toks) + 12:
                        hit_indices = list(range(head_idx[0], tail_idx[-1] + 1))
                    else:
                        mid_start = len(seg_toks) // 2 - win // 2
                        mid_idx = _find_subseq(seg_toks[mid_start : mid_start + win])
                        hit_indices = head_idx + mid_idx + tail_idx

                for idx in hit_indices:
                    matched_boxes.extend(tokens[idx][1])

            if matched_boxes:
                return _merge_line_rects_normalized(matched_boxes, page_w, page_h)

    # Stage 2 fallback: PyMuPDF `page.search_for(...)` on 4-7 word anchors
    anchor_candidates: list[str] = []
    for seg in quote_segments:
        words = [w for w in re.split(r"\s+", seg.strip()) if w]
        if not words:
            continue
        if len(words) <= 8:
            anchor_candidates.append(" ".join(words))
        else:
            anchor_candidates.append(" ".join(words[:6]))
            mid = len(words) // 2
            anchor_candidates.append(" ".join(words[max(0, mid - 3) : mid + 3]))
            anchor_candidates.append(" ".join(words[-6:]))

    found_rects: list[tuple[float, float, float, float]] = []
    for anchor in anchor_candidates:
        clean_anchor = anchor.strip(" ,.;:«»\"'()[]")
        if len(clean_anchor) < 8:
            continue
        try:
            rects = fitz_page.search_for(clean_anchor) or []
        except Exception:
            rects = []
        for r in rects[:8]:
            found_rects.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))

    return _merge_line_rects_normalized(found_rects, page_w, page_h)


def _find_pdf_clause_highlights_on_page(
    fitz_page: Any,
    clause_str: str,
) -> list[DocumentViewerHighlight]:
    """Locate normalized `[0..1]` highlight box for a clause number (e.g. `2.3`) when no quote was matched."""
    cl = clean_clause(clause_str)
    if not cl:
        return []
    rect = fitz_page.rect
    page_w, page_h = float(rect.width or 0.0), float(rect.height or 0.0)
    if page_w <= 0 or page_h <= 0:
        return []
    for probe in (f"{cl}. ", f"п. {cl}", cl):
        try:
            rects = fitz_page.search_for(probe) or []
        except Exception:
            rects = []
        if rects:
            r0 = rects[0]
            return _merge_line_rects_normalized([(float(r0.x0), float(r0.y0), float(r0.x1), float(r0.y1))], page_w, page_h)
    return []


_SCANNED_OCR_CACHE_BY_FILE: Optional[dict[str, dict[int, str]]] = None


def _get_cached_ocr_pages_for_file(file_name: str) -> dict[int, str]:
    """Load per-page OCR text from `data/ocr_cache/*.json` keyed by `file_name -> {page_num: text}`."""
    global _SCANNED_OCR_CACHE_BY_FILE
    if _SCANNED_OCR_CACHE_BY_FILE is None:
        index: dict[str, dict[int, str]] = {}
        if OCR_CACHE_DIR.exists():
            for jf in OCR_CACHE_DIR.glob("*.json"):
                try:
                    payload = json.loads(jf.read_text(encoding="utf-8"))
                    fname = str(payload.get("file_name") or "").strip()
                    p_num = int(payload.get("page_num") or 0)
                    txt = str(payload.get("text") or "")
                    if fname and p_num >= 1 and txt:
                        index.setdefault(fname, {})[p_num] = txt
                except Exception:
                    continue
        _SCANNED_OCR_CACHE_BY_FILE = index
    return _SCANNED_OCR_CACHE_BY_FILE.get(file_name, {})


def _first_valid_page_from_cpm(val: Any, total_pages: int) -> Optional[int]:
    """Extract 1-based page int from `clause_page_map` entry (`int`, `str`, or `list[int]`)."""
    cand: Any = val[0] if isinstance(val, (list, tuple)) and val else val
    if isinstance(cand, int) and 1 <= cand <= total_pages:
        return cand
    if isinstance(cand, str) and cand.strip().isdigit():
        p = int(cand.strip())
        if 1 <= p <= total_pages:
            return p
    return None


def _resolve_scanned_pdf_page_from_store(
    pipeline: IndexingPipeline,
    doc_id: str,
    file_name: str,
    total_pages: int,
    requested_page: Optional[int],
    quote_segments: list[str],
    clause_str: str,
) -> tuple[Optional[int], bool, list[DocumentViewerHighlight]]:
    """
    Resolve exact `target_page` AND proportional line-band highlight on OCR-scanned PDFs
    using `data/ocr_cache` per-page OCR text and `sections_store` `clause_page_map`.
    """
    cl = clean_clause(clause_str)
    norm_segs = [_norm_viewer_text(s) for s in quote_segments if len(_norm_viewer_text(s)) >= 8]

    # 1. Check per-page OCR text in `data/ocr_cache` (gives exact page AND vertical line position!)
    ocr_pages = _get_cached_ocr_pages_for_file(file_name)
    if ocr_pages and (norm_segs or cl):
        for cand_p in _priority_page_order(total_pages, requested_page):
            p_txt = ocr_pages.get(cand_p, "")
            if not p_txt:
                continue
            lines = [ln.strip() for ln in p_txt.splitlines() if ln.strip()]
            if not lines:
                continue
            page_norm = _norm_viewer_text(p_txt)
            matched_q = bool(norm_segs and any(ns[:36] in page_norm for ns in norm_segs))
            matched_cl = bool(
                not matched_q
                and cl
                and any(ln.startswith((f"{cl}.", f"{cl} ")) or f" {cl}." in ln for ln in lines)
            )
            if matched_q or matched_cl:
                hit_line_indices: list[int] = []
                q_words = {t for seg in quote_segments for t in _tokenize_viewer_words(seg) if len(t) >= 4}
                for l_idx, ln in enumerate(lines):
                    ln_norm = _norm_viewer_text(ln)
                    if matched_q:
                        if any(ns[:24] in ln_norm for ns in norm_segs) or (
                            len(q_words) >= 2 and sum(1 for w in q_words if w in ln_norm) >= 2
                        ):
                            hit_line_indices.append(l_idx)
                    elif cl and (ln.startswith((f"{cl}.", f"{cl} ")) or f" {cl}." in ln):
                        hit_line_indices.append(l_idx)
                        break

                tot_l = max(len(lines), 1)
                if hit_line_indices:
                    first_l = min(hit_line_indices)
                    last_l = min(max(hit_line_indices), first_l + 6)
                    y_top = max(0.06, min(0.88, 0.08 + 0.80 * (first_l / tot_l)))
                    y_bot = max(y_top + 0.032, min(0.94, 0.08 + 0.80 * ((last_l + 1.2) / tot_l)))
                else:
                    y_top, y_bot = 0.14, 0.20

                hl = [
                    DocumentViewerHighlight(
                        x=0.08,
                        y=round(y_top, 4),
                        w=0.84,
                        h=round(y_bot - y_top, 4),
                    )
                ]
                return cand_p, matched_q, hl

    # 2. Fallback to `sections_store` + `clause_page_map` (handles list[int] values!)
    sections = [
        sec for sec in pipeline.sections_store.values()
        if getattr(sec, "doc_id", "") == doc_id
    ]
    for sec in sections:
        cpm = (sec.metadata or {}).get("clause_page_map") or {}
        sec_norm = _norm_viewer_text(sec.clean_text or "")
        if norm_segs and any(ns in sec_norm for ns in norm_segs):
            if cl:
                p_cpm = _first_valid_page_from_cpm(cpm.get(cl), total_pages)
                if p_cpm is not None:
                    return p_cpm, True, [DocumentViewerHighlight(x=0.08, y=0.15, w=0.84, h=0.06)]
            for line in (sec.clean_text or "").split("\n"):
                ln_norm = _norm_viewer_text(line)
                if any(ns[:28] in ln_norm for ns in norm_segs):
                    m_cl = CLAUSE_NUMBER_RE.match(line.strip())
                    if m_cl:
                        p_cpm = _first_valid_page_from_cpm(cpm.get(m_cl.group(1)), total_pages)
                        if p_cpm is not None:
                            return p_cpm, True, [DocumentViewerHighlight(x=0.08, y=0.15, w=0.84, h=0.06)]
            if sec.page_numbers:
                p = int(sec.page_numbers[0])
                if 1 <= p <= total_pages:
                    return p, True, [DocumentViewerHighlight(x=0.08, y=0.15, w=0.84, h=0.06)]

    if cl:
        for sec in sections:
            cpm = (sec.metadata or {}).get("clause_page_map") or {}
            p_cpm = _first_valid_page_from_cpm(cpm.get(cl), total_pages)
            if p_cpm is not None:
                return p_cpm, False, [DocumentViewerHighlight(x=0.08, y=0.15, w=0.84, h=0.05)]

    return None, False, []


def _slice_rows_around_quote(
    rows: list[list[str]],
    quote_segments: list[str],
    max_rows: int = _VIEWER_MAX_EXCEL_ROWS,
) -> tuple[list[list[str]], bool]:
    """
    Cap huge Excel/Word table previews at `max_rows`, centering the window around the first row
    matching `quote_segments` so quoted cells are ALWAYS present in the viewer response!
    """
    norm_segs = [_norm_viewer_text(s) for s in quote_segments if _norm_viewer_text(s)]
    quote_toks = [t for seg in quote_segments for t in _tokenize_viewer_words(seg) if len(t) >= 4]
    matched_idx: Optional[int] = None

    if norm_segs or quote_toks:
        best_overlap = 0
        for r_idx, row in enumerate(rows):
            row_norm = _norm_viewer_text(" ".join(str(c or "") for c in row))
            if not row_norm:
                continue
            if any(ns in row_norm for ns in norm_segs if len(ns) >= 6):
                matched_idx = r_idx
                break
            if quote_toks:
                ov = sum(1 for qt in set(quote_toks) if qt in row_norm)
                if ov >= max(2, int(len(set(quote_toks)) * 0.6)) and ov > best_overlap:
                    best_overlap = ov
                    matched_idx = r_idx

    if len(rows) <= max_rows:
        return [list(r) for r in rows], (matched_idx is not None)

    if matched_idx is not None:
        half = max_rows // 2
        start = max(0, min(matched_idx - half, len(rows) - max_rows))
        return [list(r) for r in rows[start : start + max_rows]], True

    return [list(r) for r in rows[:max_rows]], False


def _extract_block_clause_no(text: str) -> str:
    """Extract leading normative clause number (`2.3`, `1.1.4`) from a block's text if present."""
    if not text:
        return ""
    first_line = text.lstrip().split("\n", 1)[0]
    m = CLAUSE_NUMBER_RE.match(first_line)
    if not m:
        return ""
    cand = m.group(1).strip(".")
    # Guard against bare years like "2026."
    if cand.isdigit() and int(cand) > 99:
        return ""
    return cand


def _block_matches_quote_or_clause(
    blk: DocumentViewerBlock,
    quote_segments: list[str],
    clause_str: str,
) -> tuple[bool, bool]:
    """Return `(matched_quote, matched_clause)` for a `DocumentViewerBlock`."""
    hay_parts = [blk.text or ""]
    if blk.headers:
        hay_parts.append(" ".join(blk.headers))
    for r in blk.rows:
        hay_parts.append(" ".join(r))
    hay_norm = _norm_viewer_text(" ".join(hay_parts))

    matched_q = False
    if quote_segments and hay_norm:
        norm_segs = [_norm_viewer_text(s) for s in quote_segments if _norm_viewer_text(s)]
        if any(ns in hay_norm for ns in norm_segs if len(ns) >= 8):
            matched_q = True
        else:
            q_toks = {t for seg in quote_segments for t in _tokenize_viewer_words(seg) if len(t) >= 4}
            if len(q_toks) >= 2:
                hit_cnt = sum(1 for t in q_toks if t in hay_norm)
                if hit_cnt >= max(2, int(len(q_toks) * 0.75)):
                    matched_q = True

    cl = clean_clause(clause_str)
    matched_cl = False
    if cl:
        if blk.clause_no and clean_clause(blk.clause_no) == cl:
            matched_cl = True
        elif hay_norm.startswith(f"{cl.lower()} ") or f" {cl.lower()}. " in f" {hay_norm} ":
            matched_cl = True

    return matched_q, matched_cl


def _build_document_view_sync(
    pipeline: IndexingPipeline,
    doc_id: str,
    page: Optional[int],
    quote: Optional[str],
    clause: Optional[str],
) -> DocumentViewerResponse:
    """Synchronous builder for `GET /api/documents/{doc_id}/view` across PDF, DOCX, DOC, XLSX, and XLS."""
    doc_meta, target_path = _resolve_document_file(pipeline, doc_id, require_disk_file=False)
    ext = (
        target_path.suffix.lower().lstrip(".")
        if target_path is not None and target_path.suffix
        else (doc_meta.file_format or "").lower().lstrip(".")
    )
    fmt = "xlsx" if ext == "xlsm" else ext
    encoded_doc_id = urllib.parse.quote(doc_id, safe="")
    download_url = f"/api/documents/{encoded_doc_id}/file"
    quote_segments = _split_quote_segments(quote or "")
    cl_clean = clean_clause(clause or "")

    # 1. PDF Document Viewer (exact rendered PNG dimensions + normalized [0..1] quote highlight overlays)
    if fmt == "pdf":
        if target_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"PDF-файл для документа '{doc_id}' не найден на диске",
            )
        pdf_doc = fitz.open(str(target_path))
        try:
            total_pages = max(1, len(pdf_doc))
            pages_out: list[DocumentViewerPage] = []
            for p_num in range(1, total_pages + 1):
                p_rect = pdf_doc[p_num - 1].rect
                pages_out.append(
                    DocumentViewerPage(
                        page_number=p_num,
                        label=f"Стр. {p_num}",
                        image_url=f"/api/documents/{encoded_doc_id}/pages/{p_num}.png",
                        width=round(float(p_rect.width), 1),
                        height=round(float(p_rect.height), 1),
                        highlights=[],
                        blocks=[],
                    )
                )

            target_page = max(1, min(total_pages, int(page))) if page is not None else 1
            matched_quote_on_page = False
            search_order = _priority_page_order(total_pages, page)

            if quote_segments:
                for cand_p in search_order:
                    hl = _find_pdf_quote_highlights_on_page(pdf_doc[cand_p - 1], quote_segments)
                    if hl:
                        pages_out[cand_p - 1].highlights = hl
                        target_page = cand_p
                        matched_quote_on_page = True
                        break

            if not matched_quote_on_page and cl_clean:
                for cand_p in search_order:
                    cl_hl = _find_pdf_clause_highlights_on_page(pdf_doc[cand_p - 1], cl_clean)
                    if cl_hl:
                        pages_out[cand_p - 1].highlights = cl_hl
                        target_page = cand_p
                        break

            # Scanned OCR PDF fallback (when native PDF page has 0 selectable glyphs)
            if not matched_quote_on_page and (quote_segments or cl_clean):
                store_p, store_matched_q, store_hl = _resolve_scanned_pdf_page_from_store(
                    pipeline=pipeline,
                    doc_id=doc_id,
                    file_name=doc_meta.file_name,
                    total_pages=total_pages,
                    requested_page=page,
                    quote_segments=quote_segments,
                    clause_str=cl_clean,
                )
                if store_p is not None:
                    target_page = store_p
                    if store_hl and not pages_out[store_p - 1].highlights:
                        pages_out[store_p - 1].highlights = store_hl
                    if store_matched_q:
                        matched_quote_on_page = True

            return DocumentViewerResponse(
                doc_id=doc_id,
                file_name=doc_meta.file_name,
                title=doc_meta.title or doc_meta.file_name,
                doc_number=doc_meta.doc_number,
                doc_date=doc_meta.doc_date,
                format="pdf",
                total_pages=total_pages,
                target_page=target_page,
                matched_quote_on_page=matched_quote_on_page,
                download_url=download_url,
                pages=pages_out,
            )
        finally:
            pdf_doc.close()

    # 2. Excel Workbooks (.xlsx / .xls): one viewer page per worksheet
    if fmt in {"xlsx", "xls"}:
        pages_out = []
        quote_matched_sheet: Optional[int] = None

        # Fast path via StructuredTableStore when sheets are already indexed in SQLite
        passports = _list_structured_tables(pipeline, doc_id)
        if passports and all(getattr(tp, "sheet_name", "") for tp in passports):
            for s_idx, tp in enumerate(passports, start=1):
                headers = [c.name for c in tp.columns]
                all_row_dicts: list[dict[str, Any]] = []
                offset = 0
                while offset < max(tp.row_count, 1) and offset < 5000:
                    q_res = _query_structured_table(
                        pipeline,
                        TableQueryRequest(table_id=tp.table_id, operation="get_rows", limit=500, offset=offset),
                    )
                    if not q_res.rows:
                        break
                    all_row_dicts.extend(q_res.rows)
                    if len(q_res.rows) < 500:
                        break
                    offset += len(q_res.rows)

                matrix_rows = [
                    [str(rd.get(col_h, "") or "") for col_h in headers]
                    for rd in all_row_dicts
                ]
                sliced_rows, row_hit = _slice_rows_around_quote(matrix_rows, quote_segments)
                if row_hit and quote_matched_sheet is None:
                    quote_matched_sheet = s_idx

                sheet_blocks: list[DocumentViewerBlock] = []
                if tp.table_title and tp.table_title != tp.sheet_name:
                    sheet_blocks.append(
                        DocumentViewerBlock(type="heading", text=tp.table_title, clause_no="", headers=[], rows=[])
                    )
                sheet_blocks.append(
                    DocumentViewerBlock(
                        type="table",
                        text=tp.table_title or tp.sheet_name,
                        clause_no="",
                        headers=headers,
                        rows=sliced_rows,
                    )
                )
                pages_out.append(
                    DocumentViewerPage(
                        page_number=s_idx,
                        label=tp.sheet_name or f"Лист {s_idx}",
                        image_url=None,
                        width=None,
                        height=None,
                        highlights=[],
                        blocks=sheet_blocks,
                    )
                )

        # Direct workbook parser path when StructuredTableStore has no sheets (or multi-block form sheet)
        if not pages_out and target_path is not None:
            parsed_blocks, _ = parse_document(target_path)
            grouped: dict[int, list[Any]] = {}
            sheet_labels: dict[int, str] = {}
            for b in parsed_blocks:
                p_no = max(1, int(b.page_num or 1))
                grouped.setdefault(p_no, []).append(b)
                sh_lbl = b.sheet_name or str((b.metadata or {}).get("sheet_name") or "")
                if sh_lbl and p_no not in sheet_labels:
                    sheet_labels[p_no] = sh_lbl

            for s_idx, p_no in enumerate(sorted(grouped) or [1], start=1):
                s_blocks: list[DocumentViewerBlock] = []
                for b in grouped.get(p_no, []):
                    meta = b.metadata or {}
                    s_hdrs = list(meta.get("structured_headers") or b.table_headers or [])
                    s_rows = list(meta.get("structured_rows") or b.table_rows or [])
                    if s_hdrs or s_rows:
                        capped_rows, row_hit = _slice_rows_around_quote(s_rows, quote_segments)
                        if row_hit and quote_matched_sheet is None:
                            quote_matched_sheet = s_idx
                        s_blocks.append(
                            DocumentViewerBlock(
                                type="table",
                                text=str(meta.get("table_title") or b.text[:240]),
                                clause_no="",
                                headers=s_hdrs,
                                rows=capped_rows,
                            )
                        )
                    elif b.text:
                        b_type = "heading" if b.block_type == "heading" else "paragraph"
                        vb = DocumentViewerBlock(type=b_type, text=b.text, clause_no="", headers=[], rows=[])
                        mq, _ = _block_matches_quote_or_clause(vb, quote_segments, cl_clean)
                        if mq and quote_matched_sheet is None:
                            quote_matched_sheet = s_idx
                        s_blocks.append(vb)
                pages_out.append(
                    DocumentViewerPage(
                        page_number=s_idx,
                        label=sheet_labels.get(p_no, f"Лист {s_idx}"),
                        image_url=None,
                        width=None,
                        height=None,
                        highlights=[],
                        blocks=s_blocks,
                    )
                )

        if not pages_out:
            pages_out.append(
                DocumentViewerPage(page_number=1, label="Лист 1", image_url=None, highlights=[], blocks=[])
            )

        total_pages = len(pages_out)
        if quote_matched_sheet is not None:
            target_page = quote_matched_sheet
            matched_quote_on_page = True
        else:
            target_page = max(1, min(total_pages, int(page))) if page is not None else 1
            matched_quote_on_page = False

        return DocumentViewerResponse(
            doc_id=doc_id,
            file_name=doc_meta.file_name,
            title=doc_meta.title or doc_meta.file_name,
            doc_number=doc_meta.doc_number,
            doc_date=doc_meta.doc_date,
            format=fmt,
            total_pages=total_pages,
            target_page=target_page,
            matched_quote_on_page=matched_quote_on_page,
            download_url=download_url,
            pages=pages_out,
        )

    # 3. Word Documents (.docx / .doc): structured archival pages grouped by `page_number` 1..N
    pages_blocks_map: dict[int, list[DocumentViewerBlock]] = {}
    if target_path is not None:
        parsed_blocks, _ = parse_document(target_path)
        for b in parsed_blocks:
            p_no = max(1, int(b.page_num or 1))
            bucket = pages_blocks_map.setdefault(p_no, [])
            if b.block_type == "table" and (b.table_headers or b.table_rows):
                capped_rows, _ = _slice_rows_around_quote(list(b.table_rows or []), quote_segments)
                bucket.append(
                    DocumentViewerBlock(
                        type="table",
                        text=b.text,
                        clause_no="",
                        headers=list(b.table_headers or []),
                        rows=capped_rows,
                    )
                )
            elif (b.text or "").strip():
                b_type = "heading" if (b.block_type == "heading" or b.heading_level) else "paragraph"
                bucket.append(
                    DocumentViewerBlock(
                        type=b_type,
                        text=b.text.strip(),
                        clause_no=_extract_block_clause_no(b.text),
                        headers=[],
                        rows=[],
                    )
                )

    # Fallback to `sections_store` if physical file was moved/unavailable
    if not pages_blocks_map:
        doc_secs = [
            sec for sec in pipeline.sections_store.values()
            if getattr(sec, "doc_id", "") == doc_id
        ]
        for idx_s, sec in enumerate(doc_secs, start=1):
            p_no = int(sec.page_numbers[0]) if sec.page_numbers else idx_s
            bucket = pages_blocks_map.setdefault(max(1, p_no), [])
            if sec.section_title:
                bucket.append(
                    DocumentViewerBlock(
                        type="heading",
                        text=sec.section_title,
                        clause_no=sec.section_num or "",
                        headers=[],
                        rows=[],
                    )
                )
            for para in (sec.clean_text or "").split("\n\n"):
                pt = para.strip()
                if pt:
                    bucket.append(
                        DocumentViewerBlock(
                            type="paragraph",
                            text=pt,
                            clause_no=_extract_block_clause_no(pt),
                            headers=[],
                            rows=[],
                        )
                    )

    max_page_num = max(pages_blocks_map.keys(), default=1)
    pages_out = []
    for p_no in range(1, max_page_num + 1):
        pages_out.append(
            DocumentViewerPage(
                page_number=p_no,
                label=f"Стр. {p_no}",
                image_url=None,
                width=None,
                height=None,
                highlights=[],
                blocks=pages_blocks_map.get(p_no, []),
            )
        )

    total_pages = max(1, len(pages_out))
    search_order = _priority_page_order(total_pages, page)
    target_page = max(1, min(total_pages, int(page))) if page is not None else 1
    matched_quote_on_page = False
    clause_matched_page: Optional[int] = None

    if quote_segments or cl_clean:
        for cand_p in search_order:
            p_obj = pages_out[cand_p - 1]
            # Check individual blocks first, then whole-page text (in case quote spans two adjacent clauses/blocks)
            for blk in p_obj.blocks:
                mq, mcl = _block_matches_quote_or_clause(blk, quote_segments, cl_clean)
                if mq:
                    target_page = cand_p
                    matched_quote_on_page = True
                    break
                if mcl and clause_matched_page is None:
                    clause_matched_page = cand_p
            if matched_quote_on_page:
                break
            if quote_segments:
                whole_page_norm = _norm_viewer_text(
                    " ".join(
                        f"{b.text} {' '.join(b.headers)} {' '.join(' '.join(r) for r in b.rows)}"
                        for b in p_obj.blocks
                    )
                )
                norm_segs = [_norm_viewer_text(s) for s in quote_segments if len(_norm_viewer_text(s)) >= 8]
                if norm_segs and any(ns in whole_page_norm for ns in norm_segs):
                    target_page = cand_p
                    matched_quote_on_page = True
                    break

    if not matched_quote_on_page and clause_matched_page is not None:
        target_page = clause_matched_page

    return DocumentViewerResponse(
        doc_id=doc_id,
        file_name=doc_meta.file_name,
        title=doc_meta.title or doc_meta.file_name,
        doc_number=doc_meta.doc_number,
        doc_date=doc_meta.doc_date,
        format=fmt or "docx",
        total_pages=total_pages,
        target_page=target_page,
        matched_quote_on_page=matched_quote_on_page,
        download_url=download_url,
        pages=pages_out,
    )


@app.get("/api/documents/{doc_id}/file", summary="Stream original document file inline for viewer/download")
async def get_document_file(doc_id: str):
    pipeline = get_pipeline()
    doc_meta, target_path = _resolve_document_file(pipeline, doc_id, require_disk_file=True)
    assert target_path is not None
    display_name = doc_meta.file_name or target_path.name
    ext = target_path.suffix.lower()
    mime_type = _DOCUMENT_MIME_TYPES.get(ext, "application/octet-stream")
    encoded_fname = urllib.parse.quote(display_name)
    return FileResponse(
        path=target_path,
        media_type=mime_type,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{encoded_fname}",
        },
    )


@app.get(
    "/api/documents/{doc_id}/pages/{page_number}.png",
    summary="Render 1-based PDF page to PNG image via PyMuPDF",
)
async def get_document_page_png(
    doc_id: str,
    page_number: int,
    dpi: int = Query(default=150, description="Render DPI (clamped 96..240)"),
):
    if page_number < 1:
        raise HTTPException(status_code=404, detail=f"Некорректный номер страницы: {page_number}")

    clamped_dpi = max(96, min(240, int(dpi)))
    pipeline = get_pipeline()
    doc_meta, target_path = _resolve_document_file(pipeline, doc_id, require_disk_file=True)
    assert target_path is not None

    if target_path.suffix.lower() != ".pdf" and (doc_meta.file_format or "").lower() != "pdf":
        raise HTTPException(
            status_code=404,
            detail=f"Документ '{doc_id}' не является PDF-файлом (формат: {doc_meta.file_format})",
        )

    def _render_png() -> bytes:
        pdf_doc = fitz.open(str(target_path))
        try:
            if page_number > len(pdf_doc):
                raise IndexError(f"Страница {page_number} выходит за пределы документа ({len(pdf_doc)} стр.)")
            fitz_page = pdf_doc[page_number - 1]
            pix = fitz_page.get_pixmap(dpi=clamped_dpi, alpha=False)
            return pix.tobytes("png")
        finally:
            pdf_doc.close()

    try:
        png_bytes = await asyncio.to_thread(_render_png)
    except IndexError as idx_exc:
        raise HTTPException(status_code=404, detail=str(idx_exc))
    except Exception as exc:
        logger.exception("Failed rendering PDF page %d for '%s': %s", page_number, doc_id, exc)
        raise HTTPException(status_code=500, detail=f"Ошибка рендеринга страницы PDF: {exc}")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get(
    "/api/documents/{doc_id}/view",
    response_model=DocumentViewerResponse,
    summary="Get structured side-by-side Document Viewer payload with exact target_page & quote highlights",
)
async def get_document_view(
    doc_id: str,
    page: Optional[int] = Query(default=None, description="Approximate 1-based page/sheet hint"),
    quote: Optional[str] = Query(default=None, description="Verbatim quote text to locate & highlight"),
    clause: Optional[str] = Query(default=None, description="Normative clause number hint (e.g. 2.3)"),
) -> DocumentViewerResponse:
    pipeline = get_pipeline()
    return await asyncio.to_thread(_build_document_view_sync, pipeline, doc_id, page, quote, clause)


@app.patch("/api/documents/{doc_id}/metadata", summary="Edit document relations, status, years, note and update Qdrant payload")
async def update_document_metadata(doc_id: str, patch: MetadataPatchRequest):
    pipeline = get_pipeline()
    patch_dict = {k: v for k, v in patch.model_dump().items() if v is not None}
    try:
        updated_meta = pipeline.update_document_metadata(doc_id, patch_dict)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Документ '{doc_id}' не найден в реестре")
    except Exception as exc:
        logger.exception("Failed to update document metadata: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ошибка обновления метаданных: {exc}")

    return updated_meta.model_dump()


class ReindexDocumentRequest(BaseModel):
    ocr_backend: Optional[str] = None
    async_task: bool = True


@app.post("/api/documents/{doc_id}/reindex", summary="Force re-index an existing document in registry")
async def reindex_document(
    doc_id: str,
    req: ReindexDocumentRequest = Body(default_factory=ReindexDocumentRequest),
):
    pipeline = get_pipeline()
    doc_meta = pipeline.get_document(doc_id)
    if doc_meta is None:
        raise HTTPException(status_code=404, detail=f"Документ '{doc_id}' не найден в реестре")

    candidates = []
    if doc_meta.file_path:
        candidates.append(Path(doc_meta.file_path))
    if doc_meta.file_name:
        candidates.append(DOCS_DIR / doc_meta.file_name)
        candidates.append(UPLOADS_DIR / doc_meta.file_name)

    target_path: Optional[Path] = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            target_path = cand.resolve()
            break

    if target_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Исходный файл для документа '{doc_id}' ({doc_meta.file_name}) не найден на диске",
        )

    clean_ocr = req.ocr_backend.strip() if req.ocr_backend and req.ocr_backend.strip() else None
    task = _get_or_create_task(task_type="reindex_doc")

    def _run_reindex():
        task.status = "running"
        task.started_at = datetime.now(timezone.utc).isoformat()
        task.file_name = target_path.name
        try:
            st = pipeline.index_file(
                file_path=target_path,
                ocr_backend=clean_ocr,
                force=True,
                progress_callback=task.emit_event,
            )
            task.status = "completed" if st.status != "error" else "failed"
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.percent = 100
            task.result = st.model_dump()
            if st.status == "error":
                task.error = st.error_message
            task.emit_event({
                "stage": "done" if st.status != "error" else "error",
                "stage_index": 6,
                "total_stages": 6,
                "stage_label": "Завершено" if st.status != "error" else "Ошибка",
                "percent": 100,
                "message": (
                    f"Переиндексация '{target_path.name}' завершена: {st.chunks_created} чанков"
                    if st.status != "error"
                    else f"Ошибка: {st.error_message}"
                ),
                "file_name": target_path.name,
                "doc_id": st.doc_id,
                "parent_count": st.parent_sections_created,
                "chunk_count": st.chunks_created,
                "tables_count": getattr(st, "tables_count", 0),
                "ocr_pages": st.ocr_pages,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.exception("Reindex document failed: %s", exc)
            task.status = "failed"
            task.error = str(exc)
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.emit_event({
                "stage": "error",
                "stage_index": task.stage_index,
                "total_stages": 6,
                "stage_label": "Ошибка",
                "percent": task.percent,
                "message": f"Ошибка переиндексации: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    if req.async_task:
        asyncio.create_task(asyncio.to_thread(_run_reindex))
        return {
            "task_id": task.task_id,
            "status": "running",
            "doc_id": doc_id,
            "file_name": target_path.name,
            "message": f"Переиндексация '{target_path.name}' запущена в фоновом режиме",
        }
    else:
        await asyncio.to_thread(_run_reindex)
        return {
            "task_id": task.task_id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
        }


@app.delete("/api/documents/{doc_id}", summary="Delete document from Qdrant, SQLite tables, sections store, and registry")
async def delete_document(doc_id: str):
    pipeline = get_pipeline()
    success = pipeline.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Документ '{doc_id}' не найден в базе")
    return {
        "success": True,
        "doc_id": doc_id,
        "message": f"Документ '{doc_id}' успешно удален из Qdrant, SQLite-таблиц, реестра и хранилища разделов",
    }


# -----------------------------------------------------------------------------
# Structured Tables Endpoints (SQLite tables_store.db)
# -----------------------------------------------------------------------------

@app.get("/api/tables", summary="List all structured TablePassports in SQLite (Excel sheets + large Word/PDF tables)")
async def list_tables(
    doc_id: Optional[str] = Query(default=None, description="Optional doc_id filter"),
):
    pipeline = get_pipeline()
    try:
        tables = await asyncio.to_thread(_list_structured_tables, pipeline, doc_id)
    except Exception as exc:
        logger.exception("Failed to list structured tables: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ошибка чтения реестра таблиц: {exc}")

    serialized = [t.model_dump() if hasattr(t, "model_dump") else t for t in tables]
    return {
        "tables": serialized,
        "total_tables": len(serialized),
        "structured_tables_count": len(serialized),
        "doc_id": doc_id,
    }


@app.post(
    "/api/tables/query",
    response_model=TableQueryResponse,
    summary="Execute deterministic filter_rows, aggregate, get_formulas, or get_rows query on SQLite table",
)
async def query_table_endpoint(req: TableQueryRequest) -> TableQueryResponse:
    pipeline = get_pipeline()
    try:
        return await asyncio.to_thread(_query_structured_table, pipeline, req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Structured table query failed for '%s': %s", req.table_id, exc)
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения запроса к таблице: {exc}")


@app.post("/api/tables/rebuild", summary="Rebuild structured SQLite tables from indexed documents using cached OCR/embeddings")
async def rebuild_structured_tables():
    pipeline = get_pipeline()
    rebuild_fn = getattr(pipeline, "rebuild_structured_tables", None)
    if not callable(rebuild_fn):
        raise HTTPException(
            status_code=501,
            detail="Метод pipeline.rebuild_structured_tables() недоступен",
        )
    try:
        return await asyncio.to_thread(rebuild_fn)
    except Exception as exc:
        logger.exception("Rebuild structured tables failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ошибка перестроения таблиц: {exc}")


@app.get("/api/tables/{table_id}", summary="Get full TablePassport schema, initial preview rows, and formula summary")
async def get_table_detail(table_id: str):
    pipeline = get_pipeline()

    def _load_detail() -> dict[str, Any]:
        passport = _get_structured_table(pipeline, table_id)
        if passport is None:
            raise KeyError(table_id)

        passport_dict = passport.model_dump() if hasattr(passport, "model_dump") else dict(passport)

        # Fetch up to 15 initial preview rows for richer interactive inspection
        preview_rows = list(passport_dict.get("preview_rows") or [])
        if passport.row_count > len(preview_rows):
            try:
                rows_resp = _query_structured_table(
                    pipeline,
                    TableQueryRequest(table_id=table_id, operation="get_rows", limit=15, offset=0),
                )
                if rows_resp.rows:
                    preview_rows = rows_resp.rows
            except Exception as prev_exc:
                logger.debug("Initial preview rows query fallback for %s: %s", table_id, prev_exc)

        formula_cols = [c.name for c in passport.columns if getattr(c, "has_formulas", False)]
        sample_formulas: list[dict[str, Any]] = []
        if passport.has_formulas:
            try:
                f_resp = _query_structured_table(
                    pipeline,
                    TableQueryRequest(table_id=table_id, operation="get_formulas", limit=15, offset=0),
                )
                sample_formulas = list(f_resp.formulas[:15])
            except Exception as f_exc:
                logger.debug("Formula summary query fallback for %s: %s", table_id, f_exc)

        formula_summary = {
            "has_formulas": bool(passport.has_formulas),
            "formula_count": int(passport.formula_count or 0),
            "formula_columns": formula_cols,
            "sample_formulas": sample_formulas,
        }

        passport_dict["preview_rows"] = preview_rows
        return {
            **passport_dict,
            "passport": passport_dict,
            "preview_rows": preview_rows,
            "formula_summary": formula_summary,
        }

    try:
        return await asyncio.to_thread(_load_detail)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Структурированная таблица '{table_id}' не найдена в базе таблиц",
        )
    except Exception as exc:
        logger.exception("Failed to load table detail for '%s': %s", table_id, exc)
        raise HTTPException(status_code=500, detail=f"Ошибка чтения паспорта таблицы: {exc}")


# -----------------------------------------------------------------------------
# Hybrid Search, Presets & Autonomous Table-Aware RAG Agent Endpoints
# -----------------------------------------------------------------------------

MIPT_QUERY_PRESETS: list[dict[str, Any]] = [
    {
        "id": "scenario_1",
        "title": "Бронирование Концертного зала (КЗ)",
        "query": "За сколько месяцев нужно подавать служебную записку на бронирование концертного зала и на какую электронную почту отправлять?",
        "expected": "п. 2.3 и 2.6: не менее 2 месяцев, kmo@mipt.ru (Регламент бронирования конц зала.pdf)",
    },
    {
        "id": "scenario_2",
        "title": "Посвящение в студенты: репетиции в КЗ",
        "query": "Сколько дней репетиций в концертном зале предоставляется физтех-школам на Посвящение в студенты?",
        "expected": "стр. 4 табл. 6.1 строка 13: не более 6 дней (Inline Parent, компактная PDF-таблица)",
    },
    {
        "id": "scenario_3",
        "title": "Поселение сотрудников в общежитие",
        "query": "Какие правила и условия установлены для поселения сотрудников МФТИ в общежития кампуса?",
        "expected": "Регламент поселения сотрудников.pdf (критерии, согласование, служебная записка)",
    },
    {
        "id": "scenario_4",
        "title": "Форма сметы: приказ 794-1 vs шаблон 2019",
        "query": "Как сейчас подается смета расходов и какие формулы, коды КВР/КОСГУ и накладные расходы применяются?",
        "expected": "Приказ N 794-1 (помесячно в 1С:ДО Приложение 1а) + коды КВР/КОСГУ и накладные 20% из шаблона 2019 года",
    },
    {
        "id": "scenario_5",
        "title": "Открытие ФЛС (Приказ 3065-1 + Распоряжение 700)",
        "query": "Какой порядок присвоения и открытия финансово-лицевого счета (ФЛС) действует с учетом последних изменений?",
        "expected": "Приказ N 3065-1 + изменения по Распоряжению N 700 от 28.12.2024",
    },
    {
        "id": "scenario_6",
        "title": "Резерв отпусков ППС на 2026-2028 гг.",
        "query": "Какой процент резерва на оплату отпусков закладывается для ППС и прочих категорий на период 2026-2028 годов?",
        "expected": "Приказ N 3699-1: 16.4% для ППС и 8.2% для остальных категорий",
    },
    {
        "id": "scenario_7",
        "title": "Иерархия ЦФО финансовой структуры 2026",
        "query": "Какая иерархия подразделений и центров финансовой ответственности (ЦФО 1-4) утверждена на 2026 год?",
        "expected": "Приложение к приказу об утверждении фин. структуры 2026г..xlsx / Приказ 3614-1 (Финансодержатель -> ЦФО 1..4)",
    },
    {
        "id": "scenario_8",
        "title": "Командировки и нормы суточных",
        "query": "Каков порядок оформления командировок работников МФТИ и размер суточных расходов?",
        "expected": "Регламент о командировании.pdf (шаги оформления) + Приказ 3699-1 (норма суточных 100 руб)",
    },
    {
        "id": "scenario_9",
        "title": "Out-of-Scope: Ипотечный кредит в банке",
        "query": "Какие документы и первоначальный взнос нужны для оформления семейной ипотеки в Сбербанке?",
        "expected": "is_out_of_scope == True (запрос вне предметной области МФТИ)",
    },
    {
        "id": "scenario_10",
        "title": "Out-of-Scope: Рецепт пасты карбонара",
        "query": "Как приготовить классическую римскую пасту карбонара с гуанчиале и сыром пекорино?",
        "expected": "is_out_of_scope == True (кулинарный запрос вне предметной области)",
    },
    {
        "id": "table_xl_cfo_frkt",
        "preset_mode": "tool_query",
        "title": "📊 XL Tool Query: Подразделения ФРКТ (ЦФО 2026)",
        "query": "Сколько подразделений подчиняется финансодержателю ФРКТ в финструктуре 2026?",
        "expected": "Автономный SQL-дозапрос query_structured_table по Фин. структуре 2026г..xlsx (точный подсчет и список)",
    },
    {
        "id": "table_xl_smeta_formulas",
        "preset_mode": "tool_query",
        "title": "🧮 XL Формулы: Накладные расходы и НДС в смете",
        "query": "Покажи все строки и формулы расчета накладных расходов и НДС в смете",
        "expected": "Вызов get_formulas / filter_rows к смете 2019.xlsx ([формула: =...], 20%) + Приказ 794-1",
    },
    {
        "id": "table_pdf_inline_kz",
        "preset_mode": "inline_parent",
        "title": "📄 PDF Inline vs XL: Репетиции на Посвящение в КЗ",
        "query": "Сколько дней репетиций дается физтех-школам на Посвящение в КЗ?",
        "expected": "Компактная PDF-таблица (< 25 строк) отдается напрямую через ParentSectionRecord (6 дней, used_table_tools=False)",
    },
]


@app.get("/api/search/presets", summary="Get standard MIPT test query presets (including XL Tool Query & Inline Table scenarios)")
async def get_query_presets():
    return MIPT_QUERY_PRESETS


@app.post("/api/search", response_model=SearchResponse, summary="Execute Hybrid RRF Search with Adaptive Table Linking")
async def execute_search(query_req: SearchQuery) -> SearchResponse:
    search_svc = get_search_service()
    try:
        return await asyncio.to_thread(search_svc.search, query_req)
    except Exception as exc:
        logger.exception("Search execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения поиска: {exc}")


@app.post(
    "/api/chatbot/context",
    response_model=ChatbotContextResponse,
    summary="Build self-contained stateless RAG context package (multi-aspect RRF + structural expansion + SQLite table hydration)",
)
async def build_chatbot_context_endpoint(req: ChatbotContextRequest) -> ChatbotContextResponse:
    search_svc = get_search_service()
    build_fn = getattr(search_svc, "build_chatbot_context", None)
    if not callable(build_fn):
        raise HTTPException(
            status_code=501,
            detail="Метод search_service.build_chatbot_context() недоступен",
        )
    try:
        if inspect.iscoroutinefunction(build_fn):
            return await build_fn(req)
        return await asyncio.to_thread(build_fn, req)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chatbot context build failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ошибка сборки контекста чат-бота: {exc}")


@app.post("/api/documents/reconcile", summary="Reconcile document_registry.json, sections_store.json, tables_store.db, and Qdrant")
async def reconcile_stores_endpoint():
    pipeline = get_pipeline()
    try:
        return await asyncio.to_thread(pipeline.reconcile_stores, True)
    except Exception as exc:
        logger.exception("Store reconciliation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ошибка сверки хранилищ: {exc}")


@app.post(
    "/api/ask",
    response_model=AgentAskResponse,
    summary="Execute Autonomous Table-Aware RAG Agent with LLM Tool Calling over SQLite Tables",
)
async def execute_agent_ask(req: AgentAskRequest) -> AgentAskResponse:
    try:
        agent = _resolve_rag_agent()
        ask_fn = getattr(agent, "ask")
        if inspect.iscoroutinefunction(ask_fn):
            response = await ask_fn(req)
        else:
            response = await asyncio.to_thread(ask_fn, req)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG Agent ask failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа ИИ-агентом: {exc}")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post(
    "/api/ask/stream",
    summary="Streamed (SSE) 4-stage RAG answer: status → sources (verified) → token… → quotes → done | error",
)
async def execute_agent_ask_stream(req: AgentAskRequest, request: Request):
    """
    Server-Sent Events over POST. The agent's sync generator runs in a worker thread and pushes
    events into an asyncio queue; when the client disconnects (or presses «Стоп»), a stop flag is
    set, the generator closes the upstream LLM stream and the worker exits.
    """
    agent = _resolve_rag_agent()
    if not hasattr(agent, "ask_stream"):
        raise HTTPException(status_code=404, detail="Потоковый режим недоступен")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()
    stop_event = threading.Event()

    def _push(item: tuple[str, Any] | None) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # loop closed
            pass

    def _worker() -> None:
        gen = agent.ask_stream(req, should_stop=stop_event.is_set)
        try:
            for item in gen:
                if stop_event.is_set():
                    break
                _push(item)
        except Exception as exc:  # pragma: no cover - surfaced to client as SSE error
            logger.exception("RAG Agent stream failed: %s", exc)
            _push(("error", {"message": f"Ошибка генерации ответа: {exc}"}))
        finally:
            gen.close()
            if stop_event.is_set():
                logger.warning("RAG stream aborted by client (disconnect / «Стоп») — upstream LLM stream closed")
            _push(None)

    async def _event_source():
        loop.run_in_executor(None, _worker)  # exits on its own once it observes stop_event
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                yield _sse(item[0], item[1])
        finally:
            stop_event.set()

    return StreamingResponse(
        _event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# -----------------------------------------------------------------------------
# Static Mount (must be last)
# -----------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.web.app:app",
        host=getattr(settings, "APP_HOST", "0.0.0.0"),
        port=int(getattr(settings, "APP_PORT", 8000)),
        reload=False,
    )
