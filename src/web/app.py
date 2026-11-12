"""FastAPI Application for MIPT Normative RAG V3.

Provides REST API and SSE endpoints for:
- System health & statistics (/api/health)
- Document upload and incremental sync (/api/documents/upload, /api/documents/sync-folder)
- Background task tracking and real-time SSE streaming (/api/tasks/{task_id}, /api/tasks/{task_id}/stream)
- Document registry, edition graph, and chunk inspection (/api/documents, /api/documents/{doc_id}/chunks)
- Metadata editing and document deletion (/api/documents/{doc_id}/metadata, /api/documents/{doc_id})
- Hybrid RRF search with Parent-Child hydration & out-of-scope detection (/api/search)
- Static SPA web interface at http://localhost:8000/
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from src.config import DOCS_DIR, UPLOADS_DIR, settings
from src.indexing.pipeline import IndexingPipeline, get_pipeline
from src.retrieval.search_service import SearchService, get_search_service
from src.schemas import (
    ChunkRecord,
    DocumentMeta,
    DocumentStatus,
    IndexingFileStatus,
    IndexingReport,
    ParentSectionRecord,
    SearchQuery,
    SearchResponse,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

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
    description="База знаний, иерархический чанкер V3 (Parent-Child + 12% Bilateral Overlap) и веб-интерфейс МФТИ",
    version="3.0.0",
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

@app.get("/api/health", summary="Get Qdrant, Polza AI & Knowledge Base Health")
async def get_health():
    pipeline = get_pipeline()
    health_info = pipeline.get_health()
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
                "message": f"Синхронизация завершена. Обработано файлов: {report.indexed_files}, пропущено: {report.skipped_files}, ошибок: {report.failed_files}",
                "parent_count": report.total_parent_sections,
                "chunk_count": report.total_chunks,
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
            # Send initial snapshot event
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

@app.get("/api/documents", summary="List all indexed documents, edition links, and report summary")
async def get_documents():
    pipeline = get_pipeline()
    docs = pipeline.get_documents()
    report = pipeline.get_indexing_report()
    return {
        "documents": [d.model_dump() for d in docs],
        "total_documents": len(docs),
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

    # Resolve file path on disk
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


@app.delete("/api/documents/{doc_id}", summary="Delete document from Qdrant, sections store, and registry")
async def delete_document(doc_id: str):
    pipeline = get_pipeline()
    success = pipeline.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Документ '{doc_id}' не найден в базе")
    return {
        "success": True,
        "doc_id": doc_id,
        "message": f"Документ '{doc_id}' успешно удален из Qdrant, реестра и хранилища разделов",
    }


# -----------------------------------------------------------------------------
# Hybrid Search & Presets Endpoints
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
        "expected": "стр. 4 табл. 6.1 строка 13: не более 6 дней (Регламент бронирования конц зала.pdf)",
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
]


@app.get("/api/search/presets", summary="Get standard MIPT test query presets")
async def get_query_presets():
    return MIPT_QUERY_PRESETS


@app.post("/api/search", response_model=SearchResponse, summary="Execute Hybrid RRF Search")
async def execute_search(query_req: SearchQuery) -> SearchResponse:
    search_svc = get_search_service()
    try:
        response = search_svc.search(query_req)
        return response
    except Exception as exc:
        logger.exception("Search execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения поиска: {exc}")


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
