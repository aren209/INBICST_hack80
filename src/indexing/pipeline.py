"""Fault-tolerant Document & Folder Indexing Pipeline with SHA-256 skip, Edition Graph, and Cross-File Deduplication."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from src.config import (
     DOCS_DIR,
    DOCUMENT_REGISTRY_PATH,
    INDEXING_REPORT_PATH,
    REGISTRY_OVERRIDES_PATH,
    SECTIONS_STORE_PATH,
    STATUS_RU_MAP,
    settings,
)
from src.indexing.embedder import HybridEmbedder
from src.indexing.qdrant_store import QdrantHybridStore
from src.schemas import (
    BlockType,
    ChunkRecord,
    DocumentMeta,
    DocumentStatus,
    IndexingFileStatus,
    IndexingReport,
    ParentSectionRecord,
    ParsedBlock,
    ZoneType,
    compute_content_hash,
    make_point_uuid,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".xlsx"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON data to disk using a temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        serialized = json.dumps(data, ensure_ascii=False, indent=2)
        tmp_path.write_text(serialized, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as rm_exc:
                logger.debug("Cleanup of temp json file failed: %s", rm_exc)
        raise


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _make_fallback_doc_id(file_name: str, doc_number: str = "") -> str:
    """Generate a deterministic, readable doc_id when not predefined in registry_overrides.yaml."""
    if doc_number:
        clean_num = re.sub(r"[^0-9a-zA-Zа-яА-Я]+", "_", doc_number).strip("_").lower()
        # Transliterate common Russian letters in doc_number if any
        translit = str.maketrans(
            "абвгдежзийклмнопрстуфхцчшщъыьэюя",
            "abvgdezhziyklmnoprstufhcchshschyeyuya"[:32],
        )
        clean_num_ascii = clean_num.translate(translit)
        short_hash = hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:6]
        return f"doc_{clean_num_ascii}_{short_hash}"
    short_hash = hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:10]
    return f"doc_{short_hash}"


def _core_doc_number(doc_number: str) -> str:
    """Extract the base document number (e.g. '1469-1 (Прил. 1)' -> '1469-1')."""
    if not doc_number:
        return ""
    base = doc_number.split("(")[0].strip()
    base = re.sub(r"^(?:№|N)\s*", "", base, flags=re.IGNORECASE).strip()
    return base


class IndexingPipeline:
    """
    Fault-tolerant orchestrator for single-file and folder ('доки/') indexing:
    - Persists & loads data/document_registry.json, data/sections_store.json, data/indexing_report.json atomically
    - Skips unchanged files by SHA-256 unless force=True
    - Maintains bidirectional edition graph (supersedes <-> superseded_by, amends <-> amended_by)
    - Regenerates/updates synthetic u00_meta chunk and Qdrant payload on graph/metadata changes
    - Performs cross-file chunk deduplication via content_hash (populating also_in_files)
    - Emits 6-stage real-time progress callbacks for Web UI
    """

    def __init__(
        self,
        qdrant_store: QdrantHybridStore | None = None,
        embedder: HybridEmbedder | None = None,
        registry_path: Path | None = None,
        sections_path: Path | None = None,
        report_path: Path | None = None,
        overrides_path: Path | None = None,
    ) -> None:
        self.qdrant = qdrant_store or QdrantHybridStore()
        self.embedder = embedder or HybridEmbedder()
        self.registry_path = registry_path or DOCUMENT_REGISTRY_PATH
        self.sections_path = sections_path or SECTIONS_STORE_PATH
        self.report_path = report_path or INDEXING_REPORT_PATH
        self.overrides_path = overrides_path or REGISTRY_OVERRIDES_PATH

        self.overrides_data: dict[str, Any] = self._load_overrides()
        self.registry: dict[str, DocumentMeta] = self._load_registry()
        self.sections_store: dict[str, ParentSectionRecord] = self._load_sections_store()
        self.last_report: IndexingReport = self._load_report()

    # ------------------------------------------------------------------
    # Persistent Storage Loaders & Savers
    # ------------------------------------------------------------------
    def _load_overrides(self) -> dict[str, Any]:
        if not self.overrides_path.exists():
            return {"documents": {}, "by_doc_number": {}}
        try:
            raw = yaml.safe_load(self.overrides_path.read_text(encoding="utf-8")) or {}
            return {
                "documents": raw.get("documents", {}) or {},
                "by_doc_number": raw.get("by_doc_number", {}) or {},
            }
        except Exception as exc:
            logger.warning("Failed to load registry_overrides.yaml: %s", exc)
            return {"documents": {}, "by_doc_number": {}}

    def _load_registry(self) -> dict[str, DocumentMeta]:
        if not self.registry_path.exists():
            return {}
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "documents" in raw and isinstance(raw["documents"], dict):
                raw_docs = raw["documents"]
            elif isinstance(raw, dict):
                raw_docs = raw
            elif isinstance(raw, list):
                raw_docs = {item.get("doc_id", f"doc_{i}"): item for i, item in enumerate(raw) if isinstance(item, dict)}
            else:
                return {}

            loaded: dict[str, DocumentMeta] = {}
            for k, val in raw_docs.items():
                if isinstance(val, dict) and ("doc_id" in val or "file_name" in val):
                    meta = DocumentMeta.model_validate(val)
                    doc_id = meta.doc_id or str(k)
                    meta.doc_id = doc_id
                    loaded[doc_id] = meta
            return loaded
        except Exception as exc:
            logger.warning("Failed to load document_registry.json: %s", exc)
            return {}

    def save_registry(self) -> None:
        payload = {doc_id: meta.model_dump() for doc_id, meta in self.registry.items()}
        _atomic_write_json(self.registry_path, payload)

    def _load_sections_store(self) -> dict[str, ParentSectionRecord]:
        if not self.sections_path.exists():
            return {}
        try:
            raw = json.loads(self.sections_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "sections" in raw and isinstance(raw["sections"], dict):
                raw_secs = raw["sections"]
            elif isinstance(raw, dict):
                raw_secs = raw
            else:
                return {}
            loaded: dict[str, ParentSectionRecord] = {}
            for sec_id, val in raw_secs.items():
                if isinstance(val, dict) and ("section_id" in val or "clean_text" in val):
                    rec = ParentSectionRecord.model_validate(val)
                    loaded[rec.section_id or str(sec_id)] = rec
            return loaded
        except Exception as exc:
            logger.warning("Failed to load sections_store.json: %s", exc)
            return {}

    def save_sections_store(self) -> None:
        payload = {sec_id: sec.model_dump() for sec_id, sec in self.sections_store.items()}
        _atomic_write_json(self.sections_path, payload)

    def _load_report(self) -> IndexingReport:
        if not self.report_path.exists():
            return IndexingReport()
        try:
            raw = json.loads(self.report_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return IndexingReport.model_validate(raw)
        except Exception as exc:
            logger.warning("Failed to load indexing_report.json: %s", exc)
        return IndexingReport()

    def save_report(self, report: IndexingReport) -> None:
        self.last_report = report
        _atomic_write_json(self.report_path, report.model_dump())

    # ------------------------------------------------------------------
    # Progress Emitter
    # ------------------------------------------------------------------
    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        stage: str,
        stage_index: int,
        stage_label: str,
        percent: int,
        message: str,
        file_name: str = "",
        doc_id: str = "",
        parent_count: int = 0,
        chunk_count: int = 0,
        ocr_pages: int = 0,
    ) -> None:
        if callback is None:
            return
        event = {
            "stage": stage,
            "stage_index": stage_index,
            "total_stages": 6,
            "stage_label": stage_label,
            "percent": max(0, min(100, int(percent))),
            "message": message,
            "file_name": file_name,
            "doc_id": doc_id,
            "parent_count": parent_count,
            "chunk_count": chunk_count,
            "ocr_pages": ocr_pages,
            "timestamp": _utc_now_iso(),
        }
        try:
            callback(event)
        except Exception as exc:
            logger.debug("Progress callback error ignored: %s", exc)

    # ------------------------------------------------------------------
    # Integration Adapters for Module 1 (Parsers) & Module 2/3 (Chunker)
    # ------------------------------------------------------------------
    def _invoke_parser(
        self,
        file_path: Path,
        ocr_backend: str | None = None,
    ) -> tuple[list[ParsedBlock], dict[str, Any]]:
        """Call src.ingestion.parsers to parse PDF/DOCX/XLSX into ParsedBlocks and parser stats."""
        from src.ingestion import parsers as parsers_mod

        parse_fn = getattr(parsers_mod, "parse_document", None) or getattr(parsers_mod, "parse_file", None)
        if parse_fn is not None:
            sig = inspect.signature(parse_fn)
            kwargs: dict[str, Any] = {}
            if "ocr_backend" in sig.parameters and ocr_backend is not None:
                kwargs["ocr_backend"] = ocr_backend
            res = parse_fn(file_path, **kwargs)
            if isinstance(res, tuple) and len(res) == 2:
                blocks_raw, meta_dict = res
                blocks = [
                    b if isinstance(b, ParsedBlock) else ParsedBlock.model_validate(b)
                    for b in (blocks_raw or [])
                ]
                return blocks, dict(meta_dict or {})
            if isinstance(res, list):
                blocks = [
                    b if isinstance(b, ParsedBlock) else ParsedBlock.model_validate(b)
                    for b in res
                ]
                return blocks, {}

        # Check class-based DocumentParser fallback if defined
        parser_cls = getattr(parsers_mod, "DocumentParser", None)
        if parser_cls is not None:
            inst = parser_cls(ocr_backend=ocr_backend) if ocr_backend else parser_cls()
            res = inst.parse(file_path)
            if isinstance(res, tuple) and len(res) == 2:
                return list(res[0]), dict(res[1] or {})
            return list(res), {}

        raise RuntimeError("No parse_document entrypoint found in src.ingestion.parsers")

    def _build_initial_doc_meta(
        self,
        file_path: Path,
        sha256: str,
        parser_meta: dict[str, Any],
        blocks: list[ParsedBlock],
        metadata_overrides: dict[str, Any] | None = None,
    ) -> DocumentMeta:
        """
        Build DocumentMeta by combining:
        1) Auto-extracted metadata via GlossaryManager.extract_document_meta (plus parser_meta)
        2) Preconfigured overrides in config/registry_overrides.yaml
        3) Explicit runtime metadata_overrides (from Web UI upload or PATCH)
        """
        from src.chunking.glossary import get_glossary_manager

        file_name = file_path.name
        gm = get_glossary_manager()
        doc_meta = gm.extract_document_meta(
            blocks_or_text=blocks,
            file_name=file_name,
            file_path=str(file_path),
            sha256=sha256,
            overrides=metadata_overrides,
        )

        # Preserve existing doc_id if file_name was already in registry
        existing_for_file = next(
            (m for m in self.registry.values() if m.file_name == file_name),
            None,
        )
        if existing_for_file and not (metadata_overrides or {}).get("doc_id"):
            doc_meta.doc_id = existing_for_file.doc_id

        pages_total = int(
            parser_meta.get("pages_total")
            or parser_meta.get("total_pages")
            or doc_meta.pages_total
            or max((b.page_num for b in blocks), default=1)
        )
        ocr_pages = int(
            parser_meta.get("ocr_pages")
            or parser_meta.get("ocr_pages_count")
            or doc_meta.ocr_pages
            or len({b.page_num for b in blocks if b.is_ocr})
        )
        doc_meta.pages_total = pages_total
        doc_meta.ocr_pages = ocr_pages
        doc_meta.file_format = file_path.suffix.lstrip(".").lower()
        doc_meta.indexed_at = _utc_now_iso()

        if parser_meta.get("quality_warning") and not doc_meta.quality_warning:
            doc_meta.quality_warning = str(parser_meta["quality_warning"])

        if metadata_overrides:
            cur_dump = doc_meta.model_dump()
            for key, val in metadata_overrides.items():
                if val is not None and val != "" and val != []:
                    cur_dump[key] = val
            doc_meta = DocumentMeta.model_validate(cur_dump)

        if not doc_meta.applicable_years and doc_meta.doc_date and len(doc_meta.doc_date) >= 4:
            try:
                y = int(doc_meta.doc_date[:4])
                doc_meta.applicable_years = [y]
            except ValueError as val_exc:
                logger.debug("Could not parse year from doc_date %s: %s", doc_meta.doc_date, val_exc)
        return doc_meta

    def _invoke_chunker(
        self,
        blocks: list[ParsedBlock],
        doc_meta: DocumentMeta,
    ) -> tuple[list[ChunkRecord], list[ParentSectionRecord], DocumentMeta]:
        """Call src.chunking.hierarchical_chunker to execute the 7-stage V3 chunking pipeline."""
        from src.chunking.hierarchical_chunker import HierarchicalChunker

        chunker = HierarchicalChunker()
        chunks, parent_sections = chunker.chunk_document(blocks=blocks, doc_meta=doc_meta)
        return chunks, parent_sections, doc_meta

    # ------------------------------------------------------------------
    # Synthetic u00_meta Builder & Updater
    # ------------------------------------------------------------------
    @staticmethod
    def build_u00_meta_chunk(doc_meta: DocumentMeta) -> ChunkRecord | None:
        """
        Build or refresh the synthetic u00_meta passport chunk (zone='meta')
        via GlossaryManager.build_u00_meta_chunk.
        """
        from src.chunking.glossary import get_glossary_manager

        return get_glossary_manager().build_u00_meta_chunk(doc_meta)

    def _ensure_u00_meta_in_chunks(
        self,
        doc_meta: DocumentMeta,
        chunks: list[ChunkRecord],
    ) -> list[ChunkRecord]:
        """Ensure exactly one up-to-date u00_meta chunk exists when doc_meta has edition/quality notes."""
        meta_chunk = self.build_u00_meta_chunk(doc_meta)
        non_meta = [c for c in chunks if c.unit_id != "u00_meta" and c.zone != ZoneType.META.value]
        if meta_chunk is not None:
            return [meta_chunk] + non_meta
        return chunks

    # ------------------------------------------------------------------
    # Bidirectional Edition Graph Reconciliation
    # ------------------------------------------------------------------
    def _find_matching_docs(self, ref: str) -> list[DocumentMeta]:
        """Find all documents in registry matching a doc_id, file_name, or doc_number reference."""
        ref_clean = ref.strip()
        if not ref_clean:
            return []
        core_ref = _core_doc_number(ref_clean)
        matches: list[DocumentMeta] = []
        for meta in self.registry.values():
            if (
                meta.doc_id == ref_clean
                or meta.file_name == ref_clean
                or (meta.doc_number and meta.doc_number == ref_clean)
                or (core_ref and _core_doc_number(meta.doc_number) == core_ref)
            ):
                matches.append(meta)
        return matches

    def _reconcile_edition_graph(self) -> set[str]:
        """
        Synchronize bidirectional edition links across all documents in self.registry:
        - If A supersedes B -> B.superseded_by includes A, and B.status becomes 'superseded'
          (unless B is 'superseded_template_reference').
        - If B amends A -> A.amended_by includes B, and A.version_note notes the amendment.
        Returns set of doc_ids whose metadata changed so their Qdrant payload and u00_meta can be updated.
        """
        changed_doc_ids: set[str] = set()

        for doc_a in list(self.registry.values()):
            ref_a = _core_doc_number(doc_a.doc_number) or doc_a.doc_number or doc_a.doc_id

            # 1. Forward supersedes: doc_a supersedes target_ref
            for target_ref in list(doc_a.supersedes):
                for doc_b in self._find_matching_docs(target_ref):
                    if doc_b.doc_id == doc_a.doc_id:
                        continue
                    modified = False
                    if ref_a not in doc_b.superseded_by:
                        doc_b.superseded_by.append(ref_a)
                        modified = True
                    if doc_b.status == DocumentStatus.ACTIVE.value:
                        # Preserve superseded_template_reference if configured in overrides
                        file_ovr = self.overrides_data.get("documents", {}).get(doc_b.file_name, {})
                        if file_ovr.get("status") == DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value:
                            doc_b.status = DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value
                        else:
                            doc_b.status = DocumentStatus.SUPERSEDED.value
                        modified = True
                    note_fragment = f"Отменен / заменен документом № {ref_a}"
                    if ref_a not in (doc_b.version_note or ""):
                        doc_b.version_note = (
                            f"{doc_b.version_note.rstrip('. ')}. {note_fragment}."
                            if doc_b.version_note
                            else f"{note_fragment} ({doc_a.title})."
                        )
                        modified = True
                    if modified:
                        changed_doc_ids.add(doc_b.doc_id)

            # 2. Forward amends: doc_a amends target_ref
            for target_ref in list(doc_a.amends):
                for doc_b in self._find_matching_docs(target_ref):
                    if doc_b.doc_id == doc_a.doc_id:
                        continue
                    modified = False
                    if ref_a not in doc_b.amended_by:
                        doc_b.amended_by.append(ref_a)
                        modified = True
                    if ref_a not in (doc_b.version_note or ""):
                        note_fragment = (
                            f"ВНИМАНИЕ (Частичное изменение): действует с учетом изменений, "
                            f"внесенных документом № {ref_a} ({doc_a.title})."
                        )
                        doc_b.version_note = (
                            f"{doc_b.version_note.rstrip('. ')}. {note_fragment}"
                            if doc_b.version_note
                            else note_fragment
                        )
                        modified = True
                    if modified:
                        changed_doc_ids.add(doc_b.doc_id)

            # 3. Reverse superseded_by: doc_a is superseded_by source_ref
            for source_ref in list(doc_a.superseded_by):
                for doc_s in self._find_matching_docs(source_ref):
                    if doc_s.doc_id == doc_a.doc_id:
                        continue
                    if ref_a and ref_a not in doc_s.supersedes and doc_a.file_name not in doc_s.supersedes:
                        doc_s.supersedes.append(ref_a)
                        changed_doc_ids.add(doc_s.doc_id)

            # 4. Reverse amended_by: doc_a is amended_by source_ref
            for source_ref in list(doc_a.amended_by):
                for doc_s in self._find_matching_docs(source_ref):
                    if doc_s.doc_id == doc_a.doc_id:
                        continue
                    if ref_a and ref_a not in doc_s.amends and doc_a.file_name not in doc_s.amends:
                        doc_s.amends.append(ref_a)
                        changed_doc_ids.add(doc_s.doc_id)

        return changed_doc_ids

    def _sync_doc_metadata_to_qdrant(self, doc_meta: DocumentMeta) -> None:
        """Push updated DocumentMeta fields and refreshed u00_meta chunk to Qdrant."""
        patch = {
            "file_name": doc_meta.file_name,
            "doc_number": doc_meta.doc_number,
            "doc_date": doc_meta.doc_date,
            "doc_type": doc_meta.doc_type,
            "title": doc_meta.title,
            "status": doc_meta.status,
            "applicable_years": list(doc_meta.applicable_years),
            "supersedes": list(doc_meta.supersedes),
            "superseded_by": list(doc_meta.superseded_by),
            "amends": list(doc_meta.amends),
            "amended_by": list(doc_meta.amended_by),
            "version_note": doc_meta.version_note,
            "quality_warning": doc_meta.quality_warning,
            "also_in_files": list(doc_meta.also_in_files),
        }
        self.qdrant.update_document_payload(doc_meta.doc_id, patch)

        meta_chunk = self.build_u00_meta_chunk(doc_meta)
        if meta_chunk is not None:
            try:
                dense_vecs, sparse_vecs = self.embedder.embed_chunks([meta_chunk.embed_text])
                self.qdrant.upsert_chunks([meta_chunk], dense_vecs, sparse_vecs, doc_meta=doc_meta)
            except Exception as exc:
                logger.warning("Failed to refresh u00_meta chunk for %s: %s", doc_meta.doc_id, exc)

    # ------------------------------------------------------------------
    # Cross-File Deduplication via content_hash
    # ------------------------------------------------------------------
    def _apply_cross_file_deduplication(
        self,
        doc_meta: DocumentMeta,
        chunks: list[ChunkRecord],
    ) -> int:
        """
        Detect identical chunks across other indexed documents by content_hash,
        linking source filenames in chunk.also_in_files and doc_meta.also_in_files.
        Returns the count of cross-file duplicated chunks found.
        """
        dup_count = 0
        other_docs = [m for d_id, m in self.registry.items() if d_id != doc_meta.doc_id]
        if not other_docs:
            return 0

        # Build content_hash -> list[(other_doc, chunk)] from Qdrant for other indexed docs
        hash_index: dict[str, list[tuple[DocumentMeta, ChunkRecord]]] = {}
        for other_meta in other_docs:
            other_chunks = self.qdrant.get_document_chunks(other_meta.doc_id)
            for oc in other_chunks:
                if oc.zone in {ZoneType.APPROVAL.value, ZoneType.META.value}:
                    continue
                chash = oc.content_hash or compute_content_hash(oc.clean_text)
                if chash:
                    hash_index.setdefault(chash, []).append((other_meta, oc))

        touched_other_docs: set[str] = set()
        for chunk in chunks:
            if chunk.zone in {ZoneType.APPROVAL.value, ZoneType.META.value}:
                continue
            chash = chunk.content_hash or compute_content_hash(chunk.clean_text)
            chunk.content_hash = chash
            if not chash or chash not in hash_index:
                continue

            dup_count += 1
            for other_meta, other_chunk in hash_index[chash]:
                if other_meta.file_name and other_meta.file_name != doc_meta.file_name:
                    if other_meta.file_name not in chunk.also_in_files:
                        chunk.also_in_files.append(other_meta.file_name)
                    if other_meta.file_name not in doc_meta.also_in_files:
                        doc_meta.also_in_files.append(other_meta.file_name)

                if doc_meta.file_name and doc_meta.file_name != other_meta.file_name:
                    if doc_meta.file_name not in other_chunk.also_in_files:
                        new_also = list(dict.fromkeys(list(other_chunk.also_in_files) + [doc_meta.file_name]))
                        other_chunk.also_in_files = new_also
                        self.qdrant.update_chunk_payload_by_uuid(
                            other_chunk.point_uuid,
                            {"also_in_files": new_also},
                        )
                    if doc_meta.file_name not in other_meta.also_in_files:
                        other_meta.also_in_files.append(doc_meta.file_name)
                        touched_other_docs.add(other_meta.doc_id)

        for other_id in touched_other_docs:
            other_m = self.registry.get(other_id)
            if other_m:
                self.qdrant.update_document_payload(
                    other_id,
                    {"also_in_files": list(other_m.also_in_files)},
                )

        return dup_count

    # ------------------------------------------------------------------
    # Single File Indexing
    # ------------------------------------------------------------------
    def index_file(
        self,
        file_path: str | Path,
        metadata_overrides: dict[str, Any] | None = None,
        ocr_backend: str | None = None,
        force: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexingFileStatus:
        """
        Fault-tolerantly index a single document (.pdf, .docx, .xlsx) through all 6 stages.
        Skips unchanged files by SHA-256 unless force=True or metadata_overrides are provided.
        """
        t0 = time.perf_counter()
        path = Path(file_path).resolve()
        file_name = path.name

        if not path.exists() or not path.is_file():
            err_msg = f"Файл не найден: {path}"
            self._emit_progress(progress_callback, "error", 1, "Ошибка", 100, err_msg, file_name=file_name)
            return IndexingFileStatus(
                file_name=file_name,
                file_path=str(path),
                status="error",
                error_message=err_msg,
                duration_sec=round(time.perf_counter() - t0, 3),
                timestamp=_utc_now_iso(),
            )

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            err_msg = f"Неподдерживаемый формат файла: {path.suffix}"
            self._emit_progress(progress_callback, "error", 1, "Ошибка", 100, err_msg, file_name=file_name)
            return IndexingFileStatus(
                file_name=file_name,
                file_path=str(path),
                status="error",
                error_message=err_msg,
                duration_sec=round(time.perf_counter() - t0, 3),
                timestamp=_utc_now_iso(),
            )

        try:
            sha256 = compute_file_sha256(path)

            # Check if file is already indexed with identical SHA-256
            existing_meta = next(
                (m for m in self.registry.values() if m.file_name == file_name or m.sha256 == sha256),
                None,
            )
            if (
                not force
                and not metadata_overrides
                and existing_meta is not None
                and existing_meta.sha256 == sha256
                and existing_meta.status != DocumentStatus.ERROR.value
                and existing_meta.chunk_count > 0
                and self.qdrant.count_points(doc_id=existing_meta.doc_id) > 0
            ):
                msg = f"Файл «{file_name}» не изменился (SHA-256 совпадает), пропуск."
                self._emit_progress(
                    progress_callback,
                    "done",
                    6,
                    "Пропущен (SHA-256)",
                    100,
                    msg,
                    file_name=file_name,
                    doc_id=existing_meta.doc_id,
                    parent_count=existing_meta.parent_count,
                    chunk_count=existing_meta.chunk_count,
                    ocr_pages=existing_meta.ocr_pages,
                )
                return IndexingFileStatus(
                    file_name=file_name,
                    file_path=str(path),
                    doc_id=existing_meta.doc_id,
                    sha256=sha256,
                    status="skipped",
                    doc_number=existing_meta.doc_number,
                    doc_date=existing_meta.doc_date,
                    pages_total=existing_meta.pages_total,
                    ocr_pages=existing_meta.ocr_pages,
                    parent_sections_created=existing_meta.parent_count,
                    chunks_created=existing_meta.chunk_count,
                    duration_sec=round(time.perf_counter() - t0, 3),
                    timestamp=_utc_now_iso(),
                )

            # Stage 1: Parsing
            self._emit_progress(
                progress_callback,
                "parsing",
                1,
                "Предразбор документа",
                12,
                f"Предразбор структуры файла «{file_name}»...",
                file_name=file_name,
            )
            eff_ocr_backend = ocr_backend or settings.OCR_BACKEND
            blocks, parser_meta = self._invoke_parser(path, ocr_backend=eff_ocr_backend)
            if not blocks:
                raise ValueError(f"Парсер не извлек ни одного текстового или табличного блока из «{file_name}»")

            # Stage 2: OCR inspection report
            doc_meta = self._build_initial_doc_meta(
                file_path=path,
                sha256=sha256,
                parser_meta=parser_meta,
                blocks=blocks,
                metadata_overrides=metadata_overrides,
            )
            self._emit_progress(
                progress_callback,
                "ocr",
                2,
                "OCR сканированных страниц",
                30,
                f"Распознано страниц: {doc_meta.pages_total} (OCR: {doc_meta.ocr_pages})",
                file_name=file_name,
                doc_id=doc_meta.doc_id,
                ocr_pages=doc_meta.ocr_pages,
            )

            # Stage 3: Hierarchical Chunking V3 (Stages 1-5: Parent-Child)
            self._emit_progress(
                progress_callback,
                "chunking",
                3,
                "Иерархический чанкинг V3",
                50,
                f"Построение 5-уровневого дерева и Parent-разделов для «{file_name}»...",
                file_name=file_name,
                doc_id=doc_meta.doc_id,
                ocr_pages=doc_meta.ocr_pages,
            )
            chunks, parent_sections, doc_meta = self._invoke_chunker(blocks, doc_meta)

            # Ensure synthetic u00_meta chunk is up-to-date
            chunks = self._ensure_u00_meta_in_chunks(doc_meta, chunks)
            doc_meta.parent_count = len(parent_sections)
            doc_meta.chunk_count = len(chunks)

            # Stage 4: Bilateral 12% Overlap & embed_text readiness
            self._emit_progress(
                progress_callback,
                "overlap",
                4,
                "Наложение 12% overlap",
                65,
                f"Сформировано {len(parent_sections)} Parent-разделов и {len(chunks)} Child-чанков с 12% overlap",
                file_name=file_name,
                doc_id=doc_meta.doc_id,
                parent_count=len(parent_sections),
                chunk_count=len(chunks),
                ocr_pages=doc_meta.ocr_pages,
            )

            # Cross-file deduplication via normalized content_hash
            dedup_count = self._apply_cross_file_deduplication(doc_meta, chunks)

            # Stage 5: Dense (baai/bge-m3) + Sparse BM25 (CRC32) Embedding
            self._emit_progress(
                progress_callback,
                "embedding",
                5,
                "Векторизация baai/bge-m3 + BM25",
                82,
                f"Вычисление Dense (1024d) и Sparse BM25 векторов для {len(chunks)} чанков...",
                file_name=file_name,
                doc_id=doc_meta.doc_id,
                parent_count=len(parent_sections),
                chunk_count=len(chunks),
                ocr_pages=doc_meta.ocr_pages,
            )
            embed_inputs = [c.embed_text or c.raw_text or c.clean_text for c in chunks]
            dense_vecs, sparse_vecs = self.embedder.embed_chunks(embed_inputs)

            # Stage 6: Qdrant Upsert + Atomic Persistence + Edition Graph Reconciliation
            self._emit_progress(
                progress_callback,
                "qdrant",
                6,
                "Запись в Qdrant и реестр",
                94,
                f"Запись {len(chunks)} точек в Qdrant и обновление графа редакций...",
                file_name=file_name,
                doc_id=doc_meta.doc_id,
                parent_count=len(parent_sections),
                chunk_count=len(chunks),
                ocr_pages=doc_meta.ocr_pages,
            )

            # Idempotent replace of old chunks & sections for doc_id
            self.qdrant.delete_by_doc_id(doc_meta.doc_id)
            old_sec_ids = [sid for sid, s in self.sections_store.items() if s.doc_id == doc_meta.doc_id]
            for sid in old_sec_ids:
                self.sections_store.pop(sid, None)

            for sec in parent_sections:
                self.sections_store[sec.section_id] = sec

            self.registry[doc_meta.doc_id] = doc_meta
            changed_graph_ids = self._reconcile_edition_graph()

            # Re-ensure u00_meta on current doc if reconciliation added links to it
            if doc_meta.doc_id in changed_graph_ids:
                chunks = self._ensure_u00_meta_in_chunks(doc_meta, chunks)
                doc_meta.chunk_count = len(chunks)
                embed_inputs = [c.embed_text or c.raw_text or c.clean_text for c in chunks]
                dense_vecs, sparse_vecs = self.embedder.embed_chunks(embed_inputs)

            self.qdrant.upsert_chunks(chunks, dense_vecs, sparse_vecs, doc_meta=doc_meta)

            # Sync any other documents affected by bidirectional edition graph links
            for other_doc_id in changed_graph_ids:
                if other_doc_id != doc_meta.doc_id and other_doc_id in self.registry:
                    self._sync_doc_metadata_to_qdrant(self.registry[other_doc_id])

            self.save_sections_store()
            self.save_registry()

            duration = round(time.perf_counter() - t0, 3)
            warnings_list: list[str] = []
            if doc_meta.quality_warning:
                warnings_list.append(doc_meta.quality_warning)
            if dedup_count > 0:
                warnings_list.append(
                    f"Обнаружено {dedup_count} кросс-дублирующихся чанков (источники объединены в also_in_files: {', '.join(doc_meta.also_in_files)})"
                )

            status_entry = IndexingFileStatus(
                file_name=file_name,
                file_path=str(path),
                doc_id=doc_meta.doc_id,
                sha256=sha256,
                status="success",
                doc_number=doc_meta.doc_number,
                doc_date=doc_meta.doc_date,
                pages_total=doc_meta.pages_total,
                ocr_pages=doc_meta.ocr_pages,
                parent_sections_created=len(parent_sections),
                chunks_created=len(chunks),
                deduplicated_chunks=dedup_count,
                duration_sec=duration,
                warnings=warnings_list,
                timestamp=_utc_now_iso(),
            )

            # Update single-file run in indexing_report.json as well
            self._append_single_file_to_report(status_entry)

            self._emit_progress(
                progress_callback,
                "done",
                6,
                "Готово",
                100,
                f"Успешно проиндексирован «{file_name}»: {len(parent_sections)} разделов, {len(chunks)} чанков ({duration:.1f} с)",
                file_name=file_name,
                doc_id=doc_meta.doc_id,
                parent_count=len(parent_sections),
                chunk_count=len(chunks),
                ocr_pages=doc_meta.ocr_pages,
            )
            return status_entry

        except Exception as exc:
            logger.exception("Error indexing file '%s': %s", file_name, exc)
            duration = round(time.perf_counter() - t0, 3)
            err_msg = f"{type(exc).__name__}: {exc}"
            self._emit_progress(
                progress_callback,
                "error",
                6,
                "Ошибка индексации",
                100,
                f"Ошибка при обработке «{file_name}»: {err_msg}",
                file_name=file_name,
            )
            err_status = IndexingFileStatus(
                file_name=file_name,
                file_path=str(path),
                status="error",
                error_message=err_msg,
                duration_sec=duration,
                timestamp=_utc_now_iso(),
            )
            self._append_single_file_to_report(err_status)
            return err_status

    def _append_single_file_to_report(self, status_entry: IndexingFileStatus) -> None:
        """Keep indexing_report.json up to date when single files are uploaded/re-indexed."""
        rep = self.last_report or IndexingReport()
        if not rep.run_id:
            rep.run_id = f"run_{uuid.uuid4().hex[:8]}"
            rep.started_at = _utc_now_iso()
        rep.finished_at = _utc_now_iso()
        # Replace previous entry for same file_name or append
        existing_idx = next(
            (i for i, f in enumerate(rep.files) if f.file_name == status_entry.file_name),
            None,
        )
        if existing_idx is not None:
            rep.files[existing_idx] = status_entry
        else:
            rep.files.append(status_entry)

        rep.total_files = len(rep.files)
        rep.indexed_files = sum(1 for f in rep.files if f.status == "success")
        rep.skipped_files = sum(1 for f in rep.files if f.status == "skipped")
        rep.failed_files = sum(1 for f in rep.files if f.status == "error")
        rep.total_parent_sections = len(self.sections_store)
        rep.total_chunks = sum(m.chunk_count for m in self.registry.values())
        rep.total_ocr_pages = sum(m.ocr_pages for m in self.registry.values())
        rep.total_duration_sec = round(sum(f.duration_sec for f in rep.files), 3)
        rep.errors = [f"{f.file_name}: {f.error_message}" for f in rep.files if f.error_message]
        all_warns: list[str] = []
        for f in rep.files:
            for w in f.warnings:
                all_warns.append(f"{f.file_name}: {w}")
        rep.warnings = all_warns
        self.save_report(rep)

    # ------------------------------------------------------------------
    # Batch / Incremental Folder Sync ('доки/')
    # ------------------------------------------------------------------
    def sync_folder(
        self,
        folder_path: str | Path | None = None,
        force: bool = False,
        ocr_backend: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexingReport:
        """
        Incrementally synchronize all supported documents (.pdf, .docx, .xlsx) in folder_path (defaults to DOCS_DIR).
        Isolates failures per file and writes a full IndexingReport to data/indexing_report.json.
        """
        t_start = time.perf_counter()
        started_at = _utc_now_iso()
        target_dir = Path(folder_path).resolve() if folder_path else DOCS_DIR.resolve()

        if not target_dir.exists() or not target_dir.is_dir():
            err_rep = IndexingReport(
                run_id=f"sync_{uuid.uuid4().hex[:8]}",
                started_at=started_at,
                finished_at=_utc_now_iso(),
                errors=[f"Директория не найдена: {target_dir}"],
            )
            self.save_report(err_rep)
            return err_rep

        files = sorted(
            [
                p
                for p in target_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTENSIONS
                and not p.name.startswith("~$")
            ],
            key=lambda p: p.name,
        )

        file_statuses: list[IndexingFileStatus] = []
        errors: list[str] = []
        warnings: list[str] = []
        total_files = len(files)

        for idx, fpath in enumerate(files, start=1):
            def _scoped_progress(ev: dict[str, Any], _idx: int = idx, _tot: int = total_files) -> None:
                if progress_callback is None:
                    return
                overall_pct = int(((_idx - 1) + (ev.get("percent", 0) / 100.0)) / max(1, _tot) * 100)
                patched = dict(ev)
                patched["percent"] = max(1, min(99 if ev.get("stage") != "done" or _idx < _tot else 100, overall_pct))
                patched["message"] = f"[{_idx}/{_tot}] {ev.get('message', '')}"
                progress_callback(patched)

            st = self.index_file(
                file_path=fpath,
                ocr_backend=ocr_backend,
                force=force,
                progress_callback=_scoped_progress if progress_callback else None,
            )
            file_statuses.append(st)
            if st.status == "error" and st.error_message:
                errors.append(f"{st.file_name}: {st.error_message}")
            for w in st.warnings:
                warnings.append(f"{st.file_name}: {w}")

        # Finalize bidirectional edition graph across all indexed documents
        changed_ids = self._reconcile_edition_graph()
        for doc_id in changed_ids:
            if doc_id in self.registry:
                self._sync_doc_metadata_to_qdrant(self.registry[doc_id])
        self.save_registry()

        indexed_cnt = sum(1 for s in file_statuses if s.status == "success")
        skipped_cnt = sum(1 for s in file_statuses if s.status == "skipped")
        failed_cnt = sum(1 for s in file_statuses if s.status == "error")
        total_dur = round(time.perf_counter() - t_start, 3)

        report = IndexingReport(
            run_id=f"sync_{uuid.uuid4().hex[:8]}",
            started_at=started_at,
            finished_at=_utc_now_iso(),
            total_files=total_files,
            indexed_files=indexed_cnt,
            skipped_files=skipped_cnt,
            failed_files=failed_cnt,
            total_parent_sections=len(self.sections_store),
            total_chunks=sum(m.chunk_count for m in self.registry.values()),
            total_ocr_pages=sum(m.ocr_pages for m in self.registry.values()),
            total_duration_sec=total_dur,
            files=file_statuses,
            errors=errors,
            warnings=warnings,
        )
        self.save_report(report)

        self._emit_progress(
            progress_callback,
            "done",
            6,
            "Синхронизация завершена",
            100,
            f"Синхронизировано {total_files} файлов: {indexed_cnt} проиндексировано, {skipped_cnt} пропущено, {failed_cnt} ошибок за {total_dur:.1f} с",
            parent_count=report.total_parent_sections,
            chunk_count=report.total_chunks,
            ocr_pages=report.total_ocr_pages,
        )
        return report

    # ------------------------------------------------------------------
    # Document Registry & Metadata Management (GET / PATCH / DELETE)
    # ------------------------------------------------------------------
    def get_documents(self) -> list[DocumentMeta]:
        """Return all registered documents sorted by doc_date desc, file_name asc."""
        return sorted(
            self.registry.values(),
            key=lambda m: (m.doc_date or "0000-00-00", m.file_name),
            reverse=True,
        )

    list_documents = get_documents

    def get_registry(self) -> dict[str, DocumentMeta]:
        return dict(self.registry)

    def get_document(self, doc_id: str) -> DocumentMeta | None:
        return self.registry.get(doc_id)

    def get_indexing_report(self) -> IndexingReport:
        return self.last_report

    def get_parent_section(self, section_id: str) -> ParentSectionRecord | None:
        return self.sections_store.get(section_id)

    def get_sections_store(self) -> dict[str, ParentSectionRecord]:
        return self.sections_store

    def get_document_chunks(self, doc_id: str) -> dict[str, Any]:
        """Return document metadata, all ParentSectionRecords, and all ChunkRecords for the Document Inspector."""
        doc_meta = self.registry.get(doc_id)
        if doc_meta is None:
            raise KeyError(f"Документ с doc_id='{doc_id}' не найден в реестре")

        sections = [
            sec for sec in self.sections_store.values() if sec.doc_id == doc_id
        ]
        sections.sort(key=lambda s: (s.unit_id, s.section_id))
        chunks = self.qdrant.get_document_chunks(doc_id)
        return {
            "document": doc_meta,
            "sections": sections,
            "chunks": chunks,
        }

    def update_document_metadata(self, doc_id: str, patch_dict: dict[str, Any]) -> DocumentMeta:
        """
        Update document metadata (status, supersedes, amends, applicable_years, version_note, etc.),
        reconcile the bidirectional edition graph across all documents, refresh u00_meta, and update Qdrant payload.
        """
        doc_meta = self.registry.get(doc_id)
        if doc_meta is None:
            raise KeyError(f"Документ с doc_id='{doc_id}' не найден в реестре")

        current_data = doc_meta.model_dump()
        for key, val in patch_dict.items():
            if val is not None and key in current_data and key != "doc_id":
                current_data[key] = val

        updated_meta = DocumentMeta.model_validate(current_data)
        self.registry[doc_id] = updated_meta

        changed_ids = self._reconcile_edition_graph()
        changed_ids.add(doc_id)

        for target_id in changed_ids:
            if target_id in self.registry:
                self._sync_doc_metadata_to_qdrant(self.registry[target_id])

        self.save_registry()
        return self.registry[doc_id]

    def delete_document(self, doc_id: str) -> bool:
        """
        Completely delete a document from Qdrant, data/sections_store.json, and data/document_registry.json,
        and clean up its references in the edition graph.
        """
        doc_meta = self.registry.pop(doc_id, None)
        self.qdrant.delete_by_doc_id(doc_id)

        sec_ids_to_remove = [sid for sid, sec in self.sections_store.items() if sec.doc_id == doc_id]
        for sid in sec_ids_to_remove:
            self.sections_store.pop(sid, None)

        if doc_meta is not None:
            ref_to_remove = _core_doc_number(doc_meta.doc_number) or doc_meta.doc_number
            refs = {doc_id, doc_meta.file_name, doc_meta.doc_number, ref_to_remove} - {""}
            for other in self.registry.values():
                modified = False
                for list_attr in ("superseded_by", "amended_by", "also_in_files"):
                    cur_list = getattr(other, list_attr)
                    filtered = [x for x in cur_list if x not in refs]
                    if len(filtered) != len(cur_list):
                        setattr(other, list_attr, filtered)
                        modified = True
                if modified:
                    self._sync_doc_metadata_to_qdrant(other)

        self.save_sections_store()
        self.save_registry()
        return doc_meta is not None

    def get_health(self) -> dict[str, Any]:
        """Return comprehensive health status for GET /api/health."""
        qdrant_health = self.qdrant.check_health()
        polza_health = self.embedder.check_health()
        overall_ok = bool(qdrant_health.get("connected")) and bool(polza_health.get("configured"))
        return {
            "status": "ok" if overall_ok else "degraded",
            "qdrant": qdrant_health,
            "polza_api": polza_health,
            "registry_docs_count": len(self.registry),
            "sections_count": len(self.sections_store),
        }


_PIPELINE_SINGLETON: IndexingPipeline | None = None


def get_pipeline() -> IndexingPipeline:
    """Return process-wide IndexingPipeline singleton."""
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = IndexingPipeline()
    return _PIPELINE_SINGLETON
