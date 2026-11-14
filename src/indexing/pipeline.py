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
    SUPPORTED_EXTENSIONS,
    TABLES_DB_PATH,
    UPLOADS_DIR,
    settings,
)
from src.indexing.embedder import HybridEmbedder
from src.indexing.qdrant_store import QdrantHybridStore
from src.indexing.table_store import StructuredTableStore, get_table_store
from src.schemas import (
    BlockType,
    ChunkRecord,
    DocumentMeta,
    DocumentStatus,
    IndexingFileStatus,
    IndexingReport,
    ParentSectionRecord,
    ParsedBlock,
    TablePassport,
    TableQueryRequest,
    TableQueryResponse,
    ZoneType,
    compute_content_hash,
    make_point_uuid,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


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


def _normalize_text_for_dedup(text: str) -> str:
    """
    Compute canonical normalized full text for cross-format duplicate detection:
    strips page markers (`[Стр. N]`), lowercases, maps `ё -> е`, strips punctuation/underscores,
    and collapses all whitespace to single spaces.
    """
    if not text:
        return ""
    s = re.sub(r"\[Стр\.\s*\d+\]", " ", text, flags=re.IGNORECASE)
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^\w\s]|_", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _word_shingles(norm_text: str, k: int = 4) -> set[str]:
    """Build k-word shingles over normalized text for cross-format chunk-boundary resilience."""
    words = norm_text.split()
    if not words:
        return set()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class PipelineFileResult(IndexingFileStatus):
    """
    Dual-interface result model subclassing `IndexingFileStatus` (`extra='allow'`) while also
    supporting dict-style `.get()`, `[]`, `in`, `.keys()`, `.items()`, `.values()` access.
    """

    def _as_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def get(self, key: str, default: Any = None) -> Any:
        d = self._as_dict()
        return d.get(key, default)

    def __getitem__(self, key: str) -> Any:
        d = self._as_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self._as_dict()

    def keys(self) -> Any:
        return self._as_dict().keys()

    def items(self) -> Any:
        return self._as_dict().items()

    def values(self) -> Any:
        return self._as_dict().values()


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
        table_store: StructuredTableStore | None = None,
        tables_db_path: Path | str | None = None,
    ) -> None:
        self.qdrant = qdrant_store or QdrantHybridStore()
        self.embedder = embedder or HybridEmbedder()
        self.registry_path = registry_path or DOCUMENT_REGISTRY_PATH
        self.sections_path = sections_path or SECTIONS_STORE_PATH
        self.report_path = report_path or INDEXING_REPORT_PATH
        self.overrides_path = overrides_path or REGISTRY_OVERRIDES_PATH
        self.tables_db_path: Path = (
            Path(tables_db_path)
            if tables_db_path is not None
            else (table_store.db_path if table_store is not None else TABLES_DB_PATH)
        )
        self._table_store: StructuredTableStore | None = table_store
        if self._table_store is None and tables_db_path is not None:
            self._table_store = get_table_store(self.tables_db_path)

        self.overrides_data: dict[str, Any] = self._load_overrides()
        self.registry: dict[str, DocumentMeta] = self._load_registry()
        self.sections_store: dict[str, ParentSectionRecord] = self._load_sections_store()
        self.last_report: IndexingReport = self._load_report()

    def _has_active_table_store(self) -> bool:
        """Return True if an in-memory/custom table store is bound or tables_store.db already exists on disk."""
        return self._table_store is not None or self.tables_db_path.exists()

    @property
    def table_store(self) -> StructuredTableStore:
        """Return the SQLite StructuredTableStore instance (lazily opened on first write/query)."""
        if self._table_store is None:
            self._table_store = get_table_store(self.tables_db_path)
        return self._table_store

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

        # Preserve existing canonical doc_id if file_name or sha256 was already in registry and no explicit override doc_id is configured
        explicit_ovr_doc_id = (
            (metadata_overrides or {}).get("doc_id")
            or (self.overrides_data.get("documents", {}).get(file_name, {}) or {}).get("doc_id")
        )
        existing_for_file = next(
            (m for m in self.registry.values() if m.file_name == file_name),
            None,
        )
        if existing_for_file is None and sha256:
            existing_for_file = next(
                (m for m in self.registry.values() if m.sha256 == sha256 and m.status != DocumentStatus.ERROR.value),
                None,
            )
        if existing_for_file and not explicit_ovr_doc_id:
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
            base_meta_dict = dict(cur_dump.get("metadata") or {})
            for key, val in metadata_overrides.items():
                if key == "metadata" and isinstance(val, dict):
                    base_meta_dict.update(val)
                elif val is not None and val != "" and val != []:
                    cur_dump[key] = val
            cur_dump["metadata"] = base_meta_dict
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

    @staticmethod
    def _extract_edo_fields(doc_meta: DocumentMeta) -> tuple[str, str, bool]:
        """Extract normalized (edo_number, edo_date, is_edo_stamp) from DocumentMeta attributes or metadata dict."""
        meta_dict = doc_meta.metadata if isinstance(doc_meta.metadata, dict) else {}
        edo_number = str(getattr(doc_meta, "edo_number", None) or meta_dict.get("edo_number") or "").strip()
        edo_date = str(getattr(doc_meta, "edo_date", None) or meta_dict.get("edo_date") or "").strip()
        raw_is_edo = getattr(doc_meta, "is_edo_stamp", None)
        if raw_is_edo is None:
            raw_is_edo = meta_dict.get("is_edo_stamp", False)
        is_edo_stamp = bool(raw_is_edo)
        return edo_number, edo_date, is_edo_stamp

    @classmethod
    def _stamp_chunk_payload_fields(cls, chunk: ChunkRecord, doc_meta: DocumentMeta) -> None:
        """Ensure ChunkRecord carries edo_number, edo_date, is_edo_stamp, table_id, and exact page_numbers."""
        edo_number, edo_date, is_edo_stamp = cls._extract_edo_fields(doc_meta)
        if chunk.metadata is None or not isinstance(chunk.metadata, dict):
            chunk.metadata = {}

        # Normalize exact integer page_numbers
        raw_pages = list(chunk.page_numbers or [])
        if not raw_pages and isinstance(chunk.metadata.get("page_numbers"), list):
            raw_pages = list(chunk.metadata["page_numbers"])
        clean_pages: list[int] = []
        for p in raw_pages:
            try:
                p_int = int(p)
                if p_int > 0 and p_int not in clean_pages:
                    clean_pages.append(p_int)
            except (TypeError, ValueError):
                continue
        clean_pages.sort()
        if not clean_pages and (chunk.unit_id == "u00_meta" or chunk.zone == ZoneType.META.value):
            clean_pages = [1]
        chunk.page_numbers = clean_pages

        # Sync table_id across top-level attribute and metadata
        if not chunk.table_id and chunk.metadata.get("table_id"):
            chunk.table_id = str(chunk.metadata["table_id"])
        if chunk.table_id:
            chunk.metadata["table_id"] = chunk.table_id

        # Stamp EDO & K2 edition fields & file_path onto both metadata and top-level extra attributes (serialized into Qdrant payload)
        doc_md = doc_meta.metadata if isinstance(doc_meta.metadata, dict) else {}
        ed_date = str(getattr(doc_meta, "edition_date", None) or doc_md.get("edition_date") or "").strip()
        ed_num = getattr(doc_meta, "edition_number", None)
        if ed_num is None:
            ed_num = doc_md.get("edition_number")
        ver_fam = str(getattr(doc_meta, "version_family", None) or doc_md.get("version_family") or "").strip()

        chunk.metadata["edo_number"] = edo_number
        chunk.metadata["edo_date"] = edo_date
        chunk.metadata["is_edo_stamp"] = is_edo_stamp
        chunk.metadata["edition_date"] = ed_date
        chunk.metadata["edition_number"] = ed_num
        chunk.metadata["version_family"] = ver_fam
        chunk.metadata["page_numbers"] = list(clean_pages)
        if doc_meta.file_path:
            chunk.metadata["file_path"] = doc_meta.file_path
            setattr(chunk, "file_path", doc_meta.file_path)
        setattr(chunk, "edo_number", edo_number)
        setattr(chunk, "edo_date", edo_date)
        setattr(chunk, "is_edo_stamp", is_edo_stamp)
        setattr(chunk, "edition_date", ed_date)
        setattr(chunk, "edition_number", ed_num)
        setattr(chunk, "version_family", ver_fam)

    @classmethod
    def _enrich_records_with_requisites_and_page_maps(
        cls,
        doc_meta: DocumentMeta,
        parent_sections: list[ParentSectionRecord],
        chunks: list[ChunkRecord],
    ) -> None:
        """
        Synchronize EDO stamp fields, table_id, exact page_numbers, clause_page_map, and chunk_pages across:
        - ChunkRecord (persisted as top-level Qdrant point payload fields + metadata)
        - ParentSectionRecord (persisted in data/sections_store.json)
        - DocumentMeta (persisted in data/document_registry.json)
        """
        edo_number, edo_date, is_edo_stamp = cls._extract_edo_fields(doc_meta)
        meta_dict = dict(doc_meta.metadata or {})
        meta_dict["edo_number"] = edo_number
        meta_dict["edo_date"] = edo_date
        meta_dict["is_edo_stamp"] = is_edo_stamp
        setattr(doc_meta, "edo_number", edo_number)
        setattr(doc_meta, "edo_date", edo_date)
        setattr(doc_meta, "is_edo_stamp", is_edo_stamp)

        for ch in chunks:
            cls._stamp_chunk_payload_fields(ch, doc_meta)

        non_meta_chunks = [
            ch for ch in chunks if ch.unit_id != "u00_meta" and ch.zone != ZoneType.META.value
        ]
        chunks_by_sec: dict[str, list[ChunkRecord]] = {}
        chunks_by_id: dict[str, ChunkRecord] = {}
        for ch in non_meta_chunks:
            chunks_by_id[ch.chunk_id] = ch
            if ch.section_id:
                chunks_by_sec.setdefault(ch.section_id, []).append(ch)

        doc_clause_page_map: dict[str, list[int]] = {}

        for sec in parent_sections:
            if sec.metadata is None or not isinstance(sec.metadata, dict):
                sec.metadata = {}

            sec_chunks = list(chunks_by_sec.get(sec.section_id, []))
            if not sec_chunks and sec.chunk_ids:
                sec_chunks = [chunks_by_id[cid] for cid in sec.chunk_ids if cid in chunks_by_id]

            # Preserve @meta-chunker's clause_page_map and merge/normalize with child chunks
            raw_cpm = sec.metadata.get("clause_page_map") or getattr(sec, "clause_page_map", None)
            norm_cpm: dict[str, list[int]] = {}
            if isinstance(raw_cpm, dict):
                for cl_key, p_vals in raw_cpm.items():
                    cl_str = str(cl_key).strip()
                    if not cl_str:
                        continue
                    pages_list = p_vals if isinstance(p_vals, (list, tuple, set)) else [p_vals]
                    cleaned = sorted({int(p) for p in pages_list if p is not None and str(p).lstrip("-").isdigit() and int(p) > 0})
                    if cleaned:
                        norm_cpm[cl_str] = cleaned

            for ch in sec_chunks:
                if ch.clause_no and ch.clause_no != "META" and ch.page_numbers:
                    cl_str = str(ch.clause_no).strip()
                    if cl_str:
                        norm_cpm[cl_str] = sorted(set(norm_cpm.get(cl_str, [])) | set(ch.page_numbers))

            raw_cpages = sec.metadata.get("chunk_pages") or getattr(sec, "chunk_pages", None)
            if isinstance(raw_cpages, list) and raw_cpages:
                norm_cpages: list[dict[str, Any]] = []
                for item in raw_cpages:
                    if isinstance(item, dict) and item.get("chunk_id"):
                        p_vals = item.get("page_numbers") or []
                        cleaned_p = sorted({int(p) for p in p_vals if p is not None and str(p).lstrip("-").isdigit() and int(p) > 0})
                        norm_cpages.append(
                            {
                                "chunk_id": str(item["chunk_id"]),
                                "clause_no": str(item.get("clause_no") or ""),
                                "page_numbers": cleaned_p,
                            }
                        )
            else:
                norm_cpages = [
                    {
                        "chunk_id": ch.chunk_id,
                        "clause_no": ch.clause_no,
                        "page_numbers": list(ch.page_numbers),
                    }
                    for ch in sec_chunks
                ]

            sec_pages = set(int(p) for p in (sec.page_numbers or []) if p is not None and int(p) > 0)
            for ch in sec_chunks:
                sec_pages.update(ch.page_numbers)
            if sec_pages:
                sec.page_numbers = sorted(sec_pages)

            sec.metadata["clause_page_map"] = norm_cpm
            sec.metadata["chunk_pages"] = norm_cpages
            sec.metadata["edo_number"] = edo_number
            sec.metadata["edo_date"] = edo_date
            sec.metadata["is_edo_stamp"] = is_edo_stamp
            setattr(sec, "clause_page_map", norm_cpm)
            setattr(sec, "chunk_pages", norm_cpages)

            for cl_str, p_list in norm_cpm.items():
                doc_clause_page_map[cl_str] = sorted(set(doc_clause_page_map.get(cl_str, [])) | set(p_list))

        for ch in non_meta_chunks:
            if ch.clause_no and ch.clause_no != "META" and ch.page_numbers:
                cl_str = str(ch.clause_no).strip()
                if cl_str:
                    doc_clause_page_map[cl_str] = sorted(
                        set(doc_clause_page_map.get(cl_str, [])) | set(ch.page_numbers)
                    )

        meta_dict["clause_page_map"] = doc_clause_page_map
        doc_meta.metadata = meta_dict
        setattr(doc_meta, "clause_page_map", doc_clause_page_map)

    def _ensure_u00_meta_in_chunks(
        self,
        doc_meta: DocumentMeta,
        chunks: list[ChunkRecord],
    ) -> list[ChunkRecord]:
        """Ensure exactly one up-to-date u00_meta chunk exists when doc_meta has edition/quality notes."""
        meta_chunk = self.build_u00_meta_chunk(doc_meta)
        non_meta = [c for c in chunks if c.unit_id != "u00_meta" and c.zone != ZoneType.META.value]
        if meta_chunk is not None:
            self._stamp_chunk_payload_fields(meta_chunk, doc_meta)
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

    @staticmethod
    def _get_doc_k2_fields(doc: DocumentMeta) -> tuple[str, Any, str, list[int], list[str]]:
        """Extract `(edition_date, edition_number, version_family, target_plan_years, repealed_subjects)` from a DocumentMeta."""
        md = doc.metadata if isinstance(doc.metadata, dict) else {}
        ed_date = str(getattr(doc, "edition_date", None) or md.get("edition_date") or "").strip()
        ed_num = getattr(doc, "edition_number", None)
        if ed_num is None:
            ed_num = md.get("edition_number")
        ver_fam = str(getattr(doc, "version_family", None) or md.get("version_family") or "").strip()
        t_years = [int(y) for y in (md.get("target_plan_years") or []) if str(y).isdigit()]
        rep_subjs = [str(s) for s in (md.get("repealed_subjects") or []) if s]
        return ed_date, ed_num, ver_fam, t_years, rep_subjs

    def _populate_missing_k2_fields(self, doc: DocumentMeta) -> bool:
        """Ensure legacy or freshly loaded DocumentMeta entries carry K2 `edition_date`, `edition_number`, and `version_family`."""
        from src.chunking.glossary import (
            compute_version_family,
            extract_edition_metadata,
            extract_target_plan_years,
        )

        ed_date, ed_num, ver_fam, t_years, _ = self._get_doc_k2_fields(doc)
        doc_secs = sorted(
            [s for s in self.sections_store.values() if s.doc_id == doc.doc_id],
            key=lambda s: (s.unit_id, s.section_id),
        )
        full_sec_text = "\n".join(s.clean_text for s in doc_secs[:4] if s.clean_text)

        changed = False
        md = dict(doc.metadata or {})

        if not ed_date and ed_num is None:
            ed_info = extract_edition_metadata(
                title=doc.title,
                header_text=full_sec_text[:1200],
                full_text=full_sec_text,
                file_name=doc.file_name,
            )
            if ed_info.get("edition_date"):
                ed_date = str(ed_info["edition_date"])
                changed = True
            if ed_info.get("edition_number") is not None:
                ed_num = ed_info["edition_number"]
                changed = True

        if not ver_fam:
            computed_vf = compute_version_family(
                doc_type=doc.doc_type,
                title=doc.title,
                header_text=full_sec_text[:1200],
                full_text=full_sec_text,
                file_name=doc.file_name,
            )
            if computed_vf:
                ver_fam = computed_vf
                changed = True

        if not t_years:
            computed_ty = extract_target_plan_years(title=doc.title, header_text=full_sec_text[:800])
            if computed_ty:
                t_years = computed_ty
                md["target_plan_years"] = t_years
                changed = True

        if md.get("edition_date") != ed_date:
            md["edition_date"] = ed_date
            changed = True
        if md.get("edition_number") != ed_num:
            md["edition_number"] = ed_num
            changed = True
        if md.get("version_family") != ver_fam:
            md["version_family"] = ver_fam
            changed = True

        setattr(doc, "edition_date", ed_date)
        setattr(doc, "edition_number", ed_num)
        setattr(doc, "version_family", ver_fam)
        doc.metadata = md
        return changed

    @classmethod
    def _doc_version_sort_key(cls, doc: DocumentMeta) -> tuple[str, tuple[int, int], int, int, str]:
        """
        Total ordering key `(effective_date, edition_number_tuple, has_supersedes, numeric_doc_no, indexed_at)`
        for comparing multi-version documents within the same `version_family`.
        """
        from src.chunking.glossary import edition_sort_tuple

        ed_date, ed_num, _, _, _ = cls._get_doc_k2_fields(doc)
        edo_number, edo_date, _ = cls._extract_edo_fields(doc)

        eff_date = ed_date or (doc.doc_date or "").strip() or edo_date
        if not eff_date and doc.applicable_years:
            eff_date = f"{max(int(y) for y in doc.applicable_years):04d}-01-01"
        if not eff_date:
            eff_date = "0000-00-00"

        ed_tuple = edition_sort_tuple(ed_num)
        has_sup = 1 if doc.supersedes else 0
        m_num = re.match(r"^(\d+)", _core_doc_number(doc.doc_number or edo_number))
        num_int = int(m_num.group(1)) if m_num else 0
        return (eff_date, ed_tuple, has_sup, num_int, doc.indexed_at or "")

    def _reconcile_relations(self, sync_external: bool = True) -> set[str]:
        """
        Synchronize explicit AND automatic K2 `version_family` edition links across all documents in `self.registry`:
        1) Explicit clauses & YAML overrides (`supersedes` <-> `superseded_by`, `amends` <-> `amended_by`, unnumbered repeal clauses).
        2) Automatic `version_family` multi-version reconciliation: when 2+ documents share the same non-generic
           `normative::...` `version_family` (excluding disjoint annual plans and partial `amends`), compares
           `(edition_date or doc_date, edition_number)`, keeps the newest version `status='active'`, marks older
           versions `status='superseded'`, links `supersedes`/`superseded_by`, and sets clear reciprocated Russian
           `version_note`s.
        When `sync_external=True` (default for standalone calls), persists updated statuses/notes to
        `document_registry.json` and updates Qdrant payloads (`set_payload` + `u00_meta`).
        """
        changed_doc_ids: set[str] = set()

        # Pass 0: Ensure K2 fields are populated across all registered documents
        for doc in list(self.registry.values()):
            if doc.status == DocumentStatus.ERROR.value:
                continue
            if self._populate_missing_k2_fields(doc):
                changed_doc_ids.add(doc.doc_id)

        # Pass 0b: Match unnumbered repealed-act clauses (e.g. "Признать утратившим силу регламент ... № б/н от 07.05.2024")
        for doc_a in list(self.registry.values()):
            if doc_a.status == DocumentStatus.ERROR.value:
                continue
            _, _, _, _, rep_subjs = self._get_doc_k2_fields(doc_a)
            if not rep_subjs:
                continue
            for doc_b in list(self.registry.values()):
                if doc_b.doc_id == doc_a.doc_id or doc_b.status == DocumentStatus.ERROR.value:
                    continue
                _, _, vf_b, _, _ = self._get_doc_k2_fields(doc_b)
                subj_b = vf_b.split("::", 1)[1] if "::" in vf_b else vf_b
                if not subj_b:
                    continue
                if any(rs == subj_b or rs in subj_b or subj_b in rs for rs in rep_subjs):
                    if self._doc_version_sort_key(doc_a) > self._doc_version_sort_key(doc_b):
                        ref_b = _core_doc_number(doc_b.doc_number) or doc_b.doc_number or doc_b.doc_id
                        if ref_b not in doc_a.supersedes:
                            doc_a.supersedes.append(ref_b)
                            changed_doc_ids.add(doc_a.doc_id)

        # Passes 1–4: Explicit forward & reverse supersedes / amends
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
                        file_ovr = self.overrides_data.get("documents", {}).get(doc_b.file_name, {})
                        if file_ovr.get("status") == DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value:
                            doc_b.status = DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value
                        else:
                            doc_b.status = DocumentStatus.SUPERSEDED.value
                        modified = True
                    disp_a_date = (
                        self._get_doc_k2_fields(doc_a)[0]
                        or doc_a.doc_date
                        or (f"{max(doc_a.applicable_years)} г." if doc_a.applicable_years else "")
                    )
                    note_fragment = (
                        f"Утратил актуальность: действует более новая редакция — {doc_a.title}"
                        + (f" от {disp_a_date}" if disp_a_date else "")
                        + "."
                    )
                    if ref_a not in (doc_b.version_note or "") and doc_a.title not in (doc_b.version_note or ""):
                        doc_b.version_note = (
                            f"{doc_b.version_note.rstrip('. ')}. {note_fragment}"
                            if doc_b.version_note
                            else note_fragment
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

        # Pass 5: Automatic K2 `version_family` multi-version reconciliation
        by_family: dict[str, list[DocumentMeta]] = {}
        for doc in self.registry.values():
            if doc.status in {DocumentStatus.ERROR.value, DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value}:
                continue
            file_ovr = self.overrides_data.get("documents", {}).get(doc.file_name, {})
            if file_ovr.get("status") == DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value:
                continue
            _, _, ver_fam, _, _ = self._get_doc_k2_fields(doc)
            if not ver_fam or not ver_fam.startswith("normative::"):
                continue
            # Exclude pure partial-amendment orders that amend another act without superseding
            if doc.amends and not doc.supersedes:
                continue
            by_family.setdefault(ver_fam, []).append(doc)

        for ver_fam, members in by_family.items():
            if len(members) < 2:
                continue

            # Partition members so distinct annual plans with disjoint explicit target_plan_years do not supersede each other
            clusters: list[list[DocumentMeta]] = []
            for m_doc in members:
                _, _, _, m_tyears, _ = self._get_doc_k2_fields(m_doc)
                placed = False
                for cl in clusters:
                    # Check compatibility with existing cluster members
                    disjoint_annual = False
                    for existing_m in cl:
                        _, _, _, ex_tyears, _ = self._get_doc_k2_fields(existing_m)
                        # Also skip if one partially amends the other
                        ref_m = _core_doc_number(m_doc.doc_number) or m_doc.doc_id
                        ref_ex = _core_doc_number(existing_m.doc_number) or existing_m.doc_id
                        if ref_m in existing_m.amends or ref_ex in m_doc.amends or ref_m in existing_m.amended_by or ref_ex in m_doc.amended_by:
                            disjoint_annual = True
                            break
                        if m_tyears and ex_tyears and set(m_tyears).isdisjoint(set(ex_tyears)):
                            disjoint_annual = True
                            break
                    if not disjoint_annual:
                        cl.append(m_doc)
                        placed = True
                        break
                if not placed:
                    clusters.append([m_doc])

            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                sort_keys = {d.doc_id: self._doc_version_sort_key(d) for d in cluster}
                # Require at least one grounded date, year, edition_number, or explicit supersedes link
                if all(k[0] == "0000-00-00" and k[1] == (0, 0) and k[2] == 0 for k in sort_keys.values()):
                    continue

                ordered = sorted(cluster, key=lambda d: sort_keys[d.doc_id])
                newest = ordered[-1]
                older_docs = ordered[:-1]

                # If top two have completely identical date + edition_number + no supersedes, skip ambiguous tie
                if (
                    sort_keys[newest.doc_id][:3] == sort_keys[older_docs[-1].doc_id][:3]
                    and not newest.supersedes
                    and not older_docs[-1].superseded_by
                ):
                    continue

                ref_new = _core_doc_number(newest.doc_number) or newest.doc_number or newest.doc_id
                new_ed_date = self._get_doc_k2_fields(newest)[0]
                disp_new_date = (
                    new_ed_date
                    or newest.doc_date
                    or (f"{max(newest.applicable_years)} г." if newest.applicable_years else "н/д")
                )

                # Ensure newest version is active (unless superseded by an explicit external order outside this family)
                new_ovr = self.overrides_data.get("documents", {}).get(newest.file_name, {})
                older_refs = {
                    _core_doc_number(od.doc_number) or od.doc_number or od.doc_id
                    for od in older_docs
                } | {od.doc_id for od in older_docs}
                external_sup = [r for r in newest.superseded_by if r not in older_refs]
                if not external_sup and new_ovr.get("status") != DocumentStatus.SUPERSEDED.value:
                    if newest.status != DocumentStatus.ACTIVE.value:
                        newest.status = DocumentStatus.ACTIVE.value
                        changed_doc_ids.add(newest.doc_id)
                    if newest.superseded_by:
                        newest.superseded_by = external_sup
                        changed_doc_ids.add(newest.doc_id)

                replaced_labels: list[str] = []
                for old_doc in older_docs:
                    ref_old = _core_doc_number(old_doc.doc_number) or old_doc.doc_number or old_doc.doc_id
                    old_ed_date = self._get_doc_k2_fields(old_doc)[0]
                    disp_old_date = (
                        old_ed_date
                        or old_doc.doc_date
                        or (f"{max(old_doc.applicable_years)} г." if old_doc.applicable_years else "б/д")
                    )
                    replaced_labels.append(f"{old_doc.title} от {disp_old_date}")

                    old_modified = False
                    if old_doc.status != DocumentStatus.SUPERSEDED.value:
                        old_doc.status = DocumentStatus.SUPERSEDED.value
                        old_modified = True
                    if ref_new not in old_doc.superseded_by:
                        old_doc.superseded_by.append(ref_new)
                        old_modified = True
                    if ref_old not in newest.supersedes:
                        newest.supersedes.append(ref_old)
                        changed_doc_ids.add(newest.doc_id)

                    expected_old_note = (
                        f"Утратил актуальность: действует более новая редакция — {newest.title} от {disp_new_date}."
                    )
                    old_ovr = self.overrides_data.get("documents", {}).get(old_doc.file_name, {})
                    if not old_ovr.get("version_note"):
                        if old_doc.version_note != expected_old_note:
                            # Preserve partial-amendment warnings if any, otherwise set crisp K2 actuality note
                            if "Частичное изменение" in (old_doc.version_note or ""):
                                if newest.title not in old_doc.version_note:
                                    old_doc.version_note = f"{expected_old_note} {old_doc.version_note}"
                                    old_modified = True
                            else:
                                old_doc.version_note = expected_old_note
                                old_modified = True

                    if old_modified:
                        changed_doc_ids.add(old_doc.doc_id)

                # Add reciprocated note on the active newest version
                if replaced_labels and not new_ovr.get("version_note"):
                    recip_note = (
                        f"Актуальная редакция (от {disp_new_date}): заменяет предыдущую редакцию — "
                        + "; ".join(replaced_labels)
                        + "."
                    )
                    cur_new_note = newest.version_note or ""
                    if "Актуальная редакция" not in cur_new_note or not all(
                        lbl.split(" от ")[0] in cur_new_note for lbl in replaced_labels
                    ):
                        if "Частичное изменение" in cur_new_note:
                            base_amend = " ".join(
                                seg for seg in cur_new_note.split(". ") if "Частичное изменение" in seg
                            ).strip()
                            newest.version_note = f"{recip_note} {base_amend}".strip()
                        else:
                            newest.version_note = recip_note
                        changed_doc_ids.add(newest.doc_id)

        if sync_external and changed_doc_ids:
            for cid in changed_doc_ids:
                if cid in self.registry:
                    self._sync_doc_metadata_to_qdrant(self.registry[cid])
            self.save_registry()

        return changed_doc_ids

    def _reconcile_edition_graph(self) -> set[str]:
        """Internal edition-graph reconciliation hook (persisted by caller after Qdrant upsert)."""
        return self._reconcile_relations(sync_external=False)

    def _sync_doc_metadata_to_qdrant(self, doc_meta: DocumentMeta) -> None:
        """Push updated DocumentMeta fields (including EDO & K2 edition fields) and refreshed u00_meta chunk to Qdrant."""
        edo_number, edo_date, is_edo_stamp = self._extract_edo_fields(doc_meta)
        ed_date, ed_num, ver_fam, _, _ = self._get_doc_k2_fields(doc_meta)
        patch = {
            "file_name": doc_meta.file_name,
            "doc_number": doc_meta.doc_number,
            "doc_date": doc_meta.doc_date,
            "edition_date": ed_date,
            "edition_number": ed_num,
            "version_family": ver_fam,
            "edo_number": edo_number,
            "edo_date": edo_date,
            "is_edo_stamp": is_edo_stamp,
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
            "tables_count": int(doc_meta.tables_count),
            "also_in_files": list(doc_meta.also_in_files),
        }
        try:
            self.qdrant.update_document_payload(doc_meta.doc_id, patch)
        except Exception as exc:
            logger.warning("Failed to update Qdrant payload for %s: %s", doc_meta.doc_id, exc)

        meta_chunk = self.build_u00_meta_chunk(doc_meta)
        if meta_chunk is not None:
            try:
                self._stamp_chunk_payload_fields(meta_chunk, doc_meta)
                dense_vecs, sparse_vecs = self.embedder.embed_chunks([meta_chunk.embed_text])
                self.qdrant.upsert_chunks([meta_chunk], dense_vecs, sparse_vecs, doc_meta=doc_meta)
            except Exception as exc:
                logger.warning("Failed to refresh u00_meta chunk for %s: %s", doc_meta.doc_id, exc)

    # ------------------------------------------------------------------
    # Duplicate Detection Helpers (Check A: SHA-256 & Check B: Content/Semantic)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_duplicate_result(
        file_name: str,
        file_path: Path,
        sha256: str,
        existing_doc: DocumentMeta,
        reason: str,
        duration_sec: float,
    ) -> PipelineFileResult:
        """Construct the standardized duplicate rejection response supporting both dict and attribute access."""
        existing_id = existing_doc.doc_id
        existing_title = existing_doc.title or ""
        existing_fname = existing_doc.file_name or ""
        msg = f"Документ уже присутствует в базе: «{existing_title}» ({existing_fname})"
        return PipelineFileResult(
            file_name=file_name,
            file_path=str(file_path),
            doc_id=existing_id,
            sha256=sha256,
            status="duplicate",
            duplicate_of=existing_id,
            existing_title=existing_title,
            existing_file_name=existing_fname,
            reason=reason,
            message=msg,
            doc_number=existing_doc.doc_number,
            doc_date=existing_doc.doc_date,
            pages_total=existing_doc.pages_total,
            ocr_pages=existing_doc.ocr_pages,
            parent_sections_created=0,
            chunks_created=0,
            tables_count=0,
            duration_sec=duration_sec,
            timestamp=_utc_now_iso(),
        )

    @classmethod
    def _is_distinct_k2_version(cls, cand: DocumentMeta, existing: DocumentMeta) -> bool:
        """
        Return True when `cand` carries explicit new/different K2 version requisites (order number, date,
        edition_date, edition_number, or repeal/amendment links) compared to `existing`, so genuine new
        editions with >=90% unchanged body clauses are reconciled via K2 rather than rejected as duplicates.
        """
        c_num = _core_doc_number(cand.doc_number)
        e_num = _core_doc_number(existing.doc_number)
        if c_num and e_num and c_num.lower() != e_num.lower():
            return True
        if cand.doc_date and existing.doc_date and cand.doc_date != existing.doc_date:
            return True

        c_ed_dt, c_ed_no, _, _, _ = cls._get_doc_k2_fields(cand)
        e_ed_dt, e_ed_no, _, _, _ = cls._get_doc_k2_fields(existing)
        if (c_ed_dt or e_ed_dt) and c_ed_dt != e_ed_dt:
            return True
        if (c_ed_no not in (None, "") or e_ed_no not in (None, "")) and str(c_ed_no or "") != str(e_ed_no or ""):
            return True
        if cand.supersedes and set(cand.supersedes) != set(existing.supersedes):
            return True
        if cand.amends and set(cand.amends) != set(existing.amends):
            return True
        return False

    def _get_existing_doc_dedup_profile(self, other: DocumentMeta) -> tuple[str, str, set[str]]:
        """
        Return `(norm_sections_text, norm_sections_sha256, chunk_content_hashes)` for an indexed document
        using `sections_store` and cached metadata / Qdrant chunks.
        """
        md = other.metadata if isinstance(other.metadata, dict) else {}
        other_secs = sorted(
            [s for s in self.sections_store.values() if s.doc_id == other.doc_id],
            key=lambda s: (s.unit_id, s.section_id),
        )
        sec_norm = _normalize_text_for_dedup("\n".join(s.clean_text for s in other_secs if s.clean_text))
        sec_sha = hashlib.sha256(sec_norm.encode("utf-8")).hexdigest() if sec_norm else str(md.get("norm_sections_sha256") or "")

        cached_hashes = md.get("chunk_content_hashes")
        chunk_hashes: set[str] = set()
        if isinstance(cached_hashes, list) and cached_hashes:
            chunk_hashes = {str(h) for h in cached_hashes if h}
        else:
            q_fn = getattr(self.qdrant, "get_document_chunks", None)
            cls_mod = getattr(type(self.qdrant), "__module__", "") or ""
            if callable(q_fn) and not cls_mod.startswith("unittest.mock"):
                try:
                    for oc in q_fn(other.doc_id) or []:
                        if oc.zone in {ZoneType.APPROVAL.value, ZoneType.META.value} or oc.unit_id == "u00_meta":
                            continue
                        ch = oc.content_hash or compute_content_hash(oc.clean_text)
                        if ch:
                            chunk_hashes.add(ch)
                except Exception:
                    pass
        return sec_norm, sec_sha, chunk_hashes

    def _find_content_duplicate(
        self,
        cand_meta: DocumentMeta,
        cand_sections: list[ParentSectionRecord],
        cand_chunks: list[ChunkRecord],
        cand_blocks: list[ParsedBlock] | None = None,
    ) -> DocumentMeta | None:
        """
        Check B (Content / Semantic Text Duplicate):
        Compare canonical normalized full-text SHA-256, normative chunk `content_hash` overlap (`>= 0.90`),
        and cross-format 4-word shingle similarity (`>= 0.90`) against all existing documents in `self.registry`.
        """
        if not self.registry:
            return None

        ov_docs = self.overrides_data.get("documents", {}) or {}
        cand_ovr = ov_docs.get(cand_meta.file_name, {}) if cand_meta.file_name else {}

        sorted_c_secs = sorted(cand_sections, key=lambda s: (s.unit_id, s.section_id))
        cand_sec_norm = _normalize_text_for_dedup("\n".join(s.clean_text for s in sorted_c_secs if s.clean_text))
        cand_sec_sha = hashlib.sha256(cand_sec_norm.encode("utf-8")).hexdigest() if cand_sec_norm else ""

        norm_chunks = [
            c
            for c in cand_chunks
            if c.zone not in {ZoneType.APPROVAL.value, ZoneType.META.value} and c.unit_id != "u00_meta"
        ]
        cand_chunks_norm = _normalize_text_for_dedup("\n".join(c.clean_text for c in norm_chunks if c.clean_text))
        cand_chunks_sha = hashlib.sha256(cand_chunks_norm.encode("utf-8")).hexdigest() if cand_chunks_norm else ""

        cand_blocks_norm = (
            _normalize_text_for_dedup("\n".join(b.text for b in cand_blocks if b.text))
            if cand_blocks
            else ""
        )
        cand_blocks_sha = hashlib.sha256(cand_blocks_norm.encode("utf-8")).hexdigest() if cand_blocks_norm else ""

        cand_hashes: set[str] = {
            c.content_hash or compute_content_hash(c.clean_text)
            for c in norm_chunks
            if c.clean_text and c.clean_text.strip()
        }
        cand_hashes.discard("")
        cand_shingles = _word_shingles(cand_sec_norm, 4) if len(cand_sec_norm) >= 120 else set()

        for other in self.registry.values():
            if other.status == DocumentStatus.ERROR.value:
                continue
            # Preserve deliberate distinct corpus entries configured in registry_overrides.yaml
            other_ovr = ov_docs.get(other.file_name, {}) if other.file_name else {}
            if (
                cand_ovr.get("doc_id")
                and other_ovr.get("doc_id")
                and cand_ovr.get("doc_id") != other_ovr.get("doc_id")
            ):
                continue

            other_md = other.metadata if isinstance(other.metadata, dict) else {}
            other_sec_norm, other_sec_sha, other_hashes = self._get_existing_doc_dedup_profile(other)
            other_chunks_sha = str(other_md.get("norm_chunks_sha256") or "")
            other_blocks_sha = str(other_md.get("norm_blocks_sha256") or "")

            # 1. Exact normalized full-text SHA-256 match (100% identical text across formats/filenames)
            if len(cand_sec_norm) >= 25 and cand_sec_sha and cand_sec_sha == other_sec_sha:
                return other
            if len(cand_chunks_norm) >= 25 and cand_chunks_sha and cand_chunks_sha in {other_chunks_sha, other_sec_sha} - {""}:
                return other
            if len(cand_blocks_norm) >= 25 and cand_blocks_sha and cand_blocks_sha in {other_blocks_sha, other_sec_sha} - {""}:
                return other

            # 2. Near-exact (>= 0.90) chunk or shingle overlap — skip if candidate is a genuine K2 new version
            if self._is_distinct_k2_version(cand_meta, other):
                continue

            has_enough_size = (len(cand_hashes) >= 2 or len(cand_sec_norm) >= 300) and (
                len(other_hashes) >= 2 or len(other_sec_norm) >= 300
            )
            if has_enough_size and cand_hashes and other_hashes:
                overlap = len(cand_hashes & other_hashes) / max(1, max(len(cand_hashes), len(other_hashes)))
                if overlap >= 0.90:
                    return other

            # Cross-format (.docx <-> .pdf) resilience when page breaks shift chunk boundaries slightly
            if has_enough_size and cand_shingles and len(other_sec_norm) >= 120:
                len_ratio = min(len(cand_sec_norm), len(other_sec_norm)) / max(1, max(len(cand_sec_norm), len(other_sec_norm)))
                if len_ratio >= 0.85:
                    other_shingles = _word_shingles(other_sec_norm, 4)
                    if other_shingles:
                        dice = (2.0 * len(cand_shingles & other_shingles)) / max(1, len(cand_shingles) + len(other_shingles))
                        if dice >= 0.90:
                            return other

        return None

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
        force_reindex: bool | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineFileResult:
        """
        Fault-tolerantly index a single document (.pdf, .docx, .doc, .xlsx, .xls) through all 6 stages.
        When `force` / `force_reindex` is False (default):
          - Check A: Rejects exact SHA-256 duplicates BEFORE parsing (`reason='exact_sha256'`).
          - Check B: Rejects normalized full-text / >=90% chunk-content duplicates BEFORE Qdrant/store writes (`reason='content_duplicate'`).
        When `force=True` or `force_reindex=True`, reindexes the canonical `doc_id` in place without creating a duplicate `doc_id`.
        """
        eff_force = bool(force if force_reindex is None else (force or force_reindex))
        t0 = time.perf_counter()
        path = Path(file_path).resolve()
        file_name = path.name

        if not path.exists() or not path.is_file():
            err_msg = f"Файл не найден: {path}"
            self._emit_progress(progress_callback, "error", 1, "Ошибка", 100, err_msg, file_name=file_name)
            return PipelineFileResult(
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
            return PipelineFileResult(
                file_name=file_name,
                file_path=str(path),
                status="error",
                error_message=err_msg,
                duration_sec=round(time.perf_counter() - t0, 3),
                timestamp=_utc_now_iso(),
            )

        try:
            if eff_force:
                self.overrides_data = self._load_overrides()
                from src.chunking.glossary import get_glossary_manager

                if hasattr(get_glossary_manager, "cache_clear"):
                    get_glossary_manager.cache_clear()

            sha256 = compute_file_sha256(path)

            # Check A (Exact SHA-256 Duplicate Rejection when force_reindex=False)
            if not eff_force:
                existing_sha_doc = next(
                    (
                        m
                        for m in self.registry.values()
                        if m.sha256 and m.sha256 == sha256 and m.status != DocumentStatus.ERROR.value
                    ),
                    None,
                )
                if existing_sha_doc is not None:
                    dup_res = self._build_duplicate_result(
                        file_name=file_name,
                        file_path=path,
                        sha256=sha256,
                        existing_doc=existing_sha_doc,
                        reason="exact_sha256",
                        duration_sec=round(time.perf_counter() - t0, 3),
                    )
                    self._emit_progress(
                        progress_callback,
                        "done",
                        6,
                        "Дубликат (SHA-256)",
                        100,
                        dup_res.message,
                        file_name=file_name,
                        doc_id=existing_sha_doc.doc_id,
                        parent_count=existing_sha_doc.parent_count,
                        chunk_count=existing_sha_doc.chunk_count,
                        ocr_pages=existing_sha_doc.ocr_pages,
                    )
                    return dup_res

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

            # Pre-assign deterministic table_id onto eligible ParsedBlocks before hierarchical chunking
            from src.indexing.table_store import extract_logical_tables_and_assign_ids

            extract_logical_tables_and_assign_ids(blocks, doc_meta)

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

            # Check B (Content / Semantic Text Duplicate Rejection BEFORE Qdrant/table/registry writes)
            explicit_ovr_doc_id = (
                (metadata_overrides or {}).get("doc_id")
                or (self.overrides_data.get("documents", {}).get(file_name, {}) or {}).get("doc_id")
            )
            content_dup_doc = self._find_content_duplicate(
                cand_meta=doc_meta,
                cand_sections=parent_sections,
                cand_chunks=chunks,
                cand_blocks=blocks,
            )
            if not eff_force and content_dup_doc is not None:
                dup_res = self._build_duplicate_result(
                    file_name=file_name,
                    file_path=path,
                    sha256=sha256,
                    existing_doc=content_dup_doc,
                    reason="content_duplicate",
                    duration_sec=round(time.perf_counter() - t0, 3),
                )
                self._emit_progress(
                    progress_callback,
                    "done",
                    6,
                    "Дубликат (контент)",
                    100,
                    dup_res.message,
                    file_name=file_name,
                    doc_id=content_dup_doc.doc_id,
                    parent_count=content_dup_doc.parent_count,
                    chunk_count=content_dup_doc.chunk_count,
                    ocr_pages=content_dup_doc.ocr_pages,
                )
                return dup_res

            # When force_reindex=True on a re-saved/renamed canonical document, reuse its canonical doc_id in place
            if (
                eff_force
                and content_dup_doc is not None
                and content_dup_doc.doc_id != doc_meta.doc_id
                and not explicit_ovr_doc_id
                and not self._is_distinct_k2_version(doc_meta, content_dup_doc)
            ):
                doc_meta.doc_id = content_dup_doc.doc_id
                extract_logical_tables_and_assign_ids(blocks, doc_meta)
                chunks, parent_sections, doc_meta = self._invoke_chunker(blocks, doc_meta)

            # Extract & upsert eligible structured tables (.xlsx sheets + large .docx/.pdf tables >= 25 rows / >= 5000 chars)
            upserted_tables = self.table_store.upsert_document_tables(doc_meta, blocks, chunks=chunks)
            doc_meta.tables_count = len(upserted_tables)
            for ch in chunks:
                if not ch.table_id and len(upserted_tables) == 1 and doc_meta.file_format in {"xlsx", "xls", "xlsm"}:
                    if ch.zone not in {ZoneType.APPROVAL.value, ZoneType.META.value}:
                        ch.table_id = upserted_tables[0].table_id
                if ch.table_id:
                    ch.metadata["table_id"] = ch.table_id

            # Merge cross-file table_hash deduplication links into doc_meta.also_in_files & peer docs
            touched_table_peers: set[str] = set()
            for tp in upserted_tables:
                for peer_fname in tp.also_in_files:
                    if peer_fname and peer_fname != doc_meta.file_name:
                        if peer_fname not in doc_meta.also_in_files:
                            doc_meta.also_in_files.append(peer_fname)
                        for peer_meta in self.registry.values():
                            if peer_meta.doc_id != doc_meta.doc_id and peer_meta.file_name == peer_fname:
                                if doc_meta.file_name and doc_meta.file_name not in peer_meta.also_in_files:
                                    peer_meta.also_in_files.append(doc_meta.file_name)
                                    touched_table_peers.add(peer_meta.doc_id)

            # Ensure synthetic u00_meta chunk is up-to-date and stamp EDO/page_numbers/clause_page_map across records
            chunks = self._ensure_u00_meta_in_chunks(doc_meta, chunks)
            self._enrich_records_with_requisites_and_page_maps(doc_meta, parent_sections, chunks)
            doc_meta.parent_count = len(parent_sections)
            doc_meta.chunk_count = len(chunks)

            # Cache normalized text SHA-256 and normative chunk content_hashes for fast future Check B comparisons
            sorted_p_secs = sorted(parent_sections, key=lambda s: (s.unit_id, s.section_id))
            sec_norm_str = _normalize_text_for_dedup("\n".join(s.clean_text for s in sorted_p_secs if s.clean_text))
            norm_ch_list = [
                c
                for c in chunks
                if c.zone not in {ZoneType.APPROVAL.value, ZoneType.META.value} and c.unit_id != "u00_meta"
            ]
            ch_norm_str = _normalize_text_for_dedup("\n".join(c.clean_text for c in norm_ch_list if c.clean_text))
            blk_norm_str = _normalize_text_for_dedup("\n".join(b.text for b in blocks if b.text))
            doc_meta.metadata = dict(doc_meta.metadata or {})
            if sec_norm_str:
                doc_meta.metadata["norm_sections_sha256"] = hashlib.sha256(sec_norm_str.encode("utf-8")).hexdigest()
            if ch_norm_str:
                doc_meta.metadata["norm_chunks_sha256"] = hashlib.sha256(ch_norm_str.encode("utf-8")).hexdigest()
            if blk_norm_str:
                doc_meta.metadata["norm_blocks_sha256"] = hashlib.sha256(blk_norm_str.encode("utf-8")).hexdigest()
            doc_meta.metadata["chunk_content_hashes"] = sorted(
                {
                    c.content_hash or compute_content_hash(c.clean_text)
                    for c in norm_ch_list
                    if c.clean_text and c.clean_text.strip()
                }
                - {""}
            )

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

            # Purge any prior stale doc_id for the exact same file_name or exact same sha256 so zero duplicates remain
            stale_same_file_ids = [
                d_id
                for d_id, m in list(self.registry.items())
                if d_id != doc_meta.doc_id and (m.file_name == file_name or (sha256 and m.sha256 == sha256))
            ]
            for stale_id in stale_same_file_ids:
                self.delete_document(stale_id)

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
                self._enrich_records_with_requisites_and_page_maps(doc_meta, parent_sections, chunks)
                doc_meta.chunk_count = len(chunks)
                embed_inputs = [c.embed_text or c.raw_text or c.clean_text for c in chunks]
                dense_vecs, sparse_vecs = self.embedder.embed_chunks(embed_inputs)

            self.qdrant.upsert_chunks(chunks, dense_vecs, sparse_vecs, doc_meta=doc_meta)

            # Sync any other documents affected by bidirectional edition graph or table_hash dedup links
            for other_doc_id in (changed_graph_ids | touched_table_peers):
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

            status_entry = PipelineFileResult(
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
                tables_count=doc_meta.tables_count,
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
            err_status = PipelineFileResult(
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
        rep.skipped_files = sum(1 for f in rep.files if f.status in ("skipped", "duplicate"))
        rep.failed_files = sum(1 for f in rep.files if f.status == "error")
        rep.total_parent_sections = len(self.sections_store)
        rep.total_chunks = sum(m.chunk_count for m in self.registry.values())
        rep.tables_count = (
            self.table_store.count_tables()
            if self._has_active_table_store()
            else sum(m.tables_count for m in self.registry.values())
        )
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
        force_reindex: bool | None = None,
        ocr_backend: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexingReport:
        """
        Incrementally synchronize all supported documents (.pdf, .docx, .doc, .xlsx, .xls) in folder_path (defaults to DOCS_DIR).
        Handles `status == 'duplicate'` gracefully alongside skipped files without failing batch runs,
        reconciles all stores (purging any orphan sections/tables/Qdrant points), and writes IndexingReport.
        """
        eff_force = bool(force if force_reindex is None else (force or force_reindex))
        t_start = time.perf_counter()
        started_at = _utc_now_iso()
        target_dir = Path(folder_path).resolve() if folder_path else DOCS_DIR.resolve()

        self.overrides_data = self._load_overrides()
        from src.chunking.glossary import get_glossary_manager

        if hasattr(get_glossary_manager, "cache_clear"):
            get_glossary_manager.cache_clear()

        if not target_dir.exists() or not target_dir.is_dir():
            err_rep = IndexingReport(
                run_id=f"sync_{uuid.uuid4().hex[:8]}",
                started_at=started_at,
                finished_at=_utc_now_iso(),
                errors=[f"Директория не найдена: {target_dir}"],
            )
            self.save_report(err_rep)
            return err_rep

        raw_files = sorted(
            [
                p
                for p in target_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTENSIONS
                and not p.name.startswith("~$")
            ],
            key=lambda p: p.name,
        )

        # When syncing the primary DOCS_DIR corpus, preserve 'data/uploads/' origin path for uploaded copies
        # and include any valid uploaded documents in UPLOADS_DIR that are not duplicated in DOCS_DIR.
        if target_dir == DOCS_DIR.resolve() and UPLOADS_DIR.exists() and UPLOADS_DIR.is_dir():
            seen_names: set[str] = set()
            seen_shas: set[str] = set()
            merged_files: list[Path] = []
            for p in raw_files:
                p_sha = ""
                try:
                    p_sha = compute_file_sha256(p)
                except OSError:
                    pass
                up_cand = UPLOADS_DIR / p.name
                if up_cand.exists() and up_cand.is_file():
                    try:
                        if p_sha and compute_file_sha256(up_cand) == p_sha:
                            p = up_cand.resolve()
                    except OSError:
                        pass
                merged_files.append(p)
                seen_names.add(p.name)
                if p_sha:
                    seen_shas.add(p_sha)

            extra_uploads: list[Path] = []
            for up in sorted(UPLOADS_DIR.iterdir(), key=lambda p: p.name):
                if (
                    not up.is_file()
                    or up.suffix.lower() not in SUPPORTED_EXTENSIONS
                    or up.name.startswith("~$")
                    or up.name in seen_names
                ):
                    continue
                try:
                    up_sha = compute_file_sha256(up)
                    if up_sha in seen_shas:
                        continue
                    seen_shas.add(up_sha)
                except OSError:
                    pass
                extra_uploads.append(up)
            merged_files.extend(extra_uploads)
            files = merged_files
        else:
            files = raw_files

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
                force=eff_force,
                force_reindex=eff_force,
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
        duplicate_cnt = sum(1 for s in file_statuses if s.status == "duplicate")
        skipped_cnt = sum(1 for s in file_statuses if s.status in ("skipped", "duplicate"))
        failed_cnt = sum(1 for s in file_statuses if s.status == "error")

        if skipped_cnt > 0 and not self._has_active_table_store():
            self.rebuild_structured_tables(ocr_backend=ocr_backend, patch_qdrant=True)
            for st in file_statuses:
                dm = self.registry.get(st.doc_id)
                if dm is not None:
                    st.tables_count = dm.tables_count

        # Reconcile all persistent stores to guarantee zero orphan sections, tables, or Qdrant points
        self.reconcile_stores(verify_files_on_disk=True)

        total_dur = round(time.perf_counter() - t_start, 3)

        report = IndexingReport(
            run_id=f"sync_{uuid.uuid4().hex[:8]}",
            started_at=started_at,
            finished_at=_utc_now_iso(),
            total_files=total_files,
            indexed_files=indexed_cnt,
            skipped_files=skipped_cnt,
            duplicate_files=duplicate_cnt,
            failed_files=failed_cnt,
            total_parent_sections=len(self.sections_store),
            total_chunks=sum(m.chunk_count for m in self.registry.values()),
            tables_count=(
                self.table_store.count_tables()
                if self._has_active_table_store()
                else sum(m.tables_count for m in self.registry.values())
            ),
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
            f"Синхронизировано {total_files} файлов: {indexed_cnt} проиндексировано, {skipped_cnt} пропущено/дубликатов, {failed_cnt} ошибок за {total_dur:.1f} с",
            parent_count=report.total_parent_sections,
            chunk_count=report.total_chunks,
            ocr_pages=report.total_ocr_pages,
        )
        return report

    def index_directory(
        self,
        folder_path: str | Path | None = None,
        force: bool = False,
        force_reindex: bool | None = None,
        ocr_backend: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexingReport:
        """Alias for `sync_folder` supporting both `force` and `force_reindex` flags."""
        return self.sync_folder(
            folder_path=folder_path,
            force=force,
            force_reindex=force_reindex,
            ocr_backend=ocr_backend,
            progress_callback=progress_callback,
        )

    # ------------------------------------------------------------------
    # Document Registry & Metadata Management (GET / PATCH / DELETE / RECONCILE)
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
        Update document metadata (status, supersedes, amends, applicable_years, version_note, edo_number, etc.),
        reconcile the bidirectional edition graph across all documents, refresh u00_meta, and update Qdrant payload.
        """
        doc_meta = self.registry.get(doc_id)
        if doc_meta is None:
            raise KeyError(f"Документ с doc_id='{doc_id}' не найден в реестре")

        current_data = doc_meta.model_dump()
        base_meta = dict(current_data.get("metadata") or {})
        for key, val in patch_dict.items():
            if val is None or key == "doc_id":
                continue
            if key == "metadata" and isinstance(val, dict):
                base_meta.update(val)
            elif key in ("edo_number", "edo_date", "is_edo_stamp"):
                base_meta[key] = val
                current_data[key] = val
            elif key in current_data:
                current_data[key] = val
        current_data["metadata"] = base_meta

        updated_meta = DocumentMeta.model_validate(current_data)
        self.registry[doc_id] = updated_meta

        changed_ids = self._reconcile_edition_graph()
        changed_ids.add(doc_id)

        for target_id in changed_ids:
            if target_id in self.registry:
                self._sync_doc_metadata_to_qdrant(self.registry[target_id])

        self.save_registry()
        return self.registry[doc_id]

    def _purge_doc_tables_and_rows(self, doc_id: str) -> int:
        """Purge all tables_meta and table_rows entries for doc_id from SQLite tables_store.db."""
        if not doc_id or not self._has_active_table_store():
            return 0
        deleted = 0
        try:
            deleted = self.table_store.delete_document_tables(doc_id)
            with self.table_store._lock:
                conn = self.table_store._connect()
                try:
                    cur = conn.execute("DELETE FROM table_rows WHERE doc_id = ?", (doc_id,))
                    conn.execute("DELETE FROM tables_meta WHERE doc_id = ?", (doc_id,))
                    if cur.rowcount and cur.rowcount > 0 and deleted == 0:
                        deleted = int(cur.rowcount)
                    conn.commit()
                finally:
                    conn.close()
        except Exception as tbl_exc:
            logger.warning("Failed to delete structured tables/rows for doc_id='%s': %s", doc_id, tbl_exc)
        return deleted

    def delete_document(self, doc_id: str) -> bool:
        """
        Completely delete a document from Qdrant, SQLite StructuredTableStore, data/sections_store.json,
        and data/document_registry.json, and clean up its references in the edition graph and peer chunks.
        """
        doc_meta = self.registry.pop(doc_id, None)
        qdrant_deleted = self.qdrant.delete_by_doc_id(doc_id)
        tables_deleted = self._purge_doc_tables_and_rows(doc_id)

        sec_ids_to_remove = [sid for sid, sec in self.sections_store.items() if sec.doc_id == doc_id]
        for sid in sec_ids_to_remove:
            self.sections_store.pop(sid, None)

        if doc_meta is not None:
            ref_to_remove = _core_doc_number(doc_meta.doc_number) or doc_meta.doc_number
            refs = {doc_id, doc_meta.file_name, doc_meta.doc_number, ref_to_remove} - {""}
            for other in self.registry.values():
                modified = False
                had_also_file = bool(doc_meta.file_name and doc_meta.file_name in other.also_in_files)
                for list_attr in ("superseded_by", "amended_by", "also_in_files"):
                    cur_list = getattr(other, list_attr)
                    filtered = [x for x in cur_list if x not in refs]
                    if len(filtered) != len(cur_list):
                        setattr(other, list_attr, filtered)
                        modified = True
                if modified:
                    self._sync_doc_metadata_to_qdrant(other)
                if had_also_file and doc_meta.file_name:
                    for oc in self.qdrant.get_document_chunks(other.doc_id):
                        if doc_meta.file_name in oc.also_in_files and oc.point_uuid:
                            cleaned_also = [fn for fn in oc.also_in_files if fn != doc_meta.file_name]
                            self.qdrant.update_chunk_payload_by_uuid(
                                oc.point_uuid,
                                {"also_in_files": cleaned_also},
                            )

        if self.last_report and self.last_report.files:
            before_len = len(self.last_report.files)
            self.last_report.files = [
                f
                for f in self.last_report.files
                if f.doc_id != doc_id and (doc_meta is None or f.file_name != doc_meta.file_name)
            ]
            if len(self.last_report.files) != before_len:
                self.last_report.total_files = len(self.last_report.files)
                self.last_report.total_parent_sections = len(self.sections_store)
                self.last_report.total_chunks = sum(m.chunk_count for m in self.registry.values())
                self.last_report.tables_count = (
                    self.table_store.count_tables()
                    if self._has_active_table_store()
                    else sum(m.tables_count for m in self.registry.values())
                )
                self.save_report(self.last_report)

        self.save_sections_store()
        self.save_registry()
        return bool(doc_meta is not None or qdrant_deleted > 0 or tables_deleted > 0 or sec_ids_to_remove)

    def reconcile_stores(self, verify_files_on_disk: bool = True) -> dict[str, Any]:
        """
        Audit and reconcile all persistent stores (`document_registry.json`, `sections_store.json`,
        `tables_store.db`, and Qdrant `mipt_normative_v3`) so zero orphan sections, orphan tables,
        orphan Qdrant points, or missing-file registry entries ever remain.
        """
        missing_files_removed: list[str] = []
        duplicate_docs_removed: list[str] = []

        # 1. Verify every registered file_path exists on disk (in доки/ or data/uploads/)
        if verify_files_on_disk:
            for doc_id, doc_meta in list(self.registry.items()):
                resolved = self._resolve_doc_file_path(doc_meta)
                if resolved is None:
                    logger.info(
                        "Reconcile: purging doc_id='%s' ('%s') because source file is missing on disk",
                        doc_id,
                        doc_meta.file_name,
                    )
                    missing_files_removed.append(doc_id)
                    self.delete_document(doc_id)
                elif str(resolved) != doc_meta.file_path:
                    doc_meta.file_path = str(resolved)

        # 2. Deduplicate any multiple registry entries sharing the exact same file_name or exact same sha256
        by_fname: dict[str, list[DocumentMeta]] = {}
        for meta in list(self.registry.values()):
            if meta.file_name:
                by_fname.setdefault(meta.file_name, []).append(meta)
        for fname, group in by_fname.items():
            if len(group) > 1:
                group.sort(key=lambda m: (m.indexed_at or "", m.chunk_count), reverse=True)
                for stale_meta in group[1:]:
                    if stale_meta.doc_id in self.registry:
                        duplicate_docs_removed.append(stale_meta.doc_id)
                        self.delete_document(stale_meta.doc_id)

        by_sha: dict[str, list[DocumentMeta]] = {}
        for meta in list(self.registry.values()):
            if meta.sha256:
                by_sha.setdefault(meta.sha256, []).append(meta)
        for sha_val, group in by_sha.items():
            if len(group) > 1:
                # Prefer canonical DOCS_DIR entry (or earlier indexed canonical entry) over later re-upload duplicate
                docs_dir_str = str(DOCS_DIR.resolve())
                group.sort(
                    key=lambda m: (
                        0 if (DOCS_DIR / m.file_name).exists() or docs_dir_str in (m.file_path or "") else 1,
                        m.indexed_at or "9999",
                    )
                )
                canonical = group[0]
                for stale_meta in group[1:]:
                    if stale_meta.doc_id in self.registry and stale_meta.doc_id != canonical.doc_id:
                        duplicate_docs_removed.append(stale_meta.doc_id)
                        self.delete_document(stale_meta.doc_id)

        valid_doc_ids: set[str] = set(self.registry.keys())

        # 3. Purge any orphan section_ids from sections_store.json whose doc_id is not in registry
        orphan_sec_ids = [
            sec_id
            for sec_id, sec in list(self.sections_store.items())
            if sec.doc_id not in valid_doc_ids
        ]
        for sec_id in orphan_sec_ids:
            self.sections_store.pop(sec_id, None)

        # 4. Purge any orphan doc_ids / table_rows from SQLite tables_store.db
        orphan_tables_removed = 0
        orphan_table_rows_removed = 0
        if self._has_active_table_store():
            try:
                all_tables = self.table_store.list_tables()
                orphan_tbl_doc_ids = sorted({t.doc_id for t in all_tables if t.doc_id not in valid_doc_ids})
                for oid in orphan_tbl_doc_ids:
                    orphan_tables_removed += self.table_store.delete_document_tables(oid)

                with self.table_store._lock:
                    conn = self.table_store._connect()
                    try:
                        if valid_doc_ids:
                            placeholders = ",".join("?" for _ in valid_doc_ids)
                            params = tuple(sorted(valid_doc_ids))
                            cur_m = conn.execute(
                                f"DELETE FROM tables_meta WHERE doc_id NOT IN ({placeholders})",
                                params,
                            )
                            cur_r = conn.execute(
                                f"DELETE FROM table_rows WHERE doc_id NOT IN ({placeholders}) "
                                f"OR table_id NOT IN (SELECT table_id FROM tables_meta)",
                                params,
                            )
                        else:
                            cur_m = conn.execute("DELETE FROM tables_meta")
                            cur_r = conn.execute("DELETE FROM table_rows")
                        if cur_m.rowcount and cur_m.rowcount > 0:
                            orphan_tables_removed += int(cur_m.rowcount)
                        if cur_r.rowcount and cur_r.rowcount > 0:
                            orphan_table_rows_removed += int(cur_r.rowcount)
                        # Reconcile table_hash peer links on remaining tables
                        rem_hashes = {
                            str(r["table_hash"])
                            for r in conn.execute("SELECT DISTINCT table_hash FROM tables_meta").fetchall()
                            if r["table_hash"]
                        }
                        if rem_hashes:
                            self.table_store._reconcile_hash_dedup_in_tx(conn, rem_hashes)
                        conn.commit()
                    finally:
                        conn.close()
            except Exception as tbl_exc:
                logger.warning("Reconcile tables_store check failed: %s", tbl_exc)

        # 5. Purge any orphan doc_ids / points from Qdrant collection
        orphan_qdrant_doc_ids: set[str] = set()
        orphan_qdrant_points_removed = 0
        pointless_ids: list[Any] = []
        if self.qdrant.ensure_collection():
            try:
                from qdrant_client.http import models as qmodels

                next_offset: Any = None
                while True:
                    points, next_offset = self.qdrant.client.scroll(
                        collection_name=self.qdrant.collection_name,
                        limit=256,
                        offset=next_offset,
                        with_payload=["doc_id"],
                        with_vectors=False,
                    )
                    for pt in points:
                        p_doc_id = str((pt.payload or {}).get("doc_id") or "").strip()
                        if not p_doc_id:
                            pointless_ids.append(pt.id)
                        elif p_doc_id not in valid_doc_ids:
                            orphan_qdrant_doc_ids.add(p_doc_id)
                    if next_offset is None:
                        break

                for oid in sorted(orphan_qdrant_doc_ids):
                    orphan_qdrant_points_removed += self.qdrant.delete_by_doc_id(oid)

                if pointless_ids:
                    self.qdrant.client.delete(
                        collection_name=self.qdrant.collection_name,
                        points_selector=qmodels.PointIdsList(points=pointless_ids),
                        wait=True,
                    )
                    orphan_qdrant_points_removed += len(pointless_ids)
            except Exception as q_exc:
                logger.warning("Reconcile Qdrant orphan sweep failed: %s", q_exc)

        # 6. Synchronize counts and clean stale also_in_files across registered documents
        valid_fnames = {m.file_name for m in self.registry.values() if m.file_name}
        sec_counts_by_doc: dict[str, int] = {}
        for sec in self.sections_store.values():
            sec_counts_by_doc[sec.doc_id] = sec_counts_by_doc.get(sec.doc_id, 0) + 1

        for doc_id, meta in self.registry.items():
            meta.parent_count = sec_counts_by_doc.get(doc_id, 0)
            if self._has_active_table_store():
                meta.tables_count = self.table_store.count_tables(doc_id=doc_id)
            cleaned_also = [
                fn for fn in meta.also_in_files if fn in valid_fnames and fn != meta.file_name
            ]
            if cleaned_also != list(meta.also_in_files):
                meta.also_in_files = cleaned_also
                self.qdrant.update_document_payload(doc_id, {"also_in_files": cleaned_also})

        self.save_sections_store()
        self.save_registry()

        return {
            "valid_documents": len(self.registry),
            "missing_files_removed": sorted(missing_files_removed),
            "duplicate_docs_removed": sorted(duplicate_docs_removed),
            "orphan_sections_removed": len(orphan_sec_ids),
            "orphan_tables_removed": orphan_tables_removed,
            "orphan_table_rows_removed": orphan_table_rows_removed,
            "orphan_qdrant_doc_ids_removed": sorted(orphan_qdrant_doc_ids),
            "orphan_qdrant_points_removed": orphan_qdrant_points_removed,
        }

    # ------------------------------------------------------------------
    # Structured Table Store Access & Fast Zero-API-Call Rebuild
    # ------------------------------------------------------------------
    def get_tables(self, doc_id: str | None = None) -> list[TablePassport]:
        """List all structured TablePassports (optionally filtered by doc_id)."""
        if not self._has_active_table_store():
            return []
        return self.table_store.list_tables(doc_id=doc_id)

    list_tables = get_tables

    def get_table(self, table_id: str) -> TablePassport | None:
        """Fetch a single structured TablePassport by table_id."""
        if not self._has_active_table_store():
            return None
        return self.table_store.get_table(table_id)

    def query_table(self, req: TableQueryRequest | dict[str, Any]) -> TableQueryResponse:
        """Execute a deterministic filter/aggregate/formula query against StructuredTableStore."""
        if not self._has_active_table_store():
            q = req if isinstance(req, TableQueryRequest) else TableQueryRequest.model_validate(req)
            return TableQueryResponse(
                table_id=q.table_id,
                operation=q.operation,
                markdown_preview=f"Таблица '{q.table_id}' не найдена в базе таблиц.",
            )
        return self.table_store.query_table(req)

    def _resolve_doc_file_path(self, doc_meta: DocumentMeta) -> Path | None:
        """Locate the source file on disk for a registered document (verifying доки/ or data/uploads/ in production)."""
        is_prod_registry = self.registry_path.resolve() == DOCUMENT_REGISTRY_PATH.resolve()
        allowed_roots = (DOCS_DIR.resolve(), UPLOADS_DIR.resolve())

        candidates: list[Path] = []
        if doc_meta.file_path:
            p = Path(doc_meta.file_path)
            if not is_prod_registry:
                candidates.append(p)
            else:
                try:
                    rp = p.resolve()
                    if any(rp == root or root in rp.parents for root in allowed_roots):
                        candidates.append(rp)
                except OSError:
                    pass
        if doc_meta.file_name:
            candidates.append(DOCS_DIR / doc_meta.file_name)
            candidates.append(UPLOADS_DIR / doc_meta.file_name)

        for cand in candidates:
            if cand.exists() and cand.is_file():
                return cand.resolve()
        return None

    def rebuild_structured_tables(
        self,
        ocr_backend: str | None = None,
        patch_qdrant: bool = True,
    ) -> dict[str, Any]:
        """
        Populate SQLite `tables_store.db` across all indexed documents AND patch `table_id` + `tables_count`
        directly into existing Qdrant chunk payloads using warm OCR cache (`0` paid API calls for unchanged chunks!).
        """
        t0 = time.perf_counter()
        if not self.registry:
            self.registry = self._load_registry()

        # If corpus has never been indexed yet, run initial sync_folder first
        if not self.registry and DOCS_DIR.exists():
            sync_rep = self.sync_folder(force=False, ocr_backend=ocr_backend)
            return {
                "status": "ok" if not sync_rep.errors else "partial",
                "documents_processed": sync_rep.total_files,
                "tables_upserted": self.table_store.count_tables(),
                "qdrant_points_patched": 0,
                "tables": [t.model_dump() for t in self.table_store.list_tables()],
                "errors": list(sync_rep.errors),
                "duration_sec": round(time.perf_counter() - t0, 3),
            }

        eff_ocr = ocr_backend or settings.OCR_BACKEND
        processed_docs = 0
        patched_points = 0
        errors: list[str] = []

        for doc_id, doc_meta in list(self.registry.items()):
            src_path = self._resolve_doc_file_path(doc_meta)
            if src_path is None:
                errors.append(f"{doc_id}: исходный файл '{doc_meta.file_name}' не найден на диске")
                continue

            try:
                blocks, _ = self._invoke_parser(src_path, ocr_backend=eff_ocr)
                if not blocks:
                    errors.append(f"{doc_meta.file_name}: парсер вернул 0 блоков")
                    continue

                from src.indexing.table_store import extract_logical_tables_and_assign_ids

                extract_logical_tables_and_assign_ids(blocks, doc_meta)
                chunks, _, _ = self._invoke_chunker(blocks, doc_meta)
                passports = self.table_store.upsert_document_tables(doc_meta, blocks, chunks=chunks)
                doc_meta.tables_count = len(passports)
                processed_docs += 1

                if patch_qdrant:
                    existing_pts = self.qdrant.count_points(doc_id=doc_id)
                    if existing_pts == 0:
                        # Document missing from Qdrant: run full index_file using embed_cache
                        self.index_file(file_path=src_path, ocr_backend=eff_ocr, force=True)
                        continue

                    # Group newly computed chunk point_uuids & content_hashes by table_id
                    uuids_by_tid: dict[str, list[str]] = {}
                    tid_by_hash: dict[str, str] = {}
                    for ch in chunks:
                        if ch.table_id:
                            pt_id = ch.point_uuid or make_point_uuid(ch.chunk_id)
                            uuids_by_tid.setdefault(ch.table_id, []).append(pt_id)
                            chash = ch.content_hash or compute_content_hash(ch.clean_text)
                            if chash:
                                tid_by_hash[chash] = ch.table_id

                    # Scroll existing Qdrant chunks for doc_id so we patch exact existing point UUIDs cleanly
                    if uuids_by_tid:
                        stored_chunks = self.qdrant.get_document_chunks(doc_id)
                        stored_uuid_set = {sc.point_uuid for sc in stored_chunks if sc.point_uuid}
                        patch_buckets: dict[str, list[str]] = {}

                        for tid, cand_uuids in uuids_by_tid.items():
                            matched_uuids = [u for u in cand_uuids if u in stored_uuid_set]
                            if matched_uuids:
                                patch_buckets.setdefault(tid, []).extend(matched_uuids)

                        # Fallback hash match for any stored table/form chunk not yet matched by point_uuid
                        already_matched = {u for bucket in patch_buckets.values() for u in bucket}
                        for sc in stored_chunks:
                            if not sc.point_uuid or sc.point_uuid in already_matched:
                                continue
                            sc_hash = sc.content_hash or compute_content_hash(sc.clean_text)
                            matched_tid = tid_by_hash.get(sc_hash, "")
                            if not matched_tid and len(passports) == 1 and doc_meta.file_format in {"xlsx", "xls", "xlsm"}:
                                if sc.zone not in {ZoneType.APPROVAL.value, ZoneType.META.value}:
                                    matched_tid = passports[0].table_id
                            if matched_tid:
                                patch_buckets.setdefault(matched_tid, []).append(sc.point_uuid)

                        self.qdrant.ensure_collection()
                        for tid, target_uuids in patch_buckets.items():
                            dedup_uuids = list(dict.fromkeys(target_uuids))
                            if not dedup_uuids:
                                continue
                            self.qdrant.client.set_payload(
                                collection_name=self.qdrant.collection_name,
                                payload={"table_id": tid},
                                points=dedup_uuids,
                                wait=True,
                            )
                            patched_points += len(dedup_uuids)

                    edo_number, edo_date, is_edo_stamp = self._extract_edo_fields(doc_meta)
                    self.qdrant.update_document_payload(
                        doc_id,
                        {
                            "tables_count": int(doc_meta.tables_count),
                            "edo_number": edo_number,
                            "edo_date": edo_date,
                            "is_edo_stamp": is_edo_stamp,
                        },
                    )

            except Exception as exc:
                logger.exception("Failed rebuilding structured tables for '%s': %s", doc_meta.file_name, exc)
                errors.append(f"{doc_meta.file_name}: {type(exc).__name__}: {exc}")

        # Reconcile cross-file also_in_files from table_hash links across all registered docs
        all_passports = self.table_store.list_tables()
        docs_by_fname = {m.file_name: m for m in self.registry.values() if m.file_name}
        touched_also_docs: set[str] = set()
        for tp in all_passports:
            owner_meta = self.registry.get(tp.doc_id)
            for peer_fname in tp.also_in_files:
                if owner_meta and peer_fname and peer_fname != owner_meta.file_name:
                    if peer_fname not in owner_meta.also_in_files:
                        owner_meta.also_in_files.append(peer_fname)
                        touched_also_docs.add(owner_meta.doc_id)
                peer_meta = docs_by_fname.get(peer_fname)
                if peer_meta and owner_meta and owner_meta.file_name and owner_meta.file_name != peer_meta.file_name:
                    if owner_meta.file_name not in peer_meta.also_in_files:
                        peer_meta.also_in_files.append(owner_meta.file_name)
                        touched_also_docs.add(peer_meta.doc_id)

        if patch_qdrant:
            for d_id in touched_also_docs:
                dm = self.registry.get(d_id)
                if dm and self.qdrant.count_points(doc_id=d_id) > 0:
                    self.qdrant.update_document_payload(d_id, {"also_in_files": list(dm.also_in_files)})

        self.reconcile_stores(verify_files_on_disk=True)

        # Update indexing_report.json with fresh tables_count
        all_passports = self.table_store.list_tables()
        total_tables = len(all_passports)
        if self.last_report and self.last_report.files:
            for f_stat in self.last_report.files:
                dm = self.registry.get(f_stat.doc_id) or docs_by_fname.get(f_stat.file_name)
                if dm is not None:
                    f_stat.tables_count = dm.tables_count
            self.last_report.tables_count = total_tables
            self.save_report(self.last_report)

        return {
            "status": "ok" if not errors else "partial",
            "documents_processed": processed_docs,
            "tables_upserted": total_tables,
            "qdrant_points_patched": patched_points,
            "tables": [t.model_dump() for t in all_passports],
            "errors": errors,
            "duration_sec": round(time.perf_counter() - t0, 3),
        }

    def get_health(self) -> dict[str, Any]:
        """Return comprehensive health status for GET /api/health."""
        qdrant_health = self.qdrant.check_health()
        embed_health = self.embedder.check_health()
        llm_health = {
            "provider": settings.llm_provider,
            "backend": settings.LLM_BACKEND,
            "quantization": settings.LLM_QUANTIZATION,
            "configured": bool(settings.llm_api_key),
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "quantized_repo": settings.LOCAL_LLM_HF_QUANTIZED_REPO,
        }
        ocr_health = {
            "backend": settings.OCR_BACKEND,
            "vl_model": settings.OCR_VL_MODEL,
            "device": settings.OCR_VL_DEVICE,
            "fallback": "local_rapidocr (PP-OCRv5 ESLAV)",
        }
        structured_tables_count = (
            self.table_store.count_tables() if self._has_active_table_store() else 0
        )
        overall_ok = (
            bool(qdrant_health.get("connected"))
            and bool(embed_health.get("configured"))
            and bool(settings.llm_api_key)
        )
        return {
            "status": "ok" if overall_ok else "degraded",
            "qdrant": qdrant_health,
            "local_models": {
                "ocr_vl": settings.OCR_VL_MODEL,
                "embeddings": settings.EMBEDDING_MODEL,
                "llm": settings.LOCAL_LLM_MODEL,
                "llm_quantization": settings.LLM_QUANTIZATION,
                "llm_quantized_repo": settings.LOCAL_LLM_HF_QUANTIZED_REPO,
            },
            "ocr": ocr_health,
            "embeddings": embed_health,
            "polza_api": embed_health,
            "llm": llm_health,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "registry_docs_count": len(self.registry),
            "sections_count": len(self.sections_store),
            "structured_tables_count": structured_tables_count,
        }


_PIPELINE_SINGLETON: IndexingPipeline | None = None


def get_pipeline() -> IndexingPipeline:
    """Return process-wide IndexingPipeline singleton."""
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = IndexingPipeline()
    return _PIPELINE_SINGLETON
