"""Pydantic v2 data schemas for MIPT Normative RAG V3 (Ingestion, Chunker V3, Indexing, Retrieval, Web UI)."""

from __future__ import annotations

import hashlib
import re
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Deterministic UUIDv5 namespace for Qdrant point IDs
QDRANT_POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def make_point_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUIDv5 string from a unique synthetic chunk_id."""
    return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, f"mipt-rag-v3::{chunk_id}"))


def compute_content_hash(text: str) -> str:
    """Compute a normalized SHA-256 content hash for cross-file chunk deduplication."""
    if not text:
        return ""
    norm = text.lower().replace("ё", "е")
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _coerce_str_list(val: Any) -> list[str]:
    """Coerce None, a single string (possibly comma-separated), or an iterable into list[str]."""
    if val is None:
        return []
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        if "," in val:
            return [x.strip() for x in val.split(",") if x.strip()]
        return [val]
    if isinstance(val, (list, tuple, set)):
        result: list[str] = []
        for item in val:
            if item is not None:
                s = str(item).strip()
                if s:
                    result.append(s)
        return result
    return [str(val).strip()]


def _coerce_int_list(val: Any) -> list[int]:
    """Coerce years from None, int, range string ('2026-2028'), comma string, or list into list[int]."""
    if val is None:
        return []
    if isinstance(val, int):
        return [val]
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        years: list[int] = []
        for part in re.split(r"[,;\s]+", val):
            part = part.strip()
            if not part:
                continue
            m_range = re.match(r"^(\d{4})\s*[-–—]\s*(\d{4})$", part)
            if m_range:
                start_y, end_y = int(m_range.group(1)), int(m_range.group(2))
                if start_y <= end_y and (end_y - start_y) <= 20:
                    years.extend(range(start_y, end_y + 1))
            elif re.match(r"^\d{4}$", part):
                years.append(int(part))
        return sorted(set(years))
    if isinstance(val, (list, tuple, set)):
        years_out: list[int] = []
        for item in val:
            years_out.extend(_coerce_int_list(item))
        return sorted(set(years_out))
    return []


class ZoneType(str, Enum):
    """Document zone classification in V3 Chunker."""

    NORMATIVE = "normative"
    TABLE = "table"
    FORM = "form"
    APPROVAL = "approval"
    META = "meta"


class BlockType(str, Enum):
    """Block structural type produced by ingestion parsers."""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    FORM = "form"
    HEADING = "heading"
    STAMP = "stamp"
    META = "meta"


class DocumentStatus(str, Enum):
    """Lifecycle status of a normative document in the registry."""

    ACTIVE = "active"
    DRAFT = "draft"
    SUPERSEDED = "superseded"
    SUPERSEDED_TEMPLATE_REFERENCE = "superseded_template_reference"
    ERROR = "error"


class ParsedBlock(BaseModel):
    """Typed structural block produced by PDF/DOCX/XLSX parsers (Module 1 -> Module 2)."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    text: str = ""
    block_type: str = BlockType.PARAGRAPH.value
    zone: ZoneType | str = ZoneType.NORMATIVE.value
    page_num: int = 1
    page_numbers: list[int] = Field(default_factory=list)
    is_bold: bool = False
    style_name: str = ""
    heading_level: int | None = None
    is_ocr: bool = False
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[list[str]] = Field(default_factory=list)
    parent_hierarchy: list[str] = Field(default_factory=list)
    sheet_name: str | None = None
    xml_num_prefix: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_page_numbers(self) -> ParsedBlock:
        if not self.page_numbers and self.page_num:
            self.page_numbers = [self.page_num]
        return self


class DocumentMeta(BaseModel):
    """Document-level registry metadata, extracted requisites, and edition graph links."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    doc_id: str = ""
    file_name: str = ""
    file_path: str = ""
    file_format: str = ""
    sha256: str = ""
    doc_number: str = ""
    doc_date: str = ""
    doc_type: str = "Документ"
    title: str = ""
    status: str = DocumentStatus.ACTIVE.value
    applicable_years: list[int] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: list[str] = Field(default_factory=list)
    amends: list[str] = Field(default_factory=list)
    amended_by: list[str] = Field(default_factory=list)
    version_note: str = ""
    quality_warning: str = ""
    pages_total: int = 0
    ocr_pages: int = 0
    parent_count: int = 0
    chunk_count: int = 0
    also_in_files: list[str] = Field(default_factory=list)
    indexed_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("supersedes", "superseded_by", "amends", "amended_by", "also_in_files", mode="before")
    @classmethod
    def _validate_str_lists(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    @field_validator("applicable_years", mode="before")
    @classmethod
    def _validate_years(cls, v: Any) -> list[int]:
        return _coerce_int_list(v)

    @model_validator(mode="after")
    def _populate_defaults(self) -> DocumentMeta:
        if not self.title and self.file_name:
            self.title = self.file_name.rsplit(".", 1)[0]
        if not self.file_format and self.file_name and "." in self.file_name:
            self.file_format = self.file_name.rsplit(".", 1)[-1].lower()
        return self


class ChunkRecord(BaseModel):
    """Child chunk record produced by the 7-stage V3 Hierarchical Chunker."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    chunk_id: str
    point_uuid: str = ""
    doc_id: str
    unit_id: str = "u01_main"
    unit_title: str = "Основной текст"
    section_id: str = ""
    section_num: str = ""
    section_title: str = ""
    clause_no: str = ""
    zone: ZoneType | str = ZoneType.NORMATIVE.value
    block_type: str = BlockType.PARAGRAPH.value
    clean_text: str = ""
    raw_text: str = ""
    embed_text: str = ""
    prev_id: str | None = None
    next_id: str | None = None
    overlap_prefix: str = ""
    overlap_suffix: str = ""
    roles: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    content_hash: str = ""
    also_in_files: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    char_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("roles", "topics", "terms", "also_in_files", mode="before")
    @classmethod
    def _validate_str_lists(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    @model_validator(mode="after")
    def _populate_derived_fields(self) -> ChunkRecord:
        if not self.point_uuid and self.chunk_id:
            self.point_uuid = make_point_uuid(self.chunk_id)
        if not self.raw_text and self.clean_text:
            self.raw_text = self.clean_text
        if not self.clean_text and self.raw_text:
            self.clean_text = self.raw_text
        if not self.char_count:
            self.char_count = len(self.clean_text)
        if not self.content_hash and self.clean_text:
            self.content_hash = compute_content_hash(self.clean_text)
        return self


class ParentSectionRecord(BaseModel):
    """Clean Parent Section record assembled in Stage 5 (before Stage 6 overlap)."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    section_id: str
    doc_id: str
    unit_id: str = "u01_main"
    unit_title: str = "Основной текст"
    section_num: str = ""
    section_title: str = ""
    clean_text: str = ""
    char_count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate_char_count(self) -> ParentSectionRecord:
        if not self.char_count and self.clean_text:
            self.char_count = len(self.clean_text)
        return self


class SearchQuery(BaseModel):
    """Hybrid search request payload."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    status_filter: list[str] | str | None = None
    year_filter: int | None = None
    doc_id_filter: str | None = None
    unit_id_filter: str | None = None
    include_parent: bool = True
    include_amendments: bool = True
    include_superseded: bool = True


class SearchHit(BaseModel):
    """Single ranked hit from hybrid RRF search with hydrated parent section & edition links."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    chunk_id: str
    point_uuid: str = ""
    doc_id: str = ""
    file_name: str = ""
    doc_number: str = ""
    doc_date: str = ""
    doc_type: str = ""
    title: str = ""
    status: str = DocumentStatus.ACTIVE.value
    applicable_years: list[int] = Field(default_factory=list)
    unit_id: str = ""
    unit_title: str = ""
    section_id: str = ""
    section_num: str = ""
    section_title: str = ""
    clause_no: str = ""
    zone: str = ZoneType.NORMATIVE.value
    score: float = 0.0
    rrf_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    raw_text: str = ""
    clean_text: str = ""
    embed_text: str = ""
    parent_section: ParentSectionRecord | None = None
    parent_text: str = ""
    roles: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    also_in_files: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    version_note: str = ""
    quality_warning: str = ""
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: list[str] = Field(default_factory=list)
    amends: list[str] = Field(default_factory=list)
    amended_by: list[str] = Field(default_factory=list)
    is_amendment: bool = False
    amends_doc_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Full response returned by Hybrid Search Service and POST /api/search."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    query: str = ""
    expanded_query: str = ""
    hits: list[SearchHit] = Field(default_factory=list)
    amendment_chunks: list[SearchHit] = Field(default_factory=list)
    version_alerts: list[str] = Field(default_factory=list)
    quality_alerts: list[str] = Field(default_factory=list)
    is_out_of_scope: bool = False
    confidence_score: float = 0.0
    top_dense_score: float = 0.0
    lexical_overlap: float = 0.0
    total_found: int = 0
    latency_ms: float = 0.0


class IndexingFileStatus(BaseModel):
    """Per-file indexing execution report entry."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    file_name: str
    file_path: str = ""
    doc_id: str = ""
    sha256: str = ""
    status: str = "success"
    doc_number: str = ""
    doc_date: str = ""
    pages_total: int = 0
    ocr_pages: int = 0
    parent_sections_created: int = 0
    chunks_created: int = 0
    deduplicated_chunks: int = 0
    duration_sec: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
    timestamp: str = ""


class IndexingReport(BaseModel):
    """Batch/incremental indexing run report persisted to data/indexing_report.json."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    total_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_parent_sections: int = 0
    total_chunks: int = 0
    total_ocr_pages: int = 0
    total_duration_sec: float = 0.0
    files: list[IndexingFileStatus] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
