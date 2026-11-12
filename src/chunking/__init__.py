"""MIPT Normative RAG V3 — Module 2 (7-Stage Hierarchical Chunker) & Module 3 (GlossaryManager)."""

from __future__ import annotations

from src.chunking.glossary import (
    DEFAULT_ABBREVIATIONS,
    ROLE_PATTERNS,
    GlossaryManager,
    get_glossary_manager,
)
from src.chunking.hierarchical_chunker import (
    ChunkerV3,
    HierarchicalChunker,
    HierarchicalChunkerV3,
    _extract_head_overlap,
    _extract_tail_overlap,
    _is_section_header,
    _merge_short_chunks,
    _parse_clause_number,
    _split_long_chunk,
    chunk_blocks,
    chunk_document,
    clean_block_text,
    flush_clause,
    is_header_requisite_table,
    is_section_header,
    is_valid_clause_number,
    parse_clause_number,
    run_chunker_v3,
)

__all__ = [
    "DEFAULT_ABBREVIATIONS",
    "ROLE_PATTERNS",
    "GlossaryManager",
    "get_glossary_manager",
    "HierarchicalChunker",
    "HierarchicalChunkerV3",
    "ChunkerV3",
    "chunk_document",
    "chunk_blocks",
    "run_chunker_v3",
    "clean_block_text",
    "is_header_requisite_table",
    "is_valid_clause_number",
    "parse_clause_number",
    "_parse_clause_number",
    "is_section_header",
    "_is_section_header",
    "_extract_tail_overlap",
    "_extract_head_overlap",
    "_split_long_chunk",
    "flush_clause",
    "_merge_short_chunks",
]
