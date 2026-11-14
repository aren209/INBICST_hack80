"""Comprehensive unit and integration tests for DeepSeek routing, .doc/.xls ingestion, and Document Viewer API."""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from typing import Any

import pymupdf as fitz
from fastapi.testclient import TestClient
import olefile
import pytest
import xlrd

from src.chunking.hierarchical_chunker import HierarchicalChunker
from src.config import SUPPORTED_EXTENSIONS, Settings, settings
from src.indexing.embedder import DenseEmbedder, HybridEmbedder, SparseBM25Encoder
from src.indexing.pipeline import IndexingPipeline
from src.indexing.qdrant_store import QdrantHybridStore
from src.indexing.table_store import StructuredTableStore
from src.ingestion.ocr_engine import HybridOCREngine
from src.ingestion.parsers import parse_document
from src.retrieval.query_pipeline import QueryPipeline
from src.retrieval.rag_agent import TableAwareRAGAgent
from src.schemas import (
    BlockType,
    DocumentMeta,
    SearchHit,
    SearchResponse,
    TableQueryRequest,
)
from src.web.app import app


# -----------------------------------------------------------------------------
# Pure-Python Binary Builders for Genuine OLE2 CFB, Word 97-2003 .doc & BIFF8 .xls
# -----------------------------------------------------------------------------


def build_ole2_cfb(streams: dict[str, bytes]) -> bytes:
    """
    Build a spec-compliant Microsoft Compound File Binary (OLE2, 512-byte sectors,
    no mini-stream cutoff so every stream >= 4096 bytes lives in regular sectors).
    """
    sector_size = 512
    # Pad streams to >= 4096 bytes (ulMiniSectorCutoff = 4096) or set cutoff = 0 so all streams use regular FAT!
    # In MS-CFB [MS-CFB 2.2], mini-stream cutoff is 0x1000 (4096): any stream < 4096 uses mini-stream UNLESS
    # we either pad streams or construct a valid Root Entry MiniStream. Padding streams is NOT valid when stream
    # byte length matters unless FIB uses exact offsets, OR we build a proper MiniStream, OR pad `WordDocument` /
    # `1Table` / `Workbook` to >= 4096 bytes (both MS Word FIB and Excel BIFF8 `EOF` ignore trailing 0x00 padding
    # after `EOF` / `fcClx+lcbClx`!). Let's verify with `olefile` and `xlrd`!
    padded_streams: list[tuple[str, bytes, int]] = []
    for name, raw in streams.items():
        orig_len = len(raw)
        # Ensure >= 4096 so OLE2 stores directly in standard 512-byte FAT sectors (no MiniFAT needed)
        padded = raw if len(raw) >= 4096 else raw + b"\x00" * (4096 - len(raw))
        if len(padded) % sector_size != 0:
            padded += b"\x00" * (sector_size - (len(padded) % sector_size))
        padded_streams.append((name, padded, len(padded)))

    # Sector layout:
    # Sector 0: Directory (up to 4 entries per 512B sector; 1 Root + N streams)
    n_entries = 1 + len(padded_streams)
    dir_sectors = math.ceil(n_entries / 4)

    stream_sector_ranges: list[tuple[int, int]] = []
    next_sec = dir_sectors
    for _name, padded, _sz in padded_streams:
        n_sec = len(padded) // sector_size
        stream_sector_ranges.append((next_sec, n_sec))
        next_sec += n_sec

    fat_sec_idx = next_sec
    total_sectors = fat_sec_idx + 1  # Single FAT sector covers up to 128 sectors (64 KB)
    assert total_sectors <= 128

    # Build FAT (128 int32 entries)
    ENDOFCHAIN = 0xFFFFFFFE
    FATSECT = 0xFFFFFFFD
    FREESECT = 0xFFFFFFFF
    fat = [FREESECT] * 128

    for d_i in range(dir_sectors):
        fat[d_i] = (d_i + 1) if (d_i + 1 < dir_sectors) else ENDOFCHAIN

    for start_sec, count_sec in stream_sector_ranges:
        for offset in range(count_sec):
            s_id = start_sec + offset
            fat[s_id] = (s_id + 1) if (offset + 1 < count_sec) else ENDOFCHAIN

    fat[fat_sec_idx] = FATSECT
    fat_bytes = struct.pack("<128I", *fat)

    # Build red-black sibling/child links in Directory (Root Entry = 0, children = 1..k chained as right-children)
    def _make_dir_entry(
        name: str,
        obj_type: int,
        left_id: int,
        right_id: int,
        child_id: int,
        start_sec: int,
        stream_size: int,
    ) -> bytes:
        utf16_name = name.encode("utf-16-le") + b"\x00\x00"
        assert len(utf16_name) <= 64
        name_buf = utf16_name.ljust(64, b"\x00")
        entry = struct.pack(
            "<64sHBBIII16sIQQIQ",
            name_buf,
            len(utf16_name),
            obj_type,  # 5=Root, 2=Stream, 0=Unallocated
            1,         # Color: black
            left_id,
            right_id,
            child_id,
            b"\x00" * 16,
            0,
            0,
            0,
            start_sec,
            stream_size,
        )
        assert len(entry) == 128
        return entry

    NOSTREAM = 0xFFFFFFFF
    dir_entries: list[bytes] = [
        _make_dir_entry(
            "Root Entry",
            5,
            NOSTREAM,
            NOSTREAM,
            1 if padded_streams else NOSTREAM,
            ENDOFCHAIN,
            0,
        )
    ]
    for idx, ((name, _padded, sz), (start_sec, _cnt)) in enumerate(
        zip(padded_streams, stream_sector_ranges), start=1
    ):
        right_sib = (idx + 1) if idx < len(padded_streams) else NOSTREAM
        dir_entries.append(
            _make_dir_entry(name, 2, NOSTREAM, right_sib, NOSTREAM, start_sec, sz)
        )

    while len(dir_entries) < dir_sectors * 4:
        dir_entries.append(b"\x00" * 128)
    dir_bytes = b"".join(dir_entries)

    # Header (512 bytes: 8B sig + 16B clsid + 5*2B + 6B + 9*4B = 76B + 109*4B DIFAT = 512B)
    difat = [fat_sec_idx] + [FREESECT] * 108
    header = (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # Signature (8B)
        + b"\x00" * 16                        # CLSID (16B)
        + struct.pack(
            "<HHHHH6sIIIIIIIII",
            0x003E,      # MinorVersion
            0x0003,      # MajorVersion (v3 = 512B sectors)
            0xFFFE,      # ByteOrder (little-endian)
            9,           # SectorShift (2^9 = 512)
            6,           # MiniSectorShift (2^6 = 64)
            b"\x00" * 6, # Reserved (6B)
            0,           # csectDir (0 for v3)
            1,           # csectFat (1 FAT sector)
            0,           # sectDirStart (sector 0)
            0,           # signature transaction
            0x1000,      # ulMiniSectorCutoff (4096)
            ENDOFCHAIN,  # sectMiniFatStart
            0,           # csectMiniFat
            ENDOFCHAIN,  # sectDifatStart
            0,           # csectDifat
        )
        + struct.pack("<109I", *difat)
    )
    assert len(header) == 512

    body = b"".join([dir_bytes, *(p for _, p, _ in padded_streams), fat_bytes])
    return header + body


def build_genuine_word97_ole2_doc() -> bytes:
    """
    Synthesize a genuine Word 97–2003 OLE2 binary `.doc` containing:
    - Piece 0 (compressed 8-bit ASCII): `"Order N 4512-1\r"`
    - Piece 1 (UTF-16LE Russian + Word field code + `\x07` 3-col table + `\x0c` page 2):
      Paragraphs, `HYPERLINK` field keeping visible `kmo@mipt.ru`, 3x3 table, and page 2 clause.
    """
    piece0_ascii = b"Order N 4512-1\r"
    piece1_ru_str = (
        "ПРИКАЗ от 15.04.2026 № 4512-1\r"
        "Об утверждении регламента лаборатории квантовой радиофизики\r"
        "1. Заявки направлять по адресу \x13 HYPERLINK \"mailto:kmo@mipt.ru\" \x14kmo@mipt.ru\x15 не позднее 5 рабочих дней.\r"
        "Код\x07Подразделение\x07Лимит, руб.\x07\x07"
        "101\x07Лаборатория квантовой оптики\x07250000\x07\x07"
        "102\x07Центр фотоники\x07480000\x07\x07"
        "\x0c"
        "2. Контроль за исполнением настоящего приказа возложить на проректора по науке.\r"
    )
    piece1_utf16 = piece1_ru_str.encode("utf-16-le")

    # WordDocument stream: 1024-byte FIB area + piece0 at offset 1024 + piece1 at offset 1056
    fib_area = bytearray(1024)
    struct.pack_into("<H", fib_area, 0x00, 0xA5EC)  # wIdent Word 97-2003
    struct.pack_into("<H", fib_area, 0x02, 0x00C1)  # nFib Word 97
    struct.pack_into("<H", fib_area, 0x0A, 0x0204)  # flags: fComplex (0x0004) | fWhichTblStm=1Table (0x0200)

    csw = 14
    cslw = 22
    cb_fclcb = 93
    struct.pack_into("<H", fib_area, 32, csw)
    pos_cslw = 34 + csw * 2  # 62
    struct.pack_into("<H", fib_area, pos_cslw, cslw)
    pos_rglw = pos_cslw + 2  # 64

    total_chars = len(piece0_ascii) + len(piece1_ru_str)
    struct.pack_into("<i", fib_area, pos_rglw + 3 * 4, total_chars)  # ccpText

    pos_cb_fclcb = pos_rglw + cslw * 4  # 152
    struct.pack_into("<H", fib_area, pos_cb_fclcb, cb_fclcb)
    pos_fclcb = pos_cb_fclcb + 2        # 154

    piece0_offset = 1024
    piece1_offset = 1056
    word_doc_stream = bytes(fib_area) + piece0_ascii.ljust(piece1_offset - piece0_offset, b"\x00") + piece1_utf16

    # Build Clx in 1Table: Pcdt (0x02) + lcbPlcPcd + aCP[0..2] + Pcd[0..1]
    cp0 = 0
    cp1 = len(piece0_ascii)
    cp2 = total_chars
    fc0_compressed = (piece0_offset * 2) | 0x40000000
    fc1_unicode = piece1_offset
    pcd0 = struct.pack("<HIH", 0, fc0_compressed, 0)
    pcd1 = struct.pack("<HIH", 0, fc1_unicode, 0)
    plc_pcd = struct.pack("<III", cp0, cp1, cp2) + pcd0 + pcd1
    clx = b"\x02" + struct.pack("<I", len(plc_pcd)) + plc_pcd

    fc_clx = 64
    table1_stream = b"\x00" * fc_clx + clx
    struct.pack_into("<II", fib_area, pos_fclcb + 33 * 8, fc_clx, len(clx))
    word_doc_stream = bytes(fib_area) + word_doc_stream[1024:]

    return build_ole2_cfb({"WordDocument": word_doc_stream, "1Table": table1_stream})


def _biff_rec(rec_id: int, payload: bytes) -> bytes:
    return struct.pack("<HH", rec_id, len(payload)) + payload


def _biff8_ustr16(text: str) -> bytes:
    """Encode a BIFF8 16-bit length UTF-16LE string (`cch` uint16 + `grbit=0x01` + UTF-16LE bytes)."""
    return struct.pack("<HB", len(text), 0x01) + text.encode("utf-16-le")


def _biff8_ustr8(text: str) -> bytes:
    """Encode a BIFF8 8-bit length UTF-16LE string (`cch` uint8 + `grbit=0x01` + UTF-16LE bytes)."""
    return struct.pack("<BB", len(text), 0x01) + text.encode("utf-16-le")


def build_genuine_biff8_xls() -> bytes:
    """
    Build a genuine BIFF8 `.xls` OLE2 workbook readable by `xlrd` (with `formatting_info=False/True`)
    and `python_calamine`, containing Russian headers, numbers, dates, and merged cells.
    """
    strings = [
        "Подразделение",       # 0
        "Статья расходов",     # 1
        "Сумма, руб.",         # 2
        "Дата утверждения",    # 3
        "ФРКТ",                # 4
        "Закупка осциллографа",# 5
        "Прикладная оптика",   # 6
    ]

    # Workbook Globals substream
    bof_globals = _biff_rec(0x0809, struct.pack("<HHHHII", 0x0600, 0x0005, 0x0DBB, 0x07CC, 0, 6))
    codepage = _biff_rec(0x0042, struct.pack("<H", 1200))
    datemode = _biff_rec(0x0022, struct.pack("<H", 0))  # 1900 date system

    # Minimal FONT (4 required by BIFF8), FORMAT (date fmt 164 = dd.mm.yyyy), and XF (0..15 + XF 16 with date format)
    font_payload = struct.pack("<HHHHHBBBB", 200, 0, 0x7FFF, 400, 0, 0, 0, 204, 0) + _biff8_ustr8("Arial")
    fonts = b"".join(_biff_rec(0x0031, font_payload) for _ in range(5))
    fmt_date = _biff_rec(0x041E, struct.pack("<H", 164) + _biff8_ustr16("dd.mm.yyyy"))

    def _xf(fmt_idx: int, is_style: bool) -> bytes:
        flags_type = 0xFFF5 if is_style else 0x0001
        return _biff_rec(0x00E0, struct.pack("<HHHHHHHHI", 0, fmt_idx, flags_type, 0x20, 0x20, 0, 0, 0, 0))

    xfs = b"".join(_xf(0, True) for _ in range(15)) + _xf(0, False) + _xf(164, False)  # XF 15=general, XF 16=date

    # SST (Shared String Table)
    sst_body = struct.pack("<II", len(strings), len(strings)) + b"".join(_biff8_ustr16(s) for s in strings)
    sst = _biff_rec(0x00FC, sst_body)
    eof_rec = _biff_rec(0x000A, b"")

    # Placeholder BOUNDSHEET (we fill exact sheet BOF byte offset right after sizing globals)
    sheet_name = "Смета 2026"
    bs_dummy = _biff_rec(0x0085, struct.pack("<IBB", 0, 0, 0) + _biff8_ustr8(sheet_name))
    globals_without_bs = bof_globals + codepage + datemode + fonts + fmt_date + xfs
    sheet_bof_offset = len(globals_without_bs) + len(bs_dummy) + len(sst) + len(eof_rec)
    boundsheet = _biff_rec(0x0085, struct.pack("<IBB", sheet_bof_offset, 0, 0) + _biff8_ustr8(sheet_name))
    globals_stream = globals_without_bs + boundsheet + sst + eof_rec

    # Worksheet substream
    bof_sheet = _biff_rec(0x0809, struct.pack("<HHHHII", 0x0600, 0x0010, 0x0DBB, 0x07CC, 0, 6))
    dims = _biff_rec(0x0200, struct.pack("<IIHHH", 0, 3, 0, 4, 0))

    def _labelsst(r: int, c: int, sst_idx: int, xf_idx: int = 15) -> bytes:
        return _biff_rec(0x00FD, struct.pack("<HHHI", r, c, xf_idx, sst_idx))

    def _number(r: int, c: int, val: float, xf_idx: int = 15) -> bytes:
        return _biff_rec(0x0203, struct.pack("<HHHd", r, c, xf_idx, val))

    # Excel serial date for 2026-05-12 in 1900 system: 46154.0
    cells = b"".join(
        [
            # Header row 0
            _labelsst(0, 0, 0),
            _labelsst(0, 1, 1),
            _labelsst(0, 2, 2),
            _labelsst(0, 3, 3),
            # Data row 1 (ФРКТ merged vertically across rows 1..2 in col 0!)
            _labelsst(1, 0, 4),
            _labelsst(1, 1, 5),
            _number(1, 2, 320000.0),
            _number(1, 3, 46154.0, xf_idx=16),
            # Data row 2 (col 0 inherits merged ФРКТ!)
            _labelsst(2, 1, 6),
            _number(2, 2, 185000.0),
            _number(2, 3, 46154.0, xf_idx=16),
        ]
    )
    # MERGEDCELLS (0x00E5): 1 range (rlo=1, rhi=2, clo=0, chi=0 inclusive in BIFF8!)
    merged = _biff_rec(0x00E5, struct.pack("<HHHHH", 1, 1, 2, 0, 0))
    sheet_stream = bof_sheet + dims + cells + merged + eof_rec

    return build_ole2_cfb({"Workbook": globals_stream + sheet_stream})


class _DummyDenseEmbedder:
    """Deterministic offline 1024-d dense embedder for fast isolated indexing tests."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for idx, t in enumerate(texts):
            vec = [0.0] * 1024
            vec[idx % 1024] = 0.6
            vec[hash(t[:32]) % 1024] = 0.8
            out.append(vec)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def check_health(self) -> dict[str, Any]:
        return {"configured": True, "provider": "mock"}


# -----------------------------------------------------------------------------
# 1. Deliverable 1: DeepSeek LLM Routing & Provider Isolation Tests
# -----------------------------------------------------------------------------


class TestDeepSeekConfigurationAndIsolation:
    def test_settings_properties_prefer_deepseek_when_key_present(self) -> None:
        s_ds = Settings(
            DEEPSEEK_API_KEY="sk-deepseek-test-key",
            DEEPSEEK_BASE_URL="https://api.deepseek.com",
            DEEPSEEK_MODEL="deepseek-chat",
            POLZA_API_KEY="pza-polza-test-key",
            POLZA_BASE_URL="https://polza.ai/api/v1",
            POLZA_LLM_MODEL="openai/gpt-4o-mini",
        )
        assert s_ds.llm_provider == "deepseek"
        assert s_ds.llm_api_key == "sk-deepseek-test-key"
        assert s_ds.llm_base_url == "https://api.deepseek.com"
        assert s_ds.llm_model == "deepseek-chat"

        s_polza = Settings(
            DEEPSEEK_API_KEY="",
            POLZA_API_KEY="pza-polza-only",
            POLZA_BASE_URL="https://polza.ai/api/v1",
            POLZA_LLM_MODEL="openai/gpt-4o-mini",
        )
        assert s_polza.llm_provider == "polza"
        assert s_polza.llm_api_key == "pza-polza-only"
        assert s_polza.llm_base_url == "https://polza.ai/api/v1"
        assert s_polza.llm_model == "openai/gpt-4o-mini"

    def test_rag_agent_and_query_pipeline_never_use_polza_slugs_on_deepseek(self) -> None:
        agent = TableAwareRAGAgent(
            api_key="sk-ds-test",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
        )
        assert agent.llm_provider == "deepseek"
        assert agent._candidate_models(None) == ["deepseek-chat"]
        # Even if a caller passes a Polza slug with '/', DeepSeek endpoint rejects it cleanly to deepseek-chat
        assert agent._candidate_models("openai/gpt-4o-mini") == ["deepseek-chat"]

        qp = QueryPipeline(
            api_key="sk-ds-test",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
        )
        assert qp.llm_provider == "deepseek"
        assert qp._candidate_models(None) == ["deepseek-chat"]
        assert qp._candidate_models("openai/gpt-4.1-mini") == ["deepseek-chat"]

    def test_embedder_and_ocr_use_local_models_by_default(self, tmp_path: Path) -> None:
        s_default = Settings(DEEPSEEK_API_KEY="", POLZA_API_KEY="")
        assert s_default.llm_provider == "local"
        assert s_default.llm_model == "Qwen/Qwen3.8-27B"
        assert s_default.LLM_QUANTIZATION == "4bit"

        emb = DenseEmbedder(cache_dir=tmp_path / "embed_cache")
        assert emb.backend == "local"
        assert emb.model == "BAAI/bge-m3"

        ocr = HybridOCREngine(cache_dir=tmp_path / "ocr_cache")
        assert ocr.vl_model_name == "PaddlePaddle/PaddleOCR-VL-1.6"


# -----------------------------------------------------------------------------
# 2. Deliverable 2: .doc (OLE2, RTF, HTML, ZIP) & .xls (BIFF8) Parsing + Indexing
# -----------------------------------------------------------------------------


class TestDocAndXlsParsingAndIndexing:
    def test_supported_extensions_include_doc_and_xls(self) -> None:
        assert {".pdf", ".docx", ".doc", ".xlsx", ".xls"}.issubset(SUPPORTED_EXTENSIONS)

    def test_parse_genuine_ole2_word97_doc_with_tables_fields_and_pages(self, tmp_path: Path) -> None:
        doc_bytes = build_genuine_word97_ole2_doc()
        doc_path = tmp_path / "Приказ 4512-1 квантовая оптика.doc"
        doc_path.write_bytes(doc_bytes)

        assert olefile.isOleFile(str(doc_path))

        blocks, meta = parse_document(doc_path, doc_id="order_4512_1")
        assert meta["file_format"] == "doc"
        assert meta["errors"] == []
        assert meta["pages_total"] == 2

        joined_text = "\n".join(b.text for b in blocks)
        assert "Order N 4512-1" in joined_text
        assert "4512-1" in joined_text
        assert "лаборатории квантовой радиофизики" in joined_text
        # Word field instruction HYPERLINK stripped, visible kmo@mipt.ru kept!
        assert "kmo@mipt.ru" in joined_text
        assert "HYPERLINK" not in joined_text

        tbl_blocks = [b for b in blocks if b.block_type == BlockType.TABLE.value and b.table_rows]
        assert len(tbl_blocks) == 1
        tbl = tbl_blocks[0]
        assert tbl.table_headers == ["Код", "Подразделение", "Лимит, руб."]
        assert ["101", "Лаборатория квантовой оптики", "250000"] in tbl.table_rows
        assert ["102", "Центр фотоники", "480000"] in tbl.table_rows

        page2_blocks = [b for b in blocks if b.page_num == 2]
        assert any("проректора по науке" in b.text for b in page2_blocks)

    def test_parse_rtf_doc_with_cp1251_unicode_and_table(self, tmp_path: Path) -> None:
        # Build RTF with \ansicpg1251 \'xx Russian escapes, \uN Unicode, and \cell/\row table
        title_cp1251 = "".join(f"\\'{b:02x}" for b in "РАСПОРЯЖЕНИЕ № 88-р от 10.03.2026".encode("cp1251"))
        clause_u = "".join(f"\\u{ord(ch)}?" for ch in "1.1. Установить срок подачи отчета — 10 рабочих дней.")
        h1 = "".join(f"\\'{b:02x}" for b in "Этап".encode("cp1251"))
        h2 = "".join(f"\\'{b:02x}" for b in "Срок".encode("cp1251"))
        v1 = "".join(f"\\'{b:02x}" for b in "Согласование".encode("cp1251"))
        v2 = "".join(f"\\'{b:02x}" for b in "3 дня".encode("cp1251"))

        rtf_content = (
            "{\\rtf1\\ansi\\ansicpg1251{\\fonttbl{\\f0 Times New Roman;}}\n"
            f"{title_cp1251}\\par\n"
            f"{clause_u}\\par\n"
            f"\\trowd\\intbl {h1}\\cell {h2}\\cell\\row\n"
            f"\\trowd\\intbl {v1}\\cell {v2}\\cell\\row\n"
            "\\pard}\n"
        ).encode("ascii")

        rtf_path = tmp_path / "Распоряжение 88-р.doc"
        rtf_path.write_bytes(rtf_content)

        blocks, meta = parse_document(rtf_path)
        assert meta["file_format"] == "doc"
        assert meta["errors"] == []
        full_txt = "\n".join(b.text for b in blocks)
        assert "РАСПОРЯЖЕНИЕ № 88-р" in full_txt
        assert "10 рабочих дней" in full_txt
        assert "Times New Roman" not in full_txt

        tbls = [b for b in blocks if b.block_type == BlockType.TABLE.value and b.table_rows]
        assert len(tbls) == 1
        assert tbls[0].table_headers == ["Этап", "Срок"]
        assert tbls[0].table_rows == [["Согласование", "3 дня"]]

    def test_parse_html_disguised_doc(self, tmp_path: Path) -> None:
        html_bytes = (
            "<!DOCTYPE html><html><head><meta charset='windows-1251'><style>body{color:black}</style></head>"
            "<body><h1>Регламент пропуска</h1>"
            "<p>2.4. Пропуск оформляется через бюро пропусков за 24 часа.</p>"
            "<table><tr><th>Тип</th><th>Кабинет</th></tr><tr><td>Временный</td><td>ГК-112</td></tr></table>"
            "</body></html>"
        ).encode("cp1251")
        p = tmp_path / "Регламент пропуска.doc"
        p.write_bytes(html_bytes)

        blocks, meta = parse_document(p)
        assert meta["file_format"] == "doc"
        txt = "\n".join(b.text for b in blocks)
        assert "бюро пропусков за 24 часа" in txt
        assert "color:black" not in txt
        tbls = [b for b in blocks if b.block_type == BlockType.TABLE.value]
        assert len(tbls) == 1
        assert tbls[0].table_rows == [["Временный", "ГК-112"]]

    def test_parse_genuine_biff8_xls_and_end_to_end_pipeline_indexing(self, tmp_path: Path) -> None:
        xls_bytes = build_genuine_biff8_xls()
        xls_path = tmp_path / "Смета лаборатории 2026.xls"
        xls_path.write_bytes(xls_bytes)

        # Verify xlrd itself recognizes it as a genuine BIFF8 workbook
        wb_check = xlrd.open_workbook(str(xls_path))
        assert wb_check.sheet_names() == ["Смета 2026"]
        wb_check.release_resources()

        blocks, meta = parse_document(xls_path, doc_id="smeta_lab_2026_xls")
        assert meta["file_format"] == "xls"
        assert meta["errors"] == []
        assert len(blocks) >= 1

        # End-to-end isolated IndexingPipeline test for BOTH .xls and .doc!
        doc_path = tmp_path / "Приказ 4512-1.doc"
        doc_path.write_bytes(build_genuine_word97_ole2_doc())

        qdrant_mem = QdrantHybridStore(in_memory=True, collection_name="test_doc_xls_col")
        table_store = StructuredTableStore(db_path=tmp_path / "test_tables.db")
        embedder = HybridEmbedder(dense_embedder=_DummyDenseEmbedder(), sparse_encoder=SparseBM25Encoder())
        pipeline = IndexingPipeline(
            qdrant_store=qdrant_mem,
            embedder=embedder,
            table_store=table_store,
            registry_path=tmp_path / "registry.json",
            sections_path=tmp_path / "sections.json",
            report_path=tmp_path / "report.json",
        )

        st_xls = pipeline.index_file(xls_path, force=True)
        assert st_xls.status == "success"
        assert st_xls.chunks_created >= 1
        assert st_xls.tables_count == 1

        # Verify .xls sheet is registered & queryable in StructuredTableStore
        xls_tables = table_store.list_tables(doc_id=st_xls.doc_id)
        assert len(xls_tables) == 1
        assert xls_tables[0].file_format == "xls"
        assert xls_tables[0].sheet_name == "Смета 2026"
        q_resp = table_store.query_table(
            TableQueryRequest(table_id=xls_tables[0].table_id, operation="filter_rows", search_text="осциллограф")
        )
        assert q_resp.matched_count == 1
        assert "320000" in str(q_resp.rows[0])

        st_doc = pipeline.index_file(doc_path, force=True)
        assert st_doc.status == "success"
        assert st_doc.chunks_created >= 2
        assert pipeline.registry[st_doc.doc_id].file_format == "doc"


# -----------------------------------------------------------------------------
# 3. Deliverable 3: Document Viewer API (/file, /pages/{n}.png, /view) & Quotes
# -----------------------------------------------------------------------------


class TestDocumentViewerAndQuoteDocId:
    def test_verify_answer_quotes_populates_doc_id_and_page_num(self) -> None:
        hit = SearchHit(
            chunk_id="reg_concert_hall_booking::c02",
            doc_id="reg_concert_hall_booking",
            file_name="Регламент бронирования конц зала.pdf",
            title="Регламент бронирования концертного зала",
            doc_number="3897-1",
            doc_date="2025-05-12",
            clause_no="2.3",
            page_numbers=[2],
            clean_text="2.3. Для бронирования концертного зала организатор подает служебную записку не менее чем за 2 месяца.",
            parent_text="2.3. Для бронирования концертного зала организатор подает служебную записку не менее чем за 2 месяца.",
            score=0.92,
        )
        sr = SearchResponse(query="бронирование кз", hits=[hit], confidence_score=0.92)
        answer = (
            "Служебную записку подают заранее:\n\n"
            "> «Для бронирования концертного зала организатор подает служебную записку не менее чем за 2 месяца.»\n"
            "> — Регламент бронирования концертного зала, № 3897-1, п. 2.3\n"
        )
        quotes = TableAwareRAGAgent._verify_answer_quotes(answer, sr)
        assert len(quotes) == 1
        q0 = quotes[0]
        assert q0["verified"] is True
        assert q0["doc_id"] == "reg_concert_hall_booking"
        assert q0["page"] == "2"
        assert q0["page_num"] == 2

    def test_viewer_endpoints_pdf_doc_xls_and_health(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Create an isolated IndexingPipeline with a 2-page synthetic PDF, a .doc, and a .xls
        pdf_path = tmp_path / "Тестовый регламент КЗ.pdf"
        pdf_doc = fitz.open()
        p1 = pdf_doc.new_page(width=595, height=842)
        p1.insert_text((72, 100), "1.1. General provisions of campus booking.", fontsize=12)
        p2 = pdf_doc.new_page(width=595, height=842)
        p2.insert_text(
            (72, 220),
            "2.3. Applications must be sent to kmo@mipt.ru at least 60 days prior.",
            fontsize=12,
        )
        pdf_doc.save(str(pdf_path))
        pdf_doc.close()

        doc_path = tmp_path / "Приказ 4512-1.doc"
        doc_path.write_bytes(build_genuine_word97_ole2_doc())

        xls_path = tmp_path / "Смета 2026.xls"
        xls_path.write_bytes(build_genuine_biff8_xls())

        qdrant_mem = QdrantHybridStore(in_memory=True, collection_name="test_viewer_col")
        table_store = StructuredTableStore(db_path=tmp_path / "viewer_tables.db")
        embedder = HybridEmbedder(dense_embedder=_DummyDenseEmbedder(), sparse_encoder=SparseBM25Encoder())
        iso_pipeline = IndexingPipeline(
            qdrant_store=qdrant_mem,
            embedder=embedder,
            table_store=table_store,
            registry_path=tmp_path / "v_registry.json",
            sections_path=tmp_path / "v_sections.json",
            report_path=tmp_path / "v_report.json",
        )

        st_pdf = iso_pipeline.index_file(pdf_path, ocr_backend="local_rapidocr", force=True)
        st_doc = iso_pipeline.index_file(doc_path, force=True)
        st_xls = iso_pipeline.index_file(xls_path, force=True)

        web_app_mod = sys.modules["src.web.app"]
        monkeypatch.setattr(web_app_mod, "get_pipeline", lambda: iso_pipeline)

        client = TestClient(app)

        # 1) /api/health exposes llm provider & model
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        h_json = r_health.json()
        assert "llm_provider" in h_json
        assert "llm_model" in h_json
        assert ".doc" in h_json["supported_extensions"]
        assert ".xls" in h_json["supported_extensions"]

        # 2) /api/documents/{doc_id}/file streams inline PDF with RFC5987 UTF-8 filename
        r_file = client.get(f"/api/documents/{st_pdf.doc_id}/file")
        assert r_file.status_code == 200
        assert r_file.headers["content-type"].startswith("application/pdf")
        assert "inline; filename*=UTF-8''" in r_file.headers["content-disposition"]

        # 3) /api/documents/{doc_id}/pages/2.png renders valid PNG; out-of-range & non-PDF return 404
        r_png = client.get(f"/api/documents/{st_pdf.doc_id}/pages/2.png?dpi=120")
        assert r_png.status_code == 200
        assert r_png.headers["content-type"] == "image/png"
        assert r_png.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert "max-age=3600" in r_png.headers.get("cache-control", "")

        assert client.get(f"/api/documents/{st_pdf.doc_id}/pages/99.png").status_code == 404
        assert client.get(f"/api/documents/{st_doc.doc_id}/pages/1.png").status_code == 200
        assert client.get(f"/api/documents/{st_xls.doc_id}/pages/1.png").status_code == 404

        # 4) /api/documents/{pdf_id}/view resolves quote on page 2 even when page=1 was passed!
        r_view_pdf = client.get(
            f"/api/documents/{st_pdf.doc_id}/view",
            params={"page": 1, "quote": "Applications must be sent to kmo@mipt.ru at least 60 days prior."},
        )
        assert r_view_pdf.status_code == 200
        vp = r_view_pdf.json()
        assert vp["format"] == "pdf"
        assert vp["total_pages"] == 2
        assert vp["target_page"] == 2
        assert vp["matched_quote_on_page"] is True
        assert vp["download_url"] == f"/api/documents/{st_pdf.doc_id}/file"
        p2_data = vp["pages"][1]
        assert p2_data["image_url"].endswith("/pages/2.png")
        assert len(p2_data["highlights"]) >= 1
        hl0 = p2_data["highlights"][0]
        assert 0.0 <= hl0["x"] <= 1.0 and 0.0 <= hl0["y"] <= 1.0
        assert 0.0 < hl0["w"] <= 1.0 and 0.0 < hl0["h"] <= 1.0

        # 5) /api/documents/{doc_id}/view for .doc resolves page 2 quote & renders structured table on page 1
        r_view_doc = client.get(
            f"/api/documents/{st_doc.doc_id}/view",
            params={"quote": "возложить на проректора по науке"},
        )
        assert r_view_doc.status_code == 200
        vd = r_view_doc.json()
        assert vd["format"] == "doc"
        assert vd["total_pages"] == 2
        assert vd["target_page"] == 2
        assert vd["matched_quote_on_page"] is True
        assert any(b["type"] == "table" and len(b["rows"]) == 2 for b in vd["pages"][0]["blocks"])

        # 6) /api/documents/{xls_id}/view for .xls renders sheet label & rows
        r_view_xls = client.get(
            f"/api/documents/{st_xls.doc_id}/view",
            params={"quote": "Закупка осциллографа"},
        )
        assert r_view_xls.status_code == 200
        vx = r_view_xls.json()
        assert vx["format"] == "xls"
        assert vx["target_page"] == 1
        assert vx["matched_quote_on_page"] is True
        assert vx["pages"][0]["label"] == "Смета 2026"
        assert vx["pages"][0]["image_url"] is None
        assert vx["pages"][0]["blocks"][0]["type"] in ("heading", "table")

        # 7) Unsupported upload extension returns clean 400
        r_bad = client.post(
            "/api/documents/upload",
            files=[("files", ("malware.exe", b"MZ\x00\x00", "application/octet-stream"))],
        )
        assert r_bad.status_code == 400

    def test_viewer_on_production_indexed_pdf_docx_and_xlsx_read_only(self) -> None:
        from src.indexing.pipeline import get_pipeline

        prod_pipeline = get_pipeline()
        if not prod_pipeline.registry:
            pytest.skip("Production registry empty")

        client = TestClient(app)

        # Real PDF: reg_concert_hall_booking
        if "reg_concert_hall_booking" in prod_pipeline.registry:
            rv = client.get(
                "/api/documents/reg_concert_hall_booking/view",
                params={"quote": "служебную записку", "clause": "2.3"},
            )
            assert rv.status_code == 200
            body = rv.json()
            assert body["format"] == "pdf"
            assert body["total_pages"] >= 2
            assert 1 <= body["target_page"] <= body["total_pages"]

        # Real DOCX & XLSX in registry
        docx_meta = next((m for m in prod_pipeline.registry.values() if m.file_format == "docx"), None)
        if docx_meta is not None:
            r_dx = client.get(f"/api/documents/{docx_meta.doc_id}/view")
            assert r_dx.status_code == 200
            assert r_dx.json()["format"] == "docx"
            assert len(r_dx.json()["pages"]) >= 1

        xlsx_meta = next((m for m in prod_pipeline.registry.values() if m.file_format == "xlsx"), None)
        if xlsx_meta is not None:
            r_xl = client.get(f"/api/documents/{xlsx_meta.doc_id}/view")
            assert r_xl.status_code == 200
            xl_body = r_xl.json()
            assert xl_body["format"] == "xlsx"
            assert len(xl_body["pages"]) >= 1
            # Verify large Excel row cap <= 250
            for p_item in xl_body["pages"]:
                for blk in p_item["blocks"]:
                    if blk["type"] == "table":
                        assert len(blk["rows"]) <= 250

