"""Comprehensive Unit Tests for Normative RAG V3 Hierarchical Chunker (Module 2 & Module 3).

Covers all 7 critical V3 edge case groups:
1. Date vs Clause disambiguation & protection (dates never parsed as clauses, 1..63 supported).
2. Duplicate clause numbers in source document (distinct chunk_id, distinct UUIDv5, no overwrite).
3. Approval zone isolation & reset (ЛИСТ СОГЛАСОВАНИЯ -> zone="approval", reset on Приложение / УТВЕРЖДЕН).
4. Section header vs Imperative order clause disambiguation.
5. Long clause splitting (>880 chars with continuation header and parts) & 3-pass short chunk merging (<160 chars).
6. Clean ParentSectionRecord (Stage 5) & 12% Bilateral Word-Boundary Overlap + Unit Isolation (Stage 6).
7. Stage 7 embed_text format, u00_meta synthetic chunk, GlossaryManager role patterns & BM25 token preservation.
"""

from __future__ import annotations

import re
import pytest

from src.config import (
    CHUNK_OVERLAP_RATIO,
    MAX_CLAUSE_CHARS,
    MAX_TABLE_FORM_CHARS,
    MIN_CLAUSE_CHARS,
    NON_ANSWER_ZONES,
    PARA_OVERLAP_MAX_CHARS,
    PARA_OVERLAP_MIN_CHARS,
    TABLE_OVERLAP_MAX_CHARS,
    TABLE_OVERLAP_MIN_CHARS,
)
from src.schemas import (
    BlockType,
    ChunkRecord,
    DocumentMeta,
    DocumentStatus,
    ParentSectionRecord,
    ParsedBlock,
    ZoneType,
    make_point_uuid,
)
from src.chunking.glossary import (
    DEFAULT_ABBREVIATIONS,
    ROLE_PATTERNS,
    GlossaryManager,
    get_glossary_manager,
)
from src.chunking.hierarchical_chunker import (
    HierarchicalChunker,
    _extract_head_overlap,
    _extract_tail_overlap,
    _is_section_header,
    _merge_short_chunks,
    _parse_clause_number,
    _split_long_chunk,
    chunk_document,
    clean_block_text,
    flush_clause,
    is_header_requisite_table,
    is_section_header,
    is_valid_clause_number,
    parse_clause_number,
)
from src.indexing.embedder import tokenize_bm25


# ===========================================================================
# 1. Date vs Clause Protection & Disambiguation
# ===========================================================================
class TestDateVsClauseProtection:
    """Verify that dates are NEVER parsed as clauses, while genuine clauses are recognized."""

    @pytest.mark.parametrize(
        "date_line",
        [
            "25.11.2021 Приказ вступил в силу со дня его подписания.",
            "01.09.2025 Начать учебный процесс в очном формате.",
            "16.06.2025 Утвердить регламент взаимодействия подразделений.",
            "28.03.2023 № 794-1",
            "11.10.2019 г. утверждена смета доходов и расходов.",
            "05.12.2024 Ввести в действие положение.",
            "25.11.2021",
            "01.09.2025",
        ],
    )
    def test_dates_at_start_never_parsed_as_clause(self, date_line: str) -> None:
        """Dates with DD.MM.YYYY must return None from parse_clause_number."""
        result = parse_clause_number(date_line)
        assert result is None, f"Date '{date_line}' was incorrectly parsed as clause: {result}"

    @pytest.mark.parametrize(
        "num_str,rest_text",
        [
            ("25.11", "2021"),
            ("01.09", "2025"),
            ("1.9", "2025 г."),
            ("25.11.2021", ""),
            ("01.09.2025", "Начало занятий"),
            ("16.06.2025", "Приказ"),
            ("01.09", "учебный год"),  # Leading zero '01' indicates day/month
            ("25.11", "2021 года"),
        ],
    )
    def test_is_valid_clause_number_rejects_dates(self, num_str: str, rest_text: str) -> None:
        """Candidate numbers representing dates must be rejected by is_valid_clause_number."""
        assert not is_valid_clause_number(num_str, rest_text=rest_text), (
            f"Candidate '{num_str}' with rest '{rest_text}' should be rejected as a date"
        )

    @pytest.mark.parametrize(
        "clause_line,expected_num,expected_body_prefix",
        [
            ("1.1. Общие положения регламента", "1.1", "Общие положения"),
            ("1.1 Общие положения без точки", "1.1", "Общие положения"),
            ("2.3.1. Требования к участникам конкурса", "2.3.1", "Требования к участникам"),
            ("п. 2.6. Концертный зал бронируется за 2 месяца", "2.6", "Концертный зал"),
            ("пункт 3.2. Порядок согласования служебной записки", "3.2", "Порядок согласования"),
            ("1. Настоящий Регламент определяет порядок...", "1", "Настоящий Регламент"),
            ("17. В случае направления в командировку...", "17", "В случае направления"),
            ("63. Настоящий Регламент вступает в силу...", "63", "Настоящий Регламент"),
        ],
    )
    def test_genuine_clauses_parsed_accurately(
        self, clause_line: str, expected_num: str, expected_body_prefix: str
    ) -> None:
        """Genuine multi-level and single-level clauses up to 63 must be recognized."""
        res = parse_clause_number(clause_line)
        assert res is not None, f"Failed to parse genuine clause: '{clause_line}'"
        num, body = res
        assert num == expected_num
        assert body.startswith(expected_body_prefix)

    def test_single_level_bounds_and_invalid_numbers(self) -> None:
        """Single-level clauses > 300 or invalid formats must be rejected."""
        assert is_valid_clause_number("63", has_trailing_dot=True)
        assert is_valid_clause_number("65", has_trailing_dot=True)
        assert is_valid_clause_number("300", has_trailing_dot=True)
        assert not is_valid_clause_number("301", has_trailing_dot=True)
        # Without trailing dot or explicit prefix, bare single digit is not a clause
        assert not is_valid_clause_number("1", has_trailing_dot=False, has_explicit_prefix=False)

    def test_full_chunker_does_not_create_date_clauses(self) -> None:
        """End-to-end chunking must not produce chunks with clause_no == '25.11' or '01.09'."""
        blocks = [
            ParsedBlock(text="Приказ № 3065-1 от 25.11.2021", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="25.11.2021 вступил в действие настоящий приказ.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="1.1. Общие положения о финансово-лицевых счетах.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="01.09.2025 года осуществляется плановый пересмотр структуры.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="1.2. Порядок открытия счетов в системе 1С:ДО.", block_type=BlockType.PARAGRAPH.value),
        ]
        doc_meta = DocumentMeta(doc_id="doc_date_test", file_name="test_order.docx", doc_number="3065-1")
        chunks, _ = chunk_document(blocks, doc_meta=doc_meta)

        clause_numbers = [c.clause_no for c in chunks if c.clause_no]
        assert "25.11" not in clause_numbers
        assert "01.09" not in clause_numbers
        assert "1.9" not in clause_numbers
        assert any("1.1" in c.clause_no for c in chunks)
        assert any("1.2" in c.clause_no for c in chunks)


# ===========================================================================
# 2. Duplicate Clause Numbers in Source Document
# ===========================================================================
class TestDuplicateClauseNumbers:
    """Verify handling of duplicate clause numbers (e.g. two clauses numbered 2.6 in 3065-1)."""

    def test_duplicate_clauses_get_distinct_chunk_id_and_point_uuid(self) -> None:
        """Two clauses both numbered 2.6 receive distinct chunk_id, distinct UUIDv5, and neither overwrites."""
        blocks = [
            ParsedBlock(
                text="2. Порядок открытия счетов",
                block_type=BlockType.HEADING.value,
                is_bold=True,
            ),
            ParsedBlock(
                text="2.6. Первый пункт с номером 2.6: Инициатор формирует служебную записку в 1С:ДО с указанием темы.",
                block_type=BlockType.PARAGRAPH.value,
            ),
            ParsedBlock(
                text="2.6. Второй пункт с номером 2.6: Финансодержатель в течение трех рабочих дней рассматривает заявку.",
                block_type=BlockType.PARAGRAPH.value,
            ),
        ]
        doc_meta = DocumentMeta(doc_id="order_3065_1", file_name="Приказ N 3065-1.docx", doc_number="3065-1")
        chunks, _ = chunk_document(blocks, doc_meta=doc_meta)

        normative_chunks = [c for c in chunks if c.zone == ZoneType.NORMATIVE.value]
        # Both chunks must be present
        cl_26_chunks = [c for c in normative_chunks if c.clause_no == "2.6"]
        assert len(cl_26_chunks) == 2, f"Expected 2 distinct chunks for duplicate clause 2.6, got {len(cl_26_chunks)}"

        ch1, ch2 = cl_26_chunks[0], cl_26_chunks[1]
        assert ch1.chunk_id != ch2.chunk_id, "chunk_id must be distinct for duplicate clauses"
        assert ch1.point_uuid != ch2.point_uuid, "point_uuid (UUIDv5) must be distinct"
        assert ch1.clause_no == "2.6"
        assert ch2.clause_no == "2.6"
        assert "Первый пункт" in ch1.clean_text
        assert "Второй пункт" in ch2.clean_text

    def test_merge_short_chunks_preserves_duplicate_clauses(self) -> None:
        """_merge_short_chunks must never merge two duplicate clauses into one."""
        ch1 = ChunkRecord(
            chunk_id="test::u01::c01",
            doc_id="test_doc",
            unit_id="u01_main",
            section_id="test::u01::s01",
            clause_no="2.6",
            clean_text="2.6. Короткий пункт А.",
            zone=ZoneType.NORMATIVE.value,
        )
        ch2 = ChunkRecord(
            chunk_id="test::u01::c02",
            doc_id="test_doc",
            unit_id="u01_main",
            section_id="test::u01::s01",
            clause_no="2.6",
            clean_text="2.6. Короткий пункт Б.",
            zone=ZoneType.NORMATIVE.value,
        )
        merged = _merge_short_chunks([ch1, ch2])
        assert len(merged) == 2, "Short duplicate clauses must not be merged together!"
        assert merged[0].chunk_id != merged[1].chunk_id


# ===========================================================================
# 3. Approval Zone Isolation & Clean Normative Reset
# ===========================================================================
class TestApprovalZoneIsolationAndReset:
    """Verify approval zone triggers and clean reset upon entering normative sections/appendices."""

    def test_approval_zone_marked_and_excluded_from_normative(self) -> None:
        """Trigger 'ЛИСТ СОГЛАСОВАНИЯ' marks subsequent blocks as zone='approval'."""
        blocks = [
            ParsedBlock(text="1.1. Нормативный пункт регламента.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="ЛИСТ СОГЛАСОВАНИЯ", block_type=BlockType.HEADING.value),
            ParsedBlock(text="Проректор по учебной работе: Иванов И.И. — Согласовано.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="Начальник юридического отдела: Петров П.П. — Согласовано.", block_type=BlockType.PARAGRAPH.value),
        ]
        doc_meta = DocumentMeta(doc_id="doc_appr_test", file_name="test_approval.docx")
        chunks, _ = chunk_document(blocks, doc_meta=doc_meta)

        approval_chunks = [c for c in chunks if c.zone == ZoneType.APPROVAL.value]
        normative_chunks = [c for c in chunks if c.zone == ZoneType.NORMATIVE.value]

        assert len(approval_chunks) >= 1
        assert all(c.zone in NON_ANSWER_ZONES for c in approval_chunks)
        assert any("Иванов И.И." in c.clean_text for c in approval_chunks)
        assert any("1.1" in c.clause_no for c in normative_chunks)

    def test_normative_mode_reset_after_approval_sheet(self) -> None:
        """Encountering 'Приложение № 1' or 'УТВЕРЖДЕН...' after approval sheet restores normative mode."""
        blocks = [
            ParsedBlock(text="1.1. Базовый порядок проведения мероприятий.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="ЛИСТ СОГЛАСОВАНИЯ", block_type=BlockType.HEADING.value),
            ParsedBlock(text="Главный бухгалтер: Сидорова С.С. — Согласовано.", block_type=BlockType.PARAGRAPH.value),
            # Structural boundary resetting to normative mode
            ParsedBlock(text="Приложение № 1 к Регламенту", block_type=BlockType.HEADING.value, is_bold=True),
            ParsedBlock(text="1.1. Порядок расчета стоимости услуг концертного зала.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="1.2. Нормы времени на репетиционные дни.", block_type=BlockType.PARAGRAPH.value),
        ]
        doc_meta = DocumentMeta(doc_id="doc_reset_test", file_name="test_reset.pdf")
        chunks, _ = chunk_document(blocks, doc_meta=doc_meta)

        # Chunks in Приложение № 1 must have zone="normative" and unit_id="u02"
        u02_chunks = [c for c in chunks if c.unit_id == "u02"]
        assert len(u02_chunks) >= 1
        assert all(c.zone == ZoneType.NORMATIVE.value for c in u02_chunks)
        assert any("расчета стоимости" in c.clean_text for c in u02_chunks)

        # Verify approval chunks are strictly isolated
        appr_chunks = [c for c in chunks if c.zone == ZoneType.APPROVAL.value]
        assert len(appr_chunks) >= 1
        assert all("Сидорова" in c.clean_text for c in appr_chunks)


# ===========================================================================
# 4. Section Header vs Imperative Order Clause Disambiguation
# ===========================================================================
class TestSectionHeaderVsImperativeClause:
    """Verify smart distinction between Parent section headers and order directives."""

    @pytest.mark.parametrize(
        "header_text,is_bold,expected",
        [
            ("1. Общие положения", True, True),
            ("2. Порядок открытия ФЛС", True, True),
            ("3. Правила бронирования Концертного зала", True, True),
            ("4. Ответственность сторон и порядок возмещения ущерба", True, True),
            ("5. Сроки и порядок подачи заявок", True, True),
            ("6. Заключительные положения", True, True),
            ("Раздел 2. Финансовые условия", False, True),
            ("I. Общие положения", False, True),
        ],
    )
    def test_valid_section_headers_recognized(self, header_text: str, is_bold: bool, expected: bool) -> None:
        """Section titles starting with standard nouns and without imperative verbs must be headers."""
        assert is_section_header(header_text, is_bold=is_bold) is expected

    @pytest.mark.parametrize(
        "clause_text,is_bold,expected",
        [
            ("2. Утвердить прилагаемую форму сметы расходов.", True, False),
            ("3. Контроль за исполнением приказа возложить на первого проректора.", True, False),
            ("1. Ввести в действие с 01.09.2025 настоящий регламент.", False, False),
            ("4. Признать утратившим силу приказ № 1469-1 от 11.10.2019.", False, False),
            ("5. Главному бухгалтеру обеспечить контроль расходования средств.", False, False),
            ("6. Управлению информационных технологий разместить приказ на сайте.", False, False),
            ("2. Поручить руководителям ЦФО предоставление отчетов;", False, False),  # Trailing semicolon
        ],
    )
    def test_imperative_order_clauses_not_section_headers(
        self, clause_text: str, is_bold: bool, expected: bool
    ) -> None:
        """Imperative clauses in order bodies must NOT be classified as section headers."""
        assert is_section_header(clause_text, is_bold=is_bold) is expected

    def test_pipeline_classifies_order_clauses_as_normative_chunks(self) -> None:
        """Pipeline must classify order clauses as normative clauses, not section titles."""
        blocks = [
            ParsedBlock(text="ПРИКАЗ", block_type=BlockType.HEADING.value),
            ParsedBlock(text="1. Утвердить Регламент первой линии техподдержки.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="2. Ввести в действие регламент с 01.09.2025.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="3. Контроль за исполнением приказа оставляю за собой.", block_type=BlockType.PARAGRAPH.value),
        ]
        doc_meta = DocumentMeta(doc_id="order_it_test", file_name="Приказ_IT.docx", doc_number="123-1")
        chunks, parents = chunk_document(blocks, doc_meta=doc_meta)

        normative_chunks = [c for c in chunks if c.zone == ZoneType.NORMATIVE.value]
        clause_nos = [c.clause_no for c in normative_chunks]
        assert any("1" in no for no in clause_nos)
        assert any("2" in no for no in clause_nos)
        assert any("3" in no for no in clause_nos)

        # None of the order clauses should become parent section titles
        for parent in parents:
            assert "Утвердить Регламент" not in parent.section_title
            assert "Контроль за исполнением" not in parent.section_title


# ===========================================================================
# 5. Long Clause Splitting & 3-Pass Short Chunk Merging
# ===========================================================================
class TestClauseSplittingAndMerging:
    """Verify long clause splitting (>880 chars) and 3-pass short chunk merging (<160 chars)."""

    def test_long_clause_splits_with_continuation_header_and_part_labeling(self) -> None:
        """A clause > 880 chars splits on sentence boundaries with continuation headers and parts."""
        sentence = (
            "Инициатор направляет заявку на открытие финансово-лицевого счета в планово-финансовый отдел. "
            "К заявке прилагается финансово-экономическое обоснование с расчетом планируемых расходов. "
            "Ответственный сотрудник ПФО проверяет соответствие кодов КВР и КОСГУ действующему бюджету. "
        )
        long_text = "2.1. " + (sentence * 6)  # ~1150 chars > 880
        assert len(long_text) > MAX_CLAUSE_CHARS

        chunk = ChunkRecord(
            chunk_id="test::u01::c01",
            doc_id="test_doc",
            unit_id="u01_main",
            clause_no="2.1",
            clean_text=long_text,
            raw_text=long_text,
            zone=ZoneType.NORMATIVE.value,
        )

        split_chunks = _split_long_chunk(chunk, max_chars=MAX_CLAUSE_CHARS, min_chars=MIN_CLAUSE_CHARS)
        assert len(split_chunks) >= 2, "Long chunk should have split into at least 2 parts"

        part1, part2 = split_chunks[0], split_chunks[1]
        assert "п. 2.1 (ч. 1)" in part1.clause_no
        assert "п. 2.1 (ч. 2)" in part2.clause_no
        assert "(Продолжение п. 2.1:" in part2.clean_text
        assert part1.chunk_id == "test::u01::c01_p1"
        assert part2.chunk_id == "test::u01::c01_p2"

        # Verify no tiny remainder < 160 chars was created
        for p in split_chunks:
            assert len(p.clean_text) >= MIN_CLAUSE_CHARS, (
                f"Part {p.chunk_id} is shorter than MIN_CLAUSE_CHARS ({len(p.clean_text)} < {MIN_CLAUSE_CHARS})"
            )

    def test_three_pass_short_chunk_merging(self) -> None:
        """Short clauses < 160 chars merge forward (1.2, 1.3) and backward (1.4, 1.5)."""
        c1 = ChunkRecord(
            chunk_id="doc::u01::c01",
            doc_id="doc",
            unit_id="u01_main",
            section_id="doc::u01::s01",
            clause_no="1.1",
            clean_text=(
                "1.1. Достаточно длинный нормативный пункт номер один общих положений настоящего регламента, "
                "превышающий минимальный установленный порог символов нарезки чанков в регламенте бронирования "
                "и не подлежащий принудительному слиянию с соседними пунктами."
            ),  # > 160 chars
            zone=ZoneType.NORMATIVE.value,
        )
        c2 = ChunkRecord(
            chunk_id="doc::u01::c02",
            doc_id="doc",
            unit_id="u01_main",
            section_id="doc::u01::s01",
            clause_no="1.2",
            clean_text="1.2. Короткий пункт два.",  # < 160 chars
            zone=ZoneType.NORMATIVE.value,
        )
        c3 = ChunkRecord(
            chunk_id="doc::u01::c03",
            doc_id="doc",
            unit_id="u01_main",
            section_id="doc::u01::s01",
            clause_no="1.3",
            clean_text=(
                "1.3. Пункт три с детальным и обстоятельным описанием установленного порядка взаимодействия "
                "структурных подразделений института при согласовании служебных записок на проведение мероприятий "
                "в Концертном зале МФТИ."
            ),  # > 160 chars
            zone=ZoneType.NORMATIVE.value,
        )
        c4 = ChunkRecord(
            chunk_id="doc::u01::c04",
            doc_id="doc",
            unit_id="u01_main",
            section_id="doc::u01::s01",
            clause_no="1.4",
            clean_text=(
                "1.4. Четвертый пункт с нормативным описанием обязанностей руководителя структурного подразделения "
                "и финансодержателя по своевременному предоставлению отчетных документов и сметных расчетов "
                "в планово-финансовый отдел института."
            ),  # > 160 chars
            zone=ZoneType.NORMATIVE.value,
        )
        c5 = ChunkRecord(
            chunk_id="doc::u01::c05",
            doc_id="doc",
            unit_id="u01_main",
            section_id="doc::u01::s01",
            clause_no="1.5",
            clean_text="1.5. Короткий пункт пять в конце раздела.",  # < 160 chars, end of section
            zone=ZoneType.NORMATIVE.value,
        )

        merged = _merge_short_chunks([c1, c2, c3, c4, c5], min_chars=160, max_chars=880)

        # c2 should have merged into c3 (forward: "1.2, 1.3")
        # c5 should have merged into c4 (backward: "1.4, 1.5")
        merged_clauses = [m.clause_no for m in merged]
        assert "1.2, 1.3" in merged_clauses, f"Forward merge failed: {merged_clauses}"
        assert "1.4, 1.5" in merged_clauses, f"Backward merge failed: {merged_clauses}"


# ===========================================================================
# 6. Clean ParentSectionRecord (Stage 5) & 12% Bilateral Word-Boundary Overlap + Unit Isolation (Stage 6)
# ===========================================================================
class TestParentSectionAndBilateralOverlap:
    """Verify Stage 5 clean ParentSectionRecord and Stage 6 12% bilateral overlap with unit isolation."""

    def test_word_boundary_overlap_never_cuts_words_or_numbers(self) -> None:
        """_extract_tail_overlap and _extract_head_overlap must cut strictly on word boundaries."""
        sample_text = (
            "В соответствии с приказом № 3065-1 от 25.11.2021 инициатор обязан предоставить "
            "расчет расходов с кодами КВР 244 и КОСГУ 226 в размере 16.4% от фонда оплаты труда."
        )
        tail = _extract_tail_overlap(sample_text, min_chars=35, max_chars=95)
        head = _extract_head_overlap(sample_text, min_chars=35, max_chars=95)

        # Tail must start at word boundary
        assert tail and tail in sample_text
        tail_start_idx = sample_text.find(tail)
        assert tail_start_idx == 0 or sample_text[tail_start_idx - 1].isspace(), (
            f"Tail overlap '{tail}' cuts word: preceding char is '{sample_text[tail_start_idx - 1]}'"
        )

        # Head must end at word boundary
        assert head and head in sample_text
        head_end_idx = head_end_idx = sample_text.find(head) + len(head)
        assert head_end_idx == len(sample_text) or sample_text[head_end_idx].isspace(), (
            f"Head overlap '{head}' cuts word: following char is '{sample_text[head_end_idx]}'"
        )

        # Verify no token like "16.4%" or "3065-1" is cut in half
        assert not re.search(r"^\d+%", tail), f"Cut percentage in tail: '{tail}'"
        assert not re.search(r"-\d+$", head), f"Cut hyphenated number in head: '{head}'"

    def test_parent_section_clean_text_assembled_before_overlap(self) -> None:
        """ParentSectionRecord.clean_text must NOT contain '...' overlap prefixes or suffixes."""
        blocks = [
            ParsedBlock(text="1. Общие положения", block_type=BlockType.HEADING.value, is_bold=True),
            ParsedBlock(
                text=(
                    "1.1. Первый пункт общих положений регламента бронирования Концертного зала устанавливает "
                    "базовые принципы и порядок предоставления помещений для проведения студенческих и научных мероприятий."
                ),
                block_type=BlockType.PARAGRAPH.value,
            ),
            ParsedBlock(
                text=(
                    "1.2. Второй пункт общих положений о сроках подачи служебных записок определяет ответственность "
                    "инициаторов за своевременное предоставление программы и сценарного плана не менее чем за два месяца."
                ),
                block_type=BlockType.PARAGRAPH.value,
            ),
        ]
        doc_meta = DocumentMeta(doc_id="doc_parent_test", file_name="reglament.docx")
        chunks, parents = chunk_document(blocks, doc_meta=doc_meta)

        assert len(parents) >= 1
        parent = parents[0]
        # Clean text of parent must contain both clauses
        assert "1.1." in parent.clean_text
        assert "1.2." in parent.clean_text
        # Parent section clean text must have NO '...' overlap markers
        assert "..." not in parent.clean_text

        # Child chunks have raw_text with bilateral overlap where applicable
        normative_chunks = [c for c in chunks if c.zone == ZoneType.NORMATIVE.value]
        assert len(normative_chunks) >= 2
        # Clean text of child chunks must also not have '...'
        for ch in normative_chunks:
            assert not ch.clean_text.startswith("...")
            assert not ch.clean_text.endswith("...")

    def test_overlap_and_id_links_never_cross_unit_boundaries(self) -> None:
        """Overlap and prev_id/next_id must NEVER cross unit_id boundaries (u01_main <-> u02)."""
        blocks = [
            # Unit 1 (Main order body)
            ParsedBlock(text="1. Утвердить Положение о командировании сотрудников МФТИ.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="2. Контроль за исполнением настоящего приказа возложить на первого проректора.", block_type=BlockType.PARAGRAPH.value),
            # Unit 2 (Appendix 1)
            ParsedBlock(text="Приложение № 1 к приказу", block_type=BlockType.HEADING.value, is_bold=True),
            ParsedBlock(text="1.1. Суточные расходы при командировании составляют 100 рублей в сутки.", block_type=BlockType.PARAGRAPH.value),
            ParsedBlock(text="1.2. Авансовый отчет предоставляется в течение трех рабочих дней.", block_type=BlockType.PARAGRAPH.value),
        ]
        doc_meta = DocumentMeta(doc_id="order_trip_test", file_name="Приказ_Командировки.docx", doc_number="591-1")
        chunks, _ = chunk_document(blocks, doc_meta=doc_meta)

        u01_chunks = [c for c in chunks if c.unit_id == "u01_main"]
        u02_chunks = [c for c in chunks if c.unit_id == "u02"]

        assert len(u01_chunks) >= 1
        assert len(u02_chunks) >= 1

        last_u01 = u01_chunks[-1]
        first_u02 = u02_chunks[0]

        # 1. prev_id / next_id isolation
        assert last_u01.next_id is None, "Last chunk of u01_main must have next_id == None"
        assert first_u02.prev_id is None, "First chunk of u02 must have prev_id == None"

        # 2. Text overlap isolation: zero text from u02 in u01 and vice-versa
        assert "Суточные расходы" not in last_u01.raw_text
        assert "Контроль за исполнением" not in first_u02.raw_text
        assert not first_u02.raw_text.startswith("...")


# ===========================================================================
# 7. Stage 7 embed_text, u00_meta, GlossaryManager & BM25 Token Preservation
# ===========================================================================
class TestStage7EmbedMetaAndGlossary:
    """Verify embed_text format, u00_meta generation, GlossaryManager features and BM25 token preservation."""

    def test_bracketed_embed_text_format(self) -> None:
        """embed_text must follow the strict bracketed header template from §1.2 Stage 7."""
        blocks = [
            ParsedBlock(text="2. Порядок открытия ФЛС", block_type=BlockType.HEADING.value, is_bold=True),
            ParsedBlock(
                text="2.3. Инициатор направляет служебную записку в планово-финансовый отдел через систему 1С:ДО.",
                block_type=BlockType.PARAGRAPH.value,
            ),
        ]
        doc_meta = DocumentMeta(
            doc_id="order_3065_1",
            file_name="Приказ N 3065-1.docx",
            doc_number="3065-1",
            doc_date="2021-11-25",
            title="Приказ об открытии счетов",
            status=DocumentStatus.ACTIVE.value,
            applicable_years=[2021, 2022, 2023, 2024, 2025, 2026],
        )
        chunks, _ = chunk_document(blocks, doc_meta=doc_meta)
        norm_chunks = [c for c in chunks if c.zone == ZoneType.NORMATIVE.value]
        assert norm_chunks, "Expected normative chunks"

        target_chunk = norm_chunks[0]
        embed = target_chunk.embed_text
        assert embed.startswith("[Документ:"), f"embed_text must start with [Документ:, got: {embed[:40]}"
        assert "№ 3065-1" in embed
        assert "2021-11-25" in embed
        assert "Редакция/Статус: Действующий" in embed
        assert "[Термины:" in embed
        assert "ФЛС" in embed or "1С:ДО" in embed
        assert target_chunk.raw_text in embed

    def test_u00_meta_generated_for_superseded_or_amended_docs(self) -> None:
        """u00_meta synthetic card must be generated when edition links or notes exist."""
        doc_meta = DocumentMeta(
            doc_id="app_1_smeta_2019",
            file_name="Приложение 1_новая смета 2019.xlsx",
            doc_number="1469-1",
            status=DocumentStatus.SUPERSEDED_TEMPLATE_REFERENCE.value,
            superseded_by=["794-1"],
            version_note="Форма сметы 2019 заменена Приказом № 794-1.",
            quality_warning="Книга сохранена без пересчета кэша формул.",
        )
        u00_chunk = get_glossary_manager().build_u00_meta_chunk(doc_meta)
        assert u00_chunk is not None
        assert u00_chunk.chunk_id == "app_1_smeta_2019::u00_meta::c000"
        assert u00_chunk.zone == ZoneType.META.value
        assert u00_chunk.zone in NON_ANSWER_ZONES
        assert "794-1" in u00_chunk.clean_text
        assert "Форма сметы 2019" in u00_chunk.clean_text

    def test_glossary_manager_role_patterns(self) -> None:
        """GlossaryManager must extract responsible roles matching all 12 ROLE_PATTERNS."""
        gm = get_glossary_manager()
        assert len(ROLE_PATTERNS) >= 12, f"Expected 12 ROLE_PATTERNS, got {len(ROLE_PATTERNS)}"

        test_cases = [
            ("Инициатор составляет заявку на подбор персонала.", "инициатор"),
            ("Организатор мероприятия несет ответственность за порядок в Концертном зале.", "организатор"),
            ("Руководитель структурного подразделения согласовывает служебную записку.", "руководитель подразделения / ЦФО"),
            ("Финансодержатель утверждает смету доходов и расходов.", "финансодержатель"),
            ("Начальник отдела ПФО проводит экспертизу кодов КОСГУ.", "начальник отдела (ПФО, КМО, УИТ и др.)"),
            ("Управление бухгалтерского учета и бухгалтерия формируют проводки.", "бухгалтерия"),
            ("Дирекция физтех-школы контролирует бюджет подразделения.", "дирекция"),
            ("Первый проректор подписывает распорядительный акт.", "проректор"),
            ("Администратор сети и технический специалист настраивают серверную стойку.", "администратор / техспециалист"),
            ("Комендант общежития проверяет санитарное состояние жилых комнат.", "комендант / управление кампусом"),
            ("Кадровая служба и рекрутер организуют собеседование кандидатов.", "кадровая служба / рекрутер"),
            ("Согласующее лицо в системе 1С:ДО визирует проект приказа.", "согласующее лицо"),
        ]

        for text, expected_role in test_cases:
            roles = gm.extract_roles(text)
            assert expected_role in roles, f"Role '{expected_role}' not extracted from: '{text}'. Found: {roles}"

    def test_glossary_manager_bidirectional_expansion_and_learning(self) -> None:
        """GlossaryManager must expand abbreviations bidirectionally and auto-learn from text."""
        gm = get_glossary_manager()

        # Bidirectional lookup
        assert gm.expand_abbr("ФЛС") == "финансово-лицевой счет"
        assert gm.expand_abbr("КЗ") == "Концертный зал"
        assert gm.expand_abbr("ЦОД") == "центр обработки данных"
        assert "ФЛС" in gm.get_abbreviation("финансово-лицевой счет")

        # Query expansion
        expanded = gm.expand_query("как подать служебку на концерт в кз")
        assert "концертный зал" in expanded.lower() or "кз" in expanded.lower()

        # Auto-learning from text with anti-garbage filter
        learn_text = (
            "Порядок работы системы автоматизации бизнес-процессов (далее — «САБП»). "
            "Сумма платежа составляет 150000 (далее - 150000 руб)."  # Garbage currency amount must be rejected!
        )
        gm.learn_from_text(learn_text)
        assert "САБП" in gm.abbreviations
        assert "150000" not in gm.abbreviations

    def test_bm25_token_preservation(self) -> None:
        """tokenize_bm25 must preserve order numbers with hyphens, percentages, emails, and clause numbers."""
        sample = (
            "По Приказу № 3065-1 от 25.11.2021 и Приказу № 3699-1 резерв отпусков составляет 16.4% для ППС "
            "и 8.2% для остальных. Заявки слать на kmo@mipt.ru через систему 1С:ДО по п. 2.6 и п. 6.1 регламента."
        )
        tokens = tokenize_bm25(sample)

        # Hyphenated order numbers
        assert "3065-1" in tokens, "Hyphenated order number '3065-1' must be preserved"
        assert "3699-1" in tokens, "Hyphenated order number '3699-1' must be preserved"

        # Percentages
        assert "16.4%" in tokens, "Percentage '16.4%' must be preserved"
        assert "8.2%" in tokens, "Percentage '8.2%' must be preserved"

        # Emails
        assert "kmo@mipt.ru" in tokens, "Email 'kmo@mipt.ru' must be preserved"

        # 1C brand
        assert "1с:до" in tokens, "Brand '1с:до' must be preserved"

        # Dotted clause numbers
        assert "2.6" in tokens, "Clause number '2.6' must be preserved"
        assert "6.1" in tokens, "Clause number '6.1' must be preserved"


# ===========================================================================
# 8. Tables, Forms, Header Unwrapping & Stage 1 Noise Cleanup
# ===========================================================================
class TestTableFormAndStage1NoiseCleanup:
    """Verify Stage 1 noise cleaning, 1x2 header table unwrapping, and Stage 3 table/form slicing."""

    def test_clean_block_text_dehyphenation_and_footers(self) -> None:
        """clean_block_text dehyphenates wrapped words and removes footer page numbers."""
        hyphen_wrapped = "порядок командиро-\nвания сотрудников"
        cleaned = clean_block_text(hyphen_wrapped)
        assert "командирования" in cleaned

        footer = "Стр. 3 из 14"
        assert clean_block_text(footer) == ""

        standalone_page = "14"
        assert clean_block_text(standalone_page) == ""

    def test_footnote_superscript_cleaned_from_order_number(self) -> None:
        """clean_block_text cleans glued footnote superscript from order number (e.g. № 8123 -> № 812)."""
        text_with_footnote = "Приказ № 8123 от 10.05.2023 года"
        cleaned = clean_block_text(text_with_footnote)
        assert "№ 812" in cleaned
        assert "№ 8123" not in cleaned

    def test_is_header_requisite_table_detection(self) -> None:
        """1x2 header frame table (e.g. 'от 28.03.2023 | № 794-1') is detected for unwrapping."""
        tbl_block = ParsedBlock(
            text="от 28.03.2023 | № 794-1",
            block_type=BlockType.TABLE.value,
            table_headers=["от 28.03.2023", "№ 794-1"],
            table_rows=[],
        )
        assert is_header_requisite_table(tbl_block)

        # Normal data table must NOT be treated as a requisite frame table
        data_tbl = ParsedBlock(
            text="",
            block_type=BlockType.TABLE.value,
            table_headers=["Код ЦФО", "Наименование подразделения", "Лимит"],
            table_rows=[["01.01", "Ректорат", "1000000"]],
        )
        assert not is_header_requisite_table(data_tbl)

    def test_table_sliced_by_eight_rows_with_header_and_hierarchy_repeated(self) -> None:
        """Wide table with 24 rows must be sliced into batches of 8 rows with headers repeated."""
        headers = ["№ п/п", "Мероприятие", "Дней репетиций"]
        rows = [[str(i), f"Мероприятие {i}", str(i % 5 + 1)] for i in range(1, 25)]  # 24 rows = 3 full 8-row slices
        table_block = ParsedBlock(
            text="",
            block_type=BlockType.TABLE.value,
            table_headers=headers,
            table_rows=rows,
            parent_hierarchy=["Физтех-школы", "КЗ"],
        )
        doc_meta = DocumentMeta(doc_id="doc_table_test", file_name="table_doc.pdf")
        chunks, _ = chunk_document([table_block], doc_meta=doc_meta)

        table_chunks = [c for c in chunks if c.zone == ZoneType.TABLE.value]
        # 24 rows sliced by 8 -> 3 chunks (8 + 8 + 8)
        assert len(table_chunks) == 3, f"Expected 3 table slices, got {len(table_chunks)}"

        for ch in table_chunks:
            # Each table chunk must repeat headers
            assert "Мероприятие" in ch.clean_text
            assert "Дней репетиций" in ch.clean_text
            # Each table chunk must repeat parent hierarchy
            assert "Физтех-школы" in ch.clean_text or "КЗ" in ch.clean_text

    def test_content_hash_consistency_and_normalization(self) -> None:
        """content_hash must normalize whitespace and 'ё' to 'е'."""
        from src.schemas import compute_content_hash

        text1 = "Порядок  открытия   ФЛС\nв системе  1С:ДО"
        text2 = "порядок открытия флс в системе 1с:до"
        assert compute_content_hash(text1) == compute_content_hash(text2)

        text_yo = "Резерв на оплату отпусков учёных"
        text_ye = "Резерв на оплату отпусков ученых"
        assert compute_content_hash(text_yo) == compute_content_hash(text_ye)

