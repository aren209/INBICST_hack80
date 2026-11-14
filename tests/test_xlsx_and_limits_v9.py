"""Unit & integration tests for v9 Excel routing/formula extraction, clause/section limits, and year guards."""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from src.chunking.glossary import GlossaryManager, _extract_applicable_years
from src.chunking.hierarchical_chunker import (
    chunk_document,
    is_section_header,
    is_valid_clause_number,
    parse_clause_number,
)
from src.config import (
    MULTI_LEVEL_FIRST_MAX,
    MULTI_LEVEL_PART_MAX,
    SINGLE_LEVEL_CLAUSE_MAX,
)
from src.indexing.table_store import (
    StructuredTableStore,
    execute_tool_call,
    extract_logical_tables_and_assign_ids,
)
from src.ingestion.parsers import (
    _grid_to_excel_sheet_blocks,
    has_smeta_kvr_header,
    parse_xlsx,
)
from src.schemas import BlockType, DocumentMeta, ParsedBlock, ZoneType


# ===========================================================================
# 1. Excel Parser Routing & Universal Formula Extraction
# ===========================================================================
class TestExcelRoutingAndFormulaExtraction:
    def test_has_smeta_kvr_header_detection(self) -> None:
        assert has_smeta_kvr_header([["Наименование показателя", "Ставка", "КВР", "КОСГУ"]]) is True
        assert has_smeta_kvr_header([["Заголовок"], ["Наименование показателя", "Вид затрат", "Сумма"]]) is True
        assert has_smeta_kvr_header([["Статья", "Январь", "Февраль", "Итого"]]) is False
        assert has_smeta_kvr_header([]) is False

    def test_arbitrary_xlsx_smeta_sheet_without_kvr_preserves_headers_and_extracts_formulas(
        self, tmp_path: Path
    ) -> None:
        xlsx_path = tmp_path / "Отдел_расходы.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Смета отдела"
        ws.append(["Статья", "Январь", "Февраль", "Итого"])
        ws.append(["Оборудование", 100, 200, "=SUM(B2:C2)"])
        ws.append(["Командировки", 50, 75, "=SUM(B3:C3)"])

        # Also test a second sheet named "Форма отчета" with a banner row and formula in a header cell
        ws2 = wb.create_sheet("Форма отчета")
        ws2.append(["Расчет затрат подразделения за 2026 год"])
        ws2.append(["Показатель", "База", 'Итого [формула: ="Итого"]'])
        ws2.append(["Материалы", 400, "=B3*1.2"])
        wb.save(xlsx_path)

        blocks, meta = parse_xlsx(xlsx_path)
        assert meta["errors"] == []

        table_blocks = [b for b in blocks if b.block_type == BlockType.TABLE.value]
        assert len(table_blocks) == 2

        blk1 = table_blocks[0]
        assert blk1.zone == ZoneType.TABLE.value
        assert blk1.table_headers == ["Статья", "Январь", "Февраль", "Итого"]
        assert not any(h.startswith("Колонка ") for h in blk1.table_headers)
        assert len(blk1.table_rows) == 2
        assert "[формула: =SUM(B2:C2)]" in blk1.table_rows[0][3]
        assert blk1.metadata["sheet_name"] == "Смета отдела"
        assert blk1.metadata["has_formulas"] is True
        assert blk1.metadata["formula_count"] == 2
        assert blk1.metadata["structured_formulas"] == [
            {
                "row_index": 0,
                "column": "Итого",
                "formula": "=SUM(B2:C2)",
                "cell": "[формула: =SUM(B2:C2)]",
            },
            {
                "row_index": 1,
                "column": "Итого",
                "formula": "=SUM(B3:C3)",
                "cell": "[формула: =SUM(B3:C3)]",
            },
        ]

        # Sheet 2: banner extracted, formula tag stripped from header label, data formula preserved
        blk2 = table_blocks[1]
        assert blk2.table_headers == ["Показатель", "База", "Итого"]
        assert blk2.metadata["table_title"] == "Расчет затрат подразделения за 2026 год"
        assert blk2.metadata["has_formulas"] is True
        assert blk2.metadata["formula_count"] == 1
        assert blk2.metadata["structured_formulas"][0]["formula"] == "=B3*1.2"
        assert blk2.metadata["structured_formulas"][0]["column"] == "Итого"

        # End-to-end StructuredTableStore + extract_logical_tables_and_assign_ids + upsert_document_tables + get_formulas
        doc_meta = DocumentMeta(
            doc_id="arbitrary_smeta_xlsx",
            file_name=xlsx_path.name,
            file_format="xlsx",
        )
        specs = extract_logical_tables_and_assign_ids(blocks, doc_meta)
        assert len(specs) == 2
        assert specs[0]["headers"] == ["Статья", "Январь", "Февраль", "Итого"]

        store = StructuredTableStore(db_path=tmp_path / "tables_v9.db")
        passports = store.upsert_document_tables(doc_meta, blocks)
        assert len(passports) == 2
        p0 = passports[0]
        assert p0.has_formulas is True
        assert p0.formula_count == 2
        assert [c.name for c in p0.columns] == ["Статья", "Январь", "Февраль", "Итого"]

        raw_res, q_res = execute_tool_call(
            "query_structured_table",
            {"table_id": p0.table_id, "operation": "get_formulas"},
            store=store,
        )
        assert q_res is not None
        formulas = [entry["formula"] for entry in raw_res["formulas"]]
        assert "=SUM(B2:C2)" in formulas
        assert "=SUM(B3:C3)" in formulas
        assert all(entry["column"] == "Итого" for entry in raw_res["formulas"])

    def test_grid_to_excel_sheet_blocks_routes_non_smeta_and_extracts_leading_eq_formulas(self) -> None:
        grid = [
            ["Статья", "Январь", "Февраль", 'Итого [формула: ="Итого"]'],
            ["Оборудование", "100", "200", "=SUM(B2:C2)"],
            ["Услуги", "50", "50", "100 [формула: =SUM(B3:C3)]"],
        ]
        # Pass formula_count=2 and sheet name "Смета отдела" — must STILL route to Data Table Sheet!
        blks = _grid_to_excel_sheet_blocks(
            sheet_name="Смета отдела",
            sheet_idx=1,
            trimmed_grid=grid,
            formula_count=2,
        )
        assert len(blks) == 1
        tbl = blks[0]
        assert tbl.block_type == BlockType.TABLE.value
        assert tbl.zone == ZoneType.TABLE.value
        assert tbl.table_headers == ["Статья", "Январь", "Февраль", "Итого"]
        assert tbl.metadata["has_formulas"] is True
        assert tbl.metadata["formula_count"] == 2
        assert [f["formula"] for f in tbl.metadata["structured_formulas"]] == [
            "=SUM(B2:C2)",
            "=SUM(B3:C3)",
        ]

    def test_real_doki_xlsx_routing_regression_read_only(self) -> None:
        from src.config import DOCS_DIR
        doki_dir = DOCS_DIR / "appendices_and_forms"
        smeta_path = doki_dir / "Приложение 1_новая смета 2019 (с незащищенными формулами).xlsx"
        fin_path = doki_dir / "Приложение к приказу об утверждении фин. структуры 2026г..xlsx"
        if not smeta_path.exists() or not fin_path.exists():
            pytest.skip("Real доки/ xlsx fixtures not available")

        smeta_blocks, smeta_meta = parse_xlsx(smeta_path)
        assert smeta_meta["errors"] == []
        assert len(smeta_blocks) == 2
        assert all(b.block_type == BlockType.FORM.value and b.zone == ZoneType.FORM.value for b in smeta_blocks)
        assert all("Наименование показателя" in b.metadata["structured_headers"] for b in smeta_blocks)
        assert all(len(b.metadata["structured_rows"]) == 196 for b in smeta_blocks)

        fin_blocks, fin_meta = parse_xlsx(fin_path)
        assert fin_meta["errors"] == []
        fin_tbls = [b for b in fin_blocks if b.block_type == BlockType.TABLE.value]
        assert len(fin_tbls) == 1
        assert fin_tbls[0].table_headers == [
            "Финансодержатель",
            "ЦФО 1 уровня",
            "ЦФО 2 уровня",
            "ЦФО 3 уровня",
            "ЦФО 4 уровня",
        ]
        assert len(fin_tbls[0].table_rows) == 524
        assert fin_tbls[0].metadata["has_formulas"] is False
        assert fin_tbls[0].metadata["formula_count"] == 0


# ===========================================================================
# 2. Clause & Section Numbering Ceilings
# ===========================================================================
class TestClauseAndSectionLimitsV9:
    def test_config_limits_raised(self) -> None:
        assert SINGLE_LEVEL_CLAUSE_MAX == 300
        assert MULTI_LEVEL_FIRST_MAX == 120
        assert MULTI_LEVEL_PART_MAX == 200

    def test_high_clauses_and_sections_accepted_while_dates_years_currency_rejected(self) -> None:
        # High single-level clauses (72., 115., 300.) accepted; 301. rejected
        for num in ("72", "115", "300"):
            res = parse_clause_number(f"{num}. Хранение документов осуществляется в архиве.")
            assert res is not None and res[0] == num
        assert parse_clause_number("301. Вне лимита.") is None

        # High multi-level clauses accepted up to 120.200
        assert parse_clause_number("72.150. Дополнительное требование.") == ("72.150", "Дополнительное требование.")
        assert parse_clause_number("120.200. Верхний предел.") == ("120.200", "Верхний предел.")
        assert parse_clause_number("121.1. Сверх первого предела.") is None
        assert parse_clause_number("1.201. Сверх второго предела.") is None

        # Section headers up to 60 accepted; > 60 rejected as section header
        assert is_section_header("35. Порядок хранения", is_bold=True) is True
        assert is_section_header("60. Заключительные положения", is_bold=False) is True
        assert is_section_header("61. Порядок хранения", is_bold=True) is False

        # Date, year, and currency guards intact
        assert parse_clause_number("01.04.2025 Приказ вступает в силу.") is None
        assert parse_clause_number("19.06.2026 Утвердить регламент.") is None
        assert is_valid_clause_number("01.04.2025", "") is False
        assert is_valid_clause_number("19.06.2026", "") is False
        assert parse_clause_number("2019 г. бюджетный период") is None
        assert parse_clause_number("2035 г. плановый горизонт") is None
        assert parse_clause_number("2025. Утвердить смету") is None
        assert parse_clause_number("150. руб. оплата") is None
        assert parse_clause_number("75.50 руб. пошлина") is None
        assert parse_clause_number("20. % ставка") is None

    def test_end_to_end_chunk_document_preserves_high_sections_and_clauses(self) -> None:
        blocks = [
            ParsedBlock(text="35. Порядок хранения", block_type=BlockType.HEADING.value, is_bold=True),
            ParsedBlock(
                text=(
                    "72. Срок хранения первичных учетных документов и регистров бухгалтерского учета "
                    "составляет не менее пяти лет после отчетного года в соответствии с графиком архивации "
                    "подразделений института."
                ),
                block_type=BlockType.PARAGRAPH.value,
            ),
            ParsedBlock(
                text=(
                    "115. Ответственность за сохранность дел, своевременную передачу описей и контроль "
                    "доступа к архивным материалам несет руководитель соответствующего структурного "
                    "подразделения института."
                ),
                block_type=BlockType.PARAGRAPH.value,
            ),
            ParsedBlock(
                text="01.04.2025 проведена плановая инвентаризация архивного фонда подразделений.",
                block_type=BlockType.PARAGRAPH.value,
            ),
            ParsedBlock(
                text="19.06.2026 утвержден график передачи дел в постоянный архив.",
                block_type=BlockType.PARAGRAPH.value,
            ),
        ]
        doc_meta = DocumentMeta(doc_id="archive_reg_v9", file_name="Регламент_архива.docx")
        chunks, parents = chunk_document(blocks, doc_meta=doc_meta)

        assert any(p.section_num == "35" and "Порядок хранения" in p.section_title for p in parents)
        clause_nos = [c.clause_no for c in chunks if c.clause_no]
        assert "72" in clause_nos
        assert "115" in clause_nos
        assert "01.04" not in clause_nos
        assert "19.06" not in clause_nos


# ===========================================================================
# 3. Year Extraction Range & Guards
# ===========================================================================
class TestGlossaryYearExtractionV9:
    def test_extract_applicable_years_range_and_guards(self) -> None:
        probe = (
            "Положение 1998 года и стратегия до 2031г.\n"
            "Приказ № 2025-1 от 1995 г. на период 2025-2031 и 2026–2030 до 2050\n"
            "Приказ N 1998-1, No 2026, № 2026, № 3065-1\n"
            "Коды: КОСГУ 2026, КВР 2020, КФО 2020, сч. 2026, 1998-2, 2025-1\n"
            "Вне диапазона: 1994, 2051"
        )
        years = _extract_applicable_years(probe)
        assert years == [1995, 1998, 2025, 2026, 2030, 2031, 2050]

    def test_extract_applicable_years_ignores_only_order_and_budget_codes(self) -> None:
        probe = (
            "Приказ № 2025-1\n"
            "Приказ N 1998-1\n"
            "Приказ № 3065-1\n"
            "Приказ № 2026\n"
            "Расходы по КОСГУ 2026 и КВР 2020 и КФО 2021 и сч. 2024\n"
            "Подпункт 2025-1 и 1998-2"
        )
        assert _extract_applicable_years(probe) == []

    def test_glossary_manager_extract_document_meta_years(self, tmp_path: Path) -> None:
        gm = GlossaryManager(
            glossary_path=tmp_path / "glossary.yaml",
            overrides_path=tmp_path / "registry_overrides.yaml",
        )
        text = (
            "ПРИКАЗ № 2025-1\n"
            "Об утверждении положения 1998 года о развитии до 2031г.\n"
            "Учет по КОСГУ 2026, ссылка на приказ N 1998-1 и приказ № 3065-1."
        )
        meta = gm.extract_document_meta(text, file_name="Приказ_№_2025_1.docx")
        assert meta.applicable_years == [1998, 2031]
