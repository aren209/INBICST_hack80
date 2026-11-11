# Навигатор сотрудника по внутренним регламентам МФТИ (INBICST_hack80)

Пайплайн мультиформатного парсинга (`PDF`, `DOCX`, `XLSX`), постраничного OCR сканов (`OCR.space API` + кэш), иерархического чанкинга по схеме **V3 (Parent-Child)** (`file -> unit -> section -> clause -> item`), локальных эмбеддингов (`intfloat/multilingual-e5-large` + лемматизированный `BM25` на `pymorphy3`) и гибридного поиска в **Qdrant**.

## Структура репозитория

- `данные/` — исходный корпус из 20 внутренних нормативных документов МФТИ (9 PDF, включая 5 сканов на 44 стр.; 8 DOCX; 3 XLSX).
- `src/`
  - `config.py` — конфигурация путей, параметров Qdrant, OCR и моделей.
  - `ingestion/registry.py` + `data/registry_overrides.yaml` — автоэкстрактор реквизитов приказов и граф версий (`supersedes`, `superseded_by`, `status`, предупреждения о дефектах исходных сканов).
  - `ingestion/ocr_engine.py` — постраничный рендеринг сканов в сжатый JPEG (`< 850 КБ`) с автоповоротом альбомных таблиц и кэшированием ответов `OCR.space API` в `data/ocr_cache/`.
  - `ingestion/parsers.py` — парсеры `DocxParser` (сохранение порядка `iterchildren()`, фильтрация `ЛИСТ СОГЛАСОВАНИЯ`), `PdfParser` (текст + таблицы + очистка сносок), `XlsxParser` (`forward-fill` иерархии ЦФО и парсинг формул смет).
  - `chunking/hierarchical_chunker.py` & `chunking/glossary.py` — иерархический чанкер `V3 (Parent-Child)` и двусторонний словарь аббревиатур МФТИ (`ФЛС`, `КФО`, `ЦФО`, `РИД`, `УИТ`, `ФЭУ` и др.).
  - `indexing/embedder.py` & `indexing/qdrant_indexer.py` — генерация локальных `Dense` (`1024 dim`) и `Sparse BM25` векторов и загрузка в Qdrant.
  - `retrieval/hybrid_search.py` — гибридный поиск в Qdrant (`Prefetch` Dense + BM25 с `exact=True` $\to$ `RRF`), мягкий бустинг актуальных версий, расширение до родительского раздела (`section`) и детекция отсутствия ответа в базе (`abstain_recommended`).
- `data/` — предрассчитанный кэш OCR всех 44 страниц сканов (`data/ocr_cache/`), кэш Dense-векторов (`data/dense_vectors_cache.npz`), дамп 888 чанков (`data/chunks_dump.json`) и 215 разделов (`data/sections_store.json`).
- `scripts/run_ingestion.py` — скрипт полной индексации корпуса в Qdrant.
- `scripts/test_queries.py` — демонстрационный прогон контрольных запросов.
- `tests/test_pipeline.py` — автотесты (`pytest`).
- `анализ_идей_чанкинга.pdf` (и `.tex`) — компактный аналитический отчёт по выбору архитектуры чанкинга и Hybrid Search.

## Быстрый старт

```bash
# 1. Запустить Qdrant в Docker
docker compose up -d

# 2. Создать окружение Python 3.11 и установить зависимости (через uv)
uv venv --python 3.11 .venv
uv pip install -r pyproject.toml

# 3. Загрузить чанки и векторы в Qdrant (использует готовый кэш в data/)
.venv/bin/python scripts/run_ingestion.py

# 4. Запустить тесты и проверочные запросы
.venv/bin/pytest tests/test_pipeline.py -v
.venv/bin/python scripts/test_queries.py
```
