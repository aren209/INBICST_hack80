# Навигатор сотрудника по внутренним регламентам МФТИ

Решение задачи 4 хакатона: **первый этап** — мультиформатный парсинг корпуса,
иерархический чанкинг и гибридный поиск по внутренним регламентам и приказам МФТИ.
Генерация ответа и пользовательский интерфейс в этот этап не входят; результат
этапа — воспроизводимый индекс и поиск, на который можно повесить LLM/API.

## Что умеет

- Читает `PDF` (текстовый слой + сканы), `DOCX` (порядок абзацев и таблиц,
  листы согласования), `XLSX` (иерархия ЦФО, формулы смет).
- Постраничный OCR сканов через `OCR.space` с дисковым кэшем
  (`data/ocr_cache/`); для корпуса из репозитория OCR уже закэширован.
- Режет документ по схеме **V3 (Parent-Child)**:
  `файл → единица (приказ/приложение) → раздел → пункт → подпункт`.
- Строит локальные эмбеддинги `intfloat/multilingual-e5-large` (1024d) и
  лемматизированный BM25 (`pymorphy3` + словарь аббревиатур МФТИ).
- Ищет в Qdrant гибридно: `Dense + BM25 → RRF`, мягкий буст действующих версий,
  расширение до родительского раздела, предупреждения о версиях/дефектах сканов
  и отказ от ответа, если в базе нет темы.

## Архитектура и поток данных

```text
data/данные/*.pdf|docx|xlsx
        │
        ▼
ingestion/parsers.py         PDF (PyMuPDF/pdfplumber), DOCX (python-docx), XLSX (openpyxl)
ingestion/ocr_engine.py      OCR.space + кэш по sha256 файла/страницы
ingestion/registry.py        реквизиты, статусы, граф версий (YAML-overrides)
        │  ParsedBlock[]
        ▼
chunking/hierarchical_chunker.py   file → unit → section → clause
chunking/glossary.py               аббревиатуры МФТИ (курируемый словарь)
        │  ChunkRecord[] + ParentSectionRecord[]
        ▼
indexing/embedder.py         e5-large (dense) + BM25 (sparse)
indexing/qdrant_indexer.py   коллекция Qdrant: именованные векторы dense/bm25
        │
        ▼
retrieval/hybrid_search.py   prefetch dense+sparse → RRF → буст версий →
                             parent-section → abstain
```

Ключевые артефакты в `data/`:

| Файл | Назначение |
|---|---|
| `chunks_dump.json` | все чанки (≈900) с метаданными и текстом |
| `sections_store.json` | родительские разделы для расширения контекста |
| `dense_vectors_cache.npz` | кэш dense-векторов (не пересчитывать при повторном запуске) |
| `bm25_stats.json` | IDF/avgdl для BM25 |
| `glossary.json` | курируемый словарь аббревиатур |
| `ocr_cache/` | постраничный OCR сканов |
| `registry_overrides.yaml` | проверенные реквизиты, статусы и связи документов |

## Требования

- Python **3.11–3.13**.
- [uv](https://docs.astral.sh/uv/) (рекомендуется) или `pip`.
- Docker + Docker Compose — для Qdrant (или локальный `qdrant` на `:6333`).
- Интернет при первом запуске: модель e5-large ONNX ≈ 2.2 ГБ.
  На слабом CPU первичная индексация 891 чанка может занять десятки минут;
  для быстрого старта можно взять mpnet-base (см. `DENSE_MODEL_NAME`).
- `OCR_SPACE_API_KEY` — только если добавляете новые сканы (для корпуса из
  репозитория работает кэш).

## Быстрый старт (Linux/macOS)

```bash
# 1. Qdrant
docker compose up -d

# 2. Окружение
uv venv --python 3.11 .venv
uv pip install -r pyproject.toml
# или: python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# 3. Переменные окружения (минимум)
export QDRANT_URL=http://localhost:6333
# export OCR_SPACE_API_KEY=...   # нужен только для новых сканов

# 4. Индексация (использует кэш OCR/векторов)
.venv/bin/python scripts/run_ingestion.py

# 5. Тесты и демонстрационные запросы
.venv/bin/pytest tests/ -v
.venv/bin/python scripts/test_queries.py
```

Остановка:

```bash
docker compose down
```

## Быстрый старт (Windows PowerShell)

```powershell
docker compose up -d
uv venv --python 3.13 .venv
uv pip install -r pyproject.toml

$env:QDRANT_URL="http://localhost:6333"
# $env:OCR_SPACE_API_KEY="..."

.\.venv\Scripts\python.exe scripts\run_ingestion.py
.\.venv\Scripts\python.exe -m pytest tests\ -v
.\.venv\Scripts\python.exe scripts\test_queries.py
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `PROJECT_ROOT` | корень репозитория | вычисляется от `src/config.py` |
| `DATA_DIR` | `<root>/данные` | исходный корпус |
| `ARTIFACTS_DATA_DIR` | `<root>/data` | кэши, дампы, реестр |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant |
| `QDRANT_COLLECTION_NAME` | `mipt_regulations` | коллекция |
| `OCR_SPACE_API_KEY` | пусто | ключ OCR.space; без него работают только закэшированные страницы |
| `OCR_SPACE_ENDPOINT` | `https://api.ocr.space/parse/image` | endpoint |
| `OCR_SMALL_NATIVE_TEXT_CHARS` | `600` | если текстового слоя меньше, а на странице есть картинка — OCR |
| `DENSE_MODEL_NAME` | `intfloat/multilingual-e5-large` | модель эмбеддингов |
| `DENSE_THREADS` | — | число потоков ONNX (для CPU) |
| `DENSE_BATCH_SIZE` | `8` | batch при индексации |
| `MODELS_CACHE_DIR` | `<root>/models_cache` | куда fastembed скачивает модель |
| `DENSE_MODEL_PATH` | — | путь к уже скачанному ONNX-каталогу (offline) |
| `DENSE_ABSTAIN_THRESHOLD` | `0.815` | порог отказа от ответа |
| `PARENT_SECTION_MAX_CHARS` | `12000` | сколько символов раздела отдавать в контекст |

Пример — `.env.example`. Сам файл `.env` автоматически не загружается:
используйте `export $(grep -v '^#' .env | xargs)` или `docker compose --env-file .env ...`.

## Docker

```bash
# только Qdrant
docker compose up -d

# Qdrant + отладочная форма на http://localhost:8501
docker compose --profile debug up -d --build
```

`Dockerfile` собирает окружение с зависимостями и Streamlit; основной пайплайн
запускается той же командой в контейнере:

```bash
docker compose run --rm debug python scripts/run_ingestion.py
```

## Отладочная форма

Нужна, чтобы загрузить один файл и посмотреть, что получилось **после OCR** и
чанкинга (без запуска полной индексации):

```bash
.venv/bin/streamlit run debug_app.py
# Windows: .\.venv\Scripts\streamlit.exe run debug_app.py
# Docker:  docker compose --profile debug up -d --build  → http://localhost:8501
```

Форма показывает:

- метаданные документа (doc_id, номер, дату, статус, версии);
- текст OCR по страницам (для сканов);
- все блоки (page/type/zone/source);
- все чанки с `raw_text` и `embed_text`;
- скачивание JSON и CSV.

Модуль разбора `src/debug/pipeline.py` не зависит от Streamlit и покрыт тестом.

## Тесты

```bash
pytest tests/ -v
```

- `test_pipeline.py` — артефакты, дубли пунктов, сноски, OCR, Qdrant и поиск;
- `test_metadata.py` — реквизиты документов и отсутствие циклов в графе версий;
- `test_debug_pipeline.py` — отладочный разбор DOCX.

Тесты с Qdrant требуют запущенного Qdrant и скачанную модель; data-only тесты
работают без модели.

## Типовые сценарии

### 1. Вопрос про суточные

Вход:

```text
Какой размер суточных при командировках по России для работников и обучающихся?
```

Выход поиска (пример):

```text
Регламент о командировании (проект 2025), п. 4.3 — «не более 100 руб.»
Приказ № 3699-1 от 18.08.2025, п. 17 — 100 руб. работникам, 50 руб. обучающимся
version_alerts: коллизия норм между проектом регламента и приказом
```

### 2. Вопрос вне корпуса

Вход:

```text
Как оформить загранпаспорт?
```

Выход:

```text
abstain_recommended = true
«В предоставленной базе регламентов нет релевантного документа по данной теме».
```

## Ограничения этапа

- Нет генерации ответа и UI: система возвращает чанки, метаданные, шаги-кандидаты
  и алерты; LLM/API — следующий этап.
- Нет реранкера: качество ограничено гибридным RRF; порог отказа калиброван на
  e5-large и корпус МФТИ.
- OCR.space — внешний сервис. Для новых сканов нужен ключ; если документы
  конфиденциальны, замените OCR на локальный (Tesseract/PaddleOCR) — код
  `OcrEngine` изолирован.
- Корпус небольшой (20 документов), поэтому ANN-оптимизации не используются:
  dense-поиск идёт с `exact=True`.

## Данные и приватность

Корпус предоставлен организаторами хакатона и считается обезличенным/открытым;
поэтому OCR-кэш и тексты документов лежат в репозитории. Если вы добавляете
реальные внутренние документы — не коммитьте `data/ocr_cache/`, `data/*.json`
и сканы, используйте отдельный приватный каталог через `DATA_DIR`.

Ключ `OCR_SPACE_API_KEY`, который был в старых коммитах, считается
скомпрометированным: смените его в личном кабинете OCR.space. В коде ключа
больше нет — только переменная окружения.
