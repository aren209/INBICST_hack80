"""Central configuration, constants, thresholds, paths, and compiled regexes for MIPT RAG V3."""

from __future__ import annotations

import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)


class Settings(BaseSettings):
    """Environment-driven application settings loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek Official API (Primary LLM for RAG Agent & Query Pipeline)
    DEEPSEEK_API_KEY: str = Field(default="")
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com")
    DEEPSEEK_MODEL: str = Field(default="deepseek-chat")

    # Polza AI API (OpenAI-compatible — Embeddings, OCR VLM, and fallback LLM)
    POLZA_API_KEY: str = Field(default="")
    POLZA_BASE_URL: str = Field(default="https://polza.ai/api/v1")
    POLZA_EMBEDDING_MODEL: str = Field(default="baai/bge-m3")
    POLZA_VLM_MODEL: str = Field(default="qwen/qwen3-vl-235b-a22b-instruct")
    POLZA_LLM_MODEL: str = Field(default="openai/gpt-4o-mini")

    # OCR Backend ("hybrid_vlm" or "local_rapidocr")
    OCR_BACKEND: str = Field(default="hybrid_vlm")

    # Qdrant Vector Store
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_COLLECTION: str = Field(default="mipt_normative_v3")

    # Embedding Parameters
    EMBEDDING_DIM: int = Field(default=1024)
    EMBEDDING_BATCH_SIZE: int = Field(default=32)

    # Directory Paths (relative to BASE_DIR if not absolute)
    DOCS_DIR_NAME: str = Field(default="доки", validation_alias="DOCS_DIR")
    DATA_DIR_NAME: str = Field(default="data", validation_alias="DATA_DIR")
    CONFIG_DIR_NAME: str = Field(default="config", validation_alias="CONFIG_DIR")

    # FastAPI Server
    APP_HOST: str = Field(default="0.0.0.0")
    APP_PORT: int = Field(default=8000)
    LOG_LEVEL: str = Field(default="INFO")

    @property
    def docs_dir(self) -> Path:
        p = Path(self.DOCS_DIR_NAME)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def data_dir(self) -> Path:
        p = Path(self.DATA_DIR_NAME)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def config_dir(self) -> Path:
        p = Path(self.CONFIG_DIR_NAME)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def llm_provider(self) -> str:
        return "deepseek" if self.DEEPSEEK_API_KEY.strip() else "polza"

    @property
    def llm_api_key(self) -> str:
        ds_key = self.DEEPSEEK_API_KEY.strip()
        return ds_key if ds_key else self.POLZA_API_KEY

    @property
    def llm_base_url(self) -> str:
        return self.DEEPSEEK_BASE_URL if self.DEEPSEEK_API_KEY.strip() else self.POLZA_BASE_URL

    @property
    def llm_model(self) -> str:
        return self.DEEPSEEK_MODEL if self.DEEPSEEK_API_KEY.strip() else self.POLZA_LLM_MODEL


settings = Settings()

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".doc", ".xlsx", ".xls"})

# ---------------------------------------------------------------------------
#Filesystem Paths
# ---------------------------------------------------------------------------
DOCS_DIR: Path = settings.docs_dir
DATA_DIR: Path = settings.data_dir
CONFIG_DIR: Path = settings.config_dir

UPLOADS_DIR: Path = DATA_DIR / "uploads"
OCR_CACHE_DIR: Path = DATA_DIR / "ocr_cache"
EMBED_CACHE_DIR: Path = DATA_DIR / "embed_cache"
QDRANT_STORAGE_DIR: Path = DATA_DIR / "qdrant_storage"

SECTIONS_STORE_PATH: Path = DATA_DIR / "sections_store.json"
DOCUMENT_REGISTRY_PATH: Path = DATA_DIR / "document_registry.json"
INDEXING_REPORT_PATH: Path = DATA_DIR / "indexing_report.json"
TABLES_DB_PATH: Path = DATA_DIR / "tables_store.db"

# Adaptive 3-Tier Structured Table Thresholds
LARGE_TABLE_MIN_ROWS: int = 25
LARGE_TABLE_MIN_CHARS: int = 5000

GLOSSARY_PATH: Path = CONFIG_DIR / "glossary.yaml"
REGISTRY_OVERRIDES_PATH: Path = CONFIG_DIR / "registry_overrides.yaml"


def ensure_Runtime_dirs() -> None:
    """Ensure all required data and cache directories exist."""
    for directory in (DATA_DIR, UPLOADS_DIR, OCR_CACHE_DIR, EMBED_CACHE_DIR, CONFIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


ensure_Runtime_dirs()

# ---------------------------------------------------------------------------
# Module 1: Ingestion & Hybrid OCR Thresholds
# ---------------------------------------------------------------------------
OCR_SMALL_NATIVE_TEXT_CHARS: int = 600
OCR_MIN_IMAGE_COVERAGE: float = 0.55
OCR_RENDER_DPI: int = 300
MAX_OCR_IMAGE_SIDE: int = 3509
VLM_MAX_TOKENS: int = 2500
VLM_TEMPERATURE: float = 0.0
VLM_TIMEOUT_SEC: float = 60.0

VLM_SYSTEM_PROMPT: str = (
    "Ты — высокоточный OCR-движок для нормативных документов МФТИ. "
    "Извлеки весь текст со страницы без сокращений и пересказа. "
    "Сохраняй точную нумерацию разделов и пунктов (например, 1.1., 2.3., 6.1.), "
    "штампы регистрации ЭДО (например, «Зарегистрировано в ЭДО МФТИ от ... № ...») "
    "и форматируй таблицы в чистый Markdown (| Колонка 1 | Колонка 2 |). "
    "КРИТИЧЕСКОЕ ПРАВИЛО: Никогда не повторяй символы подчеркивания `_`, точки `.` "
    "или пробелы для пустых линий и граф подписи в бланках — заменяй любое пустое "
    "поле для заполнения на короткий маркер `[___]`."
)

# ---------------------------------------------------------------------------
# Module 2: V3 7-Stage Hierarchical Chunker Constants
# ---------------------------------------------------------------------------
MAX_CLAUSE_CHARS: int = 880
MAX_TABLE_FORM_CHARS: int = 560
MIN_TABLE_FORM_CHARS: int = 380
MIN_CLAUSE_CHARS: int = 160
TABLE_SLICE_ROWS: int = 8
MAX_PARENT_SECTION_CHARS: int = 12_000
CONTINUATION_HEADER_PREVIEW_CHARS: int = 120

# Stage 6: 12% Bilateral Overlap Bounds (word-boundary aligned)
CHUNK_OVERLAP_RATIO: float = 0.12
PARA_OVERLAP_MIN_CHARS: int = 35
PARA_OVERLAP_MAX_CHARS: int = 95
TABLE_OVERLAP_MIN_CHARS: int = 24
TABLE_OVERLAP_MAX_CHARS: int = 48

# Non-answer zones excluded from top-k normative slots
NON_ANSWER_ZONES: set[str] = {"approval", "meta"}

# Numeric bounds for Clause vs Date disambiguation
SINGLE_LEVEL_CLAUSE_MAX: int = 300
MULTI_LEVEL_FIRST_MAX: int = 120
MULTI_LEVEL_PART_MAX: int = 200
MAX_SECTION_HEADER_CHARS: int = 180

# ---------------------------------------------------------------------------
# Module 4: Indexing, Qdrant & Retrieval Constants
# ---------------------------------------------------------------------------
DENSE_VECTOR_NAME: str = "dense"
SPARSE_VECTOR_NAME: str = "bm25"
EMBEDDING_DIM: int = settings.EMBEDDING_DIM
EMBEDDING_BATCH_SIZE: int = settings.EMBEDDING_BATCH_SIZE

# Out-of-scope detection thresholds
OUT_OF_SCOPE_DENSE_THRESHOLD: float = 0.45
OUT_OF_SCOPE_LOW_DENSE_THRESHOLD: float = 0.38
OUT_OF_SCOPE_MIN_LEXICAL_OVERLAP: float = 0.12
SUPERSEDED_SCORE_PENALTY: float = 0.82

STATUS_RU_MAP: dict[str, str] = {
    "active": "Действующий",
    "superseded": "Отменен / Утратил силу",
    "superseded_template_reference": "Справочный шаблон (бланк заменен, формулы/коды актуальны)",
    "draft": "Проект",
    "error": "Ошибка индексации",
}

# ---------------------------------------------------------------------------
# Compiled Regular Expressions for V3 Chunker & Parsers
# ---------------------------------------------------------------------------

# Stage 1: Approval zone trigger & normative reset triggers
APPROVAL_TRIGGER_RE = re.compile(
    r"^\s*(?:ЛИСТ\s+СОГЛАСОВАНИЯ|Лист\s+согласования)\b",
    re.IGNORECASE,
)

NORMATIVE_RESET_RE = re.compile(
    r"^\s*(?:Приложени[ея]\s*(?:№|N)?\s*\d*[а-яa-z]?|УТВЕРЖДЕН[ОЫА]?\b|"
    r"(?:РЕГЛАМЕНТ|ПОЛОЖЕНИЕ|ИНСТРУКЦИЯ|ПОРЯДОК)\b(?!\s*\))|КАРТОЧКА\s+ПРОЕКТА\b|"
    r"ТЕХНИЧЕСКИЙ\s+РЕГЛАМЕНТ\b(?!\s*\)))",
    re.IGNORECASE,
)

# Stage 2: Autonomous unit boundaries (u01_main -> u02 -> u03...)
UNIT_BOUNDARY_RE = re.compile(
    r"^\s*(?:"
    r"Приложени[ея]\s*(?:№|N)?\s*(?:\d+[а-яa-z]?|[а-яa-z])\b|"
    r"УТВЕРЖДЕН[ОЫА]?\s+(?:приказом|распоряжением)\b|"
    r"(?:ВНУТРЕННИЙ\s+|ТЕХНИЧЕСКИЙ\s+)?РЕГЛАМЕНТ\b(?!\s*\))|"
    r"ПОЛОЖЕНИЕ\s+о\b|"
    r"ИНСТРУКЦИЯ\s+(?:по|о)\b|"
    r"КАРТОЧКА\s+ПРОЕКТА\b"
    r")",
    re.IGNORECASE,
)

# Date prefix protection: matches DD.MM.YYYY or DD.MM.YY at start of string/token
DATE_PREFIX_RE = re.compile(
    r"^\s*(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.(?:19\d{2}|20\d{2}|\d{2})(?:\s*г\.?|\b)"
)

# Any inline date DD.MM.YYYY
DATE_INLINE_RE = re.compile(
    r"\b(0?[1-9]|[12]\d|3[01])\.(0?[1-9]|1[0-2])\.(19\d{2}|20\d{2})\b"
)

# Clause number regex (single-level "1." or multi-level "1.1.", "2.3.1")
CLAUSE_NUMBER_RE = re.compile(
    r"^\s*(?:п\.\s*|пункт\s+)?(\d+(?:\.\d+)*)\.?(?:\s+|$)(.*)",
    re.IGNORECASE | re.DOTALL,
)

# Standalone detached clause number on its own line in PDFs (e.g. "1.1." or "2.3.")
DETACHED_CLAUSE_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\.?\s*$")

# Sub-item markers (Level 5: accumulated inside parent clause)
SUBITEM_RE = re.compile(
    r"^\s*(?:[а-яa-z]\)|\d+\)|[-–—•▪◦*])\s+",
    re.IGNORECASE,
)

# Page footer / header noise
PAGE_FOOTER_RE = re.compile(
    r"^\s*(?:Стр\.?\s*\d+\s*(?:из|/)\s*\d+|Страница\s+\d+\s+из\s+\d+|-\s*\d+\s*-|\d{1,3})\s*$",
    re.IGNORECASE,
)

# Hyphenated word wrap across PDF lines
HYPHEN_WRAP_RE = re.compile(r"([а-яёА-ЯЁ]{2,})-\s*\n\s*([а-яё]{2,})")

# Footnote superscript glued to order number (e.g., "№ 8123" where 3 is a verified page footnote marker)
FOOTNOTE_ORDER_RE = re.compile(r"(№\s*\d{3})(\d)(?=\s|$|[,.;)])")

# Footnote definition line on the same page (bare single digit without dot/paren + uppercase start, or superscript)
FOOTNOTE_DEF_LINE_RE = re.compile(r"^\s*([1-9])\s+[А-ЯЁA-Z«\"(\[]")

# Section Title Starters (Level 3 Parent Section detection)
SECTION_TITLE_STARTERS: tuple[str, ...] = (
    "общие положения",
    "общие требования",
    "общие правила",
    "порядок",
    "правила",
    "требования",
    "сроки",
    "ответственность",
    "финансирование",
    "подготовка",
    "проведение",
    "термины",
    "определения",
    "сокращения",
    "обозначения",
    "заключительные положения",
    "организация",
    "контроль",
    "условия",
    "регламент",
    "структура",
    "формирование",
    "присвоение",
    "открытие",
    "ведение",
    "закрытие",
    "бронирование",
    "поселение",
    "выселение",
    "командирование",
    "возмещение",
    "оплата",
    "бюджетирование",
    "планирование",
    "согласование",
    "утверждение",
    "регистрация",
    "эксплуатация",
    "обслуживание",
    "доступ",
    "безопасность",
    "охрана труда",
    "прием",
    "подбор",
    "поиск",
    "адаптация",
    "взаимодействие",
    "права и обязанности",
    "функции",
    "задачи",
    "цели",
    "назначение",
    "область применения",
    "нормативные ссылки",
)

# Imperative Verbs Stop-list (distinguishes order clauses like "2. Утвердить..." from section titles)
IMPERATIVE_VERB_STOPWORDS: tuple[str, ...] = (
    "утвердить",
    "ввести",
    "признать",
    "установить",
    "возложить",
    "поручить",
    "отменить",
    "внести",
    "направить",
    "контрольному",
    "контроль",
    "приказываю",
    "обязать",
    "разрешить",
    "довести",
    "опубликовать",
    "разместить",
    "обеспечить",
    "производить",
    "осуществлять",
    "назначить",
    "создать",
    "считать",
    "распространить",
    "применять",
    "исполнять",
    "руководствоваться",
)
