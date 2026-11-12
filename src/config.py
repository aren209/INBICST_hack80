from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))
)
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "данные"))
ARTIFACTS_DATA_DIR = Path(os.getenv("ARTIFACTS_DATA_DIR", PROJECT_ROOT / "data"))
OCR_CACHE_DIR = ARTIFACTS_DATA_DIR / "ocr_cache"
OCR_VERIFIED_DIR = ARTIFACTS_DATA_DIR / "ocr_verified"
REGISTRY_OVERRIDES_PATH = ARTIFACTS_DATA_DIR / "registry_overrides.yaml"
GLOSSARY_PATH = ARTIFACTS_DATA_DIR / "glossary.json"
SECTIONS_STORE_PATH = ARTIFACTS_DATA_DIR / "sections_store.json"
CHUNKS_DUMP_PATH = ARTIFACTS_DATA_DIR / "chunks_dump.json"
MODELS_CACHE_DIR = Path(os.getenv("MODELS_CACHE_DIR", PROJECT_ROOT / "models_cache"))
DEBUG_TMP_DIR = Path(os.getenv("DEBUG_TMP_DIR", PROJECT_ROOT / ".debug_tmp"))

# OCR.space API Configuration
OCR_SPACE_ENDPOINT = os.getenv(
    "OCR_SPACE_ENDPOINT", "https://api.ocr.space/parse/image"
)
# Ключ задаётся ТОЛЬКО через окружение (.env / docker compose / CI).
# Публичный fallback-ключ намеренно удалён: он попадал в открытый репозиторий.
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")
OCR_MAX_IMAGE_BYTES = 850 * 1024  # < 1 MB Free Plan limit

# Если у страницы есть картинка и текстового слоя меньше этого порога,
# считаем её сканом и отправляем на OCR (иначе теряется шапка/часть страницы).
OCR_SMALL_NATIVE_TEXT_CHARS = int(os.getenv("OCR_SMALL_NATIVE_TEXT_CHARS", "600"))

# Local Embedding Model (FastEmbed ONNX multilingual).
# e5-large — основная модель корпуса (1024d). На слабом CPU первичная
# индексация может занимать десятки минут; альтернатива:
# DENSE_MODEL_NAME=sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768d).
DENSE_MODEL_NAME = os.getenv(
    "DENSE_MODEL_NAME", "intfloat/multilingual-e5-large"
)
# Маленький batch уменьшает паддинг коротких чанков до 512 токенов.
DENSE_BATCH_SIZE = int(os.getenv("DENSE_BATCH_SIZE", "8"))

# Qdrant Configuration (Docker container on 6333)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "mipt_regulations")

# Chunking hyperparameters (V3 Parent-Child)
MAX_CLAUSE_CHARS = 1200
TABLE_SLICE_ROWS = 12

# Retrieval hyperparameters
# 0.815 откалиброван на e5-large: in-domain запросы в корпусе дают >= 0.83,
# внекорпусные (загранпаспорт, билет на поезд) — 0.79–0.80.
DENSE_ABSTAIN_THRESHOLD = float(os.getenv("DENSE_ABSTAIN_THRESHOLD", "0.815"))
# Не отдаём служебные «паспорта» документов и листы согласования в ответ,
# но продолжаем использовать их для версионных/качественных предупреждений.
NON_ANSWER_ZONES = {"approval", "meta"}
PARENT_SECTION_MAX_CHARS = int(os.getenv("PARENT_SECTION_MAX_CHARS", "12000"))
