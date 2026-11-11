from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path("/home/arz/hacatonmipt")
DATA_DIR = PROJECT_ROOT / "данные"
ARTIFACTS_DATA_DIR = PROJECT_ROOT / "data"
OCR_CACHE_DIR = ARTIFACTS_DATA_DIR / "ocr_cache"
OCR_VERIFIED_DIR = ARTIFACTS_DATA_DIR / "ocr_verified"
REGISTRY_OVERRIDES_PATH = ARTIFACTS_DATA_DIR / "registry_overrides.yaml"
GLOSSARY_PATH = ARTIFACTS_DATA_DIR / "glossary.json"
SECTIONS_STORE_PATH = ARTIFACTS_DATA_DIR / "sections_store.json"
CHUNKS_DUMP_PATH = ARTIFACTS_DATA_DIR / "chunks_dump.json"
MODELS_CACHE_DIR = PROJECT_ROOT / "models_cache"

# OCR.space API Configuration
OCR_SPACE_ENDPOINT = os.getenv(
    "OCR_SPACE_ENDPOINT", "https://api.ocr.space/parse/image"
)
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "K8149675126871588957")
OCR_MAX_IMAGE_BYTES = 850 * 1024  # < 1 MB Free Plan limit

# Local Embedding Model (FastEmbed ONNX multilingual)
# Defaults to intfloat/multilingual-e5-large (1024d)
DENSE_MODEL_NAME = os.getenv(
    "DENSE_MODEL_NAME", "intfloat/multilingual-e5-large"
)

# Qdrant Configuration (Docker container on 6333)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "mipt_regulations")

# Chunking hyperparameters (V3 Parent-Child)
MAX_CLAUSE_CHARS = 1200
TABLE_SLICE_ROWS = 12
