FROM python:3.12-slim

LABEL org.opencontainers.image.title="Employee Regulations Navigator (RAG V3)" \
      org.opencontainers.image.description="On-prem RAG navigator for internal regulations & orders (PaddleOCR-VL-1.6 + BAAI/bge-m3 + 4-bit Qwen3.8-27B)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    DOCS_DIR=/app/documents \
    DATA_DIR=/app/data \
    CONFIG_DIR=/app/config

WORKDIR /app

# System dependencies for PDF/OCR (OpenCV, RapidOCR, pypdfium2), Word/Excel visual preview (LibreOffice), and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libreoffice-writer-nogui \
    libreoffice-calc-nogui \
    fonts-dejavu-core \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Pre-warm pymorphy3 Russian morphological analyzer and local RapidOCR v3 (PP-OCRv5 ESLAV fallback)
RUN python -c "\
import pymorphy3; \
morph = pymorphy3.MorphAnalyzer(); \
assert morph.parse('приказов')[0].normal_form == 'приказ'; \
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR; \
engine = RapidOCR(params={ \
    'Global.use_cls': False, \
    'Det.box_thresh': 0.3, \
    'Rec.ocr_version': OCRVersion.PPOCRV5, \
    'Rec.lang_type': LangRec.ESLAV, \
    'Rec.model_type': ModelType.MOBILE, \
}); \
print('Pre-warmed pymorphy3 and RapidOCR PP-OCRv5 ESLAV successfully!') \
"

# Copy ONLY application source code, general domain glossary, and startup scripts.
# Existing databases (data/), regulations (documents/), and corpus-specific overrides are NEVER baked into the image.
COPY config/glossary.yaml /app/config/glossary.yaml
COPY scripts/docker-entrypoint.sh /app/scripts/docker-entrypoint.sh
COPY src/ /app/src/

RUN chmod +x /app/scripts/docker-entrypoint.sh && \
    mkdir -p /app/data/uploads /app/data/ocr_cache /app/data/embed_cache /app/data/preview_cache /app/documents

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
