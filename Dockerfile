FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal system dependencies required by OpenCV / RapidOCR / pypdfium2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Pre-warm pymorphy3 Russian dictionary and RapidOCR v3 (PP-OCRv5 ESLAV MOBILE, use_cls=False)
RUN python -c "\
import pymorphy3; \
morph = pymorphy3.MorphAnalyzer(); \
assert morph.parse('приказов')[0].normal_form == 'приказ'; \
from rapidocr import RapidOCR, EngineType, LangRec, ModelType, OCRVersion; \
engine = RapidOCR(params={ \
    'Global.use_cls': False, \
    'Det.box_thresh': 0.3, \
    'Rec.ocr_version': OCRVersion.PPOCRV5, \
    'Rec.lang_type': LangRec.ESLAV, \
    'Rec.model_type': ModelType.MOBILE, \
}); \
print('Pre-warmed pymorphy3 and RapidOCR PP-OCRv5 ESLAV successfully!') \
"

COPY config/ /app/config/
COPY src/ /app/src/

RUN mkdir -p /app/data/uploads /app/data/ocr_cache /app/data/embed_cache /app/доки

EXPOSE 8000

CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
