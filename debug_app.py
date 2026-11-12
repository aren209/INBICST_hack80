from __future__ import annotations

"""Отладочная форма: загрузить документ -> увидеть OCR/блоки и чанки.

Запуск:
    streamlit run debug_app.py
"""

import csv
import io
import json
import os
from collections import defaultdict

import streamlit as st

from src.config import OCR_SPACE_API_KEY, QDRANT_URL
from src.debug.pipeline import process_uploaded_document

st.set_page_config(page_title="MIPT Navigator — debug", layout="wide")
st.title("Отладка разбора документа")
st.caption(
    "Загрузите PDF/DOCX/XLSX — форма покажет результат OCR по страницам, "
    "блоки, чанки и разделы. Основной пайплайн не меняется."
)

with st.sidebar:
    st.subheader("Окружение")
    st.write(f"OCR.space key: {'задан' if OCR_SPACE_API_KEY else 'НЕ задан'}")
    st.write(f"Qdrant: `{QDRANT_URL}`")
    st.caption(
        "Без OCR_SPACE_API_KEY новые сканы не распознаются, но для документов "
        "из корпуса работает кэш `data/ocr_cache`."
    )

uploaded = st.file_uploader(
    "Документ",
    type=["pdf", "docx", "xlsx"],
    accept_multiple_files=False,
)

col1, col2 = st.columns([1, 3])
with col1:
    run = st.button("Разобрать", type="primary", disabled=uploaded is None)
with col2:
    st.caption(
        "Метаданные берутся из `data/registry_overrides.yaml`, если имя файла "
        "совпадает; иначе — автоопределение по тексту."
    )

if run and uploaded is not None:
    with st.spinner("Читаю документ, при необходимости вызываю OCR…"):
        try:
            result = process_uploaded_document(uploaded.name, uploaded.getvalue())
            st.session_state["debug_result"] = result
        except Exception as exc:  # noqa: BLE001 - показываем пользователю
            st.error(f"Не удалось разобрать файл: {exc}")
            st.exception(exc)

result = st.session_state.get("debug_result")
if not result:
    st.info("Выберите файл и нажмите «Разобрать».")
    st.stop()

meta = result["metadata"]
stats = result["stats"]
st.subheader("1. Метаданные")
st.json(
    {
        "file_name": meta["file_name"],
        "doc_id": meta["doc_id"],
        "doc_title": meta["doc_title"],
        "doc_number": meta["doc_number"],
        "doc_date": meta["doc_date"],
        "status": meta["status"],
        "doc_type": meta["doc_type"],
        "supersedes": meta["supersedes"],
        "superseded_by": meta["superseded_by"],
        "version_note": meta["version_note"],
        "quality_warning": meta["quality_warning"],
    }
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Блоков", stats["blocks"])
c2.metric("Чанков", stats["chunks"])
c3.metric("Разделов", stats["sections"])
c4.metric("OCR-ошибок", len(stats["ocr_errors"]))
if stats["ocr_errors"]:
    with st.expander("Ошибки OCR"):
        for err in stats["ocr_errors"]:
            st.code(err)

# --- OCR по страницам -------------------------------------------------------
st.subheader("2. Текст после OCR")
ocr_pages: dict[int, list[str]] = defaultdict(list)
for block in result["blocks"]:
    if block.get("source") == "ocr":
        ocr_pages[block["page_no"]].append(block["text"])

if not ocr_pages:
    st.info("В этом документе OCR не потребовался (есть текстовый слой).")
else:
    for page_no in sorted(ocr_pages):
        with st.expander(f"Страница {page_no} (OCR)", expanded=(page_no == 1)):
            st.text("\n".join(ocr_pages[page_no]))

# --- Блоки -----------------------------------------------------------------
st.subheader("3. Блоки")
block_rows = [
    {
        "page": b["page_no"],
        "type": b["block_type"],
        "zone": b["zone"],
        "source": b["source"],
        "chars": len(b["text"]),
        "text": b["text"],
    }
    for b in result["blocks"]
]
st.dataframe(
    block_rows,
    use_container_width=True,
    hide_index=True,
    column_config={
        "text": st.column_config.TextColumn("Текст", width="large"),
    },
)

# --- Чанки -----------------------------------------------------------------
st.subheader("4. Чанки")
chunk_rows = [
    {
        "chunk_id": c["chunk_id"],
        "doc_id": c["doc_id"],
        "unit": c["unit_title"],
        "section": f'{c["section_no"]}. {c["section_title"]}',
        "clause": c["clause_no"],
        "zone": c["zone"],
        "chars": len(c["raw_text"]),
        "embed_chars": len(c["embed_text"]),
        "raw_text": c["raw_text"],
        "embed_text": c["embed_text"],
    }
    for c in result["chunks"]
]
st.dataframe(
    chunk_rows,
    use_container_width=True,
    hide_index=True,
    column_config={
        "raw_text": st.column_config.TextColumn("raw_text", width="large"),
        "embed_text": st.column_config.TextColumn("embed_text", width="large"),
    },
)

# --- Скачивание -------------------------------------------------------------
st.subheader("5. Скачать результат")
payload = json.dumps(result, ensure_ascii=False, indent=2)
st.download_button(
    "Скачать JSON (метаданные + OCR + чанки)",
    data=payload,
    file_name=f"{os.path.splitext(result['file_name'])[0]}_debug.json",
    mime="application/json",
)

buf = io.StringIO()
writer = csv.DictWriter(
    buf,
    fieldnames=[
        "chunk_id",
        "doc_id",
        "doc_title",
        "unit_title",
        "section_no",
        "section_title",
        "clause_no",
        "zone",
        "page_from",
        "page_to",
        "raw_text",
        "embed_text",
    ],
)
writer.writeheader()
for c in result["chunks"]:
    writer.writerow({k: c.get(k, "") for k in writer.fieldnames})
st.download_button(
    "Скачать чанки CSV",
    data="\ufeff" + buf.getvalue(),  # BOM для Excel
    file_name=f"{os.path.splitext(result['file_name'])[0]}_chunks.csv",
    mime="text/csv",
)
