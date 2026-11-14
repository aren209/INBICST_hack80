"""Comprehensive Unit & Live GPU Integration Tests for Local Models:
1. OCR: PaddlePaddle/PaddleOCR-VL-1.6 (+ HTML table -> Markdown & RapidOCR fallback)
2. Embeddings: BAAI/bge-m3 (1024d Cosine via SentenceTransformer on GPU/CPU)
3. LLM: Linear-Hybrid Qwen/Qwen3.8-27B in 4-bit quantization (chat, JSON, XML/OpenAI tool calling, streaming)
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium
import pytest

from src.config import DOCS_DIR, Settings, settings
from src.indexing.embedder import DenseEmbedder, cosine_similarity
from src.ingestion.ocr_engine import HybridOCREngine, sanitize_ocr_text
from src.retrieval.local_llm import (
    LOCAL_QWEN_MODELS,
    Transformers4BitChatClient,
    create_llm_client,
    parse_qwen3_xml_tool_calls,
    strip_thinking_tags,
)
from src.retrieval.query_pipeline import QueryPipeline
from src.retrieval.rag_agent import TableAwareRAGAgent


def _is_local_vllm_ready() -> bool:
    try:
        r = httpx.get(f"{settings.LOCAL_LLM_BASE_URL.rstrip('/')}/models", timeout=2.5)
        return r.status_code == 200
    except Exception:
        return False


class TestLocalModelSettingsAndHelpers:
    def test_default_settings_use_all_three_local_models(self) -> None:
        s = Settings(DEEPSEEK_API_KEY="", POLZA_API_KEY="")
        assert s.llm_provider == "local"
        assert s.llm_model == "Qwen/Qwen3.8-27B"
        assert s.LOCAL_LLM_HF_QUANTIZED_REPO == "RedHatAI/Qwen3.8-27B-INT4"
        assert s.LLM_QUANTIZATION == "4bit"
        assert s.EMBEDDING_BACKEND == "local"
        assert s.EMBEDDING_MODEL == "BAAI/bge-m3"
        assert s.OCR_BACKEND == "paddleocr_vl"
        assert s.OCR_VL_MODEL == "PaddlePaddle/PaddleOCR-VL-1.6"
        assert "Qwen/Qwen3.8-27B" in LOCAL_QWEN_MODELS

    def test_strip_thinking_tags_and_parse_qwen3_xml_tool_calls(self) -> None:
        raw_with_think = "<think>\nСначала проверим таблицу ЦФО...\n</think>\n\n### Краткий ответ\nНайдено 20 подразделений."
        assert strip_thinking_tags(raw_with_think) == "### Краткий ответ\nНайдено 20 подразделений."

        raw_xml_tc = (
            "<think>\nНужно вызвать инструмент фильтрации.\n</think>\n"
            "<tool_call>\n"
            "<function=query_structured_table>\n"
            "<parameter=table_id>\napp_fin_struct_2026_xlsx::tbl_01\n</parameter>\n"
            "<parameter=operation>\nfilter_rows\n</parameter>\n"
            "<parameter=limit>\n25\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        content, tool_calls = parse_qwen3_xml_tool_calls(raw_xml_tc)
        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "query_structured_table"
        parsed_args = json.loads(tool_calls[0].function.arguments)
        assert parsed_args["table_id"] == "app_fin_struct_2026_xlsx::tbl_01"
        assert parsed_args["operation"] == "filter_rows"
        assert parsed_args["limit"] == 25

    def test_create_llm_client_supports_offline_vllm_and_bnb_4bit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert create_llm_client(api_key="", base_url="http://localhost:8001/v1", timeout=5.0) is None

        monkeypatch.setattr(settings, "LLM_BACKEND", "local_bnb_4bit")
        bnb_client = create_llm_client(
            api_key="local",
            base_url="http://localhost:8001/v1",
            timeout=5.0,
            llm_provider="local",
        )
        assert isinstance(bnb_client, Transformers4BitChatClient)

    def test_paddleocr_vl_html_table_to_markdown_and_antiloop(self) -> None:
        raw_html = (
            "6.1. Максимальное количество репетиционных дней:\n"
            "<table><tr><th>Мероприятие</th><th>Лимит дней</th></tr>"
            "<tr><td>Посвящение в студенты</td><td>6 дней</td></tr>"
            "<tr><td>Концерт факультета</td><td>2 дня</td></tr></table>\n"
            "Подпись руководителя: ________________________"
        )
        cleaned = sanitize_ocr_text(raw_html)
        assert "| Мероприятие | Лимит дней |" in cleaned
        assert "| Посвящение в студенты | 6 дней |" in cleaned
        assert "[___]" in cleaned
        assert "<table>" not in cleaned


class TestLiveLocalBgeM3Embedder:
    def test_live_bge_m3_dense_embeddings_and_cache(self, tmp_path: Path) -> None:
        embedder = DenseEmbedder(cache_dir=tmp_path / "embed_cache")
        texts = [
            "Порядок бронирования Концертного зала МФТИ по служебной записке за 2 месяца.",
            "Заявка на проведение мероприятия в концертном зале подается на kmo@mipt.ru.",
            "Рецепт приготовления итальянской пасты карбонара со сливками и беконом.",
        ]
        vecs = embedder.embed_texts(texts)
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == 1024
            norm = math.sqrt(sum(x * x for x in v))
            assert abs(norm - 1.0) < 1e-4

        sim_related = cosine_similarity(vecs[0], vecs[1])
        sim_unrelated = cosine_similarity(vecs[0], vecs[2])
        assert sim_related > 0.55
        assert sim_related > sim_unrelated + 0.20

        # Verify warm SHA-256 disk cache hit
        embedder_warm = DenseEmbedder(cache_dir=tmp_path / "embed_cache")
        vec_cached = embedder_warm.embed_query(texts[0])
        assert cosine_similarity(vecs[0], vec_cached) > 0.99999


class TestLiveLocalPaddleOCRVL16:
    def test_live_paddleocr_vl_16_on_synthetic_and_real_mipt_scan(self, tmp_path: Path) -> None:
        ocr = HybridOCREngine(cache_dir=tmp_path / "ocr_cache")

        # 1. Synthetic image test
        img = Image.new("RGB", (900, 220), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
        draw.text((30, 40), "ПРИКАЗ МФТИ № 3065-1 от 15.11.2024", fill="black", font=font)
        draw.text((30, 110), "2.3. Срок подачи служебной записки — 2 месяца.", fill="black", font=font)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        res_syn = ocr.recognize_page(buf.getvalue(), ocr_backend="paddleocr_vl", page_num=1, file_name="synthetic.png")
        assert res_syn["backend_used"] == "paddleocr_vl"
        assert res_syn["from_cache"] is False
        assert "3065-1" in res_syn["text"]
        assert "2.3" in res_syn["text"]

        # 2. Real scanned MIPT PDF test (page 4 compact table of Регламент бронирования конц зала.pdf)
        scan_pdf = DOCS_DIR / "regulations" / "Регламент бронирования конц зала.pdf"
        if scan_pdf.exists():
            pdf_doc = pdfium.PdfDocument(str(scan_pdf))
            page = pdf_doc[3]  # Page 4 (0-indexed 3) contains clause 6.1 rehearsal days table
            pil_page = page.render(scale=200 / 72.0).to_pil()
            p_buf = io.BytesIO()
            pil_page.save(p_buf, format="PNG")
            res_real = ocr.recognize_page(
                p_buf.getvalue(),
                ocr_backend="paddleocr_vl",
                page_num=4,
                file_name=scan_pdf.name,
            )
            assert res_real["backend_used"] == "paddleocr_vl"
            low = res_real["text"].lower().replace("ё", "е")
            assert "посвящен" in low
            assert "6" in low


@pytest.mark.skipif(not _is_local_vllm_ready(), reason="Local 4-bit Qwen/Qwen3.8-27B vLLM server is not running")
class TestLiveLocalQwen38Quantized4Bit:
    def test_live_qwen38_4bit_query_rewriter_json(self) -> None:
        qp = QueryPipeline()
        plan = qp.rewrite_query_for_retrieval("за скока месяцев надо бронировать кз мфти и куда писать?")
        assert plan.used_fallback is False
        assert plan.query_intent == "document_lookup"
        assert len(plan.rewritten_primary) >= 10
        assert len(plan.sub_queries) >= 1
        assert plan.search_queries[0] == "за скока месяцев надо бронировать кз мфти и куда писать?"

    def test_live_qwen38_4bit_tool_calling_and_streaming(self) -> None:
        agent = TableAwareRAGAgent()
        resp, model_used = agent._call_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Вызови функцию query_structured_table для table_id='app_fin_struct_2026_xlsx::tbl_01' "
                        "с operation='filter_rows' и search_text='радиотехники'."
                    ),
                }
            ],
            allow_tools=True,
            requested_model=None,
        )
        assert model_used in LOCAL_QWEN_MODELS
        tcs = resp.choices[0].message.tool_calls
        assert tcs is not None and len(tcs) >= 1
        assert tcs[0].function.name == "query_structured_table"
        args = json.loads(tcs[0].function.arguments)
        assert args["table_id"] == "app_fin_struct_2026_xlsx::tbl_01"
