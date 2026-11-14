#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/qdrant_storage logs

echo "[1/3] Starting Qdrant on port 6333..."
if ! curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then
  (cd data && nohup qdrant --uri http://0.0.0.0:6333 --disable-telemetry > "$ROOT_DIR/logs/qdrant.log" 2>&1 &)
  for _ in {1..30}; do
    curl -sf http://localhost:6333/healthz >/dev/null 2>&1 && break
    sleep 1
  done
fi

echo "[2/3] Starting local 4-bit Qwen/Qwen3.8-27B (RedHatAI/Qwen3.8-27B-INT4) vLLM server on port 8001..."
if ! curl -sf http://localhost:8001/v1/models >/dev/null 2>&1; then
  nohup vllm serve "${LOCAL_LLM_HF_QUANTIZED_REPO:-RedHatAI/Qwen3.8-27B-INT4}" \
    --host 0.0.0.0 \
    --port 8001 \
    --served-model-name Qwen/Qwen3.8-27B Qwen/Qwen3.8-27B-INT4 RedHatAI/Qwen3.8-27B-INT4 \
    --language-model-only \
    --gpu-memory-utilization 0.68 \
    --max-model-len 65536 \
    --max-num-seqs 8 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$ROOT_DIR/logs/vllm_qwen38.log" 2>&1 &
  echo "Waiting for Qwen/Qwen3.8-27B 4-bit server to become ready..."
  for _ in {1..180}; do
    curl -sf http://localhost:8001/v1/models >/dev/null 2>&1 && break
    sleep 2
  done
fi

echo "[3/3] Starting MIPT RAG V3 FastAPI server (BAAI/bge-m3 + PaddleOCR-VL-1.6 + Qwen3.8-27B 4-bit) on port 8000..."
exec uvicorn src.web.app:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8000}"
