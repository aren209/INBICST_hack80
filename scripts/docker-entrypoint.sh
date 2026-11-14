#!/usr/bin/env bash
set -euo pipefail

# Ensure clean runtime directories exist (especially when fresh empty Docker volumes are mounted)
mkdir -p \
  "${DATA_DIR:-/app/data}/uploads" \
  "${DATA_DIR:-/app/data}/ocr_cache" \
  "${DATA_DIR:-/app/data}/embed_cache" \
  "${DATA_DIR:-/app/data}/preview_cache" \
  "${DOCS_DIR:-/app/documents}"

QDRANT_TARGET="${QDRANT_URL:-http://qdrant:6333}"
QDRANT_WAIT_SEC="${QDRANT_WAIT_TIMEOUT:-60}"

echo "[entrypoint] Waiting up to ${QDRANT_WAIT_SEC}s for Qdrant at ${QDRANT_TARGET}/healthz ..."
for ((i=1; i<=QDRANT_WAIT_SEC; i++)); do
  if curl -fsS "${QDRANT_TARGET%/}/healthz" >/dev/null 2>&1; then
    echo "[entrypoint] Qdrant is healthy."
    break
  fi
  if [[ "$i" -eq "$QDRANT_WAIT_SEC" ]]; then
    echo "[entrypoint] WARNING: Qdrant did not respond within ${QDRANT_WAIT_SEC}s; starting FastAPI anyway."
  fi
  sleep 1
done

# Optional auto-indexing of mounted documents directory (disabled by default for clean deployment)
if [[ "${AUTO_SYNC_ON_STARTUP:-false}" == "true" ]]; then
  echo "[entrypoint] AUTO_SYNC_ON_STARTUP=true -> running incremental sync of ${DOCS_DIR:-/app/documents}..."
  python -c "
from src.indexing.pipeline import get_pipeline
rep = get_pipeline().sync_folder(force=False)
print(f'[entrypoint] Sync complete: total={rep.total_files}, indexed={rep.indexed_files}, failed={rep.failed_files}')
" || echo "[entrypoint] WARNING: Startup sync encountered an error; continuing to start server."
fi

echo "[entrypoint] Launching application: $*"
exec "$@"
