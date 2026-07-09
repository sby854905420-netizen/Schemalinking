#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

PYTHONDONTWRITEBYTECODE=1 python -m Rag_pipline.build_index \
  --db-info-path Data/BIRD/db_info.json \
  --collection-name BIRD \
  --qdrant-path Data/BIRD/qdrant_column_index \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --batch-size 32 \
  --upsert-batch-size 128 \
  --device cuda
