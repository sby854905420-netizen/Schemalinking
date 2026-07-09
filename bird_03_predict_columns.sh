#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

PYTHONDONTWRITEBYTECODE=1 python -m Run.table2column \
  --method few_shot \
  --dataset-name BIRD \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers \
  --max-input-length 110000 \
  --max-generation-num 2048 \
  --db-info-path Data/BIRD/db_info.json \
  --qdrant-path Data/BIRD/qdrant_column_index
