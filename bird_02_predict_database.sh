#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

PYTHONDONTWRITEBYTECODE=1 python -m Run.Global_Coarse_Retrieval \
  --dataset-name BIRD \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers \
  --max-input-length 110000 \
  --max-generation-num 2048 \
  --candidate-db-top-k 3 \
  --db-selection-mode rerank
