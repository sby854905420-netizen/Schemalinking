#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

PYTHONDONTWRITEBYTECODE=1 python -m Run.sql_generator \
  --dataset-name BIRD \
  --answer-llm-name mistralai/Ministral-3-14B-Instruct-2512 \
  --provider transformers \
  --schema-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --schema-method few_shot \
  --schema-task table2column \
  --max-input-length 110000 \
  --max-generation-num 4096 \
  --dataset-path Data/BIRD/gold_sl.json \
  --db-info-path Data/BIRD/db_info.json \
  --sql-dialect "Use SQLite SQL for BIRD. Use only SQLite-compatible syntax and functions. Use table and column names exactly as shown in the schema excerpt. Quote identifiers with spaces or special characters using double quotes or backticks. Do not use Snowflake-only features such as QUALIFY, ILIKE, TRY_CAST, DATEADD, DATEDIFF, TO_DATE, TRUE/FALSE boolean literals, :: casts, or warehouse-style DATABASE.SCHEMA.TABLE qualification." \
  --include-key-columns
