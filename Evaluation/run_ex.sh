#!/usr/bin/env bash
set -euo pipefail

# 批量执行预测 SQL 和 gold SQL，并计算 EX。
# 直接修改下面的配置，不需要给本脚本传入命令行参数。

# ============================ 配置区 ============================
# MMQA 示例（当前启用）：计算 MMQA 的所有 SQL/SL 方法和模型组合。
DATASET_NAME="MMQA"
SQL_MODEL_NAME=""
SQL_METHOD="spider_agent_tc"
SL_MODEL_NAME=""
SL_METHOD=""
SNOWFLAKE_CREDENTIAL_PATH=""

# MMQA 指定实验示例：将下面的值复制到上方配置变量。
# DATASET_NAME="MMQA"
# SQL_MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
# SQL_METHOD="one_shot"
# SL_MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
# SL_METHOD="table_to_column"
# SNOWFLAKE_CREDENTIAL_PATH=""

# Spider2 指定实验示例：将下面的值复制到上方配置变量。
# DATASET_NAME="Spider2"
# SQL_MODEL_NAME="mistralai/Ministral-3-14B-Instruct-2512"
# SQL_METHOD="one_shot"
# SL_MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
# SL_METHOD="table_to_column"
# SNOWFLAKE_CREDENTIAL_PATH="snowflake_credential.json"

# 任意模型或方法变量设置为空字符串时，不对该项过滤。
# Spider2 凭据路径留空时，评估器读取 Snowflake 环境变量或项目默认凭据。
PYTHON_BIN="python"
# ===============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ARGS=(
  --dataset-name "${DATASET_NAME}"
  --output-path "results/evaluation/${DATASET_NAME}/ex.csv"
)

if [[ -n "${SQL_MODEL_NAME}" ]]; then
  ARGS+=(--sql-model-name "${SQL_MODEL_NAME}")
fi
if [[ -n "${SQL_METHOD}" ]]; then
  ARGS+=(--sql-method "${SQL_METHOD}")
fi
if [[ -n "${SL_MODEL_NAME}" ]]; then
  ARGS+=(--sl-model-name "${SL_MODEL_NAME}")
fi
if [[ -n "${SL_METHOD}" ]]; then
  ARGS+=(--sl-method "${SL_METHOD}")
fi
if [[ -n "${SNOWFLAKE_CREDENTIAL_PATH}" ]]; then
  ARGS+=(--snowflake-credential-path "${SNOWFLAKE_CREDENTIAL_PATH}")
fi

"${PYTHON_BIN}" -m Evaluation.evaluate_ex "${ARGS[@]}"
