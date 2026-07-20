#!/usr/bin/env bash
set -euo pipefail

# 批量计算 LA、EM、Recall、Avg_Ratio、Avg_token 和 Avg_time。
# 直接修改下面的配置，不需要给本脚本传入命令行参数。

# ============================ 配置区 ============================
# MMQA 示例（当前启用）：计算 MMQA 的所有方法和模型。
DATASET_NAME="BIRD"
MODEL_NAME=""
METHOD=""
DATABASE_MODEL_NAME=""

# MMQA 指定实验示例：将下面的值复制到上方配置变量。
# DATASET_NAME="MMQA"
# MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
# METHOD="table_to_column"
# DATABASE_MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"

# Spider2 指定实验示例：将下面的值复制到上方配置变量。
# DATASET_NAME="Spider2"
# MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
# METHOD="table_to_column"
# DATABASE_MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"

# 任意模型或方法变量设置为空字符串时，不对该项过滤。
PYTHON_BIN="python"
# ===============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ARGS=(
  --dataset-name "${DATASET_NAME}"
  --output-path "results/evaluation/${DATASET_NAME}/metrics.csv"
)

if [[ -n "${MODEL_NAME}" ]]; then
  ARGS+=(--model-name "${MODEL_NAME}")
fi
if [[ -n "${METHOD}" ]]; then
  ARGS+=(--method "${METHOD}")
fi
if [[ -n "${DATABASE_MODEL_NAME}" ]]; then
  ARGS+=(--database-model-name "${DATABASE_MODEL_NAME}")
fi

"${PYTHON_BIN}" -m Evaluation.evaluate_metrics "${ARGS[@]}"
