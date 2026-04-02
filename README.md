# Schemalinking

本项目的这四个脚本可分为两个阶段：

1. 数据库检索：判断问题最相关的数据库
2. Schema Linking：在目标数据库中找出相关表或相关列

对应关系如下：

- [`Run/Baseline_Database_Retrival.py`]             (./Schemalinking/Run/Baseline_Database_Retrival.py)：基线数据库检索
- [`Run/Global_Coarse_Retrieval.py`]                (./Schemalinking/Run/Global_Coarse_Retrieval.py)：检索增强版数据库检索
- [`Run/Baseline_Schema_Linking.py`]                (./Schemalinking/Run/Baseline_Schema_Linking.py)：直接列级 Schema Linking
- [`Run/table2column.py`]                           (./Schemalinking/Run/table2column.py)：先表后列的两阶段 Schema Linking

## 1. 脚本功能

[`Run/Baseline_Database_Retrival.py`]
将所有数据库摘要拼接到 prompt 中，让 LLM 直接输出最相关数据库 `predict_db_id`。

[`Run/Global_Coarse_Retrieval.py`]
先通过向量索引召回高相关列，再按数据库聚合、筛选、重排，最终输出目标数据库 `predict_db_id`。

[`Run/Baseline_Schema_Linking.py`]
在已知目标数据库后，对该数据库完整 schema 做列级 Schema Linking，输出相关列。

[`Run/table2column.py`]
先预测相关表，再仅在这些表中预测相关列。

## 2. 默认输入数据

默认数据集目录为 `Data/MMQA/`，主要依赖以下文件或目录：

- `Data/MMQA/preprocessed_data.json`
- `Data/MMQA/Database_schemas_summary.json`
- `Data/MMQA/Table_schema_csv/`
- `Data/MMQA/qdrant_column_index/`
- `Data/MMQA/Column_level_schema/`

默认配置位于 [`config.py`](/home/xubeiyu/projects/Schemalinking/config.py)。

## 3. 推荐运行流程

基线流程：

```bash
python -m Run.Baseline_Database_Retrival
python -m Run.Baseline_Schema_Linking --method few_shot
```

检索增强流程：

```bash
python Run.Global_Coarse_Retrieval
python Run.table2column --method few_shot
```

## 4. 各脚本用法

运行基线数据库检索：

```bash
python -m Run.Baseline_Database_Retrival \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

输出文件：
`Logs/<model>/Database_Retrival/baseline_database_retrival_<dataset>_<timestamp>.json`

运行全局粗检索：

```bash
python -m Run.Global_Coarse_Retrieval \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers \
  --hrc-top-p 0.02 \
  --candidate-db-top-k 3
```

输出文件：
`Logs/<model>/Database_Retrival/iterative_database_retrival_<dataset>_<timestamp>.json`

运行基线 Schema Linking：

```bash
python -m Run.Baseline_Schema_Linking \
  --method few_shot \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

如需手动指定数据库检索结果文件：

```bash
python -m Run.Baseline_Schema_Linking.py \
  --input-path Logs/<model>/Database_Retrival/iterative_database_retrival_MMQA_<timestamp>.json
```

输出文件：
`Logs/<model>/<method>_baseline_schema_linking_<dataset>_<timestamp>.json`

运行 table-to-column Schema Linking：

```bash
python -m Run.table2column \
  --method few_shot \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

如需手动指定输入文件：

```bash
python -m Run.table2column \
  --input-path Logs/<model>/Database_Retrival/iterative_database_retrival_MMQA_<timestamp>.json
```

输出文件：
`Logs/<model>/<method>_table2column_<dataset>_<timestamp>.json`

## 5. 参数说明表

### `Baseline_Database_Retrival.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--dataset-name` | 数据集名称，对应 `Data/<dataset-name>/` | `DATASET_NAME` |
| `--answer-llm-name` | 用于数据库检索的 LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | LLM 后端类型 | `PROVIDER` |
| `--max-input-length` | 模型最大输入长度 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成长度 | `MAX_GENERATEION_NUM` |
| `--database-schema-path` | 数据库摘要文件路径 | `Data/<dataset-name>/Database_schemas_summary.json` |

### `Global_Coarse_Retrieval.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | 用于候选库重排的 LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | LLM 后端类型 | `PROVIDER` |
| `--max-input-length` | 模型最大输入长度 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成长度 | `MAX_GENERATEION_NUM` |
| `--hrc-top-p` | 高相关列召回比例 | `HRC_TOP_P` |
| `--candidate-db-top-k` | 第一轮保留的候选数据库数 | `CANDIDATE_DB_TOP_K` |

### `Baseline_Schema_Linking.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--method` | Prompt 方式，支持 `zero_shot`、`few_shot` | `few_shot` |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | 用于 schema linking 的 LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | LLM 后端类型 | `PROVIDER` |
| `--max-input-length` | 模型最大输入长度 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成长度 | `MAX_GENERATEION_NUM` |
| `--input-path` | 上一步数据库检索结果文件路径 | 自动寻找最新结果 |
| `--logs-dir` | 日志根目录 | `PROJECT_ROOT/Logs` |
| `--table-schema-dir` | 数据库表结构 csv 目录 | `Data/<dataset-name>/Table_schema_csv` |
| `--output-path` | 输出文件路径 | 自动生成 |

### `table2column.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--method` | Prompt 方式，支持 `zero_shot`、`few_shot` | `few_shot` |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | 用于表筛选和列筛选的 LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | LLM 后端类型 | `PROVIDER` |
| `--max-input-length` | 模型最大输入长度 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成长度 | `MAX_GENERATEION_NUM` |
| `--input-path` | 上一步数据库检索结果文件路径 | 自动寻找最新结果 |
| `--logs-dir` | 日志根目录 | `PROJECT_ROOT/Logs` |
| `--table-schema-dir` | 数据库表结构 csv 目录 | `Data/<dataset-name>/Table_schema_csv` |
| `--output-path` | 输出文件路径 | 自动生成 |

## 6. 补充说明

- `Baseline_Schema_Linking.py` 和 `table2column.py` 的输入文件中必须包含 `predict_db_id`。
- 如果不传 `--input-path`，这两个脚本会自动从 `Logs/<model>/Database_Retrival/` 中查找最新结果文件。
- `Global_Coarse_Retrieval.py` 依赖本地向量索引、`qdrant-client`、`torch` 和 embedding 模型。
- 仓库中的文件名使用的是 `Retrival` 这一拼写，运行脚本时请按实际文件名输入。
