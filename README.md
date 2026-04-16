# Schemalinking

本仓库实现了一个两阶段的 schema linking 流程：

1. 数据库检索：先判断问题最相关的数据库。
2. Schema Linking：再在目标数据库中定位相关表或相关列。

当前代码里实际可用的核心入口共有 5 个：

| 阶段 | 脚本 | 作用 |
| --- | --- | --- |
| 离线构建 | [`Rag_pipline/build_index.py`](Rag_pipline/build_index.py) | 从 `db_info.json` 构建列级 Qdrant 向量索引 |
| 数据库检索 | [`Run/Baseline_Database_Retrival.py`](Run/Baseline_Database_Retrival.py) | 基线数据库检索，直接把所有数据库摘要拼进 prompt |
| 数据库检索 | [`Run/Global_Coarse_Retrieval.py`](Run/Global_Coarse_Retrieval.py) | 检索增强版数据库检索，先召回高相关列再做数据库重排 |
| Schema Linking | [`Run/Baseline_Schema_Linking.py`](Run/Baseline_Schema_Linking.py) | 已知目标数据库后，直接做列级 Schema Linking |
| Schema Linking | [`Run/table2column.py`](Run/table2column.py) | 先预测相关表，再在候选表中预测相关列 |

说明：

- 仓库中的文件名保留了 `Retrival` 这一拼写，运行时请按实际文件名输入。
- `Global_Coarse_Retrieval.py` 和 `table2column.py` 都依赖本地 Qdrant 列索引，因此通常需要先运行一次 `build_index.py`。

## 1. 环境准备

推荐 Python 版本：`3.11.x`

安装依赖：

```bash
pip install -r requirements.txt
```

默认配置见 [`config.py`](config.py)，包括：

- 默认数据集：`DATASET_NAME = "MMQA"`
- 默认 embedding 模型：`Qwen/Qwen3-Embedding-0.6B`
- 默认回答模型：`mistralai/Ministral-3-8B-Instruct-2512`
- 默认 provider：`transformers`

LLM provider 说明：

- `transformers`：本地或 Hugging Face 模型推理，通常需要 GPU。
- `openai`：脚本直接读取环境变量 `OPENAI_API_KEY`，不会自动加载 `.env`。
- `ollama`：需要本机已经启动 Ollama，并提前拉取对应模型。

例如使用 OpenAI provider：

```bash
export OPENAI_API_KEY=your_api_key_here
python -m Run.Baseline_Schema_Linking \
  --provider openai \
  --answer-llm-name gpt-4.1-mini
```

## 2. 数据目录约定

脚本默认按 `Data/<dataset_name>/` 的目录约定读取数据。不同脚本依赖的文件不完全相同。

### 2.1 常用文件

| 路径 | 被哪些脚本使用 | 说明 |
| --- | --- | --- |
| `Data/<dataset>/db_info.json` | `build_index.py`、`Global_Coarse_Retrieval.py`、`Baseline_Schema_Linking.py`、`table2column.py` | 数据库 schema、列描述、样例值等统一来源 |
| `Data/<dataset>/Database_schemas_summary.json` | `Baseline_Database_Retrival.py` | 基线数据库检索使用的数据库摘要 |
| `Data/<dataset>/gold_sl.json` | `Baseline_Database_Retrival.py`、`Global_Coarse_Retrieval.py` | 数据库检索阶段默认输入文件 |
| `Data/<dataset>/qdrant_column_index/` | `Global_Coarse_Retrieval.py`、`table2column.py` | 本地 Qdrant 列索引目录，通常由 `build_index.py` 生成 |
| `Data/<dataset>/documents/` | 所有会读取 hint 的脚本 | 仅对 `Spider2` 特别重要，`external_knowledge` 会被当作文档文件名去这里读取 |

### 2.2 当前仓库里的实际情况

- `MMQA`、`BIRD`、`Spider2` 已提供 `db_info.json` 和 `gold_sl.json`。
- `Baseline_Database_Retrival.py` 当前代码固定读取 `Data/<dataset>/gold_sl.json`。
- 因此 `MMQA`、`BIRD`、`Spider2` 这三个数据集都可以直接作为基线数据库检索的输入。

### 2.3 输入字段约定

根据当前脚本实现，几类输入文件至少需要包含以下字段：

| 脚本 | 最少字段 |
| --- | --- |
| `Baseline_Database_Retrival.py` | `id`、`db_id`、`question` |
| `Global_Coarse_Retrieval.py` | `id`、`db_id`、`question` |
| `Baseline_Schema_Linking.py` | `question`、`predict_db_id`，以及 `id` 或 `instance_id` |
| `table2column.py` | `question`、`predict_db_id`、`id` |

可选字段：

- `external_knowledge`：会作为 hint 注入 prompt。
- 对 `Spider2` 而言，这个字段通常不是 hint 文本本身，而是 `Data/Spider2/documents/` 下的文档文件名。

## 3. 推荐运行流程

### 3.1 先构建列级向量索引

`Global_Coarse_Retrieval.py` 和 `table2column.py` 依赖本地 Qdrant 列索引。首次运行前，建议先执行：

```bash
python -m Rag_pipline.build_index \
  --db-info-path Data/MMQA/db_info.json \
  --recreate
```

默认输出目录为：

```text
Data/<dataset>/qdrant_column_index/
```

### 3.2 基线流程

先做基线数据库检索，再直接做列级 Schema Linking：

```bash
python -m Run.Baseline_Database_Retrival \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers

python -m Run.Baseline_Schema_Linking \
  --method few_shot \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

### 3.3 检索增强流程

先做全局粗检索，再运行两阶段 table-to-column Schema Linking：

```bash
python -m Run.Global_Coarse_Retrieval \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers \
  --candidate-db-top-k 3

python -m Run.table2column \
  --method few_shot \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

### 3.4 手动指定上一步结果

`Baseline_Schema_Linking.py` 和 `table2column.py` 都支持显式传入数据库检索结果文件：

```bash
python -m Run.Baseline_Schema_Linking \
  --method few_shot \
  --input-path Logs/mistralai/Ministral-3-8B-Instruct-2512/Database_Retrival/iterative_database_retrival_MMQA_20260416_120000.json

python -m Run.table2column \
  --method few_shot \
  --input-path Logs/mistralai/Ministral-3-8B-Instruct-2512/Database_Retrival/iterative_database_retrival_MMQA_20260416_120000.json
```

如果不传 `--input-path`，这两个脚本会自动在 `Logs/<answer_llm_name>/Database_Retrival/` 下寻找该数据集最近一次的检索结果。

## 4. 输出文件约定

每个主脚本都会生成一个 `.json` 结果文件，并额外生成同名 `.log` 日志文件。

| 脚本 | 默认输出位置 |
| --- | --- |
| `Baseline_Database_Retrival.py` | `Logs/<answer_llm_name>/Database_Retrival/baseline_database_retrival_<dataset>_<timestamp>.json` |
| `Global_Coarse_Retrieval.py` | `Logs/<answer_llm_name>/Database_Retrival/iterative_database_retrival_<dataset>_<timestamp>.json` |
| `Baseline_Schema_Linking.py` | `Logs/<answer_llm_name>/<method>_baseline_schema_linking_<dataset>_<timestamp>.json` |
| `table2column.py` | `Logs/<answer_llm_name>/<method>_table2column_<dataset>_<timestamp>.json` |

说明：

- 这里的 `<answer_llm_name>` 会按模型名原样展开。
- 如果模型名里包含 `/`，例如 `mistralai/Ministral-3-8B-Instruct-2512`，那么日志目录会变成多级路径。

## 5. 各脚本参数

### 5.1 `Rag_pipline/build_index.py`

用途：从 `db_info.json` 构建列级本地 Qdrant 索引。

常用示例：

```bash
python -m Rag_pipline.build_index \
  --db-info-path Data/MMQA/db_info.json \
  --collection-name MMQA \
  --qdrant-path Data/MMQA/qdrant_column_index \
  --device cuda \
  --recreate
```

参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--db-info-path` | 输入 `db_info.json` 路径 | `Data/<DATASET_NAME>/db_info.json` |
| `--collection-name` | Qdrant collection 名称 | 默认使用数据集名称 |
| `--qdrant-path` | 本地 Qdrant 存储目录 | `<db_info_path.parent>/qdrant_column_index` |
| `--model-name` | embedding 模型名 | `EMBEDDING_MODEL_NAME` |
| `--cache-dir` | embedding 模型缓存目录 | `Llm/cache` |
| `--batch-size` | embedding 推理 batch size | `32` |
| `--upsert-batch-size` | 每轮标准化、编码并写入的列数 | `128` |
| `--device` | embedding 设备，如 `cpu` / `cuda` | 自动判断 |
| `--recreate` | 若 collection 已存在则重建 | 关闭 |

### 5.2 `Run/Baseline_Database_Retrival.py`

用途：把所有数据库摘要拼成一个 prompt，让 LLM 直接输出 `relevant_database`。

常用示例：

```bash
python -m Run.Baseline_Database_Retrival \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset-name` | 数据集名称，对应 `Data/<dataset>/` | `DATASET_NAME` |
| `--answer-llm-name` | 用于数据库检索的 LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | LLM provider，支持 `transformers` / `openai` / `ollama` | `PROVIDER` |
| `--max-input-length` | 允许的最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--database-schema-path` | 数据库摘要文件路径 | `Data/<dataset>/Database_schemas_summary.json` |

当前限制：

- 该脚本没有 `--input-path` 参数。
- 输入数据固定读取 `Data/<dataset>/gold_sl.json`。
- prompt 模板固定为 `Templates/zero_shot/find_relevant_database_baseline.txt`。

### 5.3 `Run/Global_Coarse_Retrieval.py`

用途：先做列级向量召回，再按数据库聚合、裁剪和二分类重排，输出 `predict_db_id`。

常用示例：

```bash
python -m Run.Global_Coarse_Retrieval \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers \
  --candidate-db-top-k 3 \
  --enable-progress-log
```

参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | 用于数据库重排的 LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | LLM provider | `PROVIDER` |
| `--input-path` | 输入样本文件路径 | `Data/<dataset>/gold_sl.json` |
| `--output-path` | 结果输出路径 | 自动生成 |
| `--max-input-length` | 允许的最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--candidate-db-top-k` | 第一轮重排后保留的候选数据库数量 | `CANDIDATE_DB_TOP_K` |
| `--enable-progress-log` | 输出逐样本日志 | 关闭 |

补充说明：

- 当前代码里已经没有 `--hrc-top-p` 参数。
- prompt 模板固定为 `Templates/zero_shot/binary_classification_database.txt`。
- embedding 查询模型来自 [`config.py`](config.py) 中的 `EMBEDDING_MODEL_NAME`。

### 5.4 `Run/Baseline_Schema_Linking.py`

用途：在已知 `predict_db_id` 的前提下，对整个数据库 schema 直接做列级 Schema Linking。

常用示例：

```bash
python -m Run.Baseline_Schema_Linking \
  --method few_shot \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--method` | prompt 方式，支持 `zero_shot` / `few_shot` | `few_shot` |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | Schema Linking 使用的 LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | LLM provider | `PROVIDER` |
| `--max-input-length` | 允许的最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--input-path` | 数据库检索结果路径 | 自动寻找最新结果 |
| `--logs-dir` | 日志根目录 | `PROJECT_ROOT/Logs` |
| `--db-info-path` | `db_info.json` 路径 | `Data/<dataset>/db_info.json` |
| `--output-path` | 输出文件路径 | 自动生成 |

补充说明：

- 输入文件中必须包含 `predict_db_id`。
- prompt 模板为 `Templates/<method>/baseline_schema_linking.txt`。
- 当前实现不再依赖 `Table_schema_csv/`，而是直接从 `db_info.json` 渲染 schema 文本。

### 5.5 `Run/table2column.py`

用途：先预测相关表，再在候选表范围内预测相关列。

常用示例：

```bash
python -m Run.table2column \
  --method few_shot \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--method` | prompt 方式，支持 `zero_shot` / `few_shot` | `few_shot` |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | 表筛选和列筛选使用的 LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | LLM provider | `PROVIDER` |
| `--max-input-length` | 允许的最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--input-path` | 数据库检索结果路径 | 自动寻找最新结果 |
| `--logs-dir` | 日志根目录 | `PROJECT_ROOT/Logs` |
| `--db-info-path` | `db_info.json` 路径 | `Data/<dataset>/db_info.json` |
| `--qdrant-path` | 本地 Qdrant 索引目录 | `Data/<dataset>/qdrant_column_index` |
| `--output-path` | 输出文件路径 | 自动生成 |

补充说明：

- 输入文件中必须包含 `predict_db_id`。
- 会读取 `qdrant_column_index/meta.json` 自动解析 collection 名称。
- 表级 prompt 使用 `Templates/<method>/extract_relevant_tables.txt`。
- 列级 prompt 使用 `Templates/<method>/extract_relevant_columns.txt`。

## 6. 常见注意事项

1. `build_index.py` 允许通过 `--model-name` 指定 embedding 模型，但 `Global_Coarse_Retrieval.py` 和 `table2column.py` 运行时使用的查询 embedding 模型来自 [`config.py`](config.py) 的 `EMBEDDING_MODEL_NAME`。如果两边模型不一致，可能出现向量维度不匹配或检索效果异常。
2. `Baseline_Schema_Linking.py` 与 `table2column.py` 默认都会回溯最近一次数据库检索结果，因此连续跑不同模型时，最好显式传入 `--input-path`，避免误读旧结果。
3. `Spider2` 的 hint 读取方式和其他数据集不同：`external_knowledge` 会被当作文档文件名，到 `Data/Spider2/documents/` 中读取正文。
4. 如果使用 `openai` provider，请先在当前 shell 中导出 `OPENAI_API_KEY`；仅在仓库里放 `.env` 文件并不会自动生效。
