# MDB-link

本仓库实现了一个两阶段的 schema linking 流程：

1. 数据库检索：根据自然语言问题预测最相关的数据库 `predict_db_id`。
2. Schema Linking：在预测数据库中定位相关表或相关列。

当前代码中主要有 5 个可运行入口：

| 阶段 | 脚本 | 作用 |
| --- | --- | --- |
| 离线构建 | [`Rag_pipline/build_index.py`](Rag_pipline/build_index.py) | 从 `db_info.json` 构建列级 Qdrant 向量索引 |
| 数据库检索 | [`Run/Baseline_Database_Retrival.py`](Run/Baseline_Database_Retrival.py) | 基线数据库检索：把所有数据库摘要拼进 prompt，让 LLM 输出相关数据库 |
| 数据库检索 | [`Run/Global_Coarse_Retrieval.py`](Run/Global_Coarse_Retrieval.py) | 检索增强版数据库检索：列级向量召回、数据库剪枝、LLM 二分类重排 |
| Schema Linking | [`Run/Baseline_Schema_Linking.py`](Run/Baseline_Schema_Linking.py) | 已知 `predict_db_id` 后，对目标数据库直接做列级 schema linking |
| Schema Linking | [`Run/table2column.py`](Run/table2column.py) | 先预测相关表，再在候选表范围内预测相关列 |

注意：仓库中的文件名和输出目录保留了 `Retrival` 这个拼写，运行命令和结果路径都需要按实际名称使用。

## 1. 环境准备

推荐 Python 版本：`3.11.x`。

安装依赖：

```bash
pip install -r requirements.txt
```

默认配置见 [`config.py`](config.py)：

| 配置 | 默认值 |
| --- | --- |
| `DATASET_NAME` | `MMQA` |
| `EMBEDDING_MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` |
| `ANSWER_LLM_NAME` | `mistralai/Ministral-3-8B-Instruct-2512` |
| `PROVIDER` | `transformers` |
| `MAX_INPUT_LENGTH` | `110000` |
| `MAX_GENERATEION_NUM` | `2048` |
| `TOP_KD_CAP` | `512` |
| `CANDIDATE_DB_TOP_K` | `3` |

### Provider 兼容性

`LLM` 封装支持 `transformers` 和 `openai`，但各脚本实际依赖不同：

| 脚本 | 当前可用 provider | 原因 |
| --- | --- | --- |
| `Baseline_Database_Retrival.py` | `transformers` / `openai` | 只需要普通文本生成 |
| `Global_Coarse_Retrieval.py` | `transformers` | 需要 tokenizer 计数和下一 token logits 做 yes/no 重排 |
| `Baseline_Schema_Linking.py` | `transformers` | schema 渲染器依赖 `answer_llm.tokenizer` |
| `table2column.py` | `transformers` | 依赖 tokenizer 计数、schema 裁剪和本地 embedding 检索 |
| `sql_generator.py` | `transformers` / `openai` | OpenAI provider 会提供 tokenizer-like prompt 预算接口，并可直接调用 GPT-5 系列模型 |

使用 OpenAI provider 时，代码会优先读取环境变量 `OPENAI_API_KEY`，也可以读取 `OPENAI_CREDENTIAL_PATH` 或项目根目录下的 `gpt_credential.json`。凭据 JSON 支持 `api_key`、`openai_api_key`、`OPENAI_API_KEY`、`key` 字段。

```bash
export OPENAI_API_KEY=your_api_key_here
python -m Run.Baseline_Database_Retrival \
  --dataset-name MMQA \
  --provider openai \
  --answer-llm-name gpt-4.1-mini
```

SQL generator 使用 GPT-5 mini 的示例：

```bash
python -m Run.sql_generator \
  --dataset-name MMQA \
  --provider openai \
  --answer-llm-name gpt-5-mini-2025-08-07 \
  --credential-path gpt_credential.json \
  --schema-llm-name mistralai/Ministral-3-14B-Instruct-2512
```

## 2. 数据目录约定

脚本默认从 `Data/<dataset_name>/` 读取数据。当前仓库包含 `MMQA`、`BIRD`、`Spider2` 三个数据集目录。

| 路径 | 用途 |
| --- | --- |
| `Data/<dataset>/db_info.json` | 数据库 schema、列描述、样例值、主外键等统一来源 |
| `Data/<dataset>/Database_schemas_summary.json` | 基线数据库检索使用的数据库摘要 |
| `Data/<dataset>/gold_sl.json` | 数据库检索阶段默认输入文件 |
| `Data/<dataset>/qdrant_column_index/` | 本地 Qdrant 列级索引目录，由 `build_index.py` 生成 |
| `Data/<dataset>/documents/` | Spider2 的 `external_knowledge` 文档目录 |

当前仓库里的实际情况：

- `MMQA` 和 `BIRD` 提供了 `Database_schemas_summary.json`，可以直接运行基线数据库检索。
- `Spider2` 当前没有 `Database_schemas_summary.json`，不能直接跑 `Baseline_Database_Retrival.py`，除非用 `--database-schema-path` 指向你自己准备的摘要文件。
- 三个数据集都提供了 `db_info.json` 和 `gold_sl.json`。

输入字段约定：

| 脚本 | 必需字段 |
| --- | --- |
| `Baseline_Database_Retrival.py` | `id`、`db_id`、`question` |
| `Global_Coarse_Retrieval.py` | `id`、`db_id`、`question` |
| `Baseline_Schema_Linking.py` | `question`、`predict_db_id`，以及 `id` 或 `instance_id` |
| `table2column.py` | `id`、`question`、`predict_db_id` |

可选字段：

- `external_knowledge`：作为 hint 注入 prompt。
- 对 `Spider2`，`external_knowledge` 会被当作 `Data/Spider2/documents/` 下的文件名；如果找不到对应文件，则保留原值作为 hint。

## 3. 推荐运行流程

### 3.1 构建列级向量索引

`Global_Coarse_Retrieval.py` 和 `table2column.py` 都依赖本地 Qdrant 索引。首次运行某个数据集前先执行：

```bash
python -m Rag_pipline.build_index \
  --db-info-path Data/MMQA/db_info.json \
  --collection-name MMQA \
  --qdrant-path Data/MMQA/qdrant_column_index \
  --device cuda \
  --recreate
```

默认 collection 名称会从 `db_info.json` 的父目录推断，例如 `Data/MMQA/db_info.json` 对应 `MMQA`。

重要限制：

- `Global_Coarse_Retrieval.py` 当前固定使用 `dataset_name` 作为 Qdrant collection 名称，所以构建索引时建议保持 `--collection-name` 与 `--dataset-name` 一致。
- `table2column.py` 会读取 `qdrant_column_index/meta.json` 中的 collection 名称。
- 构建索引时的 embedding 模型应与 [`config.py`](config.py) 中的 `EMBEDDING_MODEL_NAME` 保持一致，否则查询阶段可能出现向量维度不匹配或效果异常。

### 3.2 基线流程

先运行基线数据库检索，再直接做列级 Schema Linking：

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

`Baseline_Schema_Linking.py` 如果不传 `--input-path`，会自动在 `Logs/<answer_llm_name>/Database_Retrival/` 下寻找当前数据集最近一次数据库检索结果。

### 3.3 检索增强流程

先运行全局粗检索，再运行 table-to-column Schema Linking：

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

`Global_Coarse_Retrieval.py` 的粗检索逻辑：

1. 对问题做 embedding，检索全局高相关列。
2. 按数据库聚合检索命中，做基于命中数和相似度的剪枝。
3. 对候选数据库逐个构造二分类 prompt，通过 yes/no 下一 token 概率重排。
4. 如果第一轮候选数大于 `--candidate-db-top-k`，会在 top-k 数据库内再做一轮召回、剪枝和重排。

### 3.4 手动指定上一步结果

两个 Schema Linking 脚本都支持显式传入数据库检索结果：

```bash
python -m Run.Baseline_Schema_Linking \
  --method few_shot \
  --input-path Logs/mistralai/Ministral-3-8B-Instruct-2512/Database_Retrival/iterative_database_retrival_MMQA_20260416_120000.json

python -m Run.table2column \
  --method few_shot \
  --input-path Logs/mistralai/Ministral-3-8B-Instruct-2512/Database_Retrival/iterative_database_retrival_MMQA_20260416_120000.json
```

连续跑不同模型或不同数据集时，建议显式指定 `--input-path`，避免自动读取到旧结果。

## 4. 输出文件

每个主脚本都会生成 `.json` 结果文件，并通过 [`Run/logging_utils.py`](Run/logging_utils.py) 生成同名 `.log` 日志文件。

| 脚本 | 默认输出位置 |
| --- | --- |
| `Baseline_Database_Retrival.py` | `Logs/<answer_llm_name>/Database_Retrival/baseline_database_retrival_<dataset>_<timestamp>.json` |
| `Global_Coarse_Retrieval.py` | `Logs/<answer_llm_name>/Database_Retrival/iterative_database_retrival_<dataset>_<timestamp>.json` |
| `Baseline_Schema_Linking.py` | `Logs/<answer_llm_name>/<method>_baseline_schema_linking_<dataset>_<timestamp>.json` |
| `table2column.py` | `Logs/<answer_llm_name>/<method>_table2column_<dataset>_<timestamp>.json` |

说明：

- `<answer_llm_name>` 会按模型名原样展开。
- 如果模型名包含 `/`，例如 `mistralai/Ministral-3-8B-Instruct-2512`，日志目录会自然变成多级路径。
- 输出记录中会包含 `efficiency` 字段，记录当前样本的 token 使用等效率信息。

## 5. 脚本参数

### 5.1 `Rag_pipline/build_index.py`

用途：从 `db_info.json` 构建列级 Qdrant 索引。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--db-info-path` | 输入 `db_info.json` 路径 | `Data/<DATASET_NAME>/db_info.json` |
| `--collection-name` | Qdrant collection 名称 | 从 `db_info.json` 父目录推断 |
| `--qdrant-path` | 本地 Qdrant 存储目录 | `<db_info_path.parent>/qdrant_column_index` |
| `--model-name` | embedding 模型名 | `EMBEDDING_MODEL_NAME` |
| `--cache-dir` | embedding 模型缓存目录 | `Llm/cache` |
| `--batch-size` | embedding 推理 batch size | `32` |
| `--upsert-batch-size` | 每轮标准化、编码并写入的列数 | `128` |
| `--device` | embedding 设备，如 `cpu` / `cuda` | 自动判断 |
| `--recreate` | collection 已存在时重建 | 关闭 |

索引文档由表名、列名、列描述和 value descriptions 渲染而成；单条 embedding 输入最多保留 `8096` tokens。

### 5.2 `Run/Baseline_Database_Retrival.py`

用途：把所有数据库摘要拼成一个 prompt，让 LLM 输出 JSON 字段 `relevant_database`。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset-name` | 数据集名称，对应 `Data/<dataset>/` | `DATASET_NAME` |
| `--answer-llm-name` | LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | `transformers` / `openai` | `PROVIDER` |
| `--max-input-length` | 最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--database-schema-path` | 数据库摘要文件路径 | `Data/<dataset>/Database_schemas_summary.json` |

固定输入为 `Data/<dataset>/gold_sl.json`，固定 prompt 模板为 `Templates/zero_shot/find_relevant_database_baseline.txt`。

### 5.3 `Run/Global_Coarse_Retrieval.py`

用途：列级向量召回 + 数据库剪枝 + LLM yes/no 二分类重排，输出 `predict_db_id`。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | 当前实际需要 `transformers` | `PROVIDER` |
| `--input-path` | 输入样本文件路径 | `Data/<dataset>/gold_sl.json` |
| `--output-path` | 结果输出路径 | 自动生成 |
| `--max-input-length` | 最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--candidate-db-top-k` | 第一轮重排后保留的候选数据库数量 | `CANDIDATE_DB_TOP_K` |
| `--enable-progress-log` | 写入逐样本、逐步骤的详细日志 | 关闭 |

补充说明：

- 当前代码没有 `--qdrant-path` 参数，固定读取 `Data/<dataset>/qdrant_column_index`。
- 当前代码固定使用 `dataset_name` 作为 Qdrant collection 名称。
- prompt 模板固定为 `Templates/zero_shot/binary_classification_database.txt`。
- 当前 prompt 裁剪预算由 `resolve_prompt_token_cap()` 控制。实际调用未传入脚本级常量，因此使用工具函数默认值：soft cap 为 `0.85 * max_input_length`，hard cap 为 `max_input_length - 512`。日志中还会记录脚本级 `PROMPT_BUDGET_RATIO = 0.8` 和 `PROMPT_BUDGET_BUFFER = 512`。

### 5.4 `Run/Baseline_Schema_Linking.py`

用途：在已有 `predict_db_id` 的前提下，对目标数据库直接做列级 Schema Linking。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--method` | prompt 方式：`zero_shot` / `few_shot` | `few_shot` |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | 当前实际需要 `transformers` | `PROVIDER` |
| `--max-input-length` | 最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--input-path` | 数据库检索结果路径 | 自动寻找最新结果 |
| `--logs-dir` | 日志根目录，用于自动寻找输入 | `PROJECT_ROOT/Logs` |
| `--db-info-path` | `db_info.json` 路径 | `Data/<dataset>/db_info.json` |
| `--output-path` | 输出文件路径 | 自动生成 |

补充说明：

- 输入文件必须包含 `predict_db_id`。
- prompt 模板为 `Templates/<method>/baseline_schema_linking.txt`。
- 当前实现直接从 `db_info.json` 渲染 schema 文本，不再依赖 `Table_schema_csv/`。
- 如果 `predict_db_id` 为空或不在 `db_info.json` 中，对应样本会输出 `No Valid Database.`。

### 5.5 `Run/table2column.py`

用途：先预测相关表，再在候选表范围内预测相关列。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--method` | prompt 方式：`zero_shot` / `few_shot` | `few_shot` |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | 当前实际需要 `transformers` | `PROVIDER` |
| `--max-input-length` | 最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--input-path` | 数据库检索结果路径 | 自动寻找最新结果 |
| `--logs-dir` | 日志根目录，用于自动寻找输入 | `PROJECT_ROOT/Logs` |
| `--db-info-path` | `db_info.json` 路径 | `Data/<dataset>/db_info.json` |
| `--qdrant-path` | 本地 Qdrant 索引目录 | `Data/<dataset>/qdrant_column_index` |
| `--output-path` | 输出文件路径 | 自动生成 |

补充说明：

- 输入文件必须包含 `predict_db_id`。
- 表级 prompt 使用 `Templates/<method>/extract_relevant_tables.txt`。
- 列级 prompt 使用 `Templates/<method>/extract_relevant_columns.txt`。
- 表筛选阶段会优先尝试渲染完整目标数据库；如果 prompt 超过预算，会用 Qdrant 排名结果裁剪 schema。
- 列筛选阶段会优先使用预测相关表中的全部列；如果仍超预算，会回退到表筛选阶段已经裁剪过的列集合。

## 6. 常见注意事项

1. `build_index.py` 可通过 `--model-name` 指定 embedding 模型，但查询阶段使用 [`config.py`](config.py) 中的 `EMBEDDING_MODEL_NAME`，两者应保持一致。
2. `Global_Coarse_Retrieval.py` 当前没有读取 Qdrant `meta.json`，collection 名必须和 `--dataset-name` 一致。
3. `Baseline_Schema_Linking.py` 和 `table2column.py` 默认会回溯最近一次数据库检索结果；实验较多时建议显式传入 `--input-path`。
4. `Spider2` 的 hint 读取方式和其他数据集不同：`external_knowledge` 会优先作为文档文件名解析。
5. 如果使用 `openai` provider，可以导出 `OPENAI_API_KEY`，或通过 `OPENAI_CREDENTIAL_PATH` / `--credential-path` 指向 JSON 凭据文件；代码不会自动加载 `.env`。
6. 当前环境若缺少依赖，甚至 `--help` 也可能因为顶层 import 失败而无法打印；请先完成 `pip install -r requirements.txt`。
