# MDB-link

本仓库实现了一个两阶段的 schema linking 流程：

1. 数据库检索：根据自然语言问题预测最相关的数据库 `predict_db_id`。
2. Schema Linking：在预测数据库中定位相关表或相关列。

当前代码中主要有 7 个可运行入口：

| 阶段 | 脚本 | 作用 |
| --- | --- | --- |
| 离线构建 | [`Indexing/build_column_index.py`](Indexing/build_column_index.py) | 从 `db_info.json` 构建列级 Qdrant 向量索引 |
| 数据库检索 | [`Baseline/baseline_database_retrieval.py`](Baseline/baseline_database_retrieval.py) | 基线数据库检索：把所有数据库摘要拼进 prompt，让 LLM 输出相关数据库 |
| 数据库检索 | [`Run/global_coarse_retrieval.py`](Run/global_coarse_retrieval.py) | 检索增强版数据库检索：列级向量召回、数据库剪枝、LLM 二分类重排 |
| Schema Linking | [`Baseline/baseline_schema_linking.py`](Baseline/baseline_schema_linking.py) | 已知 `predict_db_id` 后，对目标数据库直接做列级 schema linking |
| Schema Linking | [`Baseline/rag_column_retrieval.py`](Baseline/rag_column_retrieval.py) | 纯向量检索 baseline |
| Schema Linking | [`Run/table_to_column.py`](Run/table_to_column.py) | 先预测相关表，再在候选表范围内预测相关列 |
| SQL 生成 | [`Run/sql_generator.py`](Run/sql_generator.py) | Spider-Agent-TC 与 one-shot SQL 生成统一入口 |

目录职责和新增文件放置规则见 [`docs/project_structure.md`](docs/project_structure.md)。新结果使用正确拼写的 `database_retrieval`；读取阶段仍兼容历史 `Database_Retrival` 目录及 `*_retrival_*.json` 文件。

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
| `baseline_database_retrieval.py` | `transformers` / `openai` | 只需要普通文本生成 |
| `global_coarse_retrieval.py` | `transformers` | 需要 tokenizer 计数和下一 token logits 做 yes/no 重排 |
| `baseline_schema_linking.py` | `transformers` | schema 渲染器依赖 `answer_llm.tokenizer` |
| `table_to_column.py` | `transformers` | 依赖 tokenizer 计数、schema 裁剪和本地 embedding 检索 |
| `sql_generator.py` | Agent：`transformers`；one-shot：`transformers` / `openai` | 默认 Spider-Agent-TC 固定使用本地 Qwen3-Coder；旧 provider 兼容性由 `--generator-mode one_shot` 保留 |

使用 OpenAI provider 时，代码会优先读取环境变量 `OPENAI_API_KEY`，也可以读取 `OPENAI_CREDENTIAL_PATH` 或项目根目录下的 `gpt_credential.json`。凭据 JSON 支持 `api_key`、`openai_api_key`、`OPENAI_API_KEY`、`key` 字段。

```bash
export OPENAI_API_KEY=your_api_key_here
python -m Baseline.baseline_database_retrieval \
  --dataset-name MMQA \
  --provider openai \
  --answer-llm-name gpt-4.1-mini
```

SQL generator 使用 GPT-5 mini 的示例：

```bash
python -m Run.sql_generator \
  --dataset-name MMQA \
  --generator-mode one_shot \
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
| `Data/<dataset>/qdrant_column_index/` | 本地 Qdrant 列级索引目录，由 `build_column_index.py` 生成 |
| `Data/<dataset>/documents/` | Spider2 的 `external_knowledge` 文档目录 |

当前仓库里的实际情况：

- `MMQA` 和 `BIRD` 提供了 `Database_schemas_summary.json`，可以直接运行基线数据库检索。
- `Spider2` 当前没有 `Database_schemas_summary.json`，不能直接跑 `baseline_database_retrieval.py`，除非用 `--database-schema-path` 指向你自己准备的摘要文件。
- 三个数据集都提供了 `db_info.json` 和 `gold_sl.json`。

输入字段约定：

| 脚本 | 必需字段 |
| --- | --- |
| `baseline_database_retrieval.py` | `id`、`db_id`、`question` |
| `global_coarse_retrieval.py` | `id`、`db_id`、`question` |
| `baseline_schema_linking.py` | `question`、`predict_db_id`，以及 `id` 或 `instance_id` |
| `table_to_column.py` | `id`、`question`、`predict_db_id` |

可选字段：

- `external_knowledge`：作为 hint 注入 prompt。
- 对 `Spider2`，`external_knowledge` 会被当作 `Data/Spider2/documents/` 下的文件名；如果找不到对应文件，则保留原值作为 hint。

## 3. 推荐运行流程

### 3.1 构建列级向量索引

`global_coarse_retrieval.py` 和 `table_to_column.py` 都依赖本地 Qdrant 索引。首次运行某个数据集前先执行：

```bash
python -m Indexing.build_column_index \
  --db-info-path Data/MMQA/db_info.json \
  --collection-name MMQA \
  --qdrant-path Data/MMQA/qdrant_column_index \
  --device cuda \
  --recreate
```

默认 collection 名称会从 `db_info.json` 的父目录推断，例如 `Data/MMQA/db_info.json` 对应 `MMQA`。

重要限制：

- `global_coarse_retrieval.py` 当前固定使用 `dataset_name` 作为 Qdrant collection 名称，所以构建索引时建议保持 `--collection-name` 与 `--dataset-name` 一致。
- `table_to_column.py` 会读取 `qdrant_column_index/meta.json` 中的 collection 名称。
- 构建索引时的 embedding 模型应与 [`config.py`](config.py) 中的 `EMBEDDING_MODEL_NAME` 保持一致，否则查询阶段可能出现向量维度不匹配或效果异常。

### 3.2 基线流程

先运行基线数据库检索，再直接做列级 Schema Linking：

```bash
python -m Baseline.baseline_database_retrieval \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers

python -m Baseline.baseline_schema_linking \
  --method few_shot \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

数据库检索把运行期文本写入 `Logs/`，稳定预测写入 `results/db/baseline/<dataset>/<model>/prediction.json`。`baseline_schema_linking.py` 只读取后者，不会扫描 `Logs/`。

### 3.3 检索增强流程

先运行全局粗检索，再运行 table-to-column Schema Linking：

```bash
python -m Run.global_coarse_retrieval \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers \
  --candidate-db-top-k 3

python -m Run.table_to_column \
  --method few_shot \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

默认衔接文件为 `results/db/global_coarse_rerank/<dataset>/<model>/prediction.json`。如果粗检索使用 `--db-selection-mode pruning`，Schema Linking 需要相应传入 `--database-method global_coarse_pruning`。

`global_coarse_retrieval.py` 的粗检索逻辑：

1. 对问题做 embedding，检索全局高相关列。
2. 按数据库聚合检索命中，做基于命中数和相似度的剪枝。
3. 对候选数据库逐个构造二分类 prompt，通过 yes/no 下一 token 概率重排。
4. 如果第一轮候选数大于 `--candidate-db-top-k`，会在 top-k 数据库内再做一轮召回、剪枝和重排。

### 3.4 指定数据库预测方法或文件

数据库预测模型与 Schema Linking 模型不同时，使用独立的 `--database-model-name`，不再依赖 Schema Linking 模型名定位输入：

```bash
python -m Run.table_to_column \
  --method few_shot \
  --answer-llm-name mistralai/Ministral-3-14B-Instruct-2512 \
  --database-method global_coarse_rerank \
  --database-model-name mistralai/Ministral-3-8B-Instruct-2512
```

`--input-path` 可显式读取某个统一 DB prediction，并具有最高优先级；旧的 Logs JSON 格式不再接受。

## 4. 输出文件

各阶段的 `.json` 预测全部写入 `results/`；检索与 Schema Linking 脚本还会通过 [`Utils/logging_utils.py`](Utils/logging_utils.py) 在 `Logs/` 生成独立的 `.log` 文本日志。

| 脚本 | 默认输出位置 |
| --- | --- |
| `baseline_database_retrieval.py` | `results/db/baseline/<dataset>/<model>/prediction.json` |
| `global_coarse_retrieval.py` | `results/db/global_coarse_<mode>/<dataset>/<model>/prediction.json` |
| `baseline_schema_linking.py` | `results/sl/prompt_baseline/<dataset>/<model>/prediction.json` |
| `table_to_column.py` | `results/sl/table_to_column/<dataset>/<model>/prediction.json` |
| `rag_column_retrieval.py` | `results/sl/rag_column_retrieval/<dataset>/<model>/prediction.json` |
| `sql_generator.py` | `results/sql/<sql_method>/<dataset>/<sql_model>/<sl_method>/<sl_model>.json` |

说明：

- 模型名中的 `/` 会被安全转换为 `__`，不会产生意外的多级预测目录。
- `Logs/` 仅包含 `.log` 文本，不包含任何供下游消费的预测结果。

数据库检索写入统一、原子更新的预测：

```text
results/db/<db_method>/<dataset>/<db_model>/prediction.json
```

其中 `db_method` 为 `baseline`、`global_coarse_rerank` 或 `global_coarse_pruning`。

Schema Linking 完成后会写入：

```text
results/sl/<sl_method>/<dataset>/<sl_model>/prediction.json
```

其中 `sl_method` 为 `prompt_baseline`、`table_to_column` 或 `rag_column_retrieval`。表名前缀严格匹配 `predict_db_id + "."` 时会在统一结果中去掉；SQL generator 读取 Spider2 统一结果时会重新组合 Snowflake 三段表名。

SQL 统一结果独立保存，不修改 Schema Linking 文件：

```text
results/sql/<sql_method>/<dataset>/<sql_model>/<sl_method>/<sl_model>.json
```

SQL generator 可通过 `--input-path` 直接读取统一 SL 文件，也可同时指定 `--sl-method` 和 `--schema-llm-name` 使用默认统一路径。

## 5. 脚本参数

### 5.1 `Indexing/build_column_index.py`

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

### 5.2 `Baseline/baseline_database_retrieval.py`

用途：把所有数据库摘要拼成一个 prompt，让 LLM 输出 JSON 字段 `relevant_database`。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset-name` | 数据集名称，对应 `Data/<dataset>/` | `DATASET_NAME` |
| `--answer-llm-name` | LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | `transformers` / `openai` | `PROVIDER` |
| `--max-input-length` | 最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--database-schema-path` | 数据库摘要文件路径 | `Data/<dataset>/Database_schemas_summary.json` |
| `--prediction-path` | 统一 DB prediction 输出路径 | `results/db/baseline/<dataset>/<model>/prediction.json` |

固定输入为 `Data/<dataset>/gold_sl.json`，固定 prompt 模板为 `Templates/zero_shot/find_relevant_database_baseline.txt`。

### 5.3 `Run/global_coarse_retrieval.py`

用途：列级向量召回 + 数据库剪枝 + LLM yes/no 二分类重排，输出 `predict_db_id`。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | 当前实际需要 `transformers` | `PROVIDER` |
| `--input-path` | 输入样本文件路径 | `Data/<dataset>/gold_sl.json` |
| `--prediction-path` | 统一 DB prediction 输出路径 | `results/db/global_coarse_<mode>/<dataset>/<model>/prediction.json` |
| `--max-input-length` | 最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--candidate-db-top-k` | 第一轮重排后保留的候选数据库数量 | `CANDIDATE_DB_TOP_K` |
| `--enable-progress-log` | 写入逐样本、逐步骤的详细日志 | 关闭 |

补充说明：

- 当前代码没有 `--qdrant-path` 参数，固定读取 `Data/<dataset>/qdrant_column_index`。
- 当前代码固定使用 `dataset_name` 作为 Qdrant collection 名称。
- prompt 模板固定为 `Templates/zero_shot/binary_classification_database.txt`。
- 当前 prompt 裁剪预算由 `resolve_prompt_token_cap()` 控制。实际调用未传入脚本级常量，因此使用工具函数默认值：soft cap 为 `0.85 * max_input_length`，hard cap 为 `max_input_length - 512`。日志中还会记录脚本级 `PROMPT_BUDGET_RATIO = 0.8` 和 `PROMPT_BUDGET_BUFFER = 512`。

### 5.4 `Baseline/baseline_schema_linking.py`

用途：在已有 `predict_db_id` 的前提下，对目标数据库直接做列级 Schema Linking。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--method` | prompt 方式：`zero_shot` / `few_shot` | `few_shot` |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | 当前实际需要 `transformers` | `PROVIDER` |
| `--max-input-length` | 最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--input-path` | 显式数据库预测路径，优先级最高 | 未设置 |
| `--database-method` | 统一数据库预测方法 | `baseline` |
| `--database-model-name` | 数据库预测模型 | 与 `--answer-llm-name` 相同 |
| `--db-info-path` | `db_info.json` 路径 | `Data/<dataset>/db_info.json` |
| `--prediction-path` | 统一 SL prediction 输出路径 | `results/sl/prompt_baseline/<dataset>/<model>/prediction.json` |

补充说明：

- 输入文件必须包含 `predict_db_id`。
- prompt 模板为 `Templates/<method>/baseline_schema_linking.txt`。
- 当前实现直接从 `db_info.json` 渲染 schema 文本，不再依赖 `Table_schema_csv/`。
- 如果 `predict_db_id` 为空或不在 `db_info.json` 中，对应样本会输出 `No Valid Database.`。

### 5.5 `Run/table_to_column.py`

用途：先预测相关表，再在候选表范围内预测相关列。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--method` | prompt 方式：`zero_shot` / `few_shot` | `few_shot` |
| `--dataset-name` | 数据集名称 | `DATASET_NAME` |
| `--answer-llm-name` | LLM 名称 | `ANSWER_LLM_NAME` |
| `--provider` | 当前实际需要 `transformers` | `PROVIDER` |
| `--max-input-length` | 最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |
| `--input-path` | 显式数据库预测路径，优先级最高 | 未设置 |
| `--database-method` | 统一数据库预测方法 | `global_coarse_rerank` |
| `--database-model-name` | 数据库预测模型 | 与 `--answer-llm-name` 相同 |
| `--db-info-path` | `db_info.json` 路径 | `Data/<dataset>/db_info.json` |
| `--qdrant-path` | 本地 Qdrant 索引目录 | `Data/<dataset>/qdrant_column_index` |
| `--prediction-path` | 统一 SL prediction 输出路径 | `results/sl/table_to_column/<dataset>/<model>/prediction.json` |

补充说明：

- 输入文件必须包含 `predict_db_id`。
- 表级 prompt 使用 `Templates/<method>/extract_relevant_tables.txt`。
- 列级 prompt 使用 `Templates/<method>/extract_relevant_columns.txt`。
- 表筛选阶段会优先尝试渲染完整目标数据库；如果 prompt 超过预算，会用 Qdrant 排名结果裁剪 schema。
- 列筛选阶段会优先使用预测相关表中的全部列；如果仍超预算，会回退到表筛选阶段已经裁剪过的列集合。
- 使用 `--disable-table-filtering` 时，预测写入 `results/ablation/wo_table_filtering/sl/table_to_column/<dataset>/<model>/prediction.json`，不会覆盖标准结果。

## 6. SQL Generator：Spider-Agent-TC 与 one-shot 对比

[`Run/sql_generator.py`](Run/sql_generator.py) 默认使用 `spider_agent_tc`：Qwen3-Coder 先输出 XML tool call，执行候选 SQL，再根据 SQLite/Snowflake 返回的样例行或错误迭代修正，最后以 `terminate` 提交 SQL。原有一次提示、一次生成的完整实现保留在 [`Run/one_shot_sql_generator.py`](Run/one_shot_sql_generator.py)，可通过 `--generator-mode one_shot` 继续运行并做消融对比。两种模式共享同一个 `SchemaLinkingAdapter`，只接收 `predict_db_id`、`predict_tables` 和 `predict_columns` 指定的 schema。

Spider-Agent-TC 会把当前数据集的执行函数和 `terminate` 作为原生 tools 传入 Qwen chat template，并用结构化 tool-call/tool-response 消息维护多轮历史。协议保持严格：不带 `<tool_call>` 的裸 SQL 或 Markdown SQL 不会作为候选查询执行。未通过 `terminate` 验证的运行会在统一 SQL 预测文件旁写入 `<stem>_agent_failures.json`，保存逐轮原始回复和错误，同时不改变预测文件的 schema。

主实验配置固定为 `Qwen/Qwen3-Coder-30B-A3B-Instruct`、Transformers BF16、单张 A100 80GB、`cuda:0`、SDPA、输入上限 24576 tokens、输出上限 4096 tokens。默认最多 10 个 Agent round，每次模型调用失败后额外重试 2 次；Spider2、MMQA 和 BIRD 使用同一套生成超参数。

执行后端：

- Spider2 使用 Snowflake。通过 `--snowflake-credential-path` 传入 JSON；支持 `account`、`username`/`user`、`password`/`pat`/`token`、`warehouse`、`role`。连接 database 始终取当前样本的 `predict_db_id`，不会采用 credential 中的默认 database 或 gold database。
- MMQA 默认在 `Data/MMQA/Sqlite_database` 定位只读 SQLite；BIRD 默认在 `Data/BIRD/Raw_data/dev_databases` 定位。可以用 `--database-root` 覆盖。SQLite 使用 `mode=ro`，禁止 `ATTACH`。

Spider2 示例：

```bash
python -m Run.sql_generator \
  --dataset-name Spider2 \
  --generator-mode spider_agent_tc \
  --input-path <results/sl/.../prediction.json> \
  --snowflake-credential-path <snowflake_credential.json> \
  --resume
```

MMQA 示例：

```bash
python -m Run.sql_generator \
  --dataset-name MMQA \
  --generator-mode spider_agent_tc \
  --input-path <results/sl/.../prediction.json> \
  --database-root Data/MMQA/Sqlite_database \
  --resume
```

运行旧方法时只需改为 `--generator-mode one_shot`。`--dry-run --limit 1` 只检查输入、Adapter、executor 路由和 prompt 预算，不加载 30B 模型，也不连接数据库。

SQL 预测文件使用统一 `predictions` 契约，每条记录包含 `id`、`predicted_sql`、`status` 和 `error`。结果逐样本原子写入，`--resume` 会按稳定 id 跳过已经完成的记录。

## 7. 常见注意事项

1. `build_column_index.py` 可通过 `--model-name` 指定 embedding 模型，但查询阶段使用 [`config.py`](config.py) 中的 `EMBEDDING_MODEL_NAME`，两者应保持一致。
2. `global_coarse_retrieval.py` 当前没有读取 Qdrant `meta.json`，collection 名必须和 `--dataset-name` 一致。
3. `baseline_schema_linking.py` 和 `table_to_column.py` 只读取 `results/db` 或显式统一输入；跨模型组合请传入 `--database-model-name`。
4. `Spider2` 的 hint 读取方式和其他数据集不同：`external_knowledge` 会优先作为文档文件名解析。
5. 如果使用 `openai` provider，可以导出 `OPENAI_API_KEY`，或通过 `OPENAI_CREDENTIAL_PATH` / `--credential-path` 指向 JSON 凭据文件；代码不会自动加载 `.env`。
6. 当前环境若缺少依赖，甚至 `--help` 也可能因为顶层 import 失败而无法打印；请先完成 `pip install -r requirements.txt`。
7. CLI 中的相对文件路径统一相对于项目根目录解析；模块命令仍建议在项目根目录执行，例如 `python -m Run.sql_generator`。
