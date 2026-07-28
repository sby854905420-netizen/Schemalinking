# MDB-link

本仓库实现了一个两阶段的 schema linking 流程：

1. 数据库检索：根据自然语言问题预测最相关的数据库 `predict_db_id`。
2. Schema Linking：在预测数据库中定位相关表或相关列。

当前代码中主要有 9 个可运行入口：

| 阶段 | 脚本 | 作用 |
| --- | --- | --- |
| 离线构建 | [`Indexing/build_column_index.py`](Indexing/build_column_index.py) | 从 `db_info.json` 构建列级 Qdrant 向量索引 |
| 数据库检索 | [`Baseline/baseline_database_retrieval.py`](Baseline/baseline_database_retrieval.py) | 基线数据库检索：把所有数据库摘要拼进 prompt，让 LLM 输出相关数据库 |
| 数据库检索 | [`Run/global_coarse_retrieval.py`](Run/global_coarse_retrieval.py) | 检索增强版数据库检索：列级向量召回、数据库剪枝、LLM 二分类重排 |
| Schema Linking | [`Baseline/baseline_schema_linking.py`](Baseline/baseline_schema_linking.py) | 已知 `predict_db_id` 后，对目标数据库直接做列级 schema linking |
| Schema Linking | [`Baseline/rag_column_retrieval.py`](Baseline/rag_column_retrieval.py) | 纯向量检索 baseline |
| Schema Linking | [`Run/table_to_column.py`](Run/table_to_column.py) | 先预测相关表，再在候选表范围内预测相关列 |
| SQL 生成 | [`Run/sql_generator.py`](Run/sql_generator.py) | Spider-Agent-TC 与 one-shot SQL 生成统一入口 |
| 评估 | [`Evaluation/evaluate_metrics.py`](Evaluation/evaluate_metrics.py) | 计算 LA、EM、Recall、Avg_Ratio、Avg_token、Avg_time |
| 评估 | [`Evaluation/evaluate_ex.py`](Evaluation/evaluate_ex.py) | 执行预测/gold SQL 并计算 EX |

目录职责和新增文件放置规则见 [`docs/project_structure.md`](docs/project_structure.md)。

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
| `HRC_RETRIEVAL_RATIO` | `0.1` |
| `ROUND1_HRC_RETRIEVAL_CAP` | `500` |
| `ROUND2_HRC_RETRIEVAL_CAP` | `50` |
| `ROUND1_DATABASE_TOP_K` | `10` |
| `ROUND1_RERANK_TOP_K` | `3` |

### Provider 兼容性

`LLM` 封装支持 `transformers` 和 `openai`，但各脚本实际依赖不同：

| 脚本 | 当前可用 provider | 原因 |
| --- | --- | --- |
| `baseline_database_retrieval.py` | `transformers` / `openai` | 只需要普通文本生成 |
| `global_coarse_retrieval.py` | `transformers` | 需要 tokenizer 计数和下一 token logits 做 yes/no 重排 |
| `baseline_schema_linking.py` | `transformers` | schema 渲染器依赖 `answer_llm.tokenizer` |
| `table_to_column.py` | `transformers` | 依赖 tokenizer 计数、schema 裁剪和本地 embedding 检索 |
| `sql_generator.py` | Agent：`transformers`；one-shot：`transformers` / `openai` | 默认 Spider-Agent-TC 固定使用本地 Qwen3-Coder；one-shot 支持普通文本生成 provider |

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
- `table_to_column.py` 和 prompt schema baseline 会按样本 id 从 `gold_sl.json` 恢复 hint；统一数据库预测文件不需要重复保存该字段。
- HRC 向量检索始终只编码 question，`external_knowledge` 不参与 HRC 选取；纯 RAG baseline 同样不使用该字段。

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

注意事项：

- 三个 Qdrant 检索入口都支持 `--qdrant-path` 和 `--collection-name`。collection 名称的解析优先级为：显式 `--collection-name`、索引目录中的 `meta.json`、`--dataset-name`。
- 构建索引时的 embedding 模型应与 [`config.py`](config.py) 中的 `EMBEDDING_MODEL_NAME` 保持一致，否则查询阶段可能出现向量维度不匹配或效果异常。

#### 在 Slurm job 中使用节点本地索引

多个 job 不应直接打开 Lustre 上的同一个嵌入式 Qdrant 目录。每个 job 启动后，先把已经构建完成且不会再写入的索引复制到节点本地临时目录，再把该目录传给检索入口：

```bash
DATASET_NAME=Spider2
SOURCE_INDEX="Data/${DATASET_NAME}/qdrant_column_index"
LOCAL_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/mdb-link-${SLURM_JOB_ID:-$$}"
LOCAL_INDEX="${LOCAL_BASE}/${DATASET_NAME}/qdrant_column_index"

mkdir -p "${LOCAL_INDEX}"
rsync -a --delete --exclude='.lock' "${SOURCE_INDEX}/" "${LOCAL_INDEX}/"

python -m Run.global_coarse_retrieval \
  --dataset-name "${DATASET_NAME}" \
  --qdrant-path "${LOCAL_INDEX}" \
  --collection-name "${DATASET_NAME}" \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

`table_to_column.py` 和 `rag_column_retrieval.py` 使用相同的 `--qdrant-path "${LOCAL_INDEX}"` 参数。任务结束后，Slurm 通常会清理节点临时目录；不要把运行时产生的 `.lock` 同步回源索引。

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

Baseline 只写稳定预测 `results/db/baseline/<dataset>/<model>/prediction.json`；不会创建文本日志或结构化决策轨迹。`baseline_schema_linking.py` 读取该预测文件。

### 3.3 检索增强流程

先运行全局粗检索，再运行 table-to-column Schema Linking：

```bash
python -m Run.global_coarse_retrieval \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers

python -m Run.table_to_column \
  --method few_shot \
  --dataset-name MMQA \
  --answer-llm-name mistralai/Ministral-3-8B-Instruct-2512 \
  --provider transformers
```

默认衔接文件为 `results/db/global_coarse_rerank/<dataset>/<model>/prediction.json`。如果粗检索使用 `--db-selection-mode pruning`，Schema Linking 需要相应传入 `--database-method global_coarse_pruning`。

`global_coarse_retrieval.py` 的粗检索逻辑：

1. 对问题做 embedding。第一轮 HRC 数量为 `ceil(全部数据库总列数 × 10%)`，上限为 500；检索在全部数据库列组成的联合空间中进行。
2. 按数据库聚合 HRC 命中，使用命中阈值/相似度 80% 分位数规则过滤，再按照 `max_score`、`score_sum`、`hit_count` 依次降序排序，最多保留 Top-10 进入第一轮 LLM 重排。
3. 如果第一轮候选数据库不超过 3 个，第一轮 LLM 直接选择 Top-1，并跳过第二轮；否则第一轮 LLM 选择 Top-3。
4. 第二轮仅在该 Top-3 数据库的联合列空间中检索，HRC 数量为 `ceil(Top-3 数据库总列数 × 10%)`，上限为 50。对结果再次做数据库剪枝后，LLM 重排选出最终 Top-1。

### 3.4 指定数据库预测方法或文件

数据库预测模型与 Schema Linking 模型不同时，使用独立的 `--database-model-name`，不再依赖 Schema Linking 模型名定位输入：

```bash
python -m Run.table_to_column \
  --method few_shot \
  --answer-llm-name mistralai/Ministral-3-14B-Instruct-2512 \
  --database-method global_coarse_rerank \
  --database-model-name mistralai/Ministral-3-8B-Instruct-2512
```

`--input-path` 可显式读取某个统一 DB prediction，并具有最高优先级；输入内容仍须满足统一预测契约。

## 4. 输出文件

各阶段的 `.json` 预测全部写入 `results/`。检索增强主流程另外写入稳定、覆盖式的结构化决策轨迹；相同方法、数据集和模型再次运行时会直接替换上一次轨迹，不创建 `run_id` 子目录。

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
- 所有入口都不创建运行期文本日志。`baseline_database_retrieval.py`、`baseline_schema_linking.py` 和 `rag_column_retrieval.py` 也不写结构化决策轨迹，只保存稳定预测结果。

结构化轨迹使用以下稳定目录：

```text
results/traces/db/global_coarse_<mode>/<dataset>/<db_model>/
├── metadata.json
├── events.jsonl
└── summary.json

results/traces/sl/table_to_column/<dataset>/<sl_model>/
├── metadata.json
├── events.jsonl
└── summary.json
```

- `metadata.json` 记录数据集、方法、模型、输入和索引等实验身份信息以及运行状态；运行级失败时额外记录精简的 `error={type,message}`。
- `events.jsonl` 逐样本记录 HRC 数量、数据库剪枝结果、LLM 重排结果、表预测、列预测、诊断字段和异常。它不记录运行时间、token 花费、HRC 的具体列、embedding、完整 prompt 或代码中固定的筛选参数。
- `summary.json` 汇总数据库、表和列阶段的保留率、准确率、召回率及首次错误阶段。
- `first_error_stage` 表示最早出现的中间退化阶段；即使下游模型偶然恢复出完整正确结果，该早期候选范围损失仍会被保留。第二轮 HRC/剪枝召回率只以实际进入第二轮的样本为分母。
- 当上游方法为 `global_coarse_rerank` 或 `global_coarse_pruning` 时，`table_to_column.py` 会校验并复制对应数据库事件，再追加表预测、列预测和样本最终结果，因此可以沿同一条轨迹定位数据库、表和列阶段的错误。以 baseline 数据库预测为输入时不创建结构化轨迹。

数据库检索写入统一、原子更新的预测：

```text
results/db/<db_method>/<dataset>/<db_model>/prediction.json
```

其中 `db_method` 为 `baseline`、`global_coarse_rerank` 或 `global_coarse_pruning`。

Schema Linking 完成后会写入：

```text
results/sl/<sl_method>/<dataset>/<sl_model>/prediction.json
```

其中本项目生成的 `sl_method` 为 `prompt_baseline`、`table_to_column` 或 `rag_column_retrieval`；SQL generator 也接受统一契约下的 `linkalign` 和 `autolink` 外部结果。SQLite 数据集会去掉严格匹配的 `predict_db_id + "."` 前缀；Spider2 只会从真正的 `DB.SCHEMA.TABLE` 三段名中去掉最外层 DB，已经是 `SCHEMA.TABLE` 的两段名保持不变。SQL generator 使用 `db_info.json` 将 Spider2 表名保守解析回 canonical 三段名；裸表名仅在目标数据库内唯一时补全，歧义时不会猜测。

SQL 统一结果独立保存，不修改 Schema Linking 文件：

```text
results/sql/<sql_method>/<dataset>/<sql_model>/<sl_method>/<sl_model>.json
```

SQL generator 可通过 `--input-path` 直接读取统一 SL 文件，也可同时指定 `--sl-method` 和 `--schema-llm-name` 使用默认统一路径。三类输入使用以下样本对齐规则：

- MDB-link 自身的 `prompt_baseline`、`table_to_column` 和 `rag_column_retrieval` 结果直接使用统一预测中的稳定 ID，不做外部 ID 转换。
- LinkAlign 会与数据源 ID 做严格对齐；当预测 ID 与 gold ID 不同（例如 MMQA 的 `two_table_*`/`three_table_*`）时，必须通过 `--dataset-path` 提供对应源数据。源数据中的 `instance_id` 与统一预测 ID 精确匹配，`gold_id` 作为最终 SQL/evaluation ID；question 必须完全一致。缺少 SL 预测的源样本会按 gold 顺序补成 `failed` 占位，不会调用 SQL 模型。
- AutoLink 的 MMQA `mmqa_N` 按严格前缀规则映射到 gold ID `N`，并校验 question；它默认使用 `Data/MMQA/gold_sl.json`，不需要额外的映射文件。缺少预测时同样写入 `failed` 占位。

ID 对齐不使用 question、位置或数据库名称作为匹配键；这些字段只用于匹配后的完整性校验，避免重复问题造成误映射。

`method: autolink` 是显式的 oracle-database 对比模式。SQL generator 使用 `Data/<dataset>/gold_sl.json` 按稳定样本 ID 定位 gold `db_id`（MMQA 将 `mmqa_N` 映射到 gold ID `N`，校验 question 完全一致，并以 gold ID 写出 SQL 结果），只保留 AutoLink `schema_linking.final.columns` 中属于该数据库的列；其他数据库的表列全部丢弃。若样本无法可靠匹配 gold 记录，或 gold 数据库没有任何预测列，该样本会在调用 SQL 模型前直接写为 `failed`。为保证不引入预测之外的列，该模式拒绝 `--include-key-columns`。

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
| `--qdrant-path` | 本地 Qdrant 索引目录 | `Data/<dataset>/qdrant_column_index` |
| `--collection-name` | Qdrant collection 名称 | `meta.json` 中的名称，缺失时使用 dataset 名 |
| `--prediction-path` | 统一 DB prediction 输出路径 | `results/db/global_coarse_<mode>/<dataset>/<model>/prediction.json` |
| `--max-input-length` | 最大输入 token 数 | `MAX_INPUT_LENGTH` |
| `--max-generation-num` | 最大生成 token 数 | `MAX_GENERATEION_NUM` |

补充说明：

- prompt 模板固定为 `Templates/zero_shot/binary_classification_database.txt`。
- prompt 裁剪预算由 `resolve_prompt_token_cap()` 控制：soft cap 为 `0.85 * max_input_length`，hard cap 为 `max_input_length - 512`。
- 第一轮 LLM 的分支规则固定为：候选数据库不超过 3 个时直接选择 Top-1，否则选择 Top-3 并进入第二轮。

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
| `--collection-name` | Qdrant collection 名称 | `meta.json` 中的名称，缺失时使用 dataset 名 |
| `--prediction-path` | 统一 SL prediction 输出路径 | `results/sl/table_to_column/<dataset>/<model>/prediction.json` |

补充说明：

- 输入文件必须包含 `predict_db_id`。
- 表级 prompt 使用 `Templates/<method>/extract_relevant_tables.txt`。
- 列级 prompt 使用 `Templates/<method>/extract_relevant_columns.txt`。
- 表筛选阶段会优先尝试渲染完整目标数据库；如果 prompt 超过预算，会用 Qdrant 排名结果裁剪 schema。
- 列筛选阶段会优先使用预测相关表中的全部列；如果仍超预算，会回退到表筛选阶段已经裁剪过的列集合。
- 使用 `--disable-table-filtering` 时，预测写入 `results/ablation/wo_table_filtering/sl/table_to_column/<dataset>/<model>/prediction.json`，不会覆盖标准结果。

## 6. SQL Generator：Spider-Agent-TC 与 one-shot 对比

[`Run/sql_generator.py`](Run/sql_generator.py) 统一实现两种 SQL 生成模式。默认的 `spider_agent_tc` 让 Qwen3-Coder 输出 XML tool call，执行候选 SQL，再根据 SQLite/Snowflake 返回的样例行或错误迭代修正，最后以 `terminate` 提交 SQL；`--generator-mode one_shot` 则执行一次提示、一次生成，用于消融对比。两种模式共享同一个 `SchemaLinkingAdapter`，只接收 `predict_db_id`、`predict_tables` 和 `predict_columns` 指定的 schema。

Spider-Agent-TC 会把当前数据集的执行函数和 `terminate` 作为原生 tools 传入 Qwen chat template，并用结构化 tool-call/tool-response 消息维护多轮历史。协议保持严格：不带 `<tool_call>` 的裸 SQL 或 Markdown SQL 不会作为候选查询执行。未通过 `terminate` 验证的运行会在统一 SQL 预测文件旁写入 `<stem>_agent_failures.json`，保存逐轮原始回复和错误。

主实验配置固定为 `Qwen/Qwen3-Coder-30B-A3B-Instruct`、Transformers BF16、单张 A100 80GB、`cuda:0`、SDPA、输入上限 24576 tokens、输出上限 4096 tokens。默认最多 20 个 Agent round，每次模型调用失败后额外重试 2 次；Spider2、MMQA 和 BIRD 使用同一套生成超参数。

执行后端：

- Spider2 使用 Snowflake。通过 `--snowflake-credential-path` 传入 JSON；支持 `account`、`username`/`user`、`password`/`pat`/`token`、`warehouse`、`role`。连接 database 始终取预处理后样本的 `predict_db_id`，不会采用 credential 中的默认 database；仅 `method: autolink` 的显式 oracle-database 模式会先把它设为 gold database。
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

LinkAlign MMQA 使用其带 `instance_id`/`gold_id` 的源数据完成 ID 对齐：

```bash
python -m Run.sql_generator \
  --dataset-name MMQA \
  --generator-mode spider_agent_tc \
  --input-path <linkalign/results/sl/linkalign/MMQA/.../prediction.json> \
  --dataset-path <linkalign/Data/MMQA/Synthesized_preprocessed_data.json>
```

旧版 LinkAlign SQL 结果使用 `two_table_*`/`three_table_*` ID；首次切换前应先停止旧生成作业，并且不要添加 `--resume`。生成器会拒绝把旧 ID 与新 ID 混写。完成首次替换后，后续运行可以正常使用 `--resume`。

AutoLink MMQA 已由默认 `gold_sl.json` 提供 `mmqa_N` 映射，只需指定预测文件：

```bash
python -m Run.sql_generator \
  --dataset-name MMQA \
  --generator-mode spider_agent_tc \
  --input-path <autolink/results/sl/autolink/MMQA/.../prediction.json> \
  --resume
```

切换到单次生成时使用 `--generator-mode one_shot`。`--dry-run --limit 1` 不加载模型或连接数据库：两种模式都会检查输入、Adapter 和各自的 prompt 预算，Agent 模式还会检查 executor 路由。

SQL 预测文件使用统一 `predictions` 契约，每条新记录包含 `id`、`predicted_sql`、`status`、`error`、`agent_rounds`、`total_tokens` 和 `elapsed_seconds`。Spider-Agent-TC 的三个统计字段分别表示该样本所有已执行 rollout 的 Agent 轮次总和、全部模型调用的输入与输出 token 总和，以及包含模型生成和 SQL 执行在内的墙钟时间；one-shot 的这三个字段为 `null`。结果逐样本原子写入，`--resume` 会按稳定 id 跳过已经完成的记录；旧版四字段记录仍可读取，但只有重新生成的样本才会获得真实统计值。

## 7. 常见注意事项

1. `build_column_index.py` 可通过 `--model-name` 指定 embedding 模型，但查询阶段使用 [`config.py`](config.py) 中的 `EMBEDDING_MODEL_NAME`，两者应保持一致。
2. 并发 Slurm job 应各自把 Qdrant 索引复制到节点本地目录，并通过 `--qdrant-path` 使用自己的副本；不要共享同一个嵌入式索引目录。
3. `baseline_schema_linking.py` 和 `table_to_column.py` 只读取 `results/db` 或显式统一输入；跨模型组合请传入 `--database-model-name`。
4. `Spider2` 的 hint 读取方式和其他数据集不同：`external_knowledge` 会优先作为文档文件名解析。
5. 如果使用 `openai` provider，可以导出 `OPENAI_API_KEY`，或通过 `OPENAI_CREDENTIAL_PATH` / `--credential-path` 指向 JSON 凭据文件；代码不会自动加载 `.env`。
6. 当前环境若缺少依赖，甚至 `--help` 也可能因为顶层 import 失败而无法打印；请先完成 `pip install -r requirements.txt`。
7. CLI 中的相对文件路径统一相对于项目根目录解析；模块命令仍建议在项目根目录执行，例如 `python -m Run.sql_generator`。
