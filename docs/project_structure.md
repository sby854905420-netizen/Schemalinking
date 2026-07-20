# Project Structure

项目按“实验入口、复用实现、静态资源、运行产物”分层：

```text
Schemalinking/
├── Baseline/                  # 可独立运行的 baseline 实验
├── Indexing/                  # 离线索引构建命令
├── Run/                       # 主流程、结果转换入口与 SQL Agent 实现
│   └── spider_agent_tc/       # SQL Agent 内部组件和执行器
├── Evaluation/                # 指标汇总与 SQL 执行准确率评估入口
├── Llm/                       # LLM 与 embedding 模型加载器
├── Utils/                     # 预测契约、结果存储及其他共享工具
├── Templates/                 # Prompt 静态资源
│   ├── few_shot/
│   ├── zero_shot/
│   └── sql_generation/
├── Data/                      # 数据集、SQLite 库、Qdrant 索引（不提交）
├── results/                   # 统一预测、评估结果与结构化轨迹（不提交）
├── tests/                     # 单元测试与小型 fixture
├── docs/                      # 设计和评估文档
├── config.py                  # 配置值与所有项目根路径
└── README.md                  # 安装、运行和参数说明
```

## 文件放置规则

- 新的对照方法或消融实验放在 `Baseline/`，文件名使用小写 snake_case。
- 主实验的可运行阶段放在 `Run/`；只服务于某个复杂入口的实现放在该入口的子包中，例如 `Run/spider_agent_tc/`。
- 数据预处理或索引构建等离线命令放在 `Indexing/`。
- 评估入口放在 `Evaluation/`，评估表统一写入 `results/evaluation/`。
- 模型加载、provider 适配放在 `Llm/`，通用格式化、路径查找和结构化轨迹工具放在 `Utils/`。
- Prompt 不写死在 Python 中，按任务放到 `Templates/<task>/`。
- 小型测试数据放在 `tests/fixtures/`；真实数据、数据库和向量索引只放在 `Data/`。
- 所有 DB、SL、SQL 预测及检索增强主流程的结构化轨迹放在 `results/`；文档放在 `docs/`。运行入口不生成独立文本日志。
- DB、SL、SQL 预测契约分别放在 `Utils/database_prediction_store.py`、`Utils/prediction_store.py` 与 `Utils/sql_prediction_store.py`。
- 不在业务模块里重复拼接 `PROJECT_ROOT / "Data"` 等路径；新增根路径先在 `config.py` 定义。

## 路径兼容策略

所有默认路径由 `config.py` 中的 `DATA_ROOT`、`TEMPLATES_ROOT`、`RESULTS_ROOT` 和 `MODEL_CACHE_ROOT` 派生。CLI 传入的相对路径通过 `resolve_project_path()` 相对于项目根目录解析，避免切换工作目录后读取错误。所有运行入口都不生成独立文本日志；三个 baseline 也不写结构化轨迹。数据库预测固定使用：

```text
results/db/<db_method>/<dataset>/<db_model>/prediction.json
```

Schema Linking 只读取上述统一 DB prediction，或读取用户通过 `--input-path` 显式提供的同格式文件。

## 结构化决策轨迹

检索增强主流程使用稳定、覆盖式的轨迹目录，不创建基于时间或 `run_id` 的子目录：

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

- `metadata.json` 保存实验身份信息和运行状态；运行级失败时额外保存精简的 `error={type,message}`。
- `events.jsonl` 保存逐样本的数据库 HRC、剪枝、重排以及表、列预测结果、诊断和异常；不保存运行时间、token 花费、HRC 的具体列、embedding、完整 prompt 或固定筛选参数。
- `summary.json` 保存数据库、表、列阶段指标和首次错误阶段汇总。
- `first_error_stage` 采用“最早中间退化”口径；第二轮召回指标只统计实际执行第二轮的样本。
- 当上游为 `global_coarse_rerank` 或 `global_coarse_pruning` 时，`table_to_column.py` 会校验并复制对应数据库事件，再追加 `table.prediction`、`column.prediction` 和样本最终事件；以上游 baseline 为输入时不创建结构化轨迹。
