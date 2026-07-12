# Project Structure

项目按“实验入口、复用实现、静态资源、运行产物”分层：

```text
Schemalinking/
├── Baseline/                  # 可独立运行的 baseline 实验
├── Indexing/                  # 离线索引构建命令
├── Run/                       # 主流程、结果转换入口与 SQL Agent 实现
│   └── spider_agent_tc/       # SQL Agent 内部组件和执行器
├── Llm/                       # LLM 与 embedding 模型加载器
├── Utils/                     # 预测契约、结果存储及其他共享工具
├── Templates/                 # Prompt 静态资源
│   ├── few_shot/
│   ├── zero_shot/
│   └── sql_generation/
├── Data/                      # 数据集、SQLite 库、Qdrant 索引（不提交）
├── Logs/                      # 仅保存运行期 .log 文本（不提交）
├── results/                   # 统一 DB、SL 与 SQL 预测（不提交）
├── tests/                     # 单元测试与小型 fixture
├── docs/                      # 设计和评估文档
├── config.py                  # 配置值与所有项目根路径
└── README.md                  # 安装、运行和参数说明
```

## 文件放置规则

- 新的对照方法或消融实验放在 `Baseline/`，文件名使用小写 snake_case。
- 主实验的可运行阶段放在 `Run/`；只服务于某个复杂入口的实现放在该入口的子包中，例如 `Run/spider_agent_tc/`。
- 数据预处理或索引构建等离线命令放在 `Indexing/`。
- 模型加载、provider 适配放在 `Llm/`，通用格式化、路径查找和日志函数放在 `Utils/`。
- Prompt 不写死在 Python 中，按任务放到 `Templates/<task>/`。
- 小型测试数据放在 `tests/fixtures/`；真实数据、数据库和向量索引只放在 `Data/`。
- 运行期文本日志放在 `Logs/`，所有 DB、SL、SQL 预测放在 `results/`，文档放在 `docs/`。
- DB、SL、SQL 预测契约分别放在 `Utils/database_prediction_store.py`、`Utils/prediction_store.py` 与 `Utils/sql_prediction_store.py`。
- 不在业务模块里重复拼接 `PROJECT_ROOT / "Data"` 等路径；新增根路径先在 `config.py` 定义。

## 路径兼容策略

所有默认路径由 `config.py` 中的 `DATA_ROOT`、`TEMPLATES_ROOT`、`LOGS_ROOT`、`RESULTS_ROOT` 和 `MODEL_CACHE_ROOT` 派生。CLI 传入的相对路径通过 `resolve_project_path()` 相对于项目根目录解析，避免切换工作目录后读取错误。

运行日志使用：

```text
Logs/<model>/<task>/<task>_<dataset>_<timestamp>.log
```

`Logs/` 不保存预测 JSON，也不作为任何预测阶段的输入。数据库预测固定使用：

```text
results/db/<db_method>/<dataset>/<db_model>/prediction.json
```

Schema Linking 只读取上述统一 DB prediction，或读取用户通过 `--input-path` 显式提供的同格式文件；不会扫描 `Logs/`。
