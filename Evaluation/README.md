# Evaluation

本目录提供两个独立评估入口：

- `evaluate_metrics.py`：计算 `LA`、`EM`、Micro `Recall`、`Avg_Ratio`、
  `Avg_token` 和 `Avg_time`，同时报告平均预测/gold 列数及覆盖率。
- `evaluate_ex.py`：在 gold 数据库执行预测 SQL 和 gold SQL，计算 `EX`。

两个入口都支持以下两种模式。所有相对路径均相对于项目根目录解析。

## 1. 按数据集批量计算所有模型

只指定数据集即可。脚本会发现该数据集下所有非空的统一预测文件：

```bash
python -m Evaluation.evaluate_metrics --dataset-name MMQA
python -m Evaluation.evaluate_ex --dataset-name MMQA
```

默认输出为：

```text
results/evaluation/<dataset>/metrics.csv
results/evaluation/<dataset>/ex.csv
```

若没有匹配的预测文件，脚本会提示跳过、写出仅含表头的 CSV，并以状态码 0
正常结束。已初始化但 `predictions` 为空的文件也默认跳过；非 EX 脚本可用
`--include-empty` 显式纳入。

## 2. 计算指定实验

非 EX 指标中，一个实验由数据集、SL 方法和 SL 模型唯一定位：

```bash
python -m Evaluation.evaluate_metrics \
  --dataset-name MMQA \
  --method table_to_column \
  --model-name Qwen/Qwen2.5-14B-Instruct
```

SQL 结果还包含 SQL 方法/模型和上游 SL 方法/模型，因此精确过滤写法为：

```bash
python -m Evaluation.evaluate_ex \
  --dataset-name MMQA \
  --sql-method one_shot \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --sl-method table_to_column \
  --sl-model-name Qwen/Qwen2.5-14B-Instruct
```

也可以直接指定唯一文件，这对于脚本化调用最简单：

```bash
python -m Evaluation.evaluate_metrics \
  --dataset-name MMQA \
  --prediction-path results/sl/table_to_column/MMQA/Qwen__Qwen2.5-14B-Instruct/prediction.json

python -m Evaluation.evaluate_ex \
  --dataset-name MMQA \
  --prediction-path results/sql/one_shot/MMQA/Qwen__Qwen2.5-14B-Instruct/table_to_column/Qwen__Qwen2.5-14B-Instruct.json
```

模型过滤同时接受原始模型名中的 `/` 和结果路径使用的 `__`。

## 计分与异常语义

- `LA` 要求预测数据库列表恰好只包含 gold `db_id`。
- `EM` 比较 `(db_id, table, column)` 集合，列顺序不影响结果。
- `Recall` 是全数据集命中 gold 列数除以 gold 列总数的 Micro Recall。
- Spider2 gold 的 `DB.SCHEMA.TABLE` 会去掉最外层 DB 后再比较；统一预测中的
  `SCHEMA.TABLE` 已是 canonical 表名，不会再次去掉与 `db_id` 同名的 schema。
  对旧产物中的裸 `TABLE`，默认仅在 `db_info.json` 的预测数据库内存在唯一匹配
  时解析为 canonical 表名；多义或跨数据库名称不会猜测。可传
  `--strict-spider2-table-names` 关闭该兼容解析。
- 质量指标和 `Avg_Ratio` 以完整 gold 集合为分母；文件内缺少的样本按空预测
  处理，并由 `coverage` 暴露完整度。
- `Avg_token`、`Avg_time` 只对 `linking_total` 中有真实数值的记录求平均；
  `token_samples`、`time_samples` 给出各自有效样本数。这样不会把缺失 usage
  错当成 0。
- EX 忽略返回列名。有 `ORDER BY` 的 gold SQL 按行顺序比较；否则使用保留重复
  次数的无序行集合比较。`NULL`、数值、文本、二进制和日期类型分别规范化。
- SQL 生成失败、执行失败、gold 执行失败、结果过大截断和结果不一致均不会被
  计为正确，并在 CSV 中分别统计。EX 的分母始终是 gold SQL 样本数。
- MMQA/BIRD 默认使用项目已有 SQLite 数据库。Spider2 需要 Snowflake connector
  和凭据，可传 `--snowflake-credential-path`，也可使用项目已有环境变量配置。
- `--sample-limit N` 仅用于 EX 冒烟测试；正式汇报时不要使用。

查看全部参数：

```bash
python -m Evaluation.evaluate_metrics --help
python -m Evaluation.evaluate_ex --help
```

也可以在 Bash 文件顶部直接修改数据集、模型和方法配置，然后运行文件。模型和
方法留空表示计算该数据集下的全部匹配结果。两个文件内都包含 MMQA 与 Spider2
配置示例，运行时不需要传入参数：

```bash
bash Evaluation/run_metrics.sh
bash Evaluation/run_ex.sh
```
