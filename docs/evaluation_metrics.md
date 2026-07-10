# Evaluation Metrics

本文档定义了 7 个评估指标，用于衡量数据库预测、schema linking 与 SQL 生成任务的表现。

## 1. LA

`LA` 用于计算预测数据库的正确率。

对于每个样本，若模型预测的数据库与标准答案数据库完全一致，则记为正确；否则记为错误。`LA` 定义为所有样本中数据库预测正确的比例：

\[
LA = \frac{\text{数据库预测正确的样本数}}{\text{样本总数}}
\]

## 2. EM

`EM` 用于计算 schema linking 预测列的完全正确率，其中列的顺序不重要。

对于每个样本，将模型预测的列集合与标准答案列集合进行比较。若两个集合完全一致，则该样本记为正确；否则记为错误。`EM` 定义为所有样本中 schema linking 完全正确的比例：

\[
EM = \frac{\text{预测列集合与标准答案列集合完全一致的样本数}}{\text{样本总数}}
\]

## 3. Recall

`Recall` 用于计算 schema linking 预测列的 Micro Recall。

该指标从全体样本的角度统计模型召回了多少标准答案列。定义为所有样本中被正确预测出的标准答案列总数，占标准答案列总数的比例：

\[
Recall = \frac{\text{所有样本中命中的正确列总数}}{\text{所有样本中的标准答案列总数}}
\]

## 4. Avg_Ratio

`Avg_Ratio` 用于同时统计 schema linking 的平均预测列数量、平均标准答案列数量，以及预测列数量与标准答案列数量之间的平均比值。

首先分别计算所有样本的平均预测列数量与平均标准答案列数量：

\[
Avg\_Pred\_Cols = \frac{1}{N}\sum_{i=1}^{N} \text{第 } i \text{ 个样本的预测列数量}
\]

\[
Avg\_Gold\_Cols = \frac{1}{N}\sum_{i=1}^{N} \text{第 } i \text{ 个样本的标准答案列数量}
\]

对于每个样本，先计算：

\[
ratio_i = \frac{\text{第 } i \text{ 个样本的预测列数量}}{\text{第 } i \text{ 个样本的标准答案列数量}}
\]

然后对所有样本的该比值取平均：

\[
Avg\_Ratio = \frac{1}{N}\sum_{i=1}^{N} ratio_i
\]

其中，`Avg_Pred_Cols` 反映模型平均每个样本预测了多少个 schema linking 列，`Avg_Gold_Cols` 反映平均每个样本包含多少个标准答案列，`Avg_Ratio` 则用于反映模型预测的列数量相对于标准答案列数量是偏多、偏少还是大致相当。

## 5. Execution Accuracy (EX)

`Execution Accuracy (EX)` 用于衡量模型生成 SQL 的执行结果是否与标准答案 SQL 的执行结果一致。该指标遵循 Spider 与 BIRD benchmark 的执行评测思想：不直接比较 SQL 字符串，而是在对应样本的标准答案数据库上分别执行预测 SQL 与 gold SQL，并比较两者返回的结果。

对于第 \(i\) 个样本，设预测 SQL 的执行结果为 \(R_i^{pred}\)，gold SQL 的执行结果为 \(R_i^{gold}\)。若两条 SQL 都能成功执行，且返回的行和值一致，则该样本记为正确：

\[
EX_i =
\begin{cases}
1, & R_i^{pred} \equiv R_i^{gold} \\
0, & \text{otherwise}
\end{cases}
\]

总体 `EX` 定义为所有样本中执行结果一致的比例：

\[
EX = \frac{1}{N}\sum_{i=1}^{N} EX_i
\]

其中，\(N\) 为参与 SQL 执行评测的样本总数。执行结果比较时，SQL 写法可以不同，只要返回结果一致即可；若 gold SQL 中包含明确的 `ORDER BY`，则结果行顺序也视为语义的一部分，否则按无序结果集合进行比较。预测 SQL 发生语法错误、运行错误、超时，或返回结果与 gold SQL 不一致时，均记为 0。

## 6. Avg_token

`Avg_token` 用于计算平均每个样本完成数据库预测和 schema linking 预测所使用的 token 数量。

对于每个样本，统计其完成数据库预测与 schema linking 预测的总 token 消耗，然后对所有样本取平均：

\[
Avg\_token = \frac{\sum_{i=1}^{N} token_i}{N}
\]

其中，`token_i` 表示第 `i` 个样本完成数据库预测和 schema linking 预测所消耗的总 token 数。

## 7. Avg_time

`Avg_time` 用于计算平均每个样本完成数据库预测和 schema linking 预测所使用的时间。

对于每个样本，统计其完成数据库预测与 schema linking 预测的总耗时，然后对所有样本取平均：

\[
Avg\_time = \frac{\sum_{i=1}^{N} time_i}{N}
\]

其中，`time_i` 表示第 `i` 个样本完成数据库预测和 schema linking 预测所花费的总时间。
