Stage 1: Column-Level Index Construction（离线阶段）

为每个数据库构建列级向量索引：
每个列作为一个索引单元
context 包含：table name, column name, data type, description, sample values
生成 embedding 并存储（保留 metadata）

输出：全局 column vector space
==================================================================================
Stage 2: – Global Coarse Retrieval
Step 1: Global Highly Relevant Column Retrieval (HRC Retrieval)
将 query 编码为向量
在全局所有列（M 个）中计算相似度
取前R% 列

输出:
Highly Relevant Columns (HRC)
-------------------------------------------
Step 2: Support-Based Database Pruning
将 HRC 按数据库分组
统计每个数据库的列数 ni
应用 count-based threshold（实验确定）,去除尾部数据库

输出
Coarsely Filtered Candidate Databases (CFCD)
--------------------------------------------
Step 3: CFCD Reranking
对 CFCD 使用三种方法之一进行重排：

Method A: Prompt-based Binary Database Reranking
    yes/no 判别
    读取 logits → softmax 概率
Method B: LLM-based Listwise Database Reranking
    多数据库联合输入
    模型直接输出排序
Method C: Embedding-based Database Reranking
    构造 database-level context
    与 query 做 embedding similarity

Top-k Candidate Selection
根据重排结果取前K个数据库

输出：
Final Candidate Databases (FCD)
======================================================
Stage 3: Round 2 – Local Refinement
Step 1: Full Column Retrieval for FCD
调取 FCD 中每个数据库的全量列
新的候选空间 = k 个数据库的所有列
----------------------------------------
Step 2: Local Column Retrieval (Refined HRC Retrieval)
在该局部空间再次做 embedding 检索
重新得到 HRC（可记为 HRC²）

输出：
Refined Highly Relevant Columns (HRC²)
------------------------------------------
Step 3: Refined Database Reranking
在 FCD 内再次进行数据库重排（可复用 Step 4 方法）
取 top-1，Final Database Selection
输出：
Target Database (最终目标数据库)
