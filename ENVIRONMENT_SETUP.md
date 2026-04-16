# 环境准备

本文档整理自 [`README.md`](README.md) 中的“环境准备”部分，便于单独查看和引用。

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
