from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_QDRANT_PATH = PROJECT_ROOT / "Data" / "MMQA" / "qdrant_column_index"

DEFAULT_SCHEMA_DIR = PROJECT_ROOT / "Data" / "MMQA" / "Column_level_schema"

DEVICE = "cuda"

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

ANSWER_LLM_NAME = "Qwen/Qwen3.5-9B"

PROVIDER = "transformers"

CONTEXT_WINDOW_SIZE = 120000

# qwen3.5 Instruct (or non-thinking) mode for general tasks
LLM_SETTINGS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
}

# LLM_SETTINGS = {
#     "temperature": 0.7,
#     "top_p": 0.8,
#     "top_k": 20,
#     "min_p": 0.0,
#     "presence_penalty": 1.5,
#     "repetition_penalty": 1.0,
# }

THINKING_MODE = False
