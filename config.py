from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_NAME = "MMQA"

DEFAULT_QDRANT_PATH = PROJECT_ROOT / "Data" / DATASET_NAME / "qdrant_column_index"

DEFAULT_SCHEMA_DIR = PROJECT_ROOT / "Data" / DATASET_NAME / "Column_level_schema"

DEVICE = "cuda"

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

ANSWER_LLM_NAME = "mistralai/Ministral-3-8B-Instruct-2512"

PROVIDER = "transformers"

MAX_GENERATEION_NUM = 2048

MAX_INPUT_LENGTH = 110000

TOP_KD_CAP = 512


CANDIDATE_DB_TOP_K = 3

BASELINE_DATABASE_RETRIVAL_QUERY_SETTINGS = {
    "temperature": 0.0,
    "repetition_penalty": 1.0,
}

BASELINE_SCHEMA_LINKING_QUERY_SETTINGS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "repetition_penalty": 1.02,
}

TABLE2COLUMN_QUERY_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
}

THINKING_MODE = False

