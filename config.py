"""Project-wide settings and filesystem locations.

All repository-owned paths are derived from this file so commands do not
depend on the caller's current working directory.
"""

from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Data"
TEMPLATES_ROOT = PROJECT_ROOT / "Templates"
LOGS_ROOT = PROJECT_ROOT / "Logs"
RESULTS_ROOT = PROJECT_ROOT / "results"
MODEL_CACHE_ROOT = PROJECT_ROOT / "Llm" / "cache"
OPENAI_CREDENTIAL_PATH = PROJECT_ROOT / "gpt_credential.json"
SNOWFLAKE_CREDENTIAL_PATH = PROJECT_ROOT / "snowflake_credential.json"



def dataset_root(dataset_name: str) -> Path:
    """Return the repository data directory for one dataset."""

    return DATA_ROOT / dataset_name


def resolve_project_path(path: Union[str, Path]) -> Path:
    """Resolve a CLI path relative to the repository root, not the shell cwd."""

    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

DATASET_NAME = "MMQA"

DEFAULT_QDRANT_PATH = dataset_root(DATASET_NAME) / "qdrant_column_index"

DEFAULT_DB_INFO_PATH = dataset_root(DATASET_NAME) / "db_info.json"

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
