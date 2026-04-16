import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from Llm.llm_loader import LLM
from config import *
from Run.logging_utils import log_run_configuration, setup_task_logger
from Utils.tools import normalize_response_text, render_prompt, resolve_hint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline database retrieval.")
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--answer-llm-name", dest="answer_llm_name", type=str, default=None)
    parser.add_argument("--provider", dest="provider", type=str, default=None)
    parser.add_argument(
        "--max-input-length",
        dest="max_input_length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-generation-num",
        dest="max_generation_num",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--database-schema-path",
        dest="database_schema_path",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def load_dataset(dataset_root: Path) -> pd.DataFrame:
    return pd.read_json(dataset_root / "gold_sl.json")


def load_database_schema(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_db_response(response_text:str) -> str:
    nor_response_text = normalize_response_text(response_text)
    try:
        response_json = json.loads(nor_response_text)
    except json.JSONDecodeError:
        return ""
    
    if not isinstance(response_json, dict):
        return ""
    
    try:
        pred_db_id = response_json["relevant_database"]
    except KeyError:
        return ""
    
    return pred_db_id
    

def append_log_entry(
    log_records: list[dict[str, Any]],
    row: pd.Series,
    response_text: str,
    answer_llm_name: str,
    provider: str,
    log_path: Path,
) -> None:
    predict_db_id = parse_db_response(response_text)
    log_records.append(
        {
            "model": answer_llm_name,
            "provider": provider,
            "id": f"{row['id']}",
            "spider_db_id": row["db_id"],
            "question": row["question"],
            "pre_db_response": response_text,
            "predict_db_id": predict_db_id
        }
    )
    log_path.write_text(json.dumps(log_records, ensure_ascii=False, indent=2), encoding="utf-8")


def database_schema_to_string(
    database_schemas: dict[str, str],
) -> str:
    schema_chunks = [database_schemas[key] for key in database_schemas.keys()]
    return ("\n" + "=" * 80 + "\n").join(schema_chunks).strip()


def run_baseline_retrieval(
    dataset_df: pd.DataFrame,
    dataset_name: str,
    documents_dir: Path,
    prompt_template: str,
    log_path: Path,
    database_schema_path: Path,
    ranking_llm: LLM,
    answer_llm_name: str,
    provider: str,
) -> int:
    database_schemas = load_database_schema(database_schema_path)
    log_records: list[dict[str, Any]] = []
    database_count = len(database_schemas)

    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        schemas_string = database_schema_to_string(database_schemas)
        prompt = render_prompt(
            prompt_template,
            DATABASE_SCHEMAS=schemas_string,
            QUESTION=row["question"],
            HINT=resolve_hint(
                row,
                dataset_name=dataset_name,
                documents_dir=documents_dir,
            ),
        )
        # prompt_token_count = ranking_llm.count_input_tokens(prompt)
        # print(f"[Baseline] id={row['id']} prompt_tokens={prompt_token_count}")
        response_text = ranking_llm.query(prompt)
        append_log_entry(
            log_records=log_records,
            row=row,
            response_text=response_text,
            answer_llm_name=answer_llm_name,
            provider=provider,
            log_path=log_path,
        )

    return database_count


def main() -> None:
    args = parse_args()

    dataset_name = args.dataset_name or DATASET_NAME
    answer_llm_name = args.answer_llm_name or ANSWER_LLM_NAME
    provider = args.provider or PROVIDER
    max_input_length = args.max_input_length or MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or MAX_GENERATEION_NUM

    dataset_root = PROJECT_ROOT / "Data" / dataset_name
    documents_dir = dataset_root / "documents"
    dataset_df = load_dataset(dataset_root)
    database_schema_path = args.database_schema_path or (dataset_root / "Database_schemas_summary.json")

    prompt_path = PROJECT_ROOT / "Templates" / "zero_shot" / "find_relevant_database_baseline.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()

    ranking_llm = LLM(
        model_name=answer_llm_name,
        provider=provider,
        max_input_length=max_input_length,
        max_generation_num=max_generation_num,
        query_settings=BASELINE_DATABASE_RETRIVAL_QUERY_SETTINGS,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = PROJECT_ROOT / "Logs" / answer_llm_name / "Database_Retrival"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"baseline_database_retrival_{dataset_name}_{run_id}.json"
    logger, logger_path = setup_task_logger("baseline_database_retrival", log_path)

    log_run_configuration(
        logger,
        task_name="Baseline Database Retrieval",
        dataset_name=dataset_name,
        data_count=len(dataset_df),
        model_name=answer_llm_name,
        provider=provider,
        result_path=log_path,
        extra_fields={
            "Prompt template": prompt_path,
            "Database schema path": database_schema_path,
            "Documents dir": documents_dir,
            "Max input length": max_input_length,
            "Max generation num": max_generation_num,
            "Logger path": logger_path,
        },
    )

    database_count = run_baseline_retrieval(
        dataset_df=dataset_df,
        dataset_name=dataset_name,
        documents_dir=documents_dir,
        prompt_template=prompt_template,
        log_path=log_path,
        database_schema_path=database_schema_path,
        ranking_llm=ranking_llm,
        answer_llm_name=answer_llm_name,
        provider=provider,
    )
    logger.info("Loaded database schema count: %s", database_count)
    logger.info("Completed %s records.", len(dataset_df))


if __name__ == "__main__":
    main()
