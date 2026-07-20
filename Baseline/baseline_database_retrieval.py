import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from Llm.llm_loader import LLM, resolve_provider
from config import *
from Utils.json_utils import normalize_response_text
from Utils.efficiency_utils import SampleEfficiencyTracker
from Utils.database_prediction_store import (
    build_database_prediction,
    build_database_prediction_path,
    initialize_database_prediction_file,
    replace_database_predictions,
    upsert_database_prediction,
)
from Utils.tools import render_prompt, resolve_hint
from Utils.artifact_paths import require_results_output


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
    parser.add_argument("--prediction-path", type=Path, default=None)
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


def save_prediction(
    row: pd.Series,
    response_text: str,
    efficiency_tracker: SampleEfficiencyTracker,
    prediction_path: Path,
) -> None:
    predict_db_id = parse_db_response(response_text)
    efficiency = efficiency_tracker.finalize()
    upsert_database_prediction(
        prediction_path,
        build_database_prediction(
            sample_id=row["id"],
            question=row["question"],
            predicted_db_id=predict_db_id,
            efficiency=efficiency,
        ),
    )


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
    database_schema_path: Path,
    ranking_llm: LLM,
    prediction_path: Path,
) -> int:
    database_schemas = load_database_schema(database_schema_path)
    database_count = len(database_schemas)

    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        efficiency_tracker = SampleEfficiencyTracker()
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
        response_text, total_tokens = ranking_llm.query_with_usage(prompt)
        efficiency_tracker.add_llm_total_tokens(total_tokens)
        save_prediction(
            row=row,
            response_text=response_text,
            efficiency_tracker=efficiency_tracker,
            prediction_path=prediction_path,
        )

    return database_count


def main() -> None:
    args = parse_args()

    dataset_name = args.dataset_name or DATASET_NAME
    answer_llm_name = args.answer_llm_name or ANSWER_LLM_NAME
    provider = resolve_provider(args.provider or PROVIDER)
    max_input_length = args.max_input_length or MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or MAX_GENERATEION_NUM

    current_dataset_root = dataset_root(dataset_name)
    documents_dir = current_dataset_root / "documents"
    dataset_df = load_dataset(current_dataset_root)
    database_schema_path = (
        resolve_project_path(args.database_schema_path)
        if args.database_schema_path
        else current_dataset_root / "Database_schemas_summary.json"
    )

    prompt_path = TEMPLATES_ROOT / "zero_shot" / "find_relevant_database_baseline.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()

    ranking_llm = LLM(
        model_name=answer_llm_name,
        provider=provider,
        max_input_length=max_input_length,
        max_generation_num=max_generation_num,
        query_settings=BASELINE_DATABASE_RETRIVAL_QUERY_SETTINGS,
    )

    prediction_path = require_results_output(
        resolve_project_path(args.prediction_path)
        if args.prediction_path
        else build_database_prediction_path("baseline", dataset_name, answer_llm_name)
    )
    initialize_database_prediction_file(
        prediction_path,
        dataset_name=dataset_name,
        method="baseline",
        model_name=answer_llm_name,
    )
    replace_database_predictions(prediction_path, [])

    database_count = run_baseline_retrieval(
        dataset_df=dataset_df,
        dataset_name=dataset_name,
        documents_dir=documents_dir,
        prompt_template=prompt_template,
        database_schema_path=database_schema_path,
        ranking_llm=ranking_llm,
        prediction_path=prediction_path,
    )
    print(
        f"Completed {len(dataset_df)} baseline database records "
        f"across {database_count} schemas."
    )


if __name__ == "__main__":
    main()
