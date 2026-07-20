from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import pandas as pd


from config import *
from Llm.llm_loader import LLM, resolve_provider
from Utils.json_utils import load_json_records
from Utils.prediction_adapter import build_prediction_from_native
from Utils.prediction_store import (
    build_prediction_path,
    initialize_prediction_file,
    replace_predictions,
    upsert_prediction,
)
from Utils.value_utils import (
    choose_external_knowledge_source,
    get_row_value,
    index_records_by_id,
)
from Utils.efficiency_utils import SampleEfficiencyTracker
from Utils.database_prediction_store import (
    DATABASE_METHODS,
    load_database_prediction_records,
    resolve_database_prediction_input,
)
from Utils.render_tools import SchemaTextRenderer
from Utils.schema_selection import DbInfoSchemaStore
from Utils.schema_prediction_utils import normalize_predicted_columns
from Utils.tools import (
    load_db_info_index,
    render_prompt,
    resolve_hint,
    resolve_supported_method,
)
from Utils.artifact_paths import require_results_output

SUPPORTED_METHODS = {"zero_shot", "few_shot"}
DEFAULT_METHOD = "few_shot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline schema linking.")
    parser.add_argument("--method", dest="method", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--answer-llm-name", dest="answer_llm_name", type=str, default=None)
    parser.add_argument("--provider", dest="provider", type=str, default=None)
    parser.add_argument("--max-input-length", dest="max_input_length", type=int, default=None)
    parser.add_argument("--max-generation-num", dest="max_generation_num", type=int, default=None)
    parser.add_argument("--input-path", dest="input_path", type=Path, default=None)
    parser.add_argument(
        "--database-method",
        choices=sorted(DATABASE_METHODS),
        default="baseline",
    )
    parser.add_argument("--database-model-name", type=str, default=None)
    parser.add_argument("--db-info-path", dest="db_info_path", type=Path, default=None)
    parser.add_argument("--prediction-path", dest="prediction_path", type=Path, default=None)
    return parser.parse_args()


def load_dataset(input_path: Path):
    return pd.DataFrame(load_database_prediction_records(input_path))


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def parse_sl_response(response: str) -> dict[str, list[str]]:
    return normalize_predicted_columns(response)

def save_prediction(
    row: Any,
    response_text: str,
    efficiency_tracker: SampleEfficiencyTracker,
    answer_llm_name: str,
    provider: str,
    prediction_path: Path,
    dataset_name: str,
    documents_dir: Path,
    source_record: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    predict_columns = parse_sl_response(response_text)
    efficiency = efficiency_tracker.finalize()
    record = {
            "model": answer_llm_name,
            "provider": provider,
            "id": f"{get_row_value(row, 'id', 'instance_id')}",
            "question": row["question"],
            "spider_db_id": get_row_value(row, "spider_db_id"),
            "predict_db_id": get_row_value(row, "predict_db_id"),
            "predict_columns_text": response_text,
            "predict_columns": predict_columns,
            "efficiency": efficiency,
        }
    if error_message:
        record["schema_error"] = error_message
    prediction = build_prediction_from_native(
        schema_record=record,
        database_record=row,
        source_record=source_record or row,
        dataset_name=dataset_name,
        method="prompt_baseline",
        documents_dir=documents_dir,
    )
    upsert_prediction(prediction_path, prediction)
    return record


def run_baseline_schema_linking(
    dataset_df: Any,
    dataset_name: str,
    documents_dir: Path,
    prompt_template: str,
    schema_store: DbInfoSchemaStore,
    answer_llm: Any,
    answer_llm_name: str,
    provider: str,
    prediction_path: Path,
    source_index: dict[str, dict[str, Any]] | None = None,
) -> int:
    from tqdm import tqdm

    processed_count = 0

    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        efficiency_tracker = SampleEfficiencyTracker()
        sample_id = get_row_value(row, "id", "instance_id")
        source_record = (source_index or {}).get(str(sample_id), {})
        predict_db_id = get_row_value(row, "predict_db_id")
        if predict_db_id is None or str(predict_db_id).strip() == "":
            save_prediction(
                row=row,
                response_text="No Valid Database.",
                efficiency_tracker=efficiency_tracker,
                answer_llm_name=answer_llm_name,
                provider=provider,
                prediction_path=prediction_path,
                source_record=source_record,
                dataset_name=dataset_name,
                documents_dir=documents_dir,
            )
            processed_count += 1
            continue
        predict_db_id = str(predict_db_id)
        try:
            column_records = schema_store.get_column_records(predict_db_id)
            if not column_records:
                save_prediction(
                    row=row,
                    response_text="No Valid Database.",
                    efficiency_tracker=efficiency_tracker,
                    answer_llm_name=answer_llm_name,
                    provider=provider,
                    prediction_path=prediction_path,
                    source_record=source_record,
                    dataset_name=dataset_name,
                    documents_dir=documents_dir,
                )
                processed_count += 1
                continue
            database_schema = schema_store.get_full_schema_text(predict_db_id)
            prompt = render_prompt(
                prompt_template,
                DATABASE_SCHEMAS=database_schema,
                QUESTION=row["question"],
                HINT=resolve_hint(
                    choose_external_knowledge_source(source_record, row),
                    dataset_name=dataset_name,
                    documents_dir=documents_dir,
                ),
            )
            response_text, total_tokens = answer_llm.query_with_usage(prompt)
            efficiency_tracker.add_llm_total_tokens(total_tokens)
            error_message = None
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            response_text = error_message

        save_prediction(
            row=row,
            response_text=response_text,
            efficiency_tracker=efficiency_tracker,
            answer_llm_name=answer_llm_name,
            provider=provider,
            prediction_path=prediction_path,
            source_record=source_record,
            dataset_name=dataset_name,
            documents_dir=documents_dir,
            error_message=error_message,
        )
        processed_count += 1
    return processed_count


def main() -> None:
    args = parse_args()

    method_name = resolve_supported_method(
        args.method, default=DEFAULT_METHOD, supported=SUPPORTED_METHODS
    )
    dataset_name = args.dataset_name or DATASET_NAME
    answer_llm_name = args.answer_llm_name or ANSWER_LLM_NAME
    database_model_name = args.database_model_name or answer_llm_name
    provider = resolve_provider(args.provider or PROVIDER)
    max_input_length = args.max_input_length or MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or MAX_GENERATEION_NUM

    current_dataset_root = dataset_root(dataset_name)
    documents_dir = current_dataset_root / "documents"
    prompt_path = TEMPLATES_ROOT / method_name / "baseline_schema_linking.txt"
    db_info_path = (
        resolve_project_path(args.db_info_path)
        if args.db_info_path
        else current_dataset_root / "db_info.json"
    )
    input_path = resolve_database_prediction_input(
        explicit_path=(
            resolve_project_path(args.input_path) if args.input_path else None
        ),
        method=args.database_method,
        dataset_name=dataset_name,
        model_name=database_model_name,
    )
    dataset_df = load_dataset(input_path)
    source_records = load_json_records(current_dataset_root / "gold_sl.json")
    source_index = {
        key: dict(value) for key, value in index_records_by_id(source_records).items()
    }
    prompt_template = load_prompt_template(prompt_path)
    db_info_index = load_db_info_index(db_info_path)
    database_model_name = (
        str(dataset_df.iloc[0].get("model") or database_model_name)
        if len(dataset_df)
        else database_model_name
    )
    prediction_path = require_results_output(
        resolve_project_path(args.prediction_path)
        if args.prediction_path
        else build_prediction_path("prompt_baseline", dataset_name, answer_llm_name)
    )
    initialize_prediction_file(
        prediction_path,
        dataset_name=dataset_name,
        method="prompt_baseline",
        database_selection_model_name=database_model_name,
        schema_linking_model_name=answer_llm_name,
    )
    replace_predictions(prediction_path, [])

    answer_llm = LLM(
        model_name=answer_llm_name,
        provider=provider,
        max_input_length=max_input_length,
        max_generation_num=max_generation_num,
        query_settings=BASELINE_SCHEMA_LINKING_QUERY_SETTINGS,
    )
    renderer = SchemaTextRenderer(tokenizer=answer_llm.tokenizer)
    schema_store = DbInfoSchemaStore(
        db_info_index=db_info_index,
        renderer=renderer,
    )

    processed_count = run_baseline_schema_linking(
        dataset_df=dataset_df,
        dataset_name=dataset_name,
        documents_dir=documents_dir,
        prompt_template=prompt_template,
        schema_store=schema_store,
        answer_llm=answer_llm,
        answer_llm_name=answer_llm_name,
        provider=provider,
        prediction_path=prediction_path,
        source_index=source_index,
    )
    print(f"Completed {processed_count} prompt-baseline schema-linking records.")


if __name__ == "__main__":
    main()
