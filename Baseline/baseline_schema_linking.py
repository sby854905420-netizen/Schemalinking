from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import pandas as pd


from config import *
from Llm.llm_loader import LLM, resolve_provider
from Utils.json_utils import atomic_write_json, load_json_records
from Utils.prediction_adapter import build_prediction_from_native
from Utils.prediction_store import (
    build_prediction_path,
    initialize_prediction_file,
    upsert_prediction,
)
from Utils.value_utils import index_records_by_id
from Utils.logging_utils import log_run_configuration, setup_task_logger
from Utils.efficiency_utils import SampleEfficiencyTracker
from Utils.render_tools import SchemaTextRenderer
from Utils.schema_selection import DbInfoSchemaStore
from Utils.schema_prediction_utils import normalize_predicted_columns
from Utils.value_utils import get_row_value
from Utils.tools import (
    load_db_info_index,
    render_prompt,
    resolve_hint,
    resolve_input_path,
    resolve_output_path,
    resolve_supported_method,
)

SUPPORTED_METHODS = {"zero_shot", "few_shot"}
INPUT_FILE_PATTERNS = (
    "baseline_database_retrieval_{dataset_name}_{timestamp}.json",
    "iterative_database_retrieval_{dataset_name}_{timestamp}.json",
    "baseline_database_retrival_{dataset_name}_{timestamp}.json",
    "iterative_database_retrival_{dataset_name}_{timestamp}.json",
)
TIMESTAMP_PATTERN_TEMPLATE = (
    r"(?:baseline_database_retrieval|iterative_database_retrieval|"
    r"baseline_database_retrival|iterative_database_retrival)_{dataset_name}_"
    r"(\d{{8}}_\d{{6}})\.json$"
)
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
    parser.add_argument("--logs-dir", dest="logs_dir", type=Path, default=None)
    parser.add_argument("--db-info-path", dest="db_info_path", type=Path, default=None)
    parser.add_argument("--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--prediction-path", dest="prediction_path", type=Path, default=None)
    return parser.parse_args()


def load_dataset(input_path: Path):

    return pd.read_json(input_path)


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def parse_sl_response(response: str) -> dict[str, list[str]]:
    return normalize_predicted_columns(response)

def append_log_entry(
    log_records: list[dict[str, Any]],
    row: Any,
    response_text: str,
    efficiency_tracker: SampleEfficiencyTracker,
    answer_llm_name: str,
    provider: str,
    output_path: Path,
    prediction_path: Path | None = None,
    source_record: dict[str, Any] | None = None,
    dataset_name: str | None = None,
    documents_dir: Path | None = None,
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
    log_records.append(record)
    atomic_write_json(output_path, log_records)
    if prediction_path is not None:
        if dataset_name is None or documents_dir is None:
            raise ValueError("Unified prediction context is incomplete.")
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
    output_path: Path,
    schema_store: DbInfoSchemaStore,
    answer_llm: Any,
    answer_llm_name: str,
    provider: str,
    prediction_path: Path | None = None,
    source_index: dict[str, dict[str, Any]] | None = None,
) -> int:
    from tqdm import tqdm

    log_records: list[dict[str, Any]] = []

    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        efficiency_tracker = SampleEfficiencyTracker()
        sample_id = get_row_value(row, "id", "instance_id")
        source_record = (source_index or {}).get(str(sample_id), {})
        predict_db_id = get_row_value(row, "predict_db_id")
        if predict_db_id is None or str(predict_db_id).strip() == "":
            append_log_entry(
                log_records=log_records,
                row=row,
                response_text="No Valid Database.",
                efficiency_tracker=efficiency_tracker,
                answer_llm_name=answer_llm_name,
                provider=provider,
                output_path=output_path,
                prediction_path=prediction_path,
                source_record=source_record,
                dataset_name=dataset_name,
                documents_dir=documents_dir,
            )
            continue
        predict_db_id = str(predict_db_id)
        try:
            column_records = schema_store.get_column_records(predict_db_id)
            if not column_records:
                append_log_entry(
                    log_records=log_records,
                    row=row,
                    response_text="No Valid Database.",
                    efficiency_tracker=efficiency_tracker,
                    answer_llm_name=answer_llm_name,
                    provider=provider,
                    output_path=output_path,
                    prediction_path=prediction_path,
                    source_record=source_record,
                    dataset_name=dataset_name,
                    documents_dir=documents_dir,
                )
                continue
            database_schema = schema_store.get_full_schema_text(predict_db_id)
            prompt = render_prompt(
                prompt_template,
                DATABASE_SCHEMAS=database_schema,
                QUESTION=row["question"],
                HINT=resolve_hint(
                    row,
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

        append_log_entry(
            log_records=log_records,
            row=row,
            response_text=response_text,
            efficiency_tracker=efficiency_tracker,
            answer_llm_name=answer_llm_name,
            provider=provider,
            output_path=output_path,
            prediction_path=prediction_path,
            source_record=source_record,
            dataset_name=dataset_name,
            documents_dir=documents_dir,
            error_message=error_message,
        )
    return len(log_records)


def main() -> None:
    args = parse_args()

    method_name = resolve_supported_method(
        args.method, default=DEFAULT_METHOD, supported=SUPPORTED_METHODS
    )
    dataset_name = args.dataset_name or DATASET_NAME
    answer_llm_name = args.answer_llm_name or ANSWER_LLM_NAME
    provider = resolve_provider(args.provider or PROVIDER)
    max_input_length = args.max_input_length or MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or MAX_GENERATEION_NUM

    current_dataset_root = dataset_root(dataset_name)
    documents_dir = current_dataset_root / "documents"
    logs_dir = resolve_project_path(args.logs_dir) if args.logs_dir else LOGS_ROOT
    prompt_path = TEMPLATES_ROOT / method_name / "baseline_schema_linking.txt"
    db_info_path = (
        resolve_project_path(args.db_info_path)
        if args.db_info_path
        else current_dataset_root / "db_info.json"
    )
    input_path = resolve_input_path(
        input_path=resolve_project_path(args.input_path) if args.input_path else None,
        logs_dir=logs_dir,
        answer_llm_name=answer_llm_name,
        dataset_name=dataset_name,
        input_file_patterns=INPUT_FILE_PATTERNS,
        timestamp_pattern_template=TIMESTAMP_PATTERN_TEMPLATE,
    )
    output_path = resolve_output_path(
        output_path=resolve_project_path(args.output_path) if args.output_path else None,
        answer_llm_name=answer_llm_name,
        dataset_name=dataset_name,
        output_stem=f"{method_name}_baseline_schema_linking",
        project_root=PROJECT_ROOT,
    )
    logger, logger_path = setup_task_logger("baseline_schema_linking", output_path)

    dataset_df = load_dataset(input_path)
    source_records = load_json_records(current_dataset_root / "gold_sl.json")
    source_index = {
        key: dict(value) for key, value in index_records_by_id(source_records).items()
    }
    prompt_template = load_prompt_template(prompt_path)
    db_info_index = load_db_info_index(db_info_path)
    database_model_name = (
        str(dataset_df.iloc[0].get("model") or answer_llm_name)
        if len(dataset_df)
        else answer_llm_name
    )
    prediction_path = (
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

    log_run_configuration(
        logger,
        task_name="Baseline Schema Linking",
        dataset_name=dataset_name,
        data_count=len(dataset_df),
        model_name=answer_llm_name,
        provider=provider,
        result_path=output_path,
        extra_fields={
            "Method": method_name,
            "Input path": input_path,
            "Prompt template": prompt_path,
            "DB info path": db_info_path,
            "Documents dir": documents_dir,
            "Max input length": max_input_length,
            "Max generation num": max_generation_num,
            "Logger path": logger_path,
            "Unified prediction path": prediction_path,
        },
    )

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
        output_path=output_path,
        schema_store=schema_store,
        answer_llm=answer_llm,
        answer_llm_name=answer_llm_name,
        provider=provider,
        prediction_path=prediction_path,
        source_index=source_index,
    )
    logger.info("Completed %s records.", processed_count)


if __name__ == "__main__":
    main()
