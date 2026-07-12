from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

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
from Utils.value_utils import index_records_by_id
from Utils.logging_utils import build_run_log_path, log_run_configuration, setup_task_logger
from Utils.efficiency_utils import SampleEfficiencyTracker
from Utils.database_prediction_store import (
    DATABASE_METHODS,
    load_database_prediction_records,
    resolve_database_prediction_input,
)
from Utils.render_tools import SchemaTextRenderer
from Utils.schema_selection import (
    DbInfoSchemaStore,
    count_prompt_tokens as count_schema_prompt_tokens,
    load_db_counts,
    resolve_schema_text_for_db,
)
from Utils.schema_prediction_utils import (
    normalize_predicted_columns,
    normalize_predicted_tables,
)
from Utils.value_utils import get_row_value
from Utils.tools import (
    get_qdrant_client,
    load_db_info_index,
    render_prompt,
    resolve_hint,
    resolve_prompt_token_cap,
    resolve_supported_method,
)
from Utils.artifact_paths import require_results_output

SUPPORTED_METHODS = {"zero_shot", "few_shot"}
DEFAULT_METHOD = "few_shot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table-to-column schema linking.")
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
        default="global_coarse_rerank",
    )
    parser.add_argument("--database-model-name", type=str, default=None)
    parser.add_argument("--db-info-path", dest="db_info_path", type=Path, default=None)
    parser.add_argument("--qdrant-path", dest="qdrant_path", type=Path, default=None)
    parser.add_argument("--prediction-path", dest="prediction_path", type=Path, default=None)
    parser.add_argument(
        "--disable-table-filtering",
        dest="disable_table_filtering",
        action="store_true",
        help="Skip table prediction and run column linking over the predicted database schema.",
    )
    return parser.parse_args()


def load_dataset(input_path: Path) -> pd.DataFrame:
    return pd.DataFrame(load_database_prediction_records(input_path))


def load_prompt_templates(method_name: str) -> dict[str, str]:
    prompt_dir = TEMPLATES_ROOT / method_name
    return {
        "table": (prompt_dir / "extract_relevant_tables.txt").read_text(encoding="utf-8").strip(),
        "column": (prompt_dir / "extract_relevant_columns.txt").read_text(encoding="utf-8").strip(),
    }


def load_qdrant_collection_name(qdrant_path: Path) -> str:
    meta_path = qdrant_path / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Could not find Qdrant metadata file at {meta_path}.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    collections = meta.get("collections")
    if not isinstance(collections, dict) or not collections:
        raise ValueError(f"Could not resolve a collection name from {meta_path}.")

    return next(iter(collections))


def filter_column_records_by_tables(
    column_records: Sequence[dict[str, Any]],
    table_names: Sequence[str],
) -> list[dict[str, Any]]:
    selected_table_names = set(normalize_predicted_tables(table_names))
    if not selected_table_names:
        return []

    return [
        record
        for record in column_records
        if str(record.get("table_name", "")).strip() in selected_table_names
    ]


def normalize_relevant_tables(
    relevant_table_list: Sequence[Any],
    available_records: Sequence[dict[str, Any]],
) -> list[str]:
    available_table_names = {
        str(record.get("table_name", "")).strip()
        for record in available_records
        if str(record.get("table_name", "")).strip()
    }

    return [
        table_name
        for table_name in normalize_predicted_tables(relevant_table_list)
        if table_name in available_table_names
    ]


def render_schema_prompt(
    prompt_template: str,
    schema_text: str,
    question: str,
    hint: str,
) -> str:
    return render_prompt(
        prompt_template,
        DATABASE_SCHEMAS=schema_text,
        QUESTION=question,
        HINT=hint,
    )


def resolve_table_prompt_schema(
    predict_db_id: str,
    question: str,
    hint: str,
    prompt_template: str,
    answer_llm: Any,
    embedder: Any,
    schema_store: DbInfoSchemaStore,
    qdrant_client: Any,
    qdrant_collection_name: str,
    db_counts: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    query_vector = embedder.encode(question, convert_to_list=True)
    return resolve_schema_text_for_db(
        query=question,
        query_vector=query_vector,
        db_id=predict_db_id,
        ranking_llm=answer_llm,
        prompt_template=prompt_template,
        hint_text=hint,
        schema_store=schema_store,
        qdrant_client=qdrant_client,
        collection_name=qdrant_collection_name,
        db_counts=db_counts,
    )


def resolve_column_prompt_records(
    predict_db_id: str,
    question: str,
    hint: str,
    relevant_table_list: Sequence[str],
    table_prompt_records: Sequence[dict[str, Any]],
    schema_store: DbInfoSchemaStore,
    answer_llm: Any,
    prompt_template: str,
) -> list[dict[str, Any]]:
    if not relevant_table_list:
        return list(table_prompt_records)

    full_db_records = schema_store.get_column_records(predict_db_id)
    selected_table_records = filter_column_records_by_tables(full_db_records, relevant_table_list)
    if not selected_table_records:
        fallback_records = filter_column_records_by_tables(table_prompt_records, relevant_table_list)
        return fallback_records or list(table_prompt_records)

    full_schema_text = schema_store.render_schema_text(predict_db_id, selected_table_records)
    prompt_tokens = count_schema_prompt_tokens(
        ranking_llm=answer_llm,
        prompt_template=prompt_template,
        schema_text=full_schema_text,
        query=question,
        hint_text=hint,
    )
    if prompt_tokens <= resolve_prompt_token_cap(answer_llm.max_input_length):
        return selected_table_records

    fallback_records = filter_column_records_by_tables(table_prompt_records, relevant_table_list)
    return fallback_records or list(table_prompt_records) or selected_table_records


def parse_table_response(response_text: str) -> list[str]:
    return normalize_predicted_tables(response_text)


def parse_column_response(response_text: str) -> dict[str, list[str]]:
    return normalize_predicted_columns(response_text)


def save_prediction(
    row: Any,
    predict_tables:list,
    predict_columns:dict,
    table_response_text: str,
    column_response_text: str,
    efficiency_tracker: SampleEfficiencyTracker,
    answer_llm_name: str,
    provider: str,
    table_filtering_enabled: bool,
    prediction_path: Path,
    dataset_name: str,
    documents_dir: Path,
    source_record: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    efficiency = efficiency_tracker.finalize()
    record = {
            "model": answer_llm_name,
            "provider": provider,
            "id": f"{get_row_value(row, 'id')}",
            "question": row["question"],
            "spider_db_id": get_row_value(row, "spider_db_id"),
            "predict_db_id": get_row_value(row, "predict_db_id"),
            "predict_tables_text":table_response_text,
            "predict_columns_text":column_response_text,
            "predict_tables": predict_tables,
            "predict_columns": predict_columns,
            "table_filtering_enabled": table_filtering_enabled,
            "efficiency": efficiency,
        }
    if error_message:
        record["schema_error"] = error_message
    prediction = build_prediction_from_native(
        schema_record=record,
        database_record=row,
        source_record=source_record or row,
        dataset_name=dataset_name,
        method="table_to_column",
        documents_dir=documents_dir,
    )
    upsert_prediction(prediction_path, prediction)
    return record


def process_table2column_sample(
    *,
    predict_db_id: str,
    question: str,
    hint: str,
    prompt_templates: dict[str, str],
    schema_store: DbInfoSchemaStore,
    full_db_records: Sequence[dict[str, Any]],
    db_counts: dict[str, int],
    embedder: Any,
    qdrant_client: Any,
    qdrant_collection_name: str,
    answer_llm: Any,
    efficiency_tracker: SampleEfficiencyTracker,
    table_filtering_enabled: bool,
) -> tuple[list[str], dict[str, Any], str, str]:
    if table_filtering_enabled:
        table_schema_text, table_prompt_records = resolve_table_prompt_schema(
            predict_db_id=predict_db_id,
            question=question,
            hint=hint,
            prompt_template=prompt_templates["table"],
            answer_llm=answer_llm,
            embedder=embedder,
            schema_store=schema_store,
            qdrant_client=qdrant_client,
            qdrant_collection_name=qdrant_collection_name,
            db_counts=db_counts,
        )
        table_prompt = render_schema_prompt(
            prompt_template=prompt_templates["table"],
            schema_text=table_schema_text,
            question=question,
            hint=hint,
        )
        table_response_text, table_total_tokens = answer_llm.query_with_usage(table_prompt)
        efficiency_tracker.add_llm_total_tokens(table_total_tokens)
        relevant_table_list = normalize_relevant_tables(
            parse_table_response(table_response_text), full_db_records
        )
        column_prompt_records = resolve_column_prompt_records(
            predict_db_id=predict_db_id,
            question=question,
            hint=hint,
            relevant_table_list=relevant_table_list,
            table_prompt_records=table_prompt_records,
            schema_store=schema_store,
            answer_llm=answer_llm,
            prompt_template=prompt_templates["column"],
        )
        column_schema_text = schema_store.render_schema_text(
            predict_db_id, column_prompt_records
        )
    else:
        table_response_text = "Table filtering disabled."
        relevant_table_list = []
        column_schema_text, _ = resolve_table_prompt_schema(
            predict_db_id=predict_db_id,
            question=question,
            hint=hint,
            prompt_template=prompt_templates["column"],
            answer_llm=answer_llm,
            embedder=embedder,
            schema_store=schema_store,
            qdrant_client=qdrant_client,
            qdrant_collection_name=qdrant_collection_name,
            db_counts=db_counts,
        )
    column_prompt = render_schema_prompt(
        prompt_template=prompt_templates["column"],
        schema_text=column_schema_text,
        question=question,
        hint=hint,
    )
    column_response_text, column_total_tokens = answer_llm.query_with_usage(column_prompt)
    efficiency_tracker.add_llm_total_tokens(column_total_tokens)
    return (
        relevant_table_list,
        parse_column_response(column_response_text),
        table_response_text,
        column_response_text,
    )


def run_table2column(
    dataset_df: pd.DataFrame,
    prompt_templates: dict[str, str],
    dataset_name: str,
    documents_dir: Path,
    schema_store: DbInfoSchemaStore,
    db_counts: dict[str, int],
    embedder: Any,
    qdrant_client: Any,
    qdrant_collection_name: str,
    answer_llm: Any,
    answer_llm_name: str,
    provider: str,
    table_filtering_enabled: bool,
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
                predict_tables=[],
                predict_columns={},
                table_response_text="No Valid Database.",
                column_response_text="No Valid Database.",
                efficiency_tracker=efficiency_tracker,
                answer_llm_name=answer_llm_name,
                provider=provider,
                table_filtering_enabled=table_filtering_enabled,
                prediction_path=prediction_path,
                source_record=source_record,
                dataset_name=dataset_name,
                documents_dir=documents_dir,
            )
            processed_count += 1
            continue

        predict_db_id = str(predict_db_id)
        try:
            question = str(row["question"])
            hint = resolve_hint(
                row,
                dataset_name=dataset_name,
                documents_dir=documents_dir,
            )
            full_db_records = schema_store.get_column_records(predict_db_id)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            save_prediction(
                row=row,
                predict_tables=[],
                predict_columns={},
                table_response_text=error_message,
                column_response_text=error_message,
                efficiency_tracker=efficiency_tracker,
                answer_llm_name=answer_llm_name,
                provider=provider,
                table_filtering_enabled=table_filtering_enabled,
                prediction_path=prediction_path,
                source_record=source_record,
                dataset_name=dataset_name,
                documents_dir=documents_dir,
                error_message=error_message,
            )
            processed_count += 1
            continue
        if not full_db_records:
            save_prediction(
                row=row,
                predict_tables=[],
                predict_columns={},
                table_response_text="No Valid Database.",
                column_response_text="No Valid Database.",
                efficiency_tracker=efficiency_tracker,
                answer_llm_name=answer_llm_name,
                provider=provider,
                table_filtering_enabled=table_filtering_enabled,
                prediction_path=prediction_path,
                source_record=source_record,
                dataset_name=dataset_name,
                documents_dir=documents_dir,
            )
            processed_count += 1
            continue

        try:
            (
                relevant_table_list,
                predict_columns,
                table_response_text,
                column_response_text,
            ) = process_table2column_sample(
                predict_db_id=predict_db_id,
                question=question,
                hint=hint,
                prompt_templates=prompt_templates,
                schema_store=schema_store,
                full_db_records=full_db_records,
                db_counts=db_counts,
                embedder=embedder,
                qdrant_client=qdrant_client,
                qdrant_collection_name=qdrant_collection_name,
                answer_llm=answer_llm,
                efficiency_tracker=efficiency_tracker,
                table_filtering_enabled=table_filtering_enabled,
            )
            error_message = None
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            relevant_table_list = []
            predict_columns = {}
            table_response_text = error_message
            column_response_text = error_message
        save_prediction(
            row=row,
            predict_tables=relevant_table_list,
            predict_columns=predict_columns,
            table_response_text=table_response_text,
            column_response_text=column_response_text,
            efficiency_tracker=efficiency_tracker,
            answer_llm_name=answer_llm_name,
            provider=provider,
            table_filtering_enabled=table_filtering_enabled,
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
    table_filtering_enabled = not args.disable_table_filtering

    current_dataset_root = dataset_root(dataset_name)
    documents_dir = current_dataset_root / "documents"
    db_info_path = resolve_project_path(args.db_info_path) if args.db_info_path else current_dataset_root / "db_info.json"
    qdrant_path = resolve_project_path(args.qdrant_path) if args.qdrant_path else current_dataset_root / "qdrant_column_index"
    db_info_index = load_db_info_index(db_info_path)
    db_counts = load_db_counts(db_info_index)
    qdrant_collection_name = load_qdrant_collection_name(qdrant_path)
    input_path = resolve_database_prediction_input(
        explicit_path=(
            resolve_project_path(args.input_path) if args.input_path else None
        ),
        method=args.database_method,
        dataset_name=dataset_name,
        model_name=database_model_name,
    )
    log_path = build_run_log_path("table2column", dataset_name, answer_llm_name)
    logger, logger_path = setup_task_logger("table2column", log_path)

    dataset_df = load_dataset(input_path)
    source_records = load_json_records(current_dataset_root / "gold_sl.json")
    source_index = {
        key: dict(value) for key, value in index_records_by_id(source_records).items()
    }
    prompt_templates = load_prompt_templates(method_name)
    database_model_name = (
        str(dataset_df.iloc[0].get("model") or database_model_name)
        if len(dataset_df)
        else database_model_name
    )
    if args.prediction_path:
        prediction_path = require_results_output(resolve_project_path(args.prediction_path))
    else:
        prediction_path = require_results_output(
            build_prediction_path(
                "table_to_column",
                dataset_name,
                answer_llm_name,
                results_root=(
                    RESULTS_ROOT / "ablation" / "wo_table_filtering"
                    if not table_filtering_enabled
                    else RESULTS_ROOT
                ),
            )
        )
    initialize_prediction_file(
        prediction_path,
        dataset_name=dataset_name,
        method="table_to_column",
        database_selection_model_name=database_model_name,
        schema_linking_model_name=answer_llm_name,
    )
    replace_predictions(prediction_path, [])

    log_run_configuration(
        logger,
        task_name="Table-to-Column Schema Linking",
        dataset_name=dataset_name,
        data_count=len(dataset_df),
        model_name=answer_llm_name,
        provider=provider,
        result_path=prediction_path,
        extra_fields={
            "Method": method_name,
            "Table filtering enabled": table_filtering_enabled,
            "Input path": input_path,
            "Table prompt template": TEMPLATES_ROOT / method_name / "extract_relevant_tables.txt",
            "Column prompt template": TEMPLATES_ROOT / method_name / "extract_relevant_columns.txt",
            "DB info path": db_info_path,
            "Qdrant path": qdrant_path,
            "Qdrant collection": qdrant_collection_name,
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
    from Llm.embedding_model_loader import EmbeddingModelLoader

    embedder = EmbeddingModelLoader()
    qdrant_client = get_qdrant_client(qdrant_path)
    renderer = SchemaTextRenderer(tokenizer=answer_llm.tokenizer)
    schema_store = DbInfoSchemaStore(
        db_info_index=db_info_index,
        renderer=renderer,
    )

    processed_count = run_table2column(
        dataset_df=dataset_df,
        prompt_templates=prompt_templates,
        dataset_name=dataset_name,
        documents_dir=documents_dir,
        schema_store=schema_store,
        db_counts=db_counts,
        embedder=embedder,
        qdrant_client=qdrant_client,
        qdrant_collection_name=qdrant_collection_name,
        answer_llm=answer_llm,
        answer_llm_name=answer_llm_name,
        provider=provider,
        table_filtering_enabled=table_filtering_enabled,
        prediction_path=prediction_path,
        source_index=source_index,
    )
    logger.info("Completed %s records.", processed_count)


if __name__ == "__main__":
    main()
