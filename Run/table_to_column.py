from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from config import (
    ANSWER_LLM_NAME,
    BASELINE_SCHEMA_LINKING_QUERY_SETTINGS,
    DATASET_NAME,
    EMBEDDING_MODEL_NAME,
    MAX_GENERATEION_NUM,
    MAX_INPUT_LENGTH,
    PROVIDER,
    RESULTS_ROOT,
    TEMPLATES_ROOT,
    dataset_root,
    resolve_project_path,
)
from Llm.llm_loader import LLM, resolve_provider
from Utils.json_utils import load_json, load_json_records
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
from Utils.qdrant_utils import resolve_qdrant_collection_name
from Utils.decision_trace import (
    TracePaths,
    append_event,
    append_events,
    build_trace_paths,
    build_trace_summary,
    finalize_trace,
    initialize_trace,
    load_events,
    mark_trace_failed,
    remove_trace,
)
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
from Utils.schema_trace_diagnostics import (
    build_column_diagnostics,
    build_table_diagnostics,
    record_table_name,
    resolve_execution_error_stage,
    resolve_first_error_stage,
    table_set,
)
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


@dataclass(frozen=True)
class TablePredictionResult:
    prompt_records: list[dict[str, Any]]
    parsed_tables: list[str]
    selected_tables: list[str]
    response_text: str


@dataclass(frozen=True)
class ColumnPredictionResult:
    prompt_records: list[dict[str, Any]]
    predicted_columns: dict[str, list[str]]
    response_text: str


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
    parser.add_argument("--collection-name", dest="collection_name", type=str, default=None)
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
    predict_tables: list[str],
    predict_columns: dict[str, list[str]],
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
        "id": str(get_row_value(row, "id", "instance_id")),
        "question": row["question"],
        "spider_db_id": get_row_value(row, "spider_db_id"),
        "predict_db_id": get_row_value(row, "predict_db_id"),
        "predict_tables_text": table_response_text,
        "predict_columns_text": column_response_text,
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


def predict_tables_for_sample(
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
) -> TablePredictionResult:
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
    parsed_tables = parse_table_response(table_response_text)
    selected_tables = normalize_relevant_tables(parsed_tables, full_db_records)
    return TablePredictionResult(
        prompt_records=list(table_prompt_records),
        parsed_tables=parsed_tables,
        selected_tables=selected_tables,
        response_text=table_response_text,
    )


def predict_columns_for_sample(
    *,
    predict_db_id: str,
    question: str,
    hint: str,
    prompt_templates: dict[str, str],
    schema_store: DbInfoSchemaStore,
    db_counts: dict[str, int],
    embedder: Any,
    qdrant_client: Any,
    qdrant_collection_name: str,
    answer_llm: Any,
    efficiency_tracker: SampleEfficiencyTracker,
    table_prediction: TablePredictionResult | None,
) -> ColumnPredictionResult:
    if table_prediction is None:
        column_schema_text, column_prompt_records = resolve_table_prompt_schema(
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
    else:
        column_prompt_records = resolve_column_prompt_records(
            predict_db_id=predict_db_id,
            question=question,
            hint=hint,
            relevant_table_list=table_prediction.selected_tables,
            table_prompt_records=table_prediction.prompt_records,
            schema_store=schema_store,
            answer_llm=answer_llm,
            prompt_template=prompt_templates["column"],
        )
        column_schema_text = schema_store.render_schema_text(
            predict_db_id, column_prompt_records
        )
    column_prompt = render_schema_prompt(
        prompt_template=prompt_templates["column"],
        schema_text=column_schema_text,
        question=question,
        hint=hint,
    )
    column_response_text, column_total_tokens = answer_llm.query_with_usage(column_prompt)
    efficiency_tracker.add_llm_total_tokens(column_total_tokens)
    return ColumnPredictionResult(
        prompt_records=list(column_prompt_records),
        predicted_columns=parse_column_response(column_response_text),
        response_text=column_response_text,
    )


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


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
    trace_events_path: Path | None = None,
    database_trace_diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    from tqdm import tqdm

    processed_count = 0

    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        efficiency_tracker = SampleEfficiencyTracker()
        sample_id = str(get_row_value(row, "id", "instance_id"))
        source_record = (source_index or {}).get(str(sample_id), {})
        database_diagnostics = (database_trace_diagnostics or {}).get(sample_id, {})
        database_first_error_stage = database_diagnostics.get("first_error_stage")
        predict_db_id = get_row_value(row, "predict_db_id")
        if predict_db_id is None or str(predict_db_id).strip() == "":
            if trace_events_path is not None:
                table_failure_diagnostics = build_table_diagnostics(
                    predict_db_id="",
                    selected_tables=[],
                    source_record=source_record,
                )
                column_failure_diagnostics = build_column_diagnostics(
                    predict_db_id="",
                    prompt_records=[],
                    predicted_columns={},
                    source_record=source_record,
                )
                append_event(
                    trace_events_path,
                    sample_id=sample_id,
                    stage="table.prediction",
                    status="skipped",
                    output={
                        "predicted_db_id": None,
                        "available_table_count": 0,
                        "prompt_column_count": 0,
                        "parsed_tables": [],
                        "selected_tables": [],
                    },
                    diagnostics=(
                        table_failure_diagnostics
                        if table_filtering_enabled
                        else {
                            "database_correct": table_failure_diagnostics[
                                "database_correct"
                            ]
                        }
                    ),
                )
                append_event(
                    trace_events_path,
                    sample_id=sample_id,
                    stage="column.prediction",
                    status="skipped",
                    output={
                        "prompt_tables": [],
                        "prompt_column_count": 0,
                        "predicted_columns": {},
                    },
                    diagnostics=column_failure_diagnostics,
                )
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
            if trace_events_path is not None:
                append_event(
                    trace_events_path,
                    sample_id=sample_id,
                    stage="sample.final",
                    status="failed",
                    output={
                        "predicted_db_id": None,
                        "predicted_tables": [],
                        "predicted_columns": {},
                    },
                    diagnostics={
                        "first_error_stage": resolve_execution_error_stage(
                            database_first_error_stage, "database.final"
                        )
                    },
                )
            processed_count += 1
            continue

        predict_db_id = str(predict_db_id)
        try:
            question = str(row["question"])
            hint = resolve_hint(
                choose_external_knowledge_source(source_record, row),
                dataset_name=dataset_name,
                documents_dir=documents_dir,
            )
            full_db_records = schema_store.get_column_records(predict_db_id)
            if not full_db_records:
                raise LookupError(
                    f"No schema columns found for predicted database {predict_db_id!r}."
                )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            error = _error_payload(exc)
            if trace_events_path is not None:
                execution_stage = (
                    "table.prediction"
                    if table_filtering_enabled
                    else "column.prediction"
                )
                table_failure_diagnostics = build_table_diagnostics(
                    predict_db_id=predict_db_id,
                    selected_tables=[],
                    source_record=source_record,
                )
                column_failure_diagnostics = build_column_diagnostics(
                    predict_db_id=predict_db_id,
                    prompt_records=[],
                    predicted_columns={},
                    source_record=source_record,
                )
                append_event(
                    trace_events_path,
                    sample_id=sample_id,
                    stage="table.prediction",
                    status="failed" if table_filtering_enabled else "skipped",
                    output={
                        "predicted_db_id": predict_db_id,
                        "available_table_count": 0,
                        "prompt_column_count": 0,
                        "parsed_tables": [],
                        "selected_tables": [],
                    },
                    diagnostics=(
                        table_failure_diagnostics
                        if table_filtering_enabled
                        else {
                            "database_correct": table_failure_diagnostics[
                                "database_correct"
                            ]
                        }
                    ),
                    error=error if table_filtering_enabled else None,
                )
                append_event(
                    trace_events_path,
                    sample_id=sample_id,
                    stage="column.prediction",
                    status="skipped" if table_filtering_enabled else "failed",
                    output={
                        "prompt_tables": [],
                        "prompt_column_count": 0,
                        "predicted_columns": {},
                    },
                    diagnostics=column_failure_diagnostics,
                    error=None if table_filtering_enabled else error,
                )
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
            if trace_events_path is not None:
                append_event(
                    trace_events_path,
                    sample_id=sample_id,
                    stage="sample.final",
                    status="failed",
                    output={
                        "predicted_db_id": predict_db_id,
                        "predicted_tables": [],
                        "predicted_columns": {},
                    },
                    diagnostics={
                        "first_error_stage": resolve_execution_error_stage(
                            database_first_error_stage, execution_stage
                        )
                    },
                    error=error,
                )
            processed_count += 1
            continue

        available_table_count = len(
            {
                record_table_name(record)
                for record in full_db_records
                if record_table_name(record)
            }
        )
        table_prediction: TablePredictionResult | None = None
        table_diagnostics: dict[str, Any] | None = None
        if table_filtering_enabled:
            try:
                table_prediction = predict_tables_for_sample(
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
                )
                table_diagnostics = build_table_diagnostics(
                    predict_db_id=predict_db_id,
                    selected_tables=table_prediction.selected_tables,
                    source_record=source_record,
                )
                if trace_events_path is not None:
                    append_event(
                        trace_events_path,
                        sample_id=sample_id,
                        stage="table.prediction",
                        output={
                            "predicted_db_id": predict_db_id,
                            "available_table_count": available_table_count,
                            "prompt_column_count": len(
                                table_prediction.prompt_records
                            ),
                            "parsed_tables": table_prediction.parsed_tables,
                            "selected_tables": table_prediction.selected_tables,
                        },
                        diagnostics=table_diagnostics,
                    )
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                error = _error_payload(exc)
                if trace_events_path is not None:
                    table_failure_diagnostics = build_table_diagnostics(
                        predict_db_id=predict_db_id,
                        selected_tables=[],
                        source_record=source_record,
                    )
                    column_failure_diagnostics = build_column_diagnostics(
                        predict_db_id=predict_db_id,
                        prompt_records=[],
                        predicted_columns={},
                        source_record=source_record,
                    )
                    append_event(
                        trace_events_path,
                        sample_id=sample_id,
                        stage="table.prediction",
                        status="failed",
                        output={
                            "predicted_db_id": predict_db_id,
                            "available_table_count": available_table_count,
                            "prompt_column_count": 0,
                            "parsed_tables": [],
                            "selected_tables": [],
                        },
                        diagnostics=table_failure_diagnostics,
                        error=error,
                    )
                    append_event(
                        trace_events_path,
                        sample_id=sample_id,
                        stage="column.prediction",
                        status="skipped",
                        output={
                            "prompt_tables": [],
                            "prompt_column_count": 0,
                            "predicted_columns": {},
                        },
                        diagnostics=column_failure_diagnostics,
                    )
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
                if trace_events_path is not None:
                    append_event(
                        trace_events_path,
                        sample_id=sample_id,
                        stage="sample.final",
                        status="failed",
                        output={
                            "predicted_db_id": predict_db_id,
                            "predicted_tables": [],
                            "predicted_columns": {},
                        },
                        diagnostics={
                            "first_error_stage": resolve_execution_error_stage(
                                database_first_error_stage, "table.prediction"
                            )
                        },
                        error=error,
                    )
                processed_count += 1
                continue
        elif trace_events_path is not None:
            append_event(
                trace_events_path,
                sample_id=sample_id,
                stage="table.prediction",
                status="skipped",
                output={
                    "predicted_db_id": predict_db_id,
                    "available_table_count": available_table_count,
                    "prompt_column_count": 0,
                    "parsed_tables": [],
                    "selected_tables": [],
                },
                diagnostics={
                    "database_correct": predict_db_id
                    == str(source_record.get("db_id") or "").strip()
                },
            )

        relevant_table_list = (
            table_prediction.selected_tables if table_prediction is not None else []
        )
        table_response_text = (
            table_prediction.response_text
            if table_prediction is not None
            else "Table filtering disabled."
        )
        try:
            column_prediction = predict_columns_for_sample(
                predict_db_id=predict_db_id,
                question=question,
                hint=hint,
                prompt_templates=prompt_templates,
                schema_store=schema_store,
                db_counts=db_counts,
                embedder=embedder,
                qdrant_client=qdrant_client,
                qdrant_collection_name=qdrant_collection_name,
                answer_llm=answer_llm,
                efficiency_tracker=efficiency_tracker,
                table_prediction=table_prediction,
            )
            predict_columns = column_prediction.predicted_columns
            column_response_text = column_prediction.response_text
            column_diagnostics = build_column_diagnostics(
                predict_db_id=predict_db_id,
                prompt_records=column_prediction.prompt_records,
                predicted_columns=predict_columns,
                source_record=source_record,
            )
            prompt_tables = sorted(
                table_set(
                    predict_db_id,
                    [
                        record_table_name(record)
                        for record in column_prediction.prompt_records
                    ],
                )
            )
            if trace_events_path is not None:
                append_event(
                    trace_events_path,
                    sample_id=sample_id,
                    stage="column.prediction",
                    output={
                        "prompt_tables": prompt_tables,
                        "prompt_column_count": len(column_prediction.prompt_records),
                        "predicted_columns": predict_columns,
                    },
                    diagnostics=column_diagnostics,
                )
                first_error_stage = resolve_first_error_stage(
                    database_first_error_stage=(
                        str(database_first_error_stage)
                        if database_first_error_stage
                        else None
                    ),
                    table_filtering_enabled=table_filtering_enabled,
                    table_diagnostics=table_diagnostics,
                    column_diagnostics=column_diagnostics,
                )
                final_event_status = "success"
                final_event_output = {
                    "predicted_db_id": predict_db_id,
                    "predicted_tables": relevant_table_list,
                    "predicted_columns": predict_columns,
                }
                final_event_diagnostics = {
                    "database_correct": column_diagnostics["database_correct"],
                    "column_exact_match": column_diagnostics[
                        "column_exact_match"
                    ],
                    "first_error_stage": first_error_stage,
                }
                final_event_error = None
            error_message = None
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            error = _error_payload(exc)
            predict_columns = {}
            column_response_text = error_message
            if trace_events_path is not None:
                column_failure_diagnostics = build_column_diagnostics(
                    predict_db_id=predict_db_id,
                    prompt_records=[],
                    predicted_columns={},
                    source_record=source_record,
                )
                append_event(
                    trace_events_path,
                    sample_id=sample_id,
                    stage="column.prediction",
                    status="failed",
                    output={
                        "prompt_tables": relevant_table_list,
                        "prompt_column_count": 0,
                        "predicted_columns": {},
                    },
                    diagnostics=column_failure_diagnostics,
                    error=error,
                )
                final_event_status = "failed"
                final_event_output = {
                    "predicted_db_id": predict_db_id,
                    "predicted_tables": relevant_table_list,
                    "predicted_columns": {},
                }
                final_event_diagnostics = {
                    "first_error_stage": resolve_execution_error_stage(
                        database_first_error_stage,
                        "column.prediction",
                        table_filtering_enabled=table_filtering_enabled,
                        table_diagnostics=table_diagnostics,
                    )
                }
                final_event_error = error
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
        if trace_events_path is not None:
            append_event(
                trace_events_path,
                sample_id=sample_id,
                stage="sample.final",
                status=final_event_status,
                output=final_event_output,
                diagnostics=final_event_diagnostics,
                error=final_event_error,
            )
        processed_count += 1

    return processed_count


def _normalized_db_id(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def validate_upstream_database_trace(
    *,
    paths: TracePaths,
    dataset_df: pd.DataFrame,
    dataset_name: str,
    database_method: str,
    database_model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not paths.metadata.is_file():
        raise FileNotFoundError(
            f"Upstream database trace metadata does not exist: {paths.metadata}"
        )
    metadata = load_json(paths.metadata)
    expected_metadata = {
        "status": "completed",
        "dataset": dataset_name,
        "method": database_method,
        "ranking_model": database_model_name,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Upstream database trace metadata does not match this run: {mismatches}"
        )

    events = load_events(paths.events)
    final_events: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["stage"] != "database.final":
            continue
        sample_id = event["sample_id"]
        if sample_id in final_events:
            raise ValueError(
                f"Duplicate database.final event for sample {sample_id!r}."
            )
        final_events[sample_id] = event

    expected_predictions = {
        str(get_row_value(row, "id", "instance_id")): _normalized_db_id(
            get_row_value(row, "predict_db_id")
        )
        for _, row in dataset_df.iterrows()
    }
    if set(final_events) != set(expected_predictions):
        missing = sorted(set(expected_predictions) - set(final_events))
        extra = sorted(set(final_events) - set(expected_predictions))
        raise ValueError(
            "Upstream database trace sample IDs do not match the prediction input: "
            f"missing={missing}, extra={extra}."
        )
    for sample_id, expected_db_id in expected_predictions.items():
        actual_db_id = _normalized_db_id(
            final_events[sample_id]["output"].get("predicted_db_id")
        )
        if actual_db_id != expected_db_id:
            raise ValueError(
                "Upstream database trace prediction does not match the input for "
                f"sample {sample_id!r}: expected {expected_db_id!r}, "
                f"got {actual_db_id!r}."
            )

    diagnostics = {
        sample_id: dict(event["diagnostics"])
        for sample_id, event in final_events.items()
    }
    return events, diagnostics


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
    db_info_path = (
        resolve_project_path(args.db_info_path)
        if args.db_info_path
        else current_dataset_root / "db_info.json"
    )
    qdrant_path = (
        resolve_project_path(args.qdrant_path)
        if args.qdrant_path
        else current_dataset_root / "qdrant_column_index"
    )
    db_info_index = load_db_info_index(db_info_path)
    db_counts = load_db_counts(db_info_index)
    qdrant_collection_name = resolve_qdrant_collection_name(
        qdrant_path,
        explicit_name=args.collection_name,
        fallback=dataset_name,
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
    prompt_templates = load_prompt_templates(method_name)
    database_model_name = (
        str(dataset_df.iloc[0].get("model") or database_model_name)
        if len(dataset_df)
        else database_model_name
    )
    experiment_results_root = (
        RESULTS_ROOT / "ablation" / "wo_table_filtering"
        if not table_filtering_enabled
        else RESULTS_ROOT
    )
    if args.prediction_path:
        prediction_path = require_results_output(
            resolve_project_path(args.prediction_path)
        )
    else:
        prediction_path = require_results_output(
            build_prediction_path(
                "table_to_column",
                dataset_name,
                answer_llm_name,
                results_root=experiment_results_root,
            )
        )
    missing_gold_ids = sorted(
        {
            str(get_row_value(row, "id", "instance_id"))
            for _, row in dataset_df.iterrows()
        }
        - set(source_index)
    )
    if missing_gold_ids:
        raise ValueError(f"Missing gold schema records for samples: {missing_gold_ids}")

    stable_trace_paths = build_trace_paths(
        "sl",
        "table_to_column",
        dataset_name,
        answer_llm_name,
        results_root=experiment_results_root,
    )
    trace_paths: TracePaths | None = None
    upstream_paths: TracePaths | None = None
    upstream_events: list[dict[str, Any]] = []
    database_trace_diagnostics: dict[str, dict[str, Any]] = {}
    if args.database_method != "baseline":
        trace_paths = stable_trace_paths
        upstream_paths = build_trace_paths(
            "db",
            args.database_method,
            dataset_name,
            database_model_name,
        )
        upstream_events, database_trace_diagnostics = validate_upstream_database_trace(
            paths=upstream_paths,
            dataset_df=dataset_df,
            dataset_name=dataset_name,
            database_method=args.database_method,
            database_model_name=database_model_name,
        )

    initialize_prediction_file(
        prediction_path,
        dataset_name=dataset_name,
        method="table_to_column",
        database_selection_model_name=database_model_name,
        schema_linking_model_name=answer_llm_name,
    )
    replace_predictions(prediction_path, [])

    if trace_paths is None:
        remove_trace(stable_trace_paths)
    else:
        initialize_trace(
            trace_paths,
            {
                "dataset": dataset_name,
                "method": "table_to_column",
                "schema_method": method_name,
                "schema_model": answer_llm_name,
                "database_method": args.database_method,
                "database_model": database_model_name,
                "input_path": str(input_path),
                "prediction_path": str(prediction_path),
                "db_info_path": str(db_info_path),
                "qdrant_path": str(qdrant_path),
                "qdrant_collection": qdrant_collection_name,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "table_filtering_enabled": table_filtering_enabled,
                "total_samples": len(dataset_df),
            },
        )

    try:
        if trace_paths is not None and upstream_paths is not None:
            append_events(trace_paths.events, upstream_events)

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
            trace_events_path=trace_paths.events if trace_paths is not None else None,
            database_trace_diagnostics=database_trace_diagnostics,
        )
        if trace_paths is not None:
            finalize_trace(
                trace_paths,
                build_trace_summary(load_events(trace_paths.events)),
            )
        print(f"Completed {processed_count} table-to-column prediction records.")
    except BaseException as exc:
        if trace_paths is not None:
            mark_trace_failed(trace_paths, exc)
        raise


if __name__ == "__main__":
    main()
