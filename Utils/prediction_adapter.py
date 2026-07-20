"""Map native MDBlink records to the unified SL prediction contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from Utils.json_utils import normalize_response_text
from Utils.prediction_store import (
    build_prediction,
    make_stage,
    normalize_columns,
    normalize_tables,
    usage_from_efficiency,
)
from Utils.value_utils import get_row_value, resolve_external_knowledge


NO_VALID_DATABASE = "No Valid Database."


def _parse_contract_response(
    response_text: Any,
    *,
    key: str,
    expected_type: type,
    fallback: Any,
) -> tuple[Any, str | None]:
    if not isinstance(response_text, str) or not response_text.strip():
        return fallback, None
    raw = response_text.strip()
    if raw == NO_VALID_DATABASE:
        return fallback, raw
    try:
        payload = json.loads(normalize_response_text(raw))
    except json.JSONDecodeError:
        return fallback, raw
    if not isinstance(payload, dict) or key not in payload or not isinstance(
        payload[key], expected_type
    ):
        return fallback, raw
    return payload[key], None


def _database_state(record: Mapping[str, Any]) -> tuple[str, list[str], str | None]:
    native_error = record.get("database_error")
    if isinstance(native_error, str) and native_error.strip():
        return "failed", [], native_error.strip()
    selected = get_row_value(record, "predict_db_id")
    selected_text = "" if selected is None else str(selected).strip()
    if selected_text:
        return "success", [selected_text], None

    raw_response = record.get("pre_db_response")
    if isinstance(raw_response, str) and raw_response.strip():
        parsed, parse_error = _parse_contract_response(
            raw_response,
            key="relevant_database",
            expected_type=str,
            fallback="",
        )
        parsed = str(parsed).strip()
        if parsed:
            return "success", [parsed], None
        if parse_error and parse_error != NO_VALID_DATABASE:
            return "failed", [], parse_error
    return "empty", [], NO_VALID_DATABASE


def _schema_state(
    record: Mapping[str, Any],
    *,
    method: str,
    db_id: str,
    dataset_name: str,
) -> tuple[
    str,
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, str]],
    str | None,
]:
    if not db_id:
        return "not_run", [], [], [], None

    native_error = record.get("schema_error")
    native_columns = record.get("predict_columns")
    columns_by_table, column_error = _parse_contract_response(
        record.get("predict_columns_text"),
        key="relevant_columns",
        expected_type=dict,
        fallback=native_columns if isinstance(native_columns, dict) else {},
    )
    if not isinstance(columns_by_table, dict):
        columns_by_table = {}

    stages: list[dict[str, Any]] = []
    table_error: str | None = None
    if method == "prompt_baseline":
        table_names: list[Any] = list(columns_by_table)
    elif method == "table_to_column" and not record.get(
        "table_filtering_enabled", True
    ):
        native_tables = record.get("predict_tables")
        table_names = native_tables if isinstance(native_tables, list) else []
    else:
        native_tables = record.get("predict_tables")
        table_names, table_error = _parse_contract_response(
            record.get("predict_tables_text"),
            key="relevant_tables",
            expected_type=list,
            fallback=native_tables if isinstance(native_tables, list) else [],
        )
        if not isinstance(table_names, list):
            table_names = []

    snowflake_three_part = str(dataset_name).strip().lower() == "spider2"
    final_tables = normalize_tables(
        db_id,
        table_names,
        snowflake_three_part=snowflake_three_part,
    )
    final_columns = normalize_columns(
        db_id,
        columns_by_table,
        snowflake_three_part=snowflake_three_part,
    )
    if method == "table_to_column" and record.get("table_filtering_enabled", True):
        table_status = "failed" if table_error else ("success" if final_tables else "empty")
        stages.append(
            make_stage("schema_pruning", table_status, tables=final_tables)
        )

    error = str(native_error).strip() if native_error else (column_error or table_error)
    if native_error or column_error:
        status = "failed"
    elif final_columns:
        status = "success"
    else:
        status = "empty"
    return status, stages, final_tables, final_columns, error


def build_prediction_from_native(
    *,
    schema_record: Mapping[str, Any],
    database_record: Mapping[str, Any] | None,
    source_record: Mapping[str, Any],
    dataset_name: str,
    method: str,
    documents_dir: Path,
    database_usage: Mapping[str, Any] | None = None,
    schema_usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sample_id = get_row_value(schema_record, "id", "instance_id")
    if sample_id is None:
        raise ValueError("Native schema result is missing id.")

    db_source = database_record if database_record is not None else schema_record
    database_status, selected_db_ids, database_error = _database_state(db_source)
    db_id = selected_db_ids[0] if selected_db_ids else ""
    schema_status, stages, final_tables, final_columns, schema_error = _schema_state(
        schema_record,
        method=method,
        db_id=db_id,
        dataset_name=dataset_name,
    )

    if database_status != "success":
        sample_status = "failed"
        error = {
            "stage": "database_selection",
            "message": database_error or NO_VALID_DATABASE,
        }
    elif schema_status == "success" and not schema_error:
        sample_status = "success"
        error = None
    else:
        sample_status = "partial"
        error = (
            {"stage": "schema_linking", "message": schema_error}
            if schema_error
            else None
        )

    if (
        method == "rag_column_retrieval"
        and database_usage is None
        and schema_usage is None
    ):
        resolved_database_usage = {"elapsed_seconds": None, "total_tokens": None}
        resolved_schema_usage = {"elapsed_seconds": None, "total_tokens": None}
    else:
        resolved_database_usage = dict(
            database_usage
            or usage_from_efficiency(
                database_record.get("efficiency")
                if database_record is not None
                else None
            )
        )
        resolved_schema_usage = dict(
            schema_usage or usage_from_efficiency(schema_record.get("efficiency"))
        )

    question = get_row_value(source_record, "question")
    if question is None:
        question = get_row_value(schema_record, "question") or ""
    return build_prediction(
        sample_id=sample_id,
        question=question,
        dataset_name=dataset_name,
        external_knowledge=resolve_external_knowledge(
            source_record,
            dataset_name=dataset_name,
            documents_dir=documents_dir,
        ),
        selected_db_ids=selected_db_ids,
        database_selection_status=database_status,
        schema_linking_status=schema_status,
        stages=stages,
        final_tables=final_tables,
        final_columns=final_columns,
        database_selection_usage=resolved_database_usage,
        schema_linking_usage=resolved_schema_usage,
        status=sample_status,
        error=error,
    )


__all__ = ["NO_VALID_DATABASE", "build_prediction_from_native"]
