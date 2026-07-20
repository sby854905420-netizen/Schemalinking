"""Pure table/column diagnostics used by structured decision traces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from Utils.prediction_store import normalize_identifier, normalize_table_name


def record_table_name(record: Mapping[str, Any]) -> str:
    metadata = record.get("meta_data")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return str(record.get("table_name") or metadata.get("table_name") or "").strip()


def _record_column_name(record: Mapping[str, Any]) -> str:
    metadata = record.get("meta_data")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return normalize_identifier(record.get("column_name") or metadata.get("column_name"))


def table_set(db_id: str, table_names: Sequence[Any]) -> set[str]:
    return {
        normalized
        for table_name in table_names
        if (normalized := normalize_table_name(db_id, table_name))
    }


def _column_set(
    db_id: str, columns_by_table: Mapping[Any, Sequence[Any]] | None
) -> set[tuple[str, str]]:
    if not isinstance(columns_by_table, Mapping):
        return set()
    result: set[tuple[str, str]] = set()
    for raw_table, raw_columns in columns_by_table.items():
        if not isinstance(raw_columns, Sequence) or isinstance(raw_columns, (str, bytes)):
            continue
        table_name = normalize_table_name(db_id, raw_table)
        for raw_column in raw_columns:
            column_name = normalize_identifier(raw_column)
            if table_name and column_name:
                result.add((table_name, column_name))
    return result


def _gold_schema(
    source_record: Mapping[str, Any],
) -> tuple[str, set[str], set[tuple[str, str]]]:
    gold_db_id = str(source_record.get("db_id") or "").strip()
    columns_by_table = source_record.get("columns_by_table")
    raw_tables = source_record.get("tables")
    gold_columns = _column_set(
        gold_db_id,
        columns_by_table if isinstance(columns_by_table, Mapping) else None,
    )
    if isinstance(raw_tables, Sequence) and not isinstance(raw_tables, (str, bytes)):
        gold_tables = table_set(gold_db_id, raw_tables)
    elif isinstance(columns_by_table, Mapping):
        gold_tables = table_set(gold_db_id, list(columns_by_table))
    else:
        gold_tables = {table for table, _ in gold_columns}
    return gold_db_id, gold_tables, gold_columns


def build_table_diagnostics(
    *,
    predict_db_id: str,
    selected_tables: Sequence[str],
    source_record: Mapping[str, Any],
) -> dict[str, Any]:
    gold_db_id, gold_tables, _ = _gold_schema(source_record)
    database_correct = predict_db_id == gold_db_id
    predicted_tables = table_set(predict_db_id, selected_tables)
    matched_tables = predicted_tables & gold_tables if database_correct else set()
    return {
        "database_correct": database_correct,
        "gold_table_count": len(gold_tables),
        "matched_table_count": len(matched_tables),
        "table_exact_match": database_correct and predicted_tables == gold_tables,
    }


def build_column_diagnostics(
    *,
    predict_db_id: str,
    prompt_records: Sequence[Mapping[str, Any]],
    predicted_columns: Mapping[Any, Sequence[Any]],
    source_record: Mapping[str, Any],
) -> dict[str, Any]:
    gold_db_id, _, gold_columns = _gold_schema(source_record)
    database_correct = predict_db_id == gold_db_id
    prompt_columns = {
        (
            normalize_table_name(predict_db_id, record_table_name(record)),
            _record_column_name(record),
        )
        for record in prompt_records
        if record_table_name(record) and _record_column_name(record)
    }
    predicted = _column_set(predict_db_id, predicted_columns)
    if database_correct:
        available_gold = gold_columns & prompt_columns
        matched = gold_columns & predicted
        matched_available = available_gold & predicted
    else:
        available_gold = set()
        matched = set()
        matched_available = set()
    return {
        "database_correct": database_correct,
        "gold_columns_available_count": len(available_gold),
        "matched_available_column_count": len(matched_available),
        "gold_column_count": len(gold_columns),
        "matched_column_count": len(matched),
        "column_exact_match": database_correct and predicted == gold_columns,
    }


def resolve_first_error_stage(
    *,
    database_first_error_stage: str | None,
    table_filtering_enabled: bool,
    table_diagnostics: Mapping[str, Any] | None,
    column_diagnostics: Mapping[str, Any],
) -> str:
    if database_first_error_stage and database_first_error_stage != "correct":
        return database_first_error_stage
    if not column_diagnostics.get("database_correct"):
        return "database.final"
    if table_filtering_enabled and not (
        table_diagnostics and table_diagnostics.get("table_exact_match")
    ):
        return "table.prediction"
    if column_diagnostics.get("gold_columns_available_count") != column_diagnostics.get(
        "gold_column_count"
    ):
        return "column.candidate_scope"
    if not column_diagnostics.get("column_exact_match"):
        return "column.prediction"
    return "correct"


def resolve_execution_error_stage(
    database_first_error_stage: Any,
    current_stage: str,
    *,
    table_filtering_enabled: bool = False,
    table_diagnostics: Mapping[str, Any] | None = None,
) -> str:
    if database_first_error_stage and str(database_first_error_stage) != "correct":
        return str(database_first_error_stage)
    if (
        current_stage == "column.prediction"
        and table_filtering_enabled
        and table_diagnostics
        and not table_diagnostics.get("table_exact_match")
    ):
        return "table.prediction"
    return current_stage


__all__ = [
    "build_column_diagnostics",
    "build_table_diagnostics",
    "record_table_name",
    "resolve_execution_error_stage",
    "resolve_first_error_stage",
    "table_set",
]
