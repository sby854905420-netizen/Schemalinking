"""Unified Schema Linking prediction storage for MDBlink variants.

The unified artifact intentionally contains database selection and schema
linking only. SQL predictions are stored separately under ``results/sql``.
Writers are expected to be called by the main process; this module provides
atomic replacement, not cross-process job coordination.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from config import RESULTS_ROOT
from Utils.json_utils import atomic_write_json, load_json, upsert_ordered_record


METHOD_NAMES = {
    "autolink",
    "linkalign",
    "prompt_baseline",
    "rag_column_retrieval",
    "table_to_column",
}
PHASE_STATUSES = {"not_run", "success", "empty", "failed", "not_supported"}
SAMPLE_STATUSES = {"success", "partial", "failed"}
STAGE_NAMES = {
    "initial_retrieval",
    "rule_expansion",
    "schema_expansion",
    "schema_pruning",
}
SQL_DIALECTS = {"sqlite", "snowflake"}


class PredictionValidationError(ValueError):
    """Raised when a unified prediction artifact violates its contract."""


def _validate_method_name(method: str) -> None:
    if method not in METHOD_NAMES:
        raise PredictionValidationError(
            f"method must be one of {sorted(METHOD_NAMES)}, got {method!r}."
        )


def safe_path_component(value: str) -> str:
    """Return the task-specified safe representation of one path component."""

    text = str(value).strip().replace("/", "__").replace("\\", "__")
    text = re.sub(r"[\s:]+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    if not text:
        raise ValueError("Path component cannot be empty after sanitization.")
    return text


def build_prediction_path(
    method: str,
    dataset_name: str,
    schema_linking_model_name: str,
    *,
    results_root: Path | str | None = None,
) -> Path:
    """Build ``results/sl/<method>/<dataset>/<model>/prediction.json``."""

    _validate_method_name(method)
    root = Path(results_root) if results_root is not None else RESULTS_ROOT
    return (
        root
        / "sl"
        / safe_path_component(method)
        / safe_path_component(dataset_name)
        / safe_path_component(schema_linking_model_name)
        / "prediction.json"
    )


def canonical_sql_dialect(dataset_name: str) -> str:
    return "snowflake" if str(dataset_name).strip().lower() == "spider2" else "sqlite"


def initialize_prediction_file(
    path: Path | str,
    *,
    dataset_name: str,
    method: str,
    database_selection_model_name: str,
    schema_linking_model_name: str,
) -> Path:
    """Create an empty artifact, or validate a compatible existing artifact."""

    _validate_method_name(method)
    prediction_path = Path(path)
    expected_header = {
        "dataset": str(dataset_name),
        "method": str(method),
        "model_names": {
            "database_selection": str(database_selection_model_name),
            "schema_linking": str(schema_linking_model_name),
        },
    }
    if prediction_path.is_file():
        payload = validate_prediction_file(prediction_path)
        actual_header = {key: payload[key] for key in expected_header}
        if actual_header != expected_header:
            raise PredictionValidationError(
                "Existing prediction file belongs to a different dataset, method, "
                "or model combination: " + str(prediction_path)
            )
        return prediction_path

    atomic_write_json(prediction_path, {**expected_header, "predictions": []})
    return prediction_path


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = load_json(path)
    except ValueError as exc:
        raise PredictionValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PredictionValidationError("Prediction file must contain a JSON object.")
    return payload


def upsert_prediction(path: Path | str, prediction: Mapping[str, Any]) -> None:
    """Insert or replace one sample while preserving its original position."""

    prediction_path = Path(path)
    payload = validate_prediction_file(prediction_path)
    normalized = dict(prediction)
    _validate_prediction(normalized, index=None)
    payload["predictions"] = upsert_ordered_record(
        payload["predictions"], normalized
    )
    validate_prediction_file(payload)
    atomic_write_json(prediction_path, payload)


def replace_predictions(
    path: Path | str, predictions: Sequence[Mapping[str, Any]]
) -> None:
    """Atomically replace all records, preserving the supplied dataset order."""

    prediction_path = Path(path)
    payload = validate_prediction_file(prediction_path)
    payload["predictions"] = [dict(prediction) for prediction in predictions]
    validate_prediction_file(payload)
    atomic_write_json(prediction_path, payload)


def validate_prediction_file(path_or_payload: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a unified prediction payload."""

    if isinstance(path_or_payload, Mapping):
        payload = dict(path_or_payload)
    else:
        payload = _load_payload(Path(path_or_payload))

    required_top_level = {"dataset", "method", "model_names", "predictions"}
    if set(payload) != required_top_level:
        raise PredictionValidationError(
            f"Top-level keys must be exactly {sorted(required_top_level)}."
        )
    if not isinstance(payload["dataset"], str) or not payload["dataset"].strip():
        raise PredictionValidationError("dataset must be a non-empty string.")
    if not isinstance(payload["method"], str):
        raise PredictionValidationError("method must be a string.")
    _validate_method_name(payload["method"])
    model_names = payload["model_names"]
    if not isinstance(model_names, dict) or set(model_names) != {
        "database_selection",
        "schema_linking",
    }:
        raise PredictionValidationError("model_names has an invalid shape.")
    if any(not isinstance(value, str) or not value.strip() for value in model_names.values()):
        raise PredictionValidationError("Both model names must be non-empty strings.")
    if not isinstance(payload["predictions"], list):
        raise PredictionValidationError("predictions must be a list.")

    seen_ids: set[str] = set()
    for index, prediction in enumerate(payload["predictions"]):
        _validate_prediction(prediction, index=index)
        sample_id = prediction["id"]
        if sample_id in seen_ids:
            raise PredictionValidationError(f"Duplicate prediction id: {sample_id}")
        seen_ids.add(sample_id)
    return payload


def _validate_prediction(prediction: Any, index: int | None) -> None:
    where = "prediction" if index is None else f"predictions[{index}]"
    required = {
        "id",
        "question",
        "sql_dialect",
        "external_knowledge",
        "database_selection",
        "schema_linking",
        "usage",
        "status",
        "error",
    }
    if not isinstance(prediction, dict) or set(prediction) != required:
        raise PredictionValidationError(f"{where} has an invalid shape.")
    if not isinstance(prediction["id"], str) or not prediction["id"]:
        raise PredictionValidationError(f"{where}.id must be a non-empty string.")
    if not isinstance(prediction["question"], str):
        raise PredictionValidationError(f"{where}.question must be a string.")
    if prediction["sql_dialect"] not in SQL_DIALECTS:
        raise PredictionValidationError(f"{where}.sql_dialect is invalid.")
    if prediction["external_knowledge"] is not None and not isinstance(
        prediction["external_knowledge"], str
    ):
        raise PredictionValidationError(f"{where}.external_knowledge is invalid.")

    database_selection = prediction["database_selection"]
    if not isinstance(database_selection, dict) or set(database_selection) != {
        "status",
        "selected_db_ids",
    }:
        raise PredictionValidationError(f"{where}.database_selection has an invalid shape.")
    _validate_phase_status(database_selection["status"], f"{where}.database_selection")
    _validate_string_list(database_selection["selected_db_ids"], f"{where}.selected_db_ids")

    schema_linking = prediction["schema_linking"]
    if not isinstance(schema_linking, dict) or set(schema_linking) != {
        "status",
        "stages",
        "final",
    }:
        raise PredictionValidationError(f"{where}.schema_linking has an invalid shape.")
    _validate_phase_status(schema_linking["status"], f"{where}.schema_linking")
    if not isinstance(schema_linking["stages"], list):
        raise PredictionValidationError(f"{where}.schema_linking.stages must be a list.")
    for stage in schema_linking["stages"]:
        _validate_stage(stage, where)
    _validate_schema_result(schema_linking["final"], f"{where}.schema_linking.final")

    _validate_usage(prediction["usage"], where)
    if prediction["status"] not in SAMPLE_STATUSES:
        raise PredictionValidationError(f"{where}.status is invalid.")
    error = prediction["error"]
    if error is not None:
        if not isinstance(error, dict) or set(error) != {"stage", "message"}:
            raise PredictionValidationError(f"{where}.error has an invalid shape.")
        if error["stage"] not in {"database_selection", "schema_linking"}:
            raise PredictionValidationError(f"{where}.error.stage is invalid.")
        if not isinstance(error["message"], str) or not error["message"]:
            raise PredictionValidationError(f"{where}.error.message is invalid.")


def _validate_phase_status(status: Any, where: str) -> None:
    if status not in PHASE_STATUSES:
        raise PredictionValidationError(f"{where}.status is invalid.")


def _validate_string_list(values: Any, where: str) -> None:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise PredictionValidationError(f"{where} must be a string list.")
    if len(values) != len(set(values)):
        raise PredictionValidationError(f"{where} contains duplicates.")


def _validate_stage(stage: Any, where: str) -> None:
    if not isinstance(stage, dict) or set(stage) != {"name", "status", "tables", "columns"}:
        raise PredictionValidationError(f"{where} contains an invalid stage.")
    if stage["name"] not in STAGE_NAMES:
        raise PredictionValidationError(f"{where} contains an invalid stage name.")
    _validate_phase_status(stage["status"], where)
    _validate_schema_result(stage, where, allow_name_status=True)


def _validate_schema_result(value: Any, where: str, allow_name_status: bool = False) -> None:
    expected = {"tables", "columns"}
    if allow_name_status:
        expected |= {"name", "status"}
    if not isinstance(value, dict) or set(value) != expected:
        raise PredictionValidationError(f"{where} has an invalid schema result.")
    tables = value["tables"]
    columns = value["columns"]
    if not isinstance(tables, list) or not isinstance(columns, list):
        raise PredictionValidationError(f"{where} tables and columns must be lists.")
    table_keys: list[tuple[str, str]] = []
    for table in tables:
        if not isinstance(table, dict) or set(table) != {"db_id", "table"}:
            raise PredictionValidationError(f"{where} contains an invalid table object.")
        table_keys.append((table["db_id"], table["table"]))
    if len(table_keys) != len(set(table_keys)):
        raise PredictionValidationError(f"{where} contains duplicate tables.")
    column_keys: list[tuple[str, str, str]] = []
    for column in columns:
        if not isinstance(column, dict) or set(column) != {"db_id", "table", "column"}:
            raise PredictionValidationError(f"{where} contains an invalid column object.")
        column_keys.append((column["db_id"], column["table"], column["column"]))
    if len(column_keys) != len(set(column_keys)):
        raise PredictionValidationError(f"{where} contains duplicate columns.")


def _validate_usage(usage: Any, where: str) -> None:
    phases = {"database_selection", "schema_linking", "linking_total"}
    if not isinstance(usage, dict) or set(usage) != phases:
        raise PredictionValidationError(f"{where}.usage has an invalid shape.")
    for phase in phases:
        values = usage[phase]
        if not isinstance(values, dict) or set(values) != {"elapsed_seconds", "total_tokens"}:
            raise PredictionValidationError(f"{where}.usage.{phase} has an invalid shape.")
        _validate_measure(values["elapsed_seconds"], allow_float=True, where=where)
        _validate_measure(values["total_tokens"], allow_float=False, where=where)
    for key in ("elapsed_seconds", "total_tokens"):
        left = usage["database_selection"][key]
        right = usage["schema_linking"][key]
        total = usage["linking_total"][key]
        expected = None if left is None or right is None else left + right
        if expected is None:
            if total is not None:
                raise PredictionValidationError(f"{where}.usage null total semantics are invalid.")
        elif not math.isclose(float(total), float(expected), rel_tol=0.0, abs_tol=1e-6):
            raise PredictionValidationError(f"{where}.usage total is inconsistent.")


def _validate_measure(value: Any, *, allow_float: bool, where: str) -> None:
    valid_type = isinstance(value, (int, float)) if allow_float else isinstance(value, int)
    if value is not None and (not valid_type or isinstance(value, bool) or value < 0):
        raise PredictionValidationError(f"{where} contains an invalid usage value.")


def _strip_display_fences(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def normalize_identifier(value: Any) -> str:
    """Remove display-only fences and a single unambiguous wrapper quote."""

    text = _strip_display_fences(value)
    for quote in ('"', "'", "`"):
        if len(text) >= 2 and text.startswith(quote) and text.endswith(quote) and text.count(quote) == 2:
            text = text[1:-1].strip()
            break
    return text


def normalize_table_name(db_id: str, table_name: Any) -> str:
    table = normalize_identifier(table_name)
    prefix = f"{db_id}."
    return table[len(prefix) :] if db_id and table.startswith(prefix) else table


def split_qualified_identifier(value: Any) -> tuple[str, ...]:
    """Split a dotted identifier without splitting dots inside Snowflake quotes.

    An empty tuple denotes a malformed identifier (for example, an unclosed
    quote or an empty component).  Components retain their original quoting so
    callers can preserve Snowflake's quoted-identifier case semantics.
    """

    text = _strip_display_fences(value)
    if not text:
        return ()
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            current.append(char)
            if char == quote:
                if quote == '"' and index + 1 < len(text) and text[index + 1] == '"':
                    current.append(text[index + 1])
                    index += 1
                else:
                    quote = None
        elif char in ('"', "`"):
            quote = char
            current.append(char)
        elif char == ".":
            component = "".join(current).strip()
            if not component:
                return ()
            parts.append(component)
            current = []
        else:
            current.append(char)
        index += 1

    component = "".join(current).strip()
    if quote is not None or not component:
        return ()
    parts.append(component)
    return tuple(parts)


def snowflake_identifier_key(value: Any) -> tuple[str, ...]:
    """Return Snowflake's semantic key for a possibly-qualified identifier.

    Unquoted components are case-insensitive and therefore normalized to upper
    case.  Double-quoted components retain exact case, including escaped quote
    handling.  A malformed identifier has an empty key.
    """

    parts = split_qualified_identifier(value)
    keys: list[str] = []
    for part in parts:
        if len(part) >= 2 and part.startswith('"') and part.endswith('"'):
            keys.append(part[1:-1].replace('""', '"'))
        elif len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            keys.append(part[1:-1])
        else:
            keys.append(part.upper())
    return tuple(keys)


def normalize_snowflake_table_name(db_id: str, table_name: Any) -> str:
    """Remove only a genuine outer DB from a Snowflake three-part name.

    Two-part names are already the unified ``SCHEMA.TABLE`` representation.
    In particular, ``PAGILA.ACTOR`` must not be shortened to ``ACTOR`` merely
    because the Spider2 database and schema happen to share the name PAGILA.
    """

    table = _strip_display_fences(table_name)
    parts = split_qualified_identifier(table)
    db_key = snowflake_identifier_key(db_id)
    if len(parts) >= 3 and db_key and snowflake_identifier_key(parts[0]) == db_key:
        return ".".join(parts[1:])
    return table


def restore_snowflake_table_name(db_id: str, table_name: Any) -> str:
    """Restore unified Snowflake names without duplicating a three-part name."""

    table = _strip_display_fences(table_name)
    parts = split_qualified_identifier(table)
    if not db_id or not parts or len(parts) >= 3:
        return table
    return f"{db_id}.{table}"


def _database_identifier_matches(
    db_id: str,
    value: Any,
    *,
    snowflake: bool,
) -> bool:
    if snowflake:
        db_key = snowflake_identifier_key(db_id)
        value_key = snowflake_identifier_key(value)
        return len(db_key) == len(value_key) == 1 and db_key == value_key
    return normalize_identifier(value) == db_id


def normalize_tables(
    db_id: str,
    table_names: Sequence[Any] | None,
    *,
    snowflake_three_part: bool = False,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_table in table_names or []:
        table = (
            normalize_snowflake_table_name(db_id, raw_table)
            if snowflake_three_part
            else normalize_table_name(db_id, raw_table)
        )
        key = (db_id, table)
        if not db_id or not table or key in seen:
            continue
        seen.add(key)
        records.append({"db_id": db_id, "table": table})
    return records


def normalize_columns(
    db_id: str,
    columns_by_table: Mapping[Any, Sequence[Any]] | None,
    *,
    snowflake_three_part: bool = False,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    if not isinstance(columns_by_table, Mapping):
        return records
    for raw_table, raw_columns in columns_by_table.items():
        if not isinstance(raw_columns, Sequence) or isinstance(raw_columns, (str, bytes)):
            continue
        table = (
            normalize_snowflake_table_name(db_id, raw_table)
            if snowflake_three_part
            else normalize_table_name(db_id, raw_table)
        )
        for raw_column in raw_columns:
            column = normalize_identifier(raw_column)
            key = (db_id, table, column)
            if not db_id or not table or not column or key in seen:
                continue
            seen.add(key)
            records.append({"db_id": db_id, "table": table, "column": column})
    return records


def make_usage(
    database_selection: Mapping[str, Any], schema_linking: Mapping[str, Any]
) -> dict[str, dict[str, float | int | None]]:
    database_usage = _normalize_usage_phase(database_selection)
    schema_usage = _normalize_usage_phase(schema_linking)
    total: dict[str, float | int | None] = {}
    for key in ("elapsed_seconds", "total_tokens"):
        left = database_usage[key]
        right = schema_usage[key]
        total[key] = None if left is None or right is None else left + right
        if key == "elapsed_seconds" and total[key] is not None:
            total[key] = round(float(total[key]), 6)
    return {
        "database_selection": database_usage,
        "schema_linking": schema_usage,
        "linking_total": total,
    }


def _normalize_usage_phase(values: Mapping[str, Any]) -> dict[str, float | int | None]:
    elapsed = values.get("elapsed_seconds")
    tokens = values.get("total_tokens")
    return {
        "elapsed_seconds": None if elapsed is None else round(float(elapsed), 6),
        "total_tokens": None if tokens is None else int(tokens),
    }


def usage_from_efficiency(efficiency: Any) -> dict[str, float | int | None]:
    if not isinstance(efficiency, Mapping):
        return {"elapsed_seconds": None, "total_tokens": None}
    return {
        "elapsed_seconds": efficiency.get("sample_elapsed_seconds"),
        "total_tokens": efficiency.get("llm_total_tokens"),
    }


def make_stage(
    name: str,
    status: str,
    *,
    tables: Sequence[dict[str, str]] | None = None,
    columns: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "tables": list(tables or []),
        "columns": list(columns or []),
    }


def build_prediction(
    *,
    sample_id: Any,
    question: Any,
    dataset_name: str,
    external_knowledge: str | None,
    selected_db_ids: Sequence[str],
    database_selection_status: str,
    schema_linking_status: str,
    stages: Sequence[Mapping[str, Any]],
    final_tables: Sequence[Mapping[str, str]],
    final_columns: Sequence[Mapping[str, str]],
    database_selection_usage: Mapping[str, Any],
    schema_linking_usage: Mapping[str, Any],
    status: str,
    error: Mapping[str, str] | None,
) -> dict[str, Any]:
    prediction = {
        "id": str(sample_id),
        "question": str(question),
        "sql_dialect": canonical_sql_dialect(dataset_name),
        "external_knowledge": external_knowledge,
        "database_selection": {
            "status": database_selection_status,
            "selected_db_ids": list(dict.fromkeys(str(value) for value in selected_db_ids if value)),
        },
        "schema_linking": {
            "status": schema_linking_status,
            "stages": [dict(stage) for stage in stages],
            "final": {
                "tables": [dict(value) for value in final_tables],
                "columns": [dict(value) for value in final_columns],
            },
        },
        "usage": make_usage(database_selection_usage, schema_linking_usage),
        "status": status,
        "error": None if error is None else dict(error),
    }
    _validate_prediction(prediction, index=None)
    return prediction


def _gold_record_id(record: Mapping[str, Any]) -> str:
    raw_id = record.get("id")
    return str(raw_id).strip() if raw_id is not None else ""


def _autolink_prediction_id(gold_id: str, dataset_name: str) -> str:
    if str(dataset_name).strip().lower() == "mmqa":
        return f"mmqa_{gold_id}" if gold_id.isdigit() else ""
    return gold_id


def _source_record_id(record: Mapping[str, Any]) -> str:
    for key in ("instance_id", "id"):
        raw_id = record.get(key)
        if raw_id is not None:
            sample_id = str(raw_id).strip()
            if sample_id:
                return sample_id
    return ""


def _canonical_source_id(record: Mapping[str, Any]) -> str:
    """Return the SQL/evaluation id represented by one dataset row."""

    raw_gold_id = record.get("gold_id")
    if raw_gold_id is not None:
        gold_id = str(raw_gold_id).strip()
        if gold_id:
            return gold_id
    return _source_record_id(record)


def _validated_linkalign_sources(
    source_records: Sequence[Mapping[str, Any]], dataset_name: str
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    ordered: list[Mapping[str, Any]] = []
    source_index: dict[str, Mapping[str, Any]] = {}
    canonical_ids: set[str] = set()
    require_gold_id = str(dataset_name).strip().lower() == "mmqa"
    for index, record in enumerate(source_records):
        if not isinstance(record, Mapping):
            raise PredictionValidationError(
                f"Dataset source row {index} must be a JSON object."
            )
        raw_instance_id = record.get("instance_id")
        instance_id = (
            str(raw_instance_id).strip() if raw_instance_id is not None else ""
        )
        if not instance_id and not require_gold_id:
            instance_id = _source_record_id(record)
        if not instance_id:
            raise PredictionValidationError(
                f"LinkAlign dataset source row {index} has no instance_id. "
                "Pass --dataset-path with the LinkAlign synthesized source data."
            )
        raw_gold_id = record.get("gold_id")
        gold_id = str(raw_gold_id).strip() if raw_gold_id is not None else ""
        if require_gold_id and not gold_id:
            raise PredictionValidationError(
                f"LinkAlign MMQA dataset source row {index} has no gold_id."
            )
        source_id = instance_id
        canonical_id = _canonical_source_id(record)
        if canonical_id in canonical_ids:
            raise PredictionValidationError(
                f"Duplicate canonical dataset id: {canonical_id!r}."
            )
        if source_id in source_index:
            raise PredictionValidationError(
                f"Duplicate dataset source id: {source_id!r}."
            )
        canonical_ids.add(canonical_id)
        ordered.append(record)
        source_index[source_id] = record
    return ordered, source_index


def _validated_gold_records(
    gold_records: Sequence[Mapping[str, Any]], dataset_name: str
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    ordered: list[Mapping[str, Any]] = []
    prediction_index: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(gold_records):
        if not isinstance(record, Mapping):
            raise PredictionValidationError(
                f"Gold dataset row {index} must be a JSON object."
            )
        gold_id = _gold_record_id(record)
        if not gold_id:
            raise PredictionValidationError(
                f"Gold dataset row {index} has no id."
            )
        prediction_id = _autolink_prediction_id(gold_id, dataset_name)
        if not prediction_id:
            raise PredictionValidationError(
                f"Gold dataset row {index} has an invalid AutoLink id: {gold_id!r}."
            )
        if prediction_id in prediction_index:
            raise PredictionValidationError(f"Duplicate gold dataset id: {gold_id!r}.")
        ordered.append(record)
        prediction_index[prediction_id] = record
    return ordered, prediction_index


def _prefailed_native_record(
    *,
    sample_id: str,
    question: Any,
    external_knowledge: Any,
    message: str,
    source_id: str = "",
    predict_db_id: str | None = None,
) -> dict[str, Any]:
    record = {
        "id": sample_id,
        "question": str(question or ""),
        "external_knowledge": external_knowledge,
        "predict_db_id": predict_db_id,
        "predict_tables": [],
        "predict_columns": {},
        "schema_input_error": message,
    }
    if source_id and source_id != sample_id:
        record["source_id"] = source_id
    return record


def _missing_prediction_record(
    source: Mapping[str, Any],
    method: str,
    *,
    source_id: str,
    canonical_id: str,
) -> dict[str, Any]:
    return _prefailed_native_record(
        sample_id=canonical_id,
        source_id=source_id,
        question=source.get("question"),
        external_knowledge=source.get("external_knowledge"),
        message=(
            f"Missing {method} schema-linking prediction for source sample "
            f"{source_id!r}."
        ),
    )


def _order_and_fill_native_records(
    records: Sequence[dict[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    source_id_for: Callable[[Mapping[str, Any]], str],
    canonical_id_for: Callable[[Mapping[str, Any]], str],
) -> list[dict[str, Any]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("id") or "").strip()
        if not sample_id:
            raise PredictionValidationError("Native schema record has no stable id.")
        if sample_id in records_by_id:
            raise PredictionValidationError(
                f"Multiple predictions map to canonical sample id {sample_id!r}."
            )
        records_by_id[sample_id] = record

    result: list[dict[str, Any]] = []
    for source in source_records:
        source_id = source_id_for(source)
        canonical_id = canonical_id_for(source)
        record = records_by_id.pop(canonical_id, None)
        result.append(
            record
            if record is not None
            else _missing_prediction_record(
                source,
                method,
                source_id=source_id,
                canonical_id=canonical_id,
            )
        )
    result.extend(records_by_id.values())
    return result


def _native_autolink_record(
    prediction: Mapping[str, Any],
    *,
    dataset_name: str,
    gold_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    sample_id = str(prediction["id"])
    gold = gold_index.get(sample_id)
    external_knowledge = prediction.get("external_knowledge")
    if gold is None:
        return _prefailed_native_record(
            sample_id=sample_id,
            question=prediction["question"],
            external_knowledge=external_knowledge,
            message=(
                f"AutoLink sample {sample_id!r} could not be matched to a gold record."
            ),
        )

    gold_sample_id = _gold_record_id(gold)

    predicted_question = str(prediction.get("question") or "").strip()
    gold_question = str(gold.get("question") or "").strip()
    if predicted_question != gold_question:
        return _prefailed_native_record(
            sample_id=gold_sample_id,
            question=prediction["question"],
            external_knowledge=external_knowledge,
            message=(
                f"AutoLink sample {sample_id!r} does not match its gold question."
            ),
        )

    if external_knowledge is None:
        external_knowledge = gold.get("external_knowledge")
    gold_db_id = normalize_identifier(gold.get("db_id"))
    if not gold_db_id:
        return _prefailed_native_record(
            sample_id=gold_sample_id,
            question=prediction["question"],
            external_knowledge=external_knowledge,
            message=(
                f"Gold record for AutoLink sample {sample_id!r} has no db_id."
            ),
        )

    snowflake = str(dataset_name).strip().lower() == "spider2"

    def restore_table(table_name: Any) -> str:
        if snowflake:
            return restore_snowflake_table_name(gold_db_id, table_name)
        table = normalize_identifier(table_name)
        return normalize_table_name(gold_db_id, table)

    final = prediction["schema_linking"]["final"]
    filtered_columns = [
        column
        for column in final["columns"]
        if _database_identifier_matches(
            gold_db_id,
            column.get("db_id"),
            snowflake=snowflake,
        )
    ]
    if not filtered_columns:
        return _prefailed_native_record(
            sample_id=gold_sample_id,
            question=prediction["question"],
            external_knowledge=external_knowledge,
            predict_db_id=gold_db_id,
            message=(
                "AutoLink predicted no columns for gold database "
                f"{gold_db_id!r}."
            ),
        )

    column_table_order: list[str] = []
    predicted_columns: dict[str, list[str]] = {}
    for column in filtered_columns:
        table = restore_table(column.get("table"))
        column_name = normalize_identifier(column.get("column"))
        if not table or not column_name:
            continue
        if table not in predicted_columns:
            column_table_order.append(table)
            predicted_columns[table] = []
        if column_name not in predicted_columns[table]:
            predicted_columns[table].append(column_name)

    if not predicted_columns:
        return _prefailed_native_record(
            sample_id=gold_sample_id,
            question=prediction["question"],
            external_knowledge=external_knowledge,
            predict_db_id=gold_db_id,
            message=(
                "AutoLink predicted no usable columns for gold database "
                f"{gold_db_id!r}."
            ),
        )

    predicted_tables: list[str] = []
    for table in final["tables"]:
        if not _database_identifier_matches(
            gold_db_id,
            table.get("db_id"),
            snowflake=snowflake,
        ):
            continue
        table_name = restore_table(table.get("table"))
        if (
            table_name in predicted_columns
            and table_name not in predicted_tables
        ):
            predicted_tables.append(table_name)
    predicted_tables.extend(
        table for table in column_table_order if table not in predicted_tables
    )

    return {
        "id": gold_sample_id,
        "question": prediction["question"],
        "external_knowledge": external_knowledge,
        "predict_db_id": gold_db_id,
        "predict_tables": predicted_tables,
        "predict_columns": predicted_columns,
    }


def _native_non_autolink_record(
    prediction: Mapping[str, Any],
    *,
    dataset_name: str,
    sample_id: str,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_id = _source_record_id(source or {})
    if source is not None:
        predicted_question = str(prediction.get("question") or "").strip()
        source_question = str(source.get("question") or "").strip()
        if predicted_question != source_question:
            return _prefailed_native_record(
                sample_id=sample_id,
                source_id=source_id,
                question=prediction.get("question"),
                external_knowledge=prediction.get("external_knowledge"),
                message=(
                    f"Schema-linking sample {prediction['id']!r} does not match "
                    f"dataset sample {source_id!r}."
                ),
            )

    selected = prediction["database_selection"]["selected_db_ids"]
    db_id = normalize_identifier(selected[0]) if selected else None
    snowflake = str(dataset_name).lower() == "spider2"

    def restore_table(table_name: str) -> str:
        if snowflake and db_id:
            return restore_snowflake_table_name(db_id, table_name)
        return normalize_identifier(table_name)

    final = prediction["schema_linking"]["final"]
    predicted_tables = [
        restore_table(item["table"])
        for item in final["tables"]
        if db_id
        and _database_identifier_matches(
            db_id,
            item.get("db_id"),
            snowflake=snowflake,
        )
    ]
    predicted_columns: dict[str, list[str]] = {}
    for item in final["columns"]:
        if not db_id or not _database_identifier_matches(
            db_id,
            item.get("db_id"),
            snowflake=snowflake,
        ):
            continue
        table = restore_table(item["table"])
        predicted_columns.setdefault(table, []).append(item["column"])
    record = {
        "id": sample_id,
        "question": prediction["question"],
        "external_knowledge": prediction["external_knowledge"],
        "predict_db_id": db_id,
        "predict_tables": predicted_tables,
        "predict_columns": predicted_columns,
    }
    if source_id and source_id != sample_id:
        record["source_id"] = source_id
    return record


def _native_non_autolink_records(
    unified: Mapping[str, Any],
    dataset_name: str,
    source_records: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    predictions = unified["predictions"]
    if unified["method"] != "linkalign":
        return [
            _native_non_autolink_record(
                prediction,
                dataset_name=dataset_name,
                sample_id=prediction["id"],
                source=None,
            )
            for prediction in predictions
        ]
    if source_records is None:
        raise PredictionValidationError(
            "LinkAlign SQL generation requires dataset source records containing "
            "the instance_id-to-gold_id mapping."
        )

    ordered_sources, source_id_index = _validated_linkalign_sources(
        source_records, dataset_name
    )
    records: list[dict[str, Any]] = []
    for prediction in predictions:
        prediction_id = str(prediction["id"])
        source = source_id_index.get(prediction_id)
        if source is None:
            raise PredictionValidationError(
                f"LinkAlign sample {prediction_id!r} is absent from the dataset "
                "source mapping. Pass --dataset-path with matching instance_id and "
                "gold_id fields."
            )

        canonical_id = _canonical_source_id(source)
        records.append(
            _native_non_autolink_record(
                prediction,
                dataset_name=dataset_name,
                sample_id=canonical_id,
                source=source,
            )
        )
    return _order_and_fill_native_records(
        records,
        ordered_sources,
        method=str(unified["method"]),
        source_id_for=_source_record_id,
        canonical_id_for=_canonical_source_id,
    )


def unified_to_native_schema_records(
    payload: Mapping[str, Any],
    dataset_name: str,
    *,
    gold_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Adapt unified SL records to the existing SQL generator input contract."""

    unified = validate_prediction_file(payload)
    if unified["dataset"].strip().lower() != str(dataset_name).strip().lower():
        raise PredictionValidationError(
            "Prediction dataset does not match the SQL generation dataset."
        )
    if unified["method"] == "autolink":
        if gold_records is None:
            raise PredictionValidationError(
                "AutoLink SQL generation requires gold records for database filtering."
            )
        ordered_gold_records, gold_index = _validated_gold_records(
            gold_records, dataset_name
        )
        records = [
            _native_autolink_record(
                prediction,
                dataset_name=dataset_name,
                gold_index=gold_index,
            )
            for prediction in unified["predictions"]
        ]
        return _order_and_fill_native_records(
            records,
            ordered_gold_records,
            method="autolink",
            source_id_for=_gold_record_id,
            canonical_id_for=_gold_record_id,
        )
    return _native_non_autolink_records(unified, dataset_name, gold_records)


__all__ = [
    "METHOD_NAMES",
    "PredictionValidationError",
    "build_prediction",
    "build_prediction_path",
    "canonical_sql_dialect",
    "initialize_prediction_file",
    "make_stage",
    "make_usage",
    "normalize_columns",
    "normalize_identifier",
    "normalize_snowflake_table_name",
    "normalize_table_name",
    "normalize_tables",
    "restore_snowflake_table_name",
    "safe_path_component",
    "snowflake_identifier_key",
    "split_qualified_identifier",
    "replace_predictions",
    "upsert_prediction",
    "usage_from_efficiency",
    "unified_to_native_schema_records",
    "validate_prediction_file",
]
