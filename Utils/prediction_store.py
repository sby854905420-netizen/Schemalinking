"""Unified Schema Linking prediction storage for MDBlink variants.

The unified artifact intentionally contains database selection and schema
linking only. SQL predictions are stored separately under ``results/sql``.
Writers are expected to be called by the main process; this module provides
atomic replacement, not cross-process job coordination.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config import RESULTS_ROOT
from Utils.json_utils import atomic_write_json, load_json, upsert_ordered_record


METHOD_NAMES = {"prompt_baseline", "table_to_column", "rag_column_retrieval"}
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


def normalize_identifier(value: Any) -> str:
    """Remove display-only fences and a single unambiguous wrapper quote."""

    text = str(value or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    for quote in ('"', "'", "`"):
        if len(text) >= 2 and text.startswith(quote) and text.endswith(quote) and text.count(quote) == 2:
            text = text[1:-1].strip()
            break
    return text


def normalize_table_name(db_id: str, table_name: Any) -> str:
    table = normalize_identifier(table_name)
    prefix = f"{db_id}."
    return table[len(prefix) :] if db_id and table.startswith(prefix) else table


def normalize_tables(db_id: str, table_names: Sequence[Any] | None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_table in table_names or []:
        table = normalize_table_name(db_id, raw_table)
        key = (db_id, table)
        if not db_id or not table or key in seen:
            continue
        seen.add(key)
        records.append({"db_id": db_id, "table": table})
    return records


def normalize_columns(
    db_id: str, columns_by_table: Mapping[Any, Sequence[Any]] | None
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    if not isinstance(columns_by_table, Mapping):
        return records
    for raw_table, raw_columns in columns_by_table.items():
        if not isinstance(raw_columns, Sequence) or isinstance(raw_columns, (str, bytes)):
            continue
        table = normalize_table_name(db_id, raw_table)
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


def unified_to_native_schema_records(
    payload: Mapping[str, Any], dataset_name: str
) -> list[dict[str, Any]]:
    """Adapt unified SL records to the existing SQL generator input contract."""

    unified = validate_prediction_file(payload)
    snowflake = str(dataset_name).lower() == "spider2"
    records: list[dict[str, Any]] = []
    for prediction in unified["predictions"]:
        selected = prediction["database_selection"]["selected_db_ids"]
        db_id = selected[0] if selected else None

        def restore_table(table_name: str) -> str:
            if snowflake and db_id and not table_name.startswith(f"{db_id}."):
                return f"{db_id}.{table_name}"
            return table_name

        final = prediction["schema_linking"]["final"]
        predicted_tables = [restore_table(item["table"]) for item in final["tables"]]
        predicted_columns: dict[str, list[str]] = {}
        for item in final["columns"]:
            table = restore_table(item["table"])
            predicted_columns.setdefault(table, []).append(item["column"])
        records.append(
            {
                "id": prediction["id"],
                "question": prediction["question"],
                "predict_db_id": db_id,
                "predict_tables": predicted_tables,
                "predict_columns": predicted_columns,
            }
        )
    return records


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
    "normalize_table_name",
    "normalize_tables",
    "safe_path_component",
    "replace_predictions",
    "upsert_prediction",
    "usage_from_efficiency",
    "unified_to_native_schema_records",
    "validate_prediction_file",
]
