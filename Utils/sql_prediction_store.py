"""Standalone SQL prediction artifacts keyed by SQL and SL configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config import RESULTS_ROOT
from Utils.json_utils import (
    atomic_write_json,
    load_json,
    upsert_ordered_record,
)
from Utils.prediction_store import METHOD_NAMES, safe_path_component


SQL_STATUSES = {"success", "empty", "failed"}


def build_sql_prediction_path(
    sql_method: str,
    dataset_name: str,
    sql_model_name: str,
    schema_linking_method: str,
    schema_linking_model_name: str,
    *,
    results_root: Path | str | None = None,
) -> Path:
    if schema_linking_method not in METHOD_NAMES:
        raise ValueError(f"Unknown schema-linking method: {schema_linking_method!r}")
    root = Path(results_root) if results_root is not None else RESULTS_ROOT
    return (
        root
        / "sql"
        / safe_path_component(sql_method)
        / safe_path_component(dataset_name)
        / safe_path_component(sql_model_name)
        / safe_path_component(schema_linking_method)
        / f"{safe_path_component(schema_linking_model_name)}.json"
    )


def initialize_sql_prediction_file(
    path: Path | str,
    *,
    dataset_name: str,
    sql_method: str,
    sql_model_name: str,
    schema_linking_method: str,
    schema_linking_model_name: str,
) -> Path:
    if schema_linking_method not in METHOD_NAMES:
        raise ValueError(f"Unknown schema-linking method: {schema_linking_method!r}")
    output_path = Path(path)
    header = {
        "dataset": str(dataset_name),
        "method": str(sql_method),
        "model_names": {
            "sql_generation": str(sql_model_name),
            "schema_linking": str(schema_linking_model_name),
        },
        "schema_linking_method": str(schema_linking_method),
    }
    if output_path.is_file():
        payload = validate_sql_prediction_file(output_path)
        if any(payload[key] != value for key, value in header.items()):
            raise ValueError(f"Existing SQL prediction has an incompatible header: {output_path}")
        return output_path
    atomic_write_json(output_path, {**header, "predictions": []})
    return output_path


def upsert_sql_prediction(
    path: Path | str,
    *,
    sample_id: Any,
    predicted_sql: str | None,
    status: str,
    error: str | None = None,
    agent_rounds: int | None = None,
    total_tokens: int | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    output_path = Path(path)
    payload = validate_sql_prediction_file(output_path)
    normalized_status = status if status in SQL_STATUSES else "failed"
    record = {
        "id": str(sample_id),
        "predicted_sql": predicted_sql if predicted_sql else None,
        "status": normalized_status,
        "error": (
            None
            if error is None
            else {"stage": "sql_generation", "message": str(error)}
        ),
        "agent_rounds": None if agent_rounds is None else int(agent_rounds),
        "total_tokens": None if total_tokens is None else int(total_tokens),
        "elapsed_seconds": (
            None if elapsed_seconds is None else round(float(elapsed_seconds), 6)
        ),
    }
    payload["predictions"] = upsert_ordered_record(payload["predictions"], record)
    validate_sql_prediction_file(payload)
    atomic_write_json(output_path, payload)


def replace_sql_predictions(
    path: Path | str, predictions: Sequence[Mapping[str, Any]]
) -> None:
    """Atomically replace all SQL prediction records."""

    output_path = Path(path)
    payload = validate_sql_prediction_file(output_path)
    payload["predictions"] = [dict(prediction) for prediction in predictions]
    validate_sql_prediction_file(payload)
    atomic_write_json(output_path, payload)


def validate_sql_prediction_file(
    path_or_payload: Path | str | Mapping[str, Any]
) -> dict[str, Any]:
    if isinstance(path_or_payload, Mapping):
        payload = dict(path_or_payload)
    else:
        payload = load_json(path_or_payload)
    required = {
        "dataset",
        "method",
        "model_names",
        "schema_linking_method",
        "predictions",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("SQL prediction file has an invalid top-level shape.")
    for key in ("dataset", "method", "schema_linking_method"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"SQL prediction {key} must be a non-empty string.")
    if payload["schema_linking_method"] not in METHOD_NAMES:
        raise ValueError("SQL prediction schema_linking_method is invalid.")
    if not isinstance(payload["model_names"], dict) or set(payload["model_names"]) != {
        "sql_generation",
        "schema_linking",
    }:
        raise ValueError("SQL prediction model_names has an invalid shape.")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in payload["model_names"].values()
    ):
        raise ValueError("SQL prediction model names must be non-empty strings.")
    if not isinstance(payload["predictions"], list):
        raise ValueError("SQL predictions must be a list.")
    seen: set[str] = set()
    for record in payload["predictions"]:
        legacy_fields = {
            "id",
            "predicted_sql",
            "status",
            "error",
        }
        metric_fields = {"agent_rounds", "total_tokens", "elapsed_seconds"}
        if not isinstance(record, dict) or frozenset(record) not in {
            frozenset(legacy_fields),
            frozenset(legacy_fields | metric_fields),
        }:
            raise ValueError("SQL prediction contains an invalid record.")
        if not isinstance(record["id"], str) or not record["id"] or record["id"] in seen:
            raise ValueError("SQL prediction ids must be non-empty and unique.")
        seen.add(record["id"])
        if record["status"] not in SQL_STATUSES:
            raise ValueError("SQL prediction status is invalid.")
        if record["predicted_sql"] is not None and not isinstance(
            record["predicted_sql"], str
        ):
            raise ValueError("predicted_sql must be a string or null.")
        if metric_fields.issubset(record):
            for key in ("agent_rounds", "total_tokens"):
                value = record[key]
                if value is not None and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise ValueError(f"SQL prediction {key} is invalid.")
            elapsed_seconds = record["elapsed_seconds"]
            if elapsed_seconds is not None and (
                not isinstance(elapsed_seconds, (int, float))
                or isinstance(elapsed_seconds, bool)
                or elapsed_seconds < 0
            ):
                raise ValueError("SQL prediction elapsed_seconds is invalid.")
        error = record["error"]
        if error is not None and (
            not isinstance(error, dict)
            or set(error) != {"stage", "message"}
            or error["stage"] != "sql_generation"
            or not isinstance(error["message"], str)
        ):
            raise ValueError("SQL prediction error is invalid.")
    return payload


__all__ = [
    "build_sql_prediction_path",
    "initialize_sql_prediction_file",
    "replace_sql_predictions",
    "upsert_sql_prediction",
    "validate_sql_prediction_file",
]
