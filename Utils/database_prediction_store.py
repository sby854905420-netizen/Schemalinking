"""Stable database-selection predictions, separate from timestamped run logs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config import RESULTS_ROOT
from Utils.json_utils import atomic_write_json, load_json, upsert_ordered_record
from Utils.prediction_store import safe_path_component, usage_from_efficiency
from Utils.value_utils import get_row_value
from Utils.artifact_paths import reject_logs_prediction_input


DATABASE_METHODS = {
    "baseline",
    "global_coarse_rerank",
    "global_coarse_pruning",
}
DATABASE_STATUSES = {"success", "empty", "failed"}


class DatabasePredictionValidationError(ValueError):
    pass


def _validate_method(method: str) -> None:
    if method not in DATABASE_METHODS:
        raise DatabasePredictionValidationError(
            f"method must be one of {sorted(DATABASE_METHODS)}, got {method!r}."
        )


def build_database_prediction_path(
    method: str,
    dataset_name: str,
    model_name: str,
    *,
    results_root: Path | str | None = None,
) -> Path:
    """Build ``results/db/<method>/<dataset>/<model>/prediction.json``."""
    _validate_method(method)
    root = Path(results_root) if results_root is not None else RESULTS_ROOT
    return (
        root
        / "db"
        / safe_path_component(method)
        / safe_path_component(dataset_name)
        / safe_path_component(model_name)
        / "prediction.json"
    )


def resolve_database_prediction_input(
    *,
    explicit_path: Path | None,
    method: str,
    dataset_name: str,
    model_name: str,
    results_root: Path | str | None = None,
) -> Path:
    """Resolve an explicit or canonical DB prediction; Logs are never inputs."""
    if explicit_path is not None:
        return reject_logs_prediction_input(explicit_path)
    unified_path = build_database_prediction_path(
        method,
        dataset_name,
        model_name,
        results_root=results_root,
    )
    if unified_path.is_file():
        return unified_path
    raise FileNotFoundError(
        f"Could not find unified database prediction: {unified_path}. "
        "Run database retrieval first or pass --input-path."
    )


def initialize_database_prediction_file(
    path: Path | str,
    *,
    dataset_name: str,
    method: str,
    model_name: str,
) -> Path:
    _validate_method(method)
    output_path = Path(path)
    header = {
        "dataset": str(dataset_name),
        "method": str(method),
        "model_name": str(model_name),
    }
    if output_path.is_file():
        payload = validate_database_prediction_file(output_path)
        if any(payload[key] != value for key, value in header.items()):
            raise DatabasePredictionValidationError(
                f"Existing database prediction has an incompatible header: {output_path}"
            )
        return output_path
    atomic_write_json(output_path, {**header, "predictions": []})
    return output_path


def build_database_prediction(
    *,
    sample_id: Any,
    question: Any,
    predicted_db_id: Any,
    efficiency: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    db_id = "" if predicted_db_id is None else str(predicted_db_id).strip()
    message = str(error).strip() if error else None
    status = "failed" if message else ("success" if db_id else "empty")
    prediction = {
        "id": str(sample_id),
        "question": str(question or ""),
        "predicted_db_id": db_id or None,
        "status": status,
        "usage": usage_from_efficiency(efficiency),
        "error": (
            None
            if message is None
            else {"stage": "database_selection", "message": message}
        ),
    }
    _validate_prediction(prediction, "prediction")
    return prediction


def database_prediction_from_native(record: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = get_row_value(record, "id", "instance_id")
    if sample_id is None:
        raise DatabasePredictionValidationError("Native database record is missing id.")
    error = record.get("database_error")
    return build_database_prediction(
        sample_id=sample_id,
        question=get_row_value(record, "question") or "",
        predicted_db_id=get_row_value(record, "predict_db_id"),
        efficiency=record.get("efficiency"),
        error=str(error) if error else None,
    )


def upsert_database_prediction(
    path: Path | str, prediction: Mapping[str, Any]
) -> None:
    output_path = Path(path)
    payload = validate_database_prediction_file(output_path)
    normalized = dict(prediction)
    _validate_prediction(normalized, "prediction")
    payload["predictions"] = upsert_ordered_record(payload["predictions"], normalized)
    validate_database_prediction_file(payload)
    atomic_write_json(output_path, payload)


def replace_database_predictions(
    path: Path | str, predictions: Sequence[Mapping[str, Any]]
) -> None:
    output_path = Path(path)
    payload = validate_database_prediction_file(output_path)
    payload["predictions"] = [dict(record) for record in predictions]
    validate_database_prediction_file(payload)
    atomic_write_json(output_path, payload)


def validate_database_prediction_file(
    path_or_payload: Path | str | Mapping[str, Any]
) -> dict[str, Any]:
    payload = (
        dict(path_or_payload)
        if isinstance(path_or_payload, Mapping)
        else load_json(path_or_payload)
    )
    required = {"dataset", "method", "model_name", "predictions"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise DatabasePredictionValidationError(
            f"Top-level keys must be exactly {sorted(required)}."
        )
    if not isinstance(payload["dataset"], str) or not payload["dataset"].strip():
        raise DatabasePredictionValidationError("dataset must be a non-empty string.")
    if not isinstance(payload["model_name"], str) or not payload["model_name"].strip():
        raise DatabasePredictionValidationError("model_name must be a non-empty string.")
    _validate_method(payload["method"])
    if not isinstance(payload["predictions"], list):
        raise DatabasePredictionValidationError("predictions must be a list.")
    seen: set[str] = set()
    for index, prediction in enumerate(payload["predictions"]):
        _validate_prediction(prediction, f"predictions[{index}]")
        if prediction["id"] in seen:
            raise DatabasePredictionValidationError(
                f"Duplicate prediction id: {prediction['id']}"
            )
        seen.add(prediction["id"])
    return payload


def _validate_prediction(prediction: Any, where: str) -> None:
    required = {"id", "question", "predicted_db_id", "status", "usage", "error"}
    if not isinstance(prediction, dict) or set(prediction) != required:
        raise DatabasePredictionValidationError(f"{where} has an invalid shape.")
    if not isinstance(prediction["id"], str) or not prediction["id"]:
        raise DatabasePredictionValidationError(f"{where}.id must be non-empty.")
    if not isinstance(prediction["question"], str):
        raise DatabasePredictionValidationError(f"{where}.question must be a string.")
    db_id = prediction["predicted_db_id"]
    if db_id is not None and (not isinstance(db_id, str) or not db_id.strip()):
        raise DatabasePredictionValidationError(f"{where}.predicted_db_id is invalid.")
    if prediction["status"] not in DATABASE_STATUSES:
        raise DatabasePredictionValidationError(f"{where}.status is invalid.")
    if prediction["status"] == "success" and db_id is None:
        raise DatabasePredictionValidationError(f"{where} success requires predicted_db_id.")
    usage = prediction["usage"]
    if not isinstance(usage, dict) or set(usage) != {"elapsed_seconds", "total_tokens"}:
        raise DatabasePredictionValidationError(f"{where}.usage has an invalid shape.")
    elapsed = usage["elapsed_seconds"]
    tokens = usage["total_tokens"]
    if elapsed is not None and (not isinstance(elapsed, (int, float)) or elapsed < 0):
        raise DatabasePredictionValidationError(f"{where}.usage elapsed is invalid.")
    if tokens is not None and (not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0):
        raise DatabasePredictionValidationError(f"{where}.usage tokens is invalid.")
    error = prediction["error"]
    if error is not None and (
        not isinstance(error, dict)
        or set(error) != {"stage", "message"}
        or error["stage"] != "database_selection"
        or not isinstance(error["message"], str)
        or not error["message"]
    ):
        raise DatabasePredictionValidationError(f"{where}.error is invalid.")


def database_predictions_to_native_records(
    path_or_payload: Path | str | Mapping[str, Any]
) -> list[dict[str, Any]]:
    payload = validate_database_prediction_file(path_or_payload)
    records: list[dict[str, Any]] = []
    for prediction in payload["predictions"]:
        records.append(
            {
                "model": payload["model_name"],
                "id": prediction["id"],
                "question": prediction["question"],
                "predict_db_id": prediction["predicted_db_id"],
                "database_error": (
                    prediction["error"]["message"] if prediction["error"] else None
                ),
                "efficiency": {
                    "sample_elapsed_seconds": prediction["usage"]["elapsed_seconds"],
                    "llm_total_tokens": prediction["usage"]["total_tokens"],
                },
            }
        )
    return records


def load_database_prediction_records(path: Path | str) -> list[dict[str, Any]]:
    """Load the unified DB artifact into the internal row contract."""
    return database_predictions_to_native_records(path)


__all__ = [
    "DATABASE_METHODS",
    "DatabasePredictionValidationError",
    "build_database_prediction",
    "build_database_prediction_path",
    "database_prediction_from_native",
    "database_predictions_to_native_records",
    "initialize_database_prediction_file",
    "load_database_prediction_records",
    "replace_database_predictions",
    "resolve_database_prediction_input",
    "upsert_database_prediction",
    "validate_database_prediction_file",
]
