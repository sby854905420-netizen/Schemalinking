"""Overwrite-only structured decision traces for the MDB-link pipeline."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from config import RESULTS_ROOT
from Utils.json_utils import atomic_write_json
from Utils.prediction_store import safe_path_component


TRACE_SCHEMA_VERSION = "1.0"
TRACE_STATUSES = {"success", "skipped", "failed"}


@dataclass(frozen=True)
class TracePaths:
    directory: Path
    metadata: Path
    events: Path
    summary: Path


def build_trace_paths(
    scope: str,
    method: str,
    dataset_name: str,
    model_name: str,
    *,
    results_root: Path | str | None = None,
) -> TracePaths:
    """Return the stable, overwrite-only trace paths for one experiment."""

    if scope not in {"db", "sl"}:
        raise ValueError(f"Unsupported trace scope: {scope!r}")
    root = Path(results_root) if results_root is not None else RESULTS_ROOT
    directory = (
        root
        / "traces"
        / scope
        / safe_path_component(method)
        / safe_path_component(dataset_name)
        / safe_path_component(model_name)
    )
    return TracePaths(
        directory=directory,
        metadata=directory / "metadata.json",
        events=directory / "events.jsonl",
        summary=directory / "summary.json",
    )


def initialize_trace(paths: TracePaths, metadata: Mapping[str, Any]) -> None:
    """Replace every trace artifact from the previous run."""

    paths.directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        **dict(metadata),
        "status": "running",
    }
    _ensure_strict_json(payload)
    atomic_write_json(paths.metadata, payload)
    paths.events.write_text("", encoding="utf-8")
    paths.summary.unlink(missing_ok=True)


def remove_trace(paths: TracePaths) -> None:
    """Remove one stable trace when the selected method must not emit traces."""

    for path in (paths.metadata, paths.events, paths.summary):
        path.unlink(missing_ok=True)
    try:
        paths.directory.rmdir()
    except (FileNotFoundError, OSError):
        pass


def append_event(
    events_path: Path | str,
    *,
    sample_id: Any,
    stage: str,
    status: str = "success",
    output: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one strict-JSON stage event and flush it immediately."""

    if status not in TRACE_STATUSES:
        raise ValueError(f"Unsupported trace event status: {status!r}")
    if not str(sample_id):
        raise ValueError("Trace event sample_id must be non-empty.")
    if not str(stage).strip():
        raise ValueError("Trace event stage must be non-empty.")
    event = {
        "sample_id": str(sample_id),
        "stage": str(stage),
        "status": status,
        "output": dict(output or {}),
        "diagnostics": dict(diagnostics or {}),
        "error": dict(error) if error else None,
    }
    _validate_event(event, "new event")
    serialized = json.dumps(event, ensure_ascii=False, allow_nan=False)
    path = Path(events_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
    return event


def load_events(events_path: Path | str) -> list[dict[str, Any]]:
    path = Path(events_path)
    if not path.is_file():
        raise FileNotFoundError(f"Structured trace does not exist: {path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        _validate_event(event, f"{path}:{line_number}")
        events.append(event)
    return events


def append_events(
    events_path: Path | str, events: Iterable[Mapping[str, Any]]
) -> int:
    """Append a previously loaded and validated event snapshot."""

    target = Path(events_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized_events: list[str] = []
    for event in events:
        event_payload = dict(event)
        _validate_event(event_payload, "event snapshot")
        serialized_events.append(
            json.dumps(event_payload, ensure_ascii=False, allow_nan=False)
        )
    with target.open("a", encoding="utf-8") as handle:
        for serialized_event in serialized_events:
            handle.write(serialized_event)
            handle.write("\n")
        handle.flush()
    return len(serialized_events)


def finalize_trace(paths: TracePaths, summary: Mapping[str, Any]) -> None:
    summary_payload = dict(summary)
    _ensure_strict_json(summary_payload)
    atomic_write_json(paths.summary, summary_payload)
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    metadata["status"] = "completed"
    _ensure_strict_json(metadata)
    atomic_write_json(paths.metadata, metadata)


def mark_trace_failed(
    paths: TracePaths, error: BaseException | Mapping[str, Any] | None = None
) -> None:
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    metadata["status"] = "failed"
    if error is not None:
        error_payload = (
            {"type": type(error).__name__, "message": str(error)}
            if isinstance(error, BaseException)
            else dict(error)
        )
        if (
            set(error_payload) != {"type", "message"}
            or not isinstance(error_payload["type"], str)
            or not error_payload["type"]
            or not isinstance(error_payload["message"], str)
        ):
            raise ValueError("Trace failure error must contain type and message.")
        metadata["error"] = error_payload
    _ensure_strict_json(metadata)
    atomic_write_json(paths.metadata, metadata)


def build_trace_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build database/table/column metrics from the compact stage diagnostics."""

    sample_ids = {
        str(event["sample_id"])
        for event in events
        if event.get("stage") == "sample.input"
    }
    final_events = [event for event in events if event.get("stage") == "sample.final"]
    database_final_events = [
        event for event in events if event.get("stage") == "database.final"
    ]

    database = {
        "round1_hrc_recall": _boolean_rate(
            events, sample_ids, "database.round1.hrc", "gold_in_output"
        ),
        "round1_filter_recall": _boolean_rate(
            events, sample_ids, "database.round1.pruning", "gold_after_filter"
        ),
        "round1_top10_recall": _boolean_rate(
            events, sample_ids, "database.round1.pruning", "gold_in_top10"
        ),
        "round1_rerank_recall": _boolean_rate(
            events, sample_ids, "database.round1.rerank", "gold_in_selected"
        ),
        "round2_hrc_recall": _boolean_rate(
            events,
            _executed_sample_ids(events, "database.round2.hrc"),
            "database.round2.hrc",
            "gold_in_output",
        ),
        "round2_pruning_recall": _boolean_rate(
            events,
            _executed_sample_ids(events, "database.round2.pruning"),
            "database.round2.pruning",
            "gold_in_selected",
        ),
        "LA": _boolean_rate(events, sample_ids, "database.final", "correct"),
    }

    table_events = [event for event in events if event.get("stage") == "table.prediction"]
    column_events = [
        event for event in events if event.get("stage") == "column.prediction"
    ]
    table = {
        "exact_match": _diagnostic_rate(table_events, "table_exact_match"),
        "recall": _micro_recall(table_events, "matched_table_count", "gold_table_count"),
        "recall_given_correct_database": _micro_recall(
            [event for event in table_events if _diagnostic(event, "database_correct")],
            "matched_table_count",
            "gold_table_count",
        ),
    }
    column = {
        "exact_match": _diagnostic_rate(column_events, "column_exact_match"),
        "recall": _micro_recall(
            column_events, "matched_column_count", "gold_column_count"
        ),
        "recall_given_correct_database": _micro_recall(
            [event for event in column_events if _diagnostic(event, "database_correct")],
            "matched_column_count",
            "gold_column_count",
        ),
        "recall_given_gold_columns_available": _micro_recall(
            column_events,
            "matched_available_column_count",
            "gold_columns_available_count",
        ),
    }

    outcome_events = final_events or database_final_events
    first_error_stage_counts = Counter(
        str(event.get("diagnostics", {}).get("first_error_stage"))
        for event in outcome_events
        if event.get("diagnostics", {}).get("first_error_stage")
        and event.get("diagnostics", {}).get("first_error_stage") != "correct"
    )
    result = {
        "total_samples": len(sample_ids),
        "database_completed_samples": len(database_final_events),
        "completed_samples": len(outcome_events),
        "failed_samples": sum(
            event.get("status") == "failed" for event in outcome_events
        ),
        "database": database,
        "first_error_stage_counts": dict(sorted(first_error_stage_counts.items())),
    }
    if table_events or column_events:
        result["table"] = table
        result["column"] = column
    return result


def _validate_event(event: Any, where: str) -> None:
    required = {"sample_id", "stage", "status", "output", "diagnostics", "error"}
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError(f"Invalid structured trace event at {where}.")
    if not isinstance(event["sample_id"], str) or not event["sample_id"]:
        raise ValueError(f"Invalid sample_id at {where}.")
    if not isinstance(event["stage"], str) or not event["stage"]:
        raise ValueError(f"Invalid stage at {where}.")
    if event["status"] not in TRACE_STATUSES:
        raise ValueError(f"Invalid status at {where}.")
    if not isinstance(event["output"], dict) or not isinstance(
        event["diagnostics"], dict
    ):
        raise ValueError(f"Invalid output/diagnostics at {where}.")
    error = event["error"]
    if error is not None and (
        not isinstance(error, dict)
        or set(error) != {"type", "message"}
        or not isinstance(error["type"], str)
        or not error["type"]
        or not isinstance(error["message"], str)
    ):
        raise ValueError(f"Invalid error at {where}.")


def _diagnostic(event: Mapping[str, Any], key: str) -> Any:
    diagnostics = event.get("diagnostics")
    return diagnostics.get(key) if isinstance(diagnostics, Mapping) else None


def _boolean_rate(
    events: Iterable[Mapping[str, Any]],
    sample_ids: set[str],
    stage: str,
    key: str,
) -> float | None:
    if not sample_ids:
        return None
    values_by_sample = {
        str(event["sample_id"]): bool(_diagnostic(event, key))
        for event in events
        if event.get("stage") == stage
        and isinstance(_diagnostic(event, key), bool)
    }
    return sum(values_by_sample.get(sample_id, False) for sample_id in sample_ids) / len(
        sample_ids
    )


def _diagnostic_rate(events: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        bool(_diagnostic(event, key))
        for event in events
        if isinstance(_diagnostic(event, key), bool)
    ]
    return sum(values) / len(values) if values else None


def _executed_sample_ids(
    events: Iterable[Mapping[str, Any]], stage: str
) -> set[str]:
    return {
        str(event["sample_id"])
        for event in events
        if event.get("stage") == stage and event.get("status") != "skipped"
    }


def _micro_recall(
    events: Sequence[Mapping[str, Any]], matched_key: str, gold_key: str
) -> float | None:
    matched = 0
    gold = 0
    for event in events:
        matched_value = _diagnostic(event, matched_key)
        gold_value = _diagnostic(event, gold_key)
        if not isinstance(matched_value, int) or isinstance(matched_value, bool):
            continue
        if not isinstance(gold_value, int) or isinstance(gold_value, bool):
            continue
        matched += matched_value
        gold += gold_value
    return matched / gold if gold else None


def _ensure_strict_json(payload: Any) -> None:
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "TracePaths",
    "append_event",
    "append_events",
    "build_trace_paths",
    "build_trace_summary",
    "finalize_trace",
    "initialize_trace",
    "load_events",
    "mark_trace_failed",
    "remove_trace",
]
