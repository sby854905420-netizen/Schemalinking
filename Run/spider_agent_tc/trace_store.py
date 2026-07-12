from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from Utils.json_utils import atomic_write_json, load_json, upsert_ordered_record


def write_agent_failure_trace(
    output_path: Path,
    *,
    sample_id: str,
    error: str,
    trace: Sequence[Mapping[str, Any]],
) -> Path:
    """Persist exact failed-round responses without changing prediction schemas."""
    trace_path = output_path.with_name(f"{output_path.stem}_agent_failures.json")
    payload: dict[str, Any] = {"failures": []}
    if trace_path.is_file():
        existing = load_json(trace_path)
        if isinstance(existing, dict) and isinstance(existing.get("failures"), list):
            payload = existing
    record = {
        "id": str(sample_id),
        "error": error,
        "rounds": [dict(item) for item in trace],
    }
    payload["failures"] = upsert_ordered_record(payload["failures"], record)
    atomic_write_json(trace_path, payload)
    return trace_path


__all__ = ["write_agent_failure_trace"]
