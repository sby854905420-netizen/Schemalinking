"""Central path boundaries for predictions and runtime logs."""

from __future__ import annotations

from pathlib import Path

from config import LOGS_ROOT, RESULTS_ROOT


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def require_results_output(path: Path | str) -> Path:
    """Require a prediction output to live under the project results root."""
    resolved = Path(path)
    if not _is_within(resolved, RESULTS_ROOT):
        raise ValueError(f"Prediction outputs must be stored under {RESULTS_ROOT}: {resolved}")
    return resolved


def reject_logs_prediction_input(path: Path | str) -> Path:
    """Reject any attempt to consume a prediction from the runtime Logs root."""
    resolved = Path(path)
    if _is_within(resolved, LOGS_ROOT):
        raise ValueError(f"Logs cannot be used as a prediction input: {resolved}")
    return resolved


__all__ = ["reject_logs_prediction_input", "require_results_output"]
