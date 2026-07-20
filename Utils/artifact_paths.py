"""Central path boundary for generated prediction artifacts."""

from __future__ import annotations

from pathlib import Path

from config import RESULTS_ROOT


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

__all__ = ["require_results_output"]
