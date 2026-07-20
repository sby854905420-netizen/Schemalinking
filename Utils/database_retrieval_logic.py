"""Dependency-light budgeting, pruning, and routing for database retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from config import HRC_RETRIEVAL_RATIO, ROUND1_DATABASE_TOP_K, ROUND1_RERANK_TOP_K


@dataclass(frozen=True)
class RetrievalBudget:
    candidate_column_count: int
    top_k: int


@dataclass(frozen=True)
class PruningResult:
    database_stats: list[dict[str, Any]]
    filtered_db_ids: list[str]
    selected_db_ids: list[str]


def resolve_hrc_budget(
    db_counts: Mapping[str, int],
    *,
    candidate_db_ids: Sequence[str] | None,
    cap: int,
    ratio: float = HRC_RETRIEVAL_RATIO,
) -> RetrievalBudget:
    """Resolve ceil(10% of candidate columns), bounded by the round cap."""

    if not 0 < ratio <= 1:
        raise ValueError("HRC retrieval ratio must be in (0, 1].")
    if cap <= 0:
        raise ValueError("HRC retrieval cap must be positive.")
    db_ids = (
        list(db_counts)
        if candidate_db_ids is None
        else list(dict.fromkeys(candidate_db_ids))
    )
    candidate_column_count = sum(
        max(0, int(db_counts.get(db_id, 0))) for db_id in db_ids
    )
    top_k = (
        min(cap, max(1, math.ceil(candidate_column_count * ratio)))
        if candidate_column_count
        else 0
    )
    return RetrievalBudget(candidate_column_count, top_k)


def resolve_round1_rerank_top_k(candidate_db_count: int) -> int:
    if candidate_db_count < 0:
        raise ValueError("Candidate database count cannot be negative.")
    return 1 if candidate_db_count <= ROUND1_RERANK_TOP_K else ROUND1_RERANK_TOP_K


def should_run_round2(candidate_db_count: int) -> bool:
    if candidate_db_count < 0:
        raise ValueError("Candidate database count cannot be negative.")
    return candidate_db_count > ROUND1_RERANK_TOP_K


def database_pruning(
    hrc_points: Sequence[Any],
    *,
    min_hit_count: int,
    min_sim_ratio: float,
    top_k: int = ROUND1_DATABASE_TOP_K,
) -> PruningResult:
    """Filter database support, rank by relevance, then retain Top-K."""

    if not 0 <= min_sim_ratio <= 1:
        raise ValueError("Similarity quantile must be in [0, 1].")
    if min_hit_count < 0 or top_k < 0:
        raise ValueError("Pruning counts cannot be negative.")

    grouped: dict[str, list[float]] = {}
    for point in hrc_points:
        db_id, score = _point_database_score(point)
        if db_id is not None and score is not None:
            grouped.setdefault(db_id, []).append(score)
    if not grouped:
        return PruningResult([], [], [])

    database_stats = [
        {
            "db_id": db_id,
            "hit_count": len(scores),
            "max_score": max(scores),
            "score_sum": sum(scores),
        }
        for db_id, scores in grouped.items()
    ]
    database_stats.sort(
        key=lambda row: (
            -row["max_score"],
            -row["score_sum"],
            -row["hit_count"],
            row["db_id"],
        )
    )
    score_threshold = _linear_quantile(
        [row["max_score"] for row in database_stats], min_sim_ratio
    )
    filtered_db_ids = [
        row["db_id"]
        for row in database_stats
        if row["hit_count"] >= min_hit_count
        or row["max_score"] >= score_threshold
    ]
    return PruningResult(
        database_stats=database_stats,
        filtered_db_ids=filtered_db_ids,
        selected_db_ids=filtered_db_ids[:top_k],
    )


def _point_database_score(point: Any) -> tuple[str | None, float | None]:
    if isinstance(point, tuple) and len(point) == 2:
        point = point[1]
    payload = getattr(point, "payload", None)
    if payload is None and isinstance(point, Mapping):
        payload = point.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    raw_db_id = payload.get("db_id")
    db_id = "" if raw_db_id is None else str(raw_db_id).strip()

    score = getattr(point, "score", None)
    if score is None and isinstance(point, Mapping):
        score = point.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return (db_id or None), None
    numeric_score = float(score)
    if not math.isfinite(numeric_score):
        return (db_id or None), None
    return (db_id or None), numeric_score


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


__all__ = [
    "PruningResult",
    "RetrievalBudget",
    "database_pruning",
    "resolve_hrc_budget",
    "resolve_round1_rerank_top_k",
    "should_run_round2",
]
