from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentInput:
    """Leakage-safe input assembled from one schema-linking prediction."""

    sample_id: str
    dataset_name: str
    question: str
    hint: str
    predict_db_id: str
    predict_tables: tuple[str, ...]
    predict_columns: dict[str, tuple[str, ...]]
    selected_column_records: tuple[dict[str, Any], ...]
    schema_text: str
    sql_dialect: str


@dataclass(frozen=True)
class ToolCall:
    function: str
    parameter_name: str
    value: str


@dataclass(frozen=True)
class ExecutionResult:
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    truncated: bool = False

    @property
    def returned_row_count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class AgentResult:
    sql: str
    status: str
    stop_reason: str
    execution_verified: bool
    rounds: int = 0
    total_tokens: int = 0
    error: str = ""
    messages: tuple[dict[str, Any], ...] = field(default_factory=tuple, repr=False)
