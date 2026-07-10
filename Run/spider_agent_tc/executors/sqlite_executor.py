from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from ..result_types import ExecutionResult
from .safety import validate_readonly_sql


def resolve_sqlite_database_path(
    database_root: Path,
    db_id: str,
    dataset_name: str,
) -> Path:
    root = Path(database_root).resolve()
    suffixes = (".sqlite", ".db")
    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.extend((root / f"{db_id}{suffix}", root / db_id / f"{db_id}{suffix}"))

    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(root)
        except ValueError:
            continue
        if resolved_candidate.is_file():
            return resolved_candidate

    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No {dataset_name} SQLite database for predicted db '{db_id}'. "
        f"Checked: {expected}. Override with --database-root."
    )


class SQLiteExecutor:
    def __init__(
        self,
        *,
        database_path: Path,
        timeout_seconds: float = 120,
        max_result_rows: int = 20,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.max_result_rows = int(max_result_rows)

    def execute(self, sql: str) -> ExecutionResult:
        statement = validate_readonly_sql(sql)
        if not self.database_path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {self.database_path}")

        uri = f"file:{self.database_path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
        deadline = time.monotonic() + self.timeout_seconds

        def abort_when_timed_out() -> int:
            return 1 if time.monotonic() > deadline else 0

        try:
            connection.execute("PRAGMA query_only = ON")

            denied_actions = {
                sqlite3.SQLITE_ATTACH,
                sqlite3.SQLITE_DETACH,
                sqlite3.SQLITE_PRAGMA,
            }

            def authorize(
                action: int,
                arg1: str | None,
                _arg2: str | None,
                _db: str | None,
                _source: str | None,
            ) -> int:
                if action in denied_actions:
                    return sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_READ and str(arg1 or "").lower() in {
                    "sqlite_master",
                    "sqlite_schema",
                }:
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            connection.set_authorizer(authorize)
            connection.set_progress_handler(abort_when_timed_out, 1000)
            cursor = connection.execute(statement)
            columns = tuple(description[0] for description in (cursor.description or ()))
            fetched = cursor.fetchmany(self.max_result_rows + 1)
            truncated = len(fetched) > self.max_result_rows
            rows = tuple(tuple(row) for row in fetched[: self.max_result_rows])
            return ExecutionResult(columns=columns, rows=rows, truncated=truncated)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise TimeoutError(
                    f"SQLite query exceeded {self.timeout_seconds:.1f}s."
                ) from exc
            raise
        finally:
            connection.close()

    def close(self) -> None:
        return None


__all__ = ["SQLiteExecutor", "resolve_sqlite_database_path"]
