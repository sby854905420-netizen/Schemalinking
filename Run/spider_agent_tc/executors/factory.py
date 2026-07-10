from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import SNOWFLAKE_CREDENTIAL_PATH, dataset_root, resolve_project_path

from .snowflake_executor import SnowflakeExecutor
from .sqlite_executor import SQLiteExecutor, resolve_sqlite_database_path


@dataclass(frozen=True)
class ExecutorRoute:
    dataset_name: str
    dialect_name: str
    execute_function: str
    backend: str


ROUTES = {
    "spider2": ExecutorRoute("Spider2", "Snowflake", "execute_snowflake_sql", "snowflake"),
    "mmqa": ExecutorRoute("MMQA", "SQLite", "execute_sqlite_sql", "sqlite"),
    "bird": ExecutorRoute("BIRD", "SQLite", "execute_sqlite_sql", "sqlite"),
}


def executor_route_for_dataset(dataset_name: str) -> ExecutorRoute:
    route = ROUTES.get(dataset_name.lower())
    if route is None:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Register its dialect/executor in ROUTES."
        )
    return route


def default_database_root(dataset_name: str) -> Path | None:
    normalized = dataset_name.lower()
    if normalized == "mmqa":
        return dataset_root("MMQA") / "Sqlite_database"
    if normalized == "bird":
        return dataset_root("BIRD") / "Raw_data" / "dev_databases"
    return None


class ExecutorFactory:
    def __init__(
        self,
        *,
        dataset_name: str,
        database_root: Path | None,
        snowflake_credential_path: Path | None,
        timeout_seconds: float,
        max_result_rows: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.route = executor_route_for_dataset(dataset_name)
        self.database_root = (
            resolve_project_path(database_root)
            if database_root is not None
            else default_database_root(dataset_name)
        )
        default_credential = SNOWFLAKE_CREDENTIAL_PATH
        self.snowflake_credential_path = (
            resolve_project_path(snowflake_credential_path)
            if snowflake_credential_path is not None
            else (default_credential if default_credential.is_file() else None)
        )
        self.timeout_seconds = timeout_seconds
        self.max_result_rows = max_result_rows

    def describe(self, predict_db_id: str) -> dict[str, str]:
        description = {
            "dataset": self.dataset_name,
            "dialect": self.route.dialect_name,
            "backend": self.route.backend,
            "predict_db_id": predict_db_id,
        }
        if self.database_root is not None:
            description["database_root"] = str(self.database_root)
        return description

    def create(self, predict_db_id: str) -> Any:
        if self.route.backend == "snowflake":
            executor = SnowflakeExecutor(
                credential_path=self.snowflake_credential_path,
                predict_db_id=predict_db_id,
                timeout_seconds=int(self.timeout_seconds),
                max_result_rows=self.max_result_rows,
            )
            executor.validate_configuration()
            return executor
        if self.database_root is None:
            raise ValueError(
                f"{self.dataset_name} requires --database-root for SQLite execution."
            )
        path = resolve_sqlite_database_path(
            self.database_root, predict_db_id, self.dataset_name
        )
        return SQLiteExecutor(
            database_path=path,
            timeout_seconds=self.timeout_seconds,
            max_result_rows=self.max_result_rows,
        )


__all__ = [
    "ExecutorFactory",
    "ExecutorRoute",
    "ROUTES",
    "default_database_root",
    "executor_route_for_dataset",
]
