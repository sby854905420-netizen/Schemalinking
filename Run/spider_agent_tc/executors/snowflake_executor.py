from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..result_types import ExecutionResult
from .safety import validate_readonly_sql


def load_snowflake_credentials(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Snowflake credential file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Snowflake credential file must contain a JSON object.")
    return payload


def snowflake_connection_params(
    credential_path: Path | None,
    *,
    predict_db_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    credentials = load_snowflake_credentials(credential_path)
    user = os.environ.get("SNOWFLAKE_USER") or credentials.get("user") or credentials.get(
        "username"
    )
    password = (
        os.environ.get("SNOWFLAKE_PAT")
        or os.environ.get("SNOWFLAKE_PASSWORD")
        or credentials.get("password")
        or credentials.get("pat")
        or credentials.get("token")
    )
    account = os.environ.get("SNOWFLAKE_ACCOUNT") or credentials.get("account")
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE") or credentials.get("warehouse")
    role = os.environ.get("SNOWFLAKE_ROLE") or credentials.get("role")
    missing = [
        name
        for name, value in {
            "account": account,
            "user": user,
            "password/PAT": password,
            "warehouse": warehouse,
            "role": role,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError("Missing Snowflake credential fields: " + ", ".join(missing))

    # The database is deliberately forced from the prepared predict_db_id; a
    # credential-file default database is never used. AutoLink's explicit oracle
    # filtering, when selected, materializes the gold database upstream.
    return {
        "account": account,
        "user": user,
        "password": password,
        "warehouse": warehouse,
        "role": role,
        "database": predict_db_id,
        "login_timeout": int(os.environ.get("SNOWFLAKE_LOGIN_TIMEOUT", "30")),
        "network_timeout": timeout_seconds,
        "client_session_keep_alive": False,
        "session_parameters": {
            "QUERY_TAG": os.environ.get("SNOWFLAKE_QUERY_TAG", "mdblink_spider_agent_tc"),
            "STATEMENT_TIMEOUT_IN_SECONDS": timeout_seconds,
        },
    }


class SnowflakeExecutor:
    def __init__(
        self,
        *,
        credential_path: Path | None,
        predict_db_id: str,
        timeout_seconds: int = 120,
        max_result_rows: int = 20,
    ) -> None:
        self.credential_path = credential_path
        self.predict_db_id = predict_db_id
        self.timeout_seconds = int(timeout_seconds)
        self.max_result_rows = int(max_result_rows)
        self._connection = None

    def validate_configuration(self) -> None:
        try:
            import snowflake.connector  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Spider2 execution requires snowflake-connector-python. "
                "Install the optional Snowflake dependency."
            ) from exc
        snowflake_connection_params(
            self.credential_path,
            predict_db_id=self.predict_db_id,
            timeout_seconds=self.timeout_seconds,
        )

    def _connect(self) -> Any:
        if self._connection is not None:
            return self._connection
        import snowflake.connector
        self._connection = snowflake.connector.connect(
            **snowflake_connection_params(
                self.credential_path,
                predict_db_id=self.predict_db_id,
                timeout_seconds=self.timeout_seconds,
            )
        )
        return self._connection

    def execute(self, sql: str) -> ExecutionResult:
        statement = validate_readonly_sql(sql)
        cursor = self._connect().cursor()
        try:
            try:
                cursor.execute(statement, timeout=self.timeout_seconds)
            except TypeError:
                cursor.execute(statement)
            columns = tuple(description[0] for description in (cursor.description or ()))
            fetched = cursor.fetchmany(self.max_result_rows + 1)
            truncated = len(fetched) > self.max_result_rows
            rows = tuple(tuple(row) for row in fetched[: self.max_result_rows])
            return ExecutionResult(columns=columns, rows=rows, truncated=truncated)
        finally:
            cursor.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


__all__ = [
    "SnowflakeExecutor",
    "load_snowflake_credentials",
    "snowflake_connection_params",
]
