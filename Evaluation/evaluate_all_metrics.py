from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any


SCHEMA_LOG_PATTERN = re.compile(
    r"^(?P<method>.+?)_(?P<schema_task>baseline_schema_linking|table2column)_"
    r"(?P<dataset>.+?)_(?P<timestamp>\d{8}_\d{6})\.json$"
)
DB_RETRIEVAL_FILE_PATTERN = re.compile(
    r"^(?P<kind>baseline|iterative)_database_retrival_"
    r"(?P<dataset>.+?)_(?P<timestamp>\d{8}_\d{6})\.json$"
)
SQL_RESULT_FILE_PATTERN = re.compile(
    r"^sql_generation_(?P<dataset>.+?)_(?P<timestamp>\d{8}_\d{6})\.json$"
)
ORDER_BY_PATTERN = re.compile(r"\border\s+by\b", re.IGNORECASE)
WRITE_OR_SESSION_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|"
    r"COPY|PUT|REMOVE|CALL|USE"
    r")\b",
    re.IGNORECASE,
)

CSV_FIELDS = [
    "metric_task",
    "dataset",
    "model",
    "provider",
    "method",
    "schema_task",
    "schema_source_model",
    "timestamp",
    "samples",
    "LA",
    "EM",
    "Recall",
    "Avg_Pred_Cols",
    "Avg_Gold_Cols",
    "Avg_Ratio",
    "EX",
    "correct",
    "Avg_token",
    "Avg_time",
    "Avg_token_database_retrieval",
    "Avg_time_database_retrieval",
    "Avg_token_schema_linking",
    "Avg_time_schema_linking",
    "missing_gold_count",
    "missing_db_retrieval_count",
    "skipped_count",
    "execution_error_count",
    "wrong_result_count",
    "log_file",
    "database_retrieval_log_file",
    "error",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LA, EM, Recall, Avg_Ratio, EX, Avg_token, and Avg_time "
            "from Schemalinking logs."
        )
    )
    parser.add_argument("--logs-dir", type=Path, default=project_root / "Logs")
    parser.add_argument("--data-dir", type=Path, default=project_root / "Data")
    parser.add_argument(
        "--sql-results-dir",
        type=Path,
        default=None,
        help="Directory containing sql_generation_<dataset>_<timestamp>.json logs.",
    )
    parser.add_argument(
        "--sqlite-db-dir",
        type=Path,
        default=None,
        help="Directory containing MMQA SQLite databases.",
    )
    parser.add_argument(
        "--snowflake-credential-file",
        type=Path,
        default=None,
        help="JSON credential file for optional Spider2 Snowflake EX evaluation.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to evaluate, e.g. MMQA Spider2.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["schema", "sql"],
        default=["schema", "sql"],
        help="Metric groups to run.",
    )
    parser.add_argument(
        "--db-retrieval-match",
        choices=["auto", "before-schema", "latest"],
        default="auto",
        help=(
            "How to pair schema-linking logs with database-retrieval logs for "
            "Avg_token/Avg_time. auto prefers baseline retrieval for "
            "baseline_schema_linking and iterative retrieval for table2column."
        ),
    )
    parser.add_argument(
        "--enable-snowflake",
        action="store_true",
        help="Enable Spider2 EX evaluation through Snowflake.",
    )
    parser.add_argument(
        "--skip-snowflake",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--sqlite-timeout", type=float, default=30.0)
    parser.add_argument("--snowflake-timeout", type=int, default=120)
    parser.add_argument(
        "--limit-sql-samples",
        type=int,
        default=None,
        help="Optional debugging limit per SQL generation log.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "Evluation" / "all_evaluation_metrics.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root / "Evluation" / "all_evaluation_metrics.json",
    )
    parser.add_argument(
        "--details-json",
        type=Path,
        default=project_root / "Evluation" / "all_evaluation_details.json",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def selected_dataset(dataset_name: str, selected_datasets: set[str] | None) -> bool:
    return selected_datasets is None or dataset_name.lower() in selected_datasets


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return [record for record in data if isinstance(record, dict)]


def resolve_schema_metadata(path: Path) -> dict[str, str] | None:
    match = SCHEMA_LOG_PATTERN.match(path.name)
    if match is None:
        return None
    return match.groupdict()


def resolve_db_retrieval_metadata(path: Path) -> dict[str, str] | None:
    match = DB_RETRIEVAL_FILE_PATTERN.match(path.name)
    if match is None:
        return None
    return match.groupdict()


def resolve_sql_result_metadata(path: Path) -> dict[str, str] | None:
    match = SQL_RESULT_FILE_PATTERN.match(path.name)
    if match is None:
        return None
    return match.groupdict()


def iter_schema_logs(logs_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in logs_dir.rglob("*.json")
        if resolve_schema_metadata(path) is not None
    )


def iter_sql_result_logs(sql_results_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sql_results_dir.glob("*.json")
        if resolve_sql_result_metadata(path) is not None
    )


def load_gold_schema_by_id(data_dir: Path, dataset_name: str) -> dict[str, dict[str, Any]]:
    gold_path = data_dir / dataset_name / "gold_sl.json"
    gold_records = load_json_array(gold_path)
    gold_by_id: dict[str, dict[str, Any]] = {}
    for record in gold_records:
        sample_id = normalize_text(record.get("id"))
        if sample_id:
            gold_by_id[sample_id] = record
    return gold_by_id


def load_gold_sql_by_id(data_dir: Path, dataset_name: str) -> dict[str, str]:
    gold_path = data_dir / dataset_name / "gold_sql.json"
    data = load_json(gold_path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected an id-to-SQL JSON object: {gold_path}")
    return {
        normalize_text(sample_id): normalize_text(sql)
        for sample_id, sql in data.items()
        if normalize_text(sample_id)
    }


def load_gold_db_by_id(data_dir: Path, dataset_name: str) -> dict[str, str]:
    return {
        sample_id: normalize_text(record.get("db_id"))
        for sample_id, record in load_gold_schema_by_id(data_dir, dataset_name).items()
        if normalize_text(record.get("db_id"))
    }


def coerce_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def split_table_column(value: str) -> tuple[str, str]:
    if "." not in value:
        return "", value
    table_name, column_name = value.rsplit(".", 1)
    return table_name.strip(), column_name.strip()


def make_column_set(columns_by_table: Any) -> set[tuple[str, str]]:
    column_set: set[tuple[str, str]] = set()

    if isinstance(columns_by_table, dict):
        nested = columns_by_table.get("relevant_columns")
        if isinstance(nested, dict):
            return make_column_set(nested)

        for table_name, columns in columns_by_table.items():
            normalized_table = normalize_text(table_name)
            if not normalized_table or normalized_table == "relevant_columns":
                continue

            if isinstance(columns, dict):
                column_values = [
                    key for key, is_selected in columns.items() if is_selected
                ]
            else:
                column_values = coerce_sequence(columns)

            for column_name in column_values:
                normalized_column = normalize_text(column_name)
                if normalized_column:
                    column_set.add((normalized_table, normalized_column))
        return column_set

    if isinstance(columns_by_table, list):
        for item in columns_by_table:
            if isinstance(item, dict):
                table_name = normalize_text(
                    item.get("table_name") or item.get("table") or item.get("tableName")
                )
                column_name = normalize_text(
                    item.get("column_name") or item.get("column") or item.get("columnName")
                )
            else:
                table_name, column_name = split_table_column(normalize_text(item))

            if column_name:
                column_set.add((table_name, column_name))

    return column_set


def read_efficiency(record: dict[str, Any]) -> dict[str, float]:
    efficiency = record.get("efficiency")
    if not isinstance(efficiency, dict):
        return {"tokens": 0.0, "time": 0.0}
    return {
        "tokens": float(efficiency.get("llm_total_tokens") or 0),
        "time": float(efficiency.get("sample_elapsed_seconds") or 0),
    }


def load_efficiency_by_id(path: Path) -> dict[str, dict[str, float]]:
    efficiency_by_id: dict[str, dict[str, float]] = {}
    for record in load_json_array(path):
        sample_id = normalize_text(record.get("id"))
        if sample_id:
            efficiency_by_id[sample_id] = read_efficiency(record)
    return efficiency_by_id


def find_db_retrieval_log(
    schema_log_path: Path,
    dataset_name: str,
    schema_task: str,
    schema_timestamp: str,
    match_policy: str,
) -> Path | None:
    db_retrieval_dir = schema_log_path.parent / "Database_Retrival"
    candidates: list[dict[str, Any]] = []
    for path in sorted(db_retrieval_dir.glob("*.json")):
        metadata = resolve_db_retrieval_metadata(path)
        if metadata is None or metadata["dataset"] != dataset_name:
            continue
        candidates.append(
            {
                "kind": metadata["kind"],
                "timestamp": metadata["timestamp"],
                "path": path,
            }
        )

    if not candidates:
        return None

    filtered = candidates
    if match_policy in {"auto", "before-schema"}:
        before_schema = [
            candidate
            for candidate in candidates
            if candidate["timestamp"] <= schema_timestamp
        ]
        if before_schema:
            filtered = before_schema

    if match_policy == "auto":
        preferred_kind = (
            "baseline" if schema_task == "baseline_schema_linking" else "iterative"
        )
        preferred = [
            candidate
            for candidate in filtered
            if candidate["kind"] == preferred_kind
        ]
        if preferred:
            filtered = preferred

    return max(filtered, key=lambda candidate: candidate["timestamp"])["path"]


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def evaluate_schema_log(
    path: Path,
    data_dir: Path,
    db_retrieval_match: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = resolve_schema_metadata(path)
    if metadata is None:
        raise ValueError(f"Not a schema-linking log: {path}")

    dataset_name = metadata["dataset"]
    gold_by_id = load_gold_schema_by_id(data_dir, dataset_name)
    predictions = load_json_array(path)
    db_retrieval_log_path = find_db_retrieval_log(
        schema_log_path=path,
        dataset_name=dataset_name,
        schema_task=metadata["schema_task"],
        schema_timestamp=metadata["timestamp"],
        match_policy=db_retrieval_match,
    )
    db_efficiency_by_id = (
        load_efficiency_by_id(db_retrieval_log_path)
        if db_retrieval_log_path is not None
        else {}
    )

    samples = 0
    db_correct = 0
    exact_match_correct = 0
    hit_columns = 0
    total_gold_columns = 0
    total_pred_columns = 0
    avg_ratio_values: list[float] = []
    schema_linking_token_values: list[float] = []
    schema_linking_time_values: list[float] = []
    db_retrieval_token_values: list[float] = []
    db_retrieval_time_values: list[float] = []
    total_token_values: list[float] = []
    total_time_values: list[float] = []
    missing_gold_ids: list[str] = []
    missing_db_retrieval_ids: list[str] = []
    details: list[dict[str, Any]] = []

    model_name = ""
    provider = ""

    for prediction in predictions:
        sample_id = normalize_text(prediction.get("id"))
        gold = gold_by_id.get(sample_id)
        if gold is None:
            missing_gold_ids.append(sample_id)
            details.append(
                {
                    "metric_task": "schema_linking",
                    "log_file": str(path),
                    "dataset": dataset_name,
                    "id": sample_id,
                    "status": "missing_gold",
                }
            )
            continue

        samples += 1
        model_name = model_name or normalize_text(prediction.get("model"))
        provider = provider or normalize_text(prediction.get("provider"))

        gold_db = normalize_text(gold.get("db_id"))
        pred_db = normalize_text(prediction.get("predict_db_id"))
        sample_db_correct = int(pred_db == gold_db)
        db_correct += sample_db_correct

        gold_columns = make_column_set(gold.get("columns_by_table"))
        pred_columns = make_column_set(prediction.get("predict_columns"))
        sample_exact_match = int(pred_columns == gold_columns)
        exact_match_correct += sample_exact_match

        sample_hits = len(pred_columns & gold_columns)
        hit_columns += sample_hits
        total_gold_columns += len(gold_columns)
        total_pred_columns += len(pred_columns)
        if gold_columns:
            avg_ratio_values.append(len(pred_columns) / len(gold_columns))

        schema_efficiency = read_efficiency(prediction)
        db_efficiency = db_efficiency_by_id.get(sample_id)
        if db_efficiency is None:
            missing_db_retrieval_ids.append(sample_id)
            db_efficiency = {"tokens": 0.0, "time": 0.0}

        schema_linking_token_values.append(schema_efficiency["tokens"])
        schema_linking_time_values.append(schema_efficiency["time"])
        db_retrieval_token_values.append(db_efficiency["tokens"])
        db_retrieval_time_values.append(db_efficiency["time"])
        total_token_values.append(db_efficiency["tokens"] + schema_efficiency["tokens"])
        total_time_values.append(db_efficiency["time"] + schema_efficiency["time"])

        details.append(
            {
                "metric_task": "schema_linking",
                "log_file": str(path),
                "database_retrieval_log_file": (
                    str(db_retrieval_log_path) if db_retrieval_log_path else ""
                ),
                "dataset": dataset_name,
                "id": sample_id,
                "status": "evaluated",
                "db_correct": sample_db_correct,
                "exact_match": sample_exact_match,
                "hit_columns": sample_hits,
                "gold_column_count": len(gold_columns),
                "pred_column_count": len(pred_columns),
                "schema_linking_tokens": schema_efficiency["tokens"],
                "schema_linking_time": schema_efficiency["time"],
                "database_retrieval_tokens": db_efficiency["tokens"],
                "database_retrieval_time": db_efficiency["time"],
            }
        )

    if samples == 0:
        raise ValueError(
            f"No prediction ids in {path} matched {data_dir / dataset_name / 'gold_sl.json'}"
        )

    summary = {
        "metric_task": "schema_linking",
        "dataset": dataset_name,
        "model": model_name,
        "provider": provider,
        "method": metadata["method"],
        "schema_task": metadata["schema_task"],
        "schema_source_model": "",
        "timestamp": metadata["timestamp"],
        "samples": samples,
        "missing_gold_count": len(missing_gold_ids),
        "missing_db_retrieval_count": len(missing_db_retrieval_ids),
        "LA": db_correct / samples,
        "EM": exact_match_correct / samples,
        "Recall": hit_columns / total_gold_columns if total_gold_columns else 0.0,
        "Avg_Pred_Cols": total_pred_columns / samples,
        "Avg_Gold_Cols": total_gold_columns / samples,
        "Avg_Ratio": mean_or_zero(avg_ratio_values),
        "EX": "",
        "correct": "",
        "Avg_token": mean_or_zero(total_token_values),
        "Avg_time": mean_or_zero(total_time_values),
        "Avg_token_database_retrieval": mean_or_zero(db_retrieval_token_values),
        "Avg_time_database_retrieval": mean_or_zero(db_retrieval_time_values),
        "Avg_token_schema_linking": mean_or_zero(schema_linking_token_values),
        "Avg_time_schema_linking": mean_or_zero(schema_linking_time_values),
        "skipped_count": "",
        "execution_error_count": "",
        "wrong_result_count": "",
        "log_file": str(path),
        "database_retrieval_log_file": (
            str(db_retrieval_log_path) if db_retrieval_log_path else ""
        ),
        "error": "",
    }
    return summary, details


def strip_sql_comments(sql: str) -> str:
    text = sql.strip()
    while True:
        if text.startswith("--"):
            _, _, text = text.partition("\n")
            text = text.strip()
            continue
        if text.startswith("/*"):
            end = text.find("*/")
            if end == -1:
                return ""
            text = text[end + 2 :].strip()
            continue
        return text


def clean_single_statement(sql: str) -> str:
    return strip_sql_comments(sql).rstrip().rstrip(";").strip()


def validate_readonly_sql(sql: str) -> str:
    statement = clean_single_statement(sql)
    if not statement:
        raise ValueError("Empty SQL query.")
    if ";" in statement:
        raise ValueError("Only one SQL statement is allowed.")

    first_keyword_match = re.match(r"([A-Za-z]+)", statement)
    first_keyword = first_keyword_match.group(1).upper() if first_keyword_match else ""
    if first_keyword not in {"SELECT", "WITH"}:
        raise ValueError("Only SELECT/WITH queries are allowed.")
    if WRITE_OR_SESSION_KEYWORDS.search(statement):
        raise ValueError("Write, DDL, session, and administrative statements are not allowed.")
    return statement


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Decimal):
        if value.is_nan():
            return "NaN"
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, list):
        return tuple(normalize_scalar(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted((key, normalize_scalar(item)) for key, item in value.items())
        )
    return value


def normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [tuple(normalize_scalar(value) for value in row) for row in rows]


def rows_match(
    predicted_rows: list[tuple[Any, ...]],
    gold_rows: list[tuple[Any, ...]],
    order_matters: bool,
) -> bool:
    predicted_rows = normalize_rows(predicted_rows)
    gold_rows = normalize_rows(gold_rows)

    if len(predicted_rows) != len(gold_rows):
        return False
    if not predicted_rows and not gold_rows:
        return True
    if len(predicted_rows[0]) != len(gold_rows[0]):
        return False
    if order_matters:
        return predicted_rows == gold_rows
    return Counter(predicted_rows) == Counter(gold_rows)


def sqlite_db_path(sqlite_db_dir: Path, db_id: str) -> Path:
    return sqlite_db_dir / f"{db_id}.sqlite"


def execute_sqlite(
    sql: str,
    db_path: Path,
    db_id: str,
    timeout_seconds: float,
) -> list[tuple[Any, ...]]:
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    statement = validate_readonly_sql(sql)
    conn = sqlite3.connect(str(db_path))
    deadline = time.monotonic() + timeout_seconds

    def abort_when_timed_out() -> int:
        return 1 if time.monotonic() > deadline else 0

    try:
        conn.execute(f"ATTACH DATABASE ? AS {quote_identifier(db_id)}", (str(db_path),))
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(abort_when_timed_out, 1000)
        cursor = conn.execute(statement)
        return cursor.fetchall()
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise TimeoutError(f"SQLite query exceeded {timeout_seconds:.1f}s") from exc
        raise
    finally:
        conn.close()


def load_snowflake_credentials(credential_file: Path) -> dict[str, Any]:
    if credential_file.is_file():
        data = load_json(credential_file)
        if isinstance(data, dict):
            return data
    return {}


def snowflake_connection_params(
    credential_file: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    credentials = load_snowflake_credentials(credential_file)
    user = os.environ.get("SNOWFLAKE_USER") or credentials.get("user") or credentials.get("username")
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

    params: dict[str, Any] = {
        "account": account,
        "user": user,
        "password": password,
        "warehouse": warehouse,
        "role": role,
        "login_timeout": int(os.environ.get("SNOWFLAKE_LOGIN_TIMEOUT", "30")),
        "network_timeout": timeout_seconds,
        "client_session_keep_alive": False,
        "session_parameters": {
            "QUERY_TAG": os.environ.get("SNOWFLAKE_QUERY_TAG", "schemalinking_ex_evaluation"),
            "STATEMENT_TIMEOUT_IN_SECONDS": timeout_seconds,
        },
    }

    for optional_key in ("database", "schema"):
        value = os.environ.get(f"SNOWFLAKE_{optional_key.upper()}") or credentials.get(optional_key)
        if value:
            params[optional_key] = value

    return params


class SnowflakeExecutor:
    def __init__(self, credential_file: Path, timeout_seconds: int) -> None:
        import snowflake.connector

        self._connector = snowflake.connector
        self._credential_file = credential_file
        self._timeout_seconds = timeout_seconds
        self._conn = None

    def connect(self) -> None:
        self._conn = self._connector.connect(
            **snowflake_connection_params(self._credential_file, self._timeout_seconds)
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str) -> list[tuple[Any, ...]]:
        if self._conn is None:
            raise RuntimeError("Snowflake connection is not open.")
        statement = validate_readonly_sql(sql)
        cursor = self._conn.cursor()
        try:
            cursor.execute(statement)
            return cursor.fetchall()
        finally:
            cursor.close()


def execute_pair(
    predicted_sql: str,
    gold_sql: str,
    dataset_name: str,
    db_id: str,
    sqlite_db_dir: Path,
    sqlite_timeout: float,
    snowflake_executor: SnowflakeExecutor | None,
    snowflake_unavailable_reason: str,
) -> dict[str, Any]:
    order_matters = ORDER_BY_PATTERN.search(gold_sql) is not None
    try:
        if dataset_name.lower() == "mmqa":
            db_path = sqlite_db_path(sqlite_db_dir, db_id)
            gold_rows = execute_sqlite(gold_sql, db_path, db_id, sqlite_timeout)
            predicted_rows = execute_sqlite(predicted_sql, db_path, db_id, sqlite_timeout)
        elif dataset_name.lower() == "spider2":
            if snowflake_executor is None:
                return {
                    "status": "skipped",
                    "correct": 0,
                    "error": snowflake_unavailable_reason,
                }
            gold_rows = snowflake_executor.execute(gold_sql)
            predicted_rows = snowflake_executor.execute(predicted_sql)
        else:
            return {
                "status": "skipped",
                "correct": 0,
                "error": f"Unsupported dataset for execution: {dataset_name}",
            }
    except Exception as exc:
        return {
            "status": "execution_error",
            "correct": 0,
            "error": str(exc),
        }

    correct = rows_match(predicted_rows, gold_rows, order_matters)
    return {
        "status": "correct" if correct else "wrong_result",
        "correct": int(correct),
        "predicted_row_count": len(predicted_rows),
        "gold_row_count": len(gold_rows),
        "order_matters": order_matters,
    }


def evaluate_sql_log(
    path: Path,
    data_dir: Path,
    sqlite_db_dir: Path,
    sqlite_timeout: float,
    snowflake_executor: SnowflakeExecutor | None,
    snowflake_unavailable_reason: str,
    limit_sql_samples: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = resolve_sql_result_metadata(path)
    if metadata is None:
        raise ValueError(f"Not a SQL generation log: {path}")

    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Expected a SQL generation result object: {path}")

    run_info = payload.get("run_info") if isinstance(payload.get("run_info"), dict) else {}
    dataset_name = normalize_text(run_info.get("dataset_name") or metadata["dataset"])
    gold_sql_by_id = load_gold_sql_by_id(data_dir, dataset_name)
    gold_db_by_id = load_gold_db_by_id(data_dir, dataset_name)

    records = [
        record for record in payload["results"] if isinstance(record, dict)
    ]
    if limit_sql_samples is not None:
        records = records[:limit_sql_samples]

    samples = 0
    correct = 0
    missing_gold = 0
    skipped = 0
    execution_errors = 0
    wrong_results = 0
    details: list[dict[str, Any]] = []

    for record in records:
        sample_id = normalize_text(record.get("id"))
        predicted_sql = normalize_text(record.get("predict_sql"))
        gold_sql = gold_sql_by_id.get(sample_id)
        db_id = gold_db_by_id.get(sample_id) or normalize_text(record.get("gold_db_id"))

        if not sample_id or not gold_sql or not db_id:
            missing_gold += 1
            details.append(
                {
                    "metric_task": "execution_accuracy",
                    "log_file": str(path),
                    "dataset": dataset_name,
                    "id": sample_id,
                    "status": "missing_gold",
                    "correct": 0,
                }
            )
            continue

        samples += 1
        result = execute_pair(
            predicted_sql=predicted_sql,
            gold_sql=gold_sql,
            dataset_name=dataset_name,
            db_id=db_id,
            sqlite_db_dir=sqlite_db_dir,
            sqlite_timeout=sqlite_timeout,
            snowflake_executor=snowflake_executor,
            snowflake_unavailable_reason=snowflake_unavailable_reason,
        )
        correct += int(result.get("correct", 0))
        status = normalize_text(result.get("status"))
        if status == "skipped":
            skipped += 1
        elif status == "execution_error":
            execution_errors += 1
        elif status == "wrong_result":
            wrong_results += 1

        details.append(
            {
                "metric_task": "execution_accuracy",
                "log_file": str(path),
                "dataset": dataset_name,
                "id": sample_id,
                "db_id": db_id,
                **result,
            }
        )

    summary = {
        "metric_task": "execution_accuracy",
        "dataset": dataset_name,
        "model": normalize_text(run_info.get("model")),
        "provider": normalize_text(run_info.get("provider")),
        "method": "",
        "schema_task": "",
        "schema_source_model": normalize_text(run_info.get("schema_source_model")),
        "timestamp": metadata["timestamp"],
        "samples": samples,
        "LA": "",
        "EM": "",
        "Recall": "",
        "Avg_Pred_Cols": "",
        "Avg_Gold_Cols": "",
        "Avg_Ratio": "",
        "EX": correct / samples if samples else 0.0,
        "correct": correct,
        "Avg_token": "",
        "Avg_time": "",
        "Avg_token_database_retrieval": "",
        "Avg_time_database_retrieval": "",
        "Avg_token_schema_linking": "",
        "Avg_time_schema_linking": "",
        "missing_gold_count": missing_gold,
        "missing_db_retrieval_count": "",
        "skipped_count": skipped,
        "execution_error_count": execution_errors,
        "wrong_result_count": wrong_results,
        "log_file": str(path),
        "database_retrieval_log_file": "",
        "error": "",
    }
    return summary, details


def resolve_sql_results_dir(args: argparse.Namespace) -> Path:
    if args.sql_results_dir is not None:
        return args.sql_results_dir
    return args.logs_dir / "sql_results"


def resolve_sqlite_db_dir(args: argparse.Namespace, project_root: Path) -> Path:
    if args.sqlite_db_dir is not None:
        return args.sqlite_db_dir

    candidates = [
        project_root / "Data" / "MMQA" / "Sqlite_database",
        project_root.parent / "autolink" / "Data" / "MMQA" / "Sqlite_database",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[-1]


def resolve_snowflake_credential_file(args: argparse.Namespace, project_root: Path) -> Path:
    if args.snowflake_credential_file is not None:
        return args.snowflake_credential_file

    candidates = [
        project_root / "snowflake_credential.json",
        project_root.parent / "autolink" / "snowflake_credential.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def print_schema_summary(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"{row['dataset']} {row['schema_task']} {row['timestamp']}: "
            f"LA={row['LA']:.4f}, EM={row['EM']:.4f}, "
            f"Recall={row['Recall']:.4f}, Avg_Ratio={row['Avg_Ratio']:.4f}"
        )


def print_execution_summary(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"{row['dataset']} sql_generation {row['timestamp']}: "
            f"EX={row['EX']:.4f} ({row['correct']}/{row['samples']}), "
            f"errors={row['execution_error_count']}, skipped={row['skipped_count']}"
        )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    selected_datasets = (
        {normalize_text(dataset).lower() for dataset in args.datasets}
        if args.datasets
        else None
    )

    schema_summaries: list[dict[str, Any]] = []
    schema_details: list[dict[str, Any]] = []
    if "schema" in args.tasks:
        for path in iter_schema_logs(args.logs_dir):
            metadata = resolve_schema_metadata(path)
            if metadata is None:
                continue
            if not selected_dataset(metadata["dataset"], selected_datasets):
                continue
            summary, details = evaluate_schema_log(
                path=path,
                data_dir=args.data_dir,
                db_retrieval_match=args.db_retrieval_match,
            )
            schema_summaries.append(summary)
            schema_details.extend(details)

    execution_summaries: list[dict[str, Any]] = []
    execution_details: list[dict[str, Any]] = []
    if "sql" in args.tasks:
        sql_results_dir = resolve_sql_results_dir(args)
        sqlite_db_dir = resolve_sqlite_db_dir(args, project_root)
        snowflake_credential_file = resolve_snowflake_credential_file(args, project_root)
        sql_log_paths = []
        for path in iter_sql_result_logs(sql_results_dir):
            metadata = resolve_sql_result_metadata(path)
            if metadata is None:
                continue
            if selected_dataset(metadata["dataset"], selected_datasets):
                sql_log_paths.append(path)

        needs_snowflake = any(
            (resolve_sql_result_metadata(path) or {}).get("dataset", "").lower() == "spider2"
            for path in sql_log_paths
        )
        run_snowflake = needs_snowflake and args.enable_snowflake and not args.skip_snowflake
        snowflake_executor: SnowflakeExecutor | None = None
        snowflake_unavailable_reason = (
            "Snowflake evaluation is disabled. Re-run with --enable-snowflake "
            "and valid credentials to evaluate Spider2 EX."
        )
        try:
            if run_snowflake:
                try:
                    snowflake_executor = SnowflakeExecutor(
                        snowflake_credential_file,
                        args.snowflake_timeout,
                    )
                    snowflake_executor.connect()
                    snowflake_unavailable_reason = ""
                except Exception as exc:
                    snowflake_executor = None
                    snowflake_unavailable_reason = f"Snowflake unavailable: {exc}"

            for path in sql_log_paths:
                summary, details = evaluate_sql_log(
                    path=path,
                    data_dir=args.data_dir,
                    sqlite_db_dir=sqlite_db_dir,
                    sqlite_timeout=args.sqlite_timeout,
                    snowflake_executor=snowflake_executor,
                    snowflake_unavailable_reason=snowflake_unavailable_reason,
                    limit_sql_samples=args.limit_sql_samples,
                )
                execution_summaries.append(summary)
                execution_details.extend(details)
        finally:
            if snowflake_executor is not None:
                snowflake_executor.close()

    all_summaries = schema_summaries + execution_summaries
    output_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_linking": schema_summaries,
        "execution_accuracy": execution_summaries,
    }
    details_payload = {
        "generated_at": output_payload["generated_at"],
        "schema_linking": schema_details,
        "execution_accuracy": execution_details,
    }

    write_csv(all_summaries, args.output_csv)
    write_json(output_payload, args.output_json)
    write_json(details_payload, args.details_json)

    print(f"Schema linking logs evaluated: {len(schema_summaries)}")
    print(f"SQL generation logs evaluated: {len(execution_summaries)}")
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote details: {args.details_json}")
    print_schema_summary(schema_summaries)
    print_execution_summary(execution_summaries)


if __name__ == "__main__":
    main()
