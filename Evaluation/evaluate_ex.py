"""Compute execution accuracy (EX) for MDB-link SQL predictions.

The evaluator executes gold and predicted SQL against the gold database.  In
batch mode it evaluates all SQL artifacts for one dataset sample-by-sample, so
each gold query is executed only once and shared across model comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_NAME, DATA_ROOT, RESULTS_ROOT, resolve_project_path
from Run.spider_agent_tc.executors.factory import ExecutorFactory


TABLE_COLUMNS = (
    "dataset",
    "sql_method",
    "sql_model",
    "schema_method",
    "schema_model",
    "total_samples",
    "prediction_records",
    "matched_predictions",
    "coverage",
    "correct",
    "EX",
    "missing_predictions",
    "generation_failures",
    "gold_errors",
    "execution_errors",
    "truncated_results",
    "mismatches",
    "extra_predictions",
    "prediction_file",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute predicted and gold SQL and compute Execution Accuracy (EX)."
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help=f"Dataset to evaluate (default: {DATASET_NAME}).",
    )
    parser.add_argument(
        "--model-name",
        "--sql-model-name",
        dest="sql_model_name",
        help="SQL-generation model filter (both original and '__' path names work).",
    )
    parser.add_argument("--sql-method", help="SQL-generation method filter.")
    parser.add_argument("--sl-model-name", help="Schema-linking model filter.")
    parser.add_argument("--sl-method", help="Schema-linking method filter.")
    parser.add_argument(
        "--prediction-path",
        type=Path,
        help="Evaluate one exact SQL prediction JSON instead of discovery.",
    )
    parser.add_argument(
        "--gold-sql-path",
        type=Path,
        help="Gold SQL mapping (default: Data/<dataset>/gold_sql.json).",
    )
    parser.add_argument(
        "--gold-sl-path",
        type=Path,
        help="Gold schema rows used to locate db_id (default: gold_sl.json).",
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        help="Override the MMQA/BIRD SQLite database root.",
    )
    parser.add_argument(
        "--snowflake-credential-path",
        type=Path,
        help="Spider2 Snowflake credential JSON; environment variables also work.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout for each SQL statement (default: 120).",
    )
    parser.add_argument(
        "--max-result-rows",
        type=int,
        default=100000,
        help=(
            "Maximum rows fetched per query. A larger result is counted as "
            "truncated, never as correct (default: 100000)."
        ),
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Evaluate only the first N gold samples (intended for smoke tests).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N samples; use 0 to disable (default: 100).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="CSV output path (default: results/evaluation/<dataset>/ex.csv).",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Decimal places used in CSV and console output (default: 6).",
    )
    return parser


def _safe_component(value: str) -> str:
    text = str(value).strip().replace("/", "__").replace("\\", "__")
    text = re.sub(r"[\s:]+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-")


def _name_matches(actual: str, requested: str | None) -> bool:
    if not requested:
        return True
    actual_text = str(actual).strip().casefold()
    requested_text = str(requested).strip().casefold()
    return actual_text == requested_text or _safe_component(actual_text) == _safe_component(
        requested_text
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc


def load_sql_prediction(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    required = {
        "dataset",
        "method",
        "model_names",
        "schema_linking_method",
        "predictions",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"Not a unified SQL prediction artifact: {path}")
    models = payload["model_names"]
    if not isinstance(models, dict) or set(models) != {
        "sql_generation",
        "schema_linking",
    }:
        raise ValueError(f"Invalid model_names in {path}")
    if not isinstance(payload["predictions"], list):
        raise ValueError(f"predictions must be a list: {path}")
    seen: set[str] = set()
    for index, prediction in enumerate(payload["predictions"]):
        if not isinstance(prediction, dict) or not str(prediction.get("id", "")):
            raise ValueError(f"Invalid SQL prediction record {index}: {path}")
        sample_id = str(prediction["id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate SQL prediction id {sample_id!r}: {path}")
        seen.add(sample_id)
    return payload


def discover_predictions(
    *,
    dataset_name: str,
    sql_model_name: str | None,
    sql_method: str | None,
    sl_model_name: str | None,
    sl_method: str | None,
    prediction_path: Path | None,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    warnings: list[str] = []
    if prediction_path is not None:
        candidates = [resolve_project_path(prediction_path)]
    else:
        candidates = sorted((RESULTS_ROOT / "sql").glob("*/*/*/*/*.json"))

    selected: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        if path.name.endswith("_agent_failures.json"):
            continue
        if not path.is_file():
            warnings.append(f"skip missing prediction file: {path}")
            continue
        try:
            payload = load_sql_prediction(path)
        except ValueError as exc:
            warnings.append(f"skip invalid prediction file: {exc}")
            continue
        if str(payload["dataset"]).casefold() != dataset_name.casefold():
            continue
        if sql_method and str(payload["method"]).casefold() != sql_method.casefold():
            continue
        if sl_method and str(payload["schema_linking_method"]).casefold() != sl_method.casefold():
            continue
        models = payload["model_names"]
        if not _name_matches(models["sql_generation"], sql_model_name):
            continue
        if not _name_matches(models["schema_linking"], sl_model_name):
            continue
        selected.append((path, payload))
    return selected, warnings


def load_gold_sql(path: Path) -> list[tuple[str, str]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Gold SQL file must be an id-to-SQL JSON object: {path}")
    rows: list[tuple[str, str]] = []
    for raw_id, raw_sql in payload.items():
        sample_id = str(raw_id)
        if not sample_id or not isinstance(raw_sql, str) or not raw_sql.strip():
            raise ValueError(f"Invalid gold SQL for id {sample_id!r}: {path}")
        rows.append((sample_id, raw_sql))
    return rows


def load_gold_databases(path: Path) -> dict[str, str]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Gold schema file must contain a JSON list: {path}")
    result: dict[str, str] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid gold schema row {index}: {path}")
        sample_id = str(row.get("id", ""))
        db_id = row.get("db_id")
        if not sample_id or not isinstance(db_id, str) or not db_id:
            raise ValueError(f"Invalid gold schema id/db_id at row {index}: {path}")
        if sample_id in result:
            raise ValueError(f"Duplicate gold schema id {sample_id!r}: {path}")
        result[sample_id] = db_id
    return result


def _strip_literals_and_comments(sql: str) -> str:
    """Remove quoted text/comments before looking for an ORDER BY keyword."""

    text = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    text = re.sub(r"--[^\r\n]*", " ", text)
    text = re.sub(r"'(?:''|[^'])*'", " ", text)
    text = re.sub(r'"(?:""|[^"])*"', " ", text)
    text = re.sub(r"`(?:``|[^`])*`", " ", text)
    return text


def gold_requires_order(sql: str) -> bool:
    return (
        re.search(
            r"\border\s+by\b",
            _strip_literals_and_comments(sql),
            re.IGNORECASE,
        )
        is not None
    )


def _canonical_number(value: Any) -> tuple[str, str]:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ("number", str(value).lower())
    try:
        number = Decimal(str(int(value) if isinstance(value, bool) else value))
    except (InvalidOperation, TypeError, ValueError):
        return ("number", str(value))
    if number == 0:
        number = Decimal(0)
    return ("number", str(number.normalize()))


def canonical_cell(value: Any) -> tuple[str, str]:
    if value is None:
        return ("null", "")
    if isinstance(value, (bool, int, float, Decimal)):
        return _canonical_number(value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, (datetime, date, time)):
        return ("datetime", value.isoformat())
    if isinstance(value, str):
        return ("text", value)
    return (type(value).__name__, str(value))


def canonical_rows(result: Any) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(tuple(canonical_cell(cell) for cell in row) for row in result.rows)


def results_equal(predicted: Any, gold: Any, *, order_matters: bool) -> bool:
    if predicted.truncated or gold.truncated:
        return False
    predicted_rows = canonical_rows(predicted)
    gold_rows = canonical_rows(gold)
    if order_matters:
        return predicted_rows == gold_rows
    return Counter(predicted_rows) == Counter(gold_rows)


def _initial_stat(path: Path, payload: Mapping[str, Any], gold_ids: set[str]) -> dict[str, Any]:
    predictions = payload["predictions"]
    prediction_ids = {str(record["id"]) for record in predictions}
    models = payload["model_names"]
    relative_path = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    return {
        "dataset": payload["dataset"],
        "sql_method": payload["method"],
        "sql_model": models["sql_generation"],
        "schema_method": payload["schema_linking_method"],
        "schema_model": models["schema_linking"],
        "prediction_records": len(predictions),
        "matched_predictions": len(prediction_ids & gold_ids),
        "correct": 0,
        "missing_predictions": 0,
        "generation_failures": 0,
        "gold_errors": 0,
        "execution_errors": 0,
        "truncated_results": 0,
        "mismatches": 0,
        "extra_predictions": len(prediction_ids - gold_ids),
        "prediction_file": str(relative_path),
    }


def _valid_predicted_sql(record: Mapping[str, Any] | None) -> str | None:
    if record is None or record.get("status") != "success":
        return None
    sql = record.get("predicted_sql")
    return sql.strip() if isinstance(sql, str) and sql.strip() else None


def evaluate_artifacts(
    *,
    artifacts: Sequence[tuple[Path, Mapping[str, Any]]],
    gold_sql_rows: Sequence[tuple[str, str]],
    reference_gold_ids: set[str] | None,
    gold_databases: Mapping[str, str],
    factory: ExecutorFactory,
    progress_every: int,
) -> list[dict[str, Any]]:
    gold_ids = {sample_id for sample_id, _ in gold_sql_rows}
    extra_reference_ids = reference_gold_ids if reference_gold_ids is not None else gold_ids
    prediction_maps = [
        {str(record["id"]): record for record in payload["predictions"]}
        for _, payload in artifacts
    ]
    stats = [_initial_stat(path, payload, extra_reference_ids) for path, payload in artifacts]
    for stat, prediction_map in zip(stats, prediction_maps):
        stat["matched_predictions"] = len(set(prediction_map) & gold_ids)

    for position, (sample_id, gold_sql) in enumerate(gold_sql_rows, start=1):
        runnable: list[tuple[int, str]] = []
        for index, prediction_map in enumerate(prediction_maps):
            record = prediction_map.get(sample_id)
            if record is None:
                stats[index]["missing_predictions"] += 1
                continue
            predicted_sql = _valid_predicted_sql(record)
            if predicted_sql is None:
                stats[index]["generation_failures"] += 1
                continue
            runnable.append((index, predicted_sql))

        if runnable:
            db_id = gold_databases.get(sample_id)
            if not db_id:
                for index, _ in runnable:
                    stats[index]["gold_errors"] += 1
            else:
                executor = None
                try:
                    executor = factory.create(db_id)
                    gold_result = executor.execute(gold_sql)
                    if gold_result.truncated:
                        for index, _ in runnable:
                            stats[index]["truncated_results"] += 1
                    else:
                        order_matters = gold_requires_order(gold_sql)
                        for index, predicted_sql in runnable:
                            try:
                                predicted_result = executor.execute(predicted_sql)
                                if predicted_result.truncated:
                                    stats[index]["truncated_results"] += 1
                                elif results_equal(
                                    predicted_result,
                                    gold_result,
                                    order_matters=order_matters,
                                ):
                                    stats[index]["correct"] += 1
                                else:
                                    stats[index]["mismatches"] += 1
                            except Exception:
                                stats[index]["execution_errors"] += 1
                except Exception:
                    for index, _ in runnable:
                        stats[index]["gold_errors"] += 1
                finally:
                    if executor is not None:
                        executor.close()

        if progress_every > 0 and (
            position % progress_every == 0 or position == len(gold_sql_rows)
        ):
            print(f"Evaluated {position}/{len(gold_sql_rows)} gold samples...", file=sys.stderr)

    total = len(gold_sql_rows)
    for stat in stats:
        stat["total_samples"] = total
        stat["coverage"] = stat["matched_predictions"] / total if total else None
        stat["EX"] = stat["correct"] / total if total else None
    return stats


def _rounded_rows(rows: Sequence[Mapping[str, Any]], digits: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = {}
        for key in TABLE_COLUMNS:
            value = row.get(key)
            output[key] = round(value, digits) if isinstance(value, float) else value
        result.append(output)
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def print_table(rows: Sequence[Mapping[str, Any]]) -> None:
    display_columns = (
        "sql_method",
        "sql_model",
        "schema_method",
        "schema_model",
        "total_samples",
        "coverage",
        "correct",
        "EX",
        "execution_errors",
        "mismatches",
    )
    if not rows:
        print("No matching non-empty SQL prediction artifacts; nothing to evaluate.")
        return
    text_rows = [
        {key: "-" if row.get(key) is None else str(row.get(key)) for key in display_columns}
        for row in rows
    ]
    widths = {
        key: max(len(key), *(len(row[key]) for row in text_rows))
        for key in display_columns
    }
    print("  ".join(key.ljust(widths[key]) for key in display_columns))
    print("  ".join("-" * widths[key] for key in display_columns))
    for row in text_rows:
        print("  ".join(row[key].ljust(widths[key]) for key in display_columns))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive.")
    if args.max_result_rows <= 0:
        raise SystemExit("--max-result-rows must be positive.")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise SystemExit("--sample-limit must be positive.")
    if args.progress_every < 0 or args.digits < 0:
        raise SystemExit("--progress-every and --digits must be non-negative.")

    artifacts, warnings = discover_predictions(
        dataset_name=args.dataset_name,
        sql_model_name=args.sql_model_name,
        sql_method=args.sql_method,
        sl_model_name=args.sl_model_name,
        sl_method=args.sl_method,
        prediction_path=args.prediction_path,
    )
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    non_empty_artifacts: list[tuple[Path, dict[str, Any]]] = []
    for path, payload in artifacts:
        if not payload["predictions"]:
            print(f"Skip empty SQL prediction artifact: {path}", file=sys.stderr)
            continue
        non_empty_artifacts.append((path, payload))
    artifacts = non_empty_artifacts

    output_path = (
        resolve_project_path(args.output_path)
        if args.output_path
        else RESULTS_ROOT / "evaluation" / args.dataset_name / "ex.csv"
    )
    if not artifacts:
        write_csv(output_path, [])
        print_table([])
        print(f"\nSaved 0 evaluation row(s) to {output_path}")
        return 0

    gold_sql_path = (
        resolve_project_path(args.gold_sql_path)
        if args.gold_sql_path
        else DATA_ROOT / args.dataset_name / "gold_sql.json"
    )
    gold_sl_path = (
        resolve_project_path(args.gold_sl_path)
        if args.gold_sl_path
        else DATA_ROOT / args.dataset_name / "gold_sl.json"
    )
    if not gold_sql_path.is_file() or not gold_sl_path.is_file():
        print(
            f"Skip: missing gold file ({gold_sql_path} or {gold_sl_path}).",
            file=sys.stderr,
        )
        return 0
    try:
        gold_sql_rows = load_gold_sql(gold_sql_path)
        gold_databases = load_gold_databases(gold_sl_path)
    except ValueError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    reference_gold_ids = {sample_id for sample_id, _ in gold_sql_rows}
    if args.sample_limit is not None:
        gold_sql_rows = gold_sql_rows[: args.sample_limit]

    try:
        import sqlparse  # noqa: F401
    except ImportError:
        print(
            "Evaluation failed: sqlparse is required for safe SQL execution. "
            "Install the project requirements first.",
            file=sys.stderr,
        )
        return 2

    factory = ExecutorFactory(
        dataset_name=args.dataset_name,
        database_root=(
            resolve_project_path(args.database_root) if args.database_root else None
        ),
        snowflake_credential_path=(
            resolve_project_path(args.snowflake_credential_path)
            if args.snowflake_credential_path
            else None
        ),
        timeout_seconds=args.timeout_seconds,
        max_result_rows=args.max_result_rows,
    )
    if factory.route.backend == "snowflake" and gold_sql_rows:
        first_db_id = gold_databases.get(gold_sql_rows[0][0])
        if first_db_id:
            probe = None
            try:
                probe = factory.create(first_db_id)
            except Exception as exc:
                print(f"Evaluation failed: {exc}", file=sys.stderr)
                return 2
            finally:
                if probe is not None:
                    probe.close()
    rows = evaluate_artifacts(
        artifacts=artifacts,
        gold_sql_rows=gold_sql_rows,
        reference_gold_ids=reference_gold_ids,
        gold_databases=gold_databases,
        factory=factory,
        progress_every=args.progress_every,
    )
    rows.sort(
        key=lambda row: (
            row["sql_method"],
            row["sql_model"],
            row["schema_method"],
            row["schema_model"],
        )
    )
    rounded = _rounded_rows(rows, args.digits)
    write_csv(output_path, rounded)
    print_table(rounded)
    print(f"\nSaved {len(rounded)} evaluation row(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
