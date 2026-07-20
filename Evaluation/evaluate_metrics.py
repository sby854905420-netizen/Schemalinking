"""Evaluate DB selection and schema-linking prediction artifacts.

Default mode discovers every non-empty schema-linking artifact for one
dataset. Supplying model/method filters (or an exact prediction path) narrows
the table to a particular experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_NAME, DATA_ROOT, RESULTS_ROOT, resolve_project_path
from Utils.prediction_store import (
    normalize_snowflake_table_name,
    snowflake_identifier_key,
    split_qualified_identifier,
)


TABLE_COLUMNS = (
    "dataset",
    "method",
    "database_model",
    "schema_model",
    "total_samples",
    "prediction_records",
    "matched_predictions",
    "coverage",
    "LA",
    "EM",
    "Recall",
    "Avg_Pred_Cols",
    "Avg_Gold_Cols",
    "Avg_Ratio",
    "Avg_token",
    "Avg_time",
    "token_samples",
    "time_samples",
    "extra_predictions",
    "prediction_file",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute LA, EM, micro Recall, Avg_Ratio, Avg_token and Avg_time "
            "from unified schema-linking predictions."
        )
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help=f"Dataset to evaluate (default: {DATASET_NAME}).",
    )
    parser.add_argument(
        "--model-name",
        help=(
            "Schema-linking model filter. Both original names such as "
            "Qwen/Qwen2.5-14B-Instruct and safe path names using '__' work."
        ),
    )
    parser.add_argument("--method", help="Schema-linking method filter.")
    parser.add_argument(
        "--database-model-name",
        help="Optional database-selection model filter.",
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        help="Evaluate one exact unified SL prediction JSON instead of discovery.",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        help="Gold schema-linking JSON (default: Data/<dataset>/gold_sl.json).",
    )
    parser.add_argument(
        "--db-info-path",
        type=Path,
        help=(
            "Spider2 db_info JSON used for conservative table-name resolution "
            "(default: Data/Spider2/db_info.json)."
        ),
    )
    parser.add_argument(
        "--strict-spider2-table-names",
        action="store_true",
        help=(
            "Disable db_info-backed unique alias resolution and compare only "
            "canonical table strings."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "CSV output path (default: "
            "results/evaluation/<dataset>/metrics.csv)."
        ),
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include initialized artifacts whose predictions list is empty.",
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


def load_gold(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Gold schema file must contain a JSON list: {path}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"Gold row {index} is not an object: {path}")
        sample_id = str(raw.get("id", ""))
        db_id = raw.get("db_id")
        columns = raw.get("columns_by_table")
        if not sample_id or not isinstance(db_id, str) or not isinstance(columns, dict):
            raise ValueError(f"Gold row {index} has an invalid id/db_id/columns_by_table.")
        if sample_id in seen:
            raise ValueError(f"Duplicate gold id {sample_id!r}: {path}")
        seen.add(sample_id)
        rows.append(raw)
    return rows


def load_prediction(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    required = {"dataset", "method", "model_names", "predictions"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"Not a unified schema-linking prediction artifact: {path}")
    model_names = payload["model_names"]
    if not isinstance(model_names, dict) or set(model_names) != {
        "database_selection",
        "schema_linking",
    }:
        raise ValueError(f"Invalid model_names in {path}")
    if not isinstance(payload["predictions"], list):
        raise ValueError(f"predictions must be a list: {path}")
    seen: set[str] = set()
    for index, prediction in enumerate(payload["predictions"]):
        if not isinstance(prediction, dict) or not str(prediction.get("id", "")):
            raise ValueError(f"Invalid prediction record {index}: {path}")
        sample_id = str(prediction["id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate prediction id {sample_id!r}: {path}")
        seen.add(sample_id)
    return payload


def discover_predictions(
    *,
    dataset_name: str,
    model_name: str | None,
    method: str | None,
    database_model_name: str | None,
    prediction_path: Path | None,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    warnings: list[str] = []
    if prediction_path is not None:
        candidates = [resolve_project_path(prediction_path)]
    else:
        candidates = sorted((RESULTS_ROOT / "sl").glob("*/*/*/prediction.json"))

    selected: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        if not path.is_file():
            warnings.append(f"skip missing prediction file: {path}")
            continue
        try:
            payload = load_prediction(path)
        except ValueError as exc:
            warnings.append(f"skip invalid prediction file: {exc}")
            continue
        if str(payload["dataset"]).casefold() != dataset_name.casefold():
            continue
        if method and str(payload["method"]).casefold() != method.casefold():
            continue
        models = payload["model_names"]
        if not _name_matches(models["schema_linking"], model_name):
            continue
        if not _name_matches(models["database_selection"], database_model_name):
            continue
        selected.append((path, payload))
    return selected, warnings


def _normalize_identifier(value: Any) -> str:
    text = str(value or "").strip()
    for quote in ('"', "'", "`"):
        if (
            len(text) >= 2
            and text.startswith(quote)
            and text.endswith(quote)
            and text.count(quote) == 2
        ):
            return text[1:-1].strip()
    return text


def _normalize_gold_table(db_id: str, value: Any) -> str:
    """Convert a source ``DB.SCHEMA.TABLE`` name to ``SCHEMA.TABLE``.

    Unified prediction artifacts already store canonical table names with the
    outer database component removed.  Applying this conversion to prediction
    tables as well would incorrectly turn ``DB.TABLE`` into ``TABLE`` whenever
    a Spider2 database and schema share the same name.
    """

    table = _normalize_identifier(value)
    prefix = f"{db_id}."
    return table[len(prefix) :] if db_id and table.startswith(prefix) else table


class Spider2TableResolver:
    """Resolve Spider2 aliases to canonical ``SCHEMA.TABLE`` identifiers."""

    def __init__(self, table_names_by_db: Mapping[str, Sequence[str]]) -> None:
        self._full: dict[str, dict[tuple[str, ...], set[str]]] = {}
        self._canonical: dict[str, dict[tuple[str, ...], set[str]]] = {}
        self._basename: dict[str, dict[str, set[str]]] = {}
        for raw_db_id, raw_tables in table_names_by_db.items():
            db_id = str(raw_db_id).strip()
            if not db_id:
                continue
            full_index: dict[tuple[str, ...], set[str]] = {}
            canonical_index: dict[tuple[str, ...], set[str]] = {}
            basename_index: dict[str, set[str]] = {}
            for raw_table in raw_tables:
                full_table = str(raw_table).strip()
                full_key = snowflake_identifier_key(full_table)
                canonical_table = normalize_snowflake_table_name(db_id, full_table)
                canonical_key = snowflake_identifier_key(canonical_table)
                if not full_key or not canonical_key:
                    continue
                full_index.setdefault(full_key, set()).add(canonical_table)
                canonical_index.setdefault(canonical_key, set()).add(canonical_table)
                basename_index.setdefault(canonical_key[-1], set()).add(
                    canonical_table
                )
            self._full[db_id] = full_index
            self._canonical[db_id] = canonical_index
            self._basename[db_id] = basename_index

    def resolve(self, db_id: str, value: Any) -> str:
        table = _normalize_identifier(value)
        parts = split_qualified_identifier(value)
        key = snowflake_identifier_key(value)
        if not parts or not key or db_id not in self._full:
            return table

        if len(parts) >= 3:
            matches = self._full[db_id].get(key, set())
        elif len(parts) == 2:
            matches = self._canonical[db_id].get(key, set())
            if not matches:
                db_key = snowflake_identifier_key(db_id)
                if db_key and key[0] == db_key[0]:
                    matches = self._basename[db_id].get(key[-1], set())
        else:
            matches = self._basename[db_id].get(key[0], set())
        return next(iter(matches)) if len(matches) == 1 else table


def load_spider2_table_resolver(path: Path) -> Spider2TableResolver:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Spider2 db_info must contain a JSON list: {path}")
    table_names_by_db: dict[str, list[str]] = {}
    for index, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Spider2 db_info row {index} is not an object: {path}")
        db_id = str(entry.get("db_id") or "").strip()
        table_names = entry.get("table_names")
        if not db_id or not isinstance(table_names, list):
            raise ValueError(f"Spider2 db_info row {index} has invalid table names.")
        table_names_by_db[db_id] = [str(table) for table in table_names]
    return Spider2TableResolver(table_names_by_db)


def gold_column_set(
    row: Mapping[str, Any],
    table_resolver: Spider2TableResolver | None = None,
) -> set[tuple[str, str, str]]:
    db_id = str(row["db_id"])
    result: set[tuple[str, str, str]] = set()
    for raw_table, raw_columns in row["columns_by_table"].items():
        if not isinstance(raw_columns, Sequence) or isinstance(raw_columns, (str, bytes)):
            continue
        table = (
            table_resolver.resolve(db_id, raw_table)
            if table_resolver is not None
            else _normalize_gold_table(db_id, raw_table)
        )
        for raw_column in raw_columns:
            column = _normalize_identifier(raw_column)
            if table and column:
                result.add((db_id, table, column))
    return result


def predicted_column_set(
    record: Mapping[str, Any] | None,
    table_resolver: Spider2TableResolver | None = None,
) -> set[tuple[str, str, str]]:
    if not record:
        return set()
    linking = record.get("schema_linking")
    final = linking.get("final") if isinstance(linking, Mapping) else None
    columns = final.get("columns") if isinstance(final, Mapping) else None
    if not isinstance(columns, list):
        return set()
    result: set[tuple[str, str, str]] = set()
    for item in columns:
        if not isinstance(item, Mapping):
            continue
        db_id = _normalize_identifier(item.get("db_id"))
        # The unified SL contract already stores tables as SCHEMA.TABLE for
        # Spider2 (and as TABLE for SQLite datasets).  Do not strip db_id a
        # second time here.
        table = (
            table_resolver.resolve(db_id, item.get("table"))
            if table_resolver is not None
            else _normalize_identifier(item.get("table"))
        )
        column = _normalize_identifier(item.get("column"))
        if db_id and table and column:
            result.add((db_id, table, column))
    return result


def _selected_databases(record: Mapping[str, Any] | None) -> list[str]:
    if not record:
        return []
    selection = record.get("database_selection")
    values = selection.get("selected_db_ids") if isinstance(selection, Mapping) else None
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _usage_value(record: Mapping[str, Any], key: str) -> float | None:
    usage = record.get("usage")
    total = usage.get("linking_total") if isinstance(usage, Mapping) else None
    value = total.get(key) if isinstance(total, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def evaluate_payload(
    payload: Mapping[str, Any],
    gold_rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    table_resolver: Spider2TableResolver | None = None,
) -> dict[str, Any]:
    predictions = payload["predictions"]
    prediction_by_id = {str(record["id"]): record for record in predictions}
    gold_ids = {str(row["id"]) for row in gold_rows}

    la_correct = 0
    em_correct = 0
    recalled = 0
    gold_column_total = 0
    predicted_column_total = 0
    ratio_total = 0.0
    matched_predictions = 0
    token_values: list[float] = []
    time_values: list[float] = []

    for gold in gold_rows:
        sample_id = str(gold["id"])
        prediction = prediction_by_id.get(sample_id)
        if prediction is not None:
            matched_predictions += 1
            token_value = _usage_value(prediction, "total_tokens")
            time_value = _usage_value(prediction, "elapsed_seconds")
            if token_value is not None:
                token_values.append(token_value)
            if time_value is not None:
                time_values.append(time_value)

        if _selected_databases(prediction) == [str(gold["db_id"])]:
            la_correct += 1

        gold_columns = gold_column_set(gold, table_resolver)
        predicted_columns = predicted_column_set(prediction, table_resolver)
        if predicted_columns == gold_columns:
            em_correct += 1
        recalled += len(predicted_columns & gold_columns)
        gold_column_total += len(gold_columns)
        predicted_column_total += len(predicted_columns)
        if gold_columns:
            ratio_total += len(predicted_columns) / len(gold_columns)

    total = len(gold_rows)
    model_names = payload["model_names"]
    relative_path = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    return {
        "dataset": payload["dataset"],
        "method": payload["method"],
        "database_model": model_names["database_selection"],
        "schema_model": model_names["schema_linking"],
        "total_samples": total,
        "prediction_records": len(predictions),
        "matched_predictions": matched_predictions,
        "coverage": matched_predictions / total if total else None,
        "LA": la_correct / total if total else None,
        "EM": em_correct / total if total else None,
        "Recall": recalled / gold_column_total if gold_column_total else None,
        "Avg_Pred_Cols": predicted_column_total / total if total else None,
        "Avg_Gold_Cols": gold_column_total / total if total else None,
        "Avg_Ratio": ratio_total / total if total else None,
        "Avg_token": sum(token_values) / len(token_values) if token_values else None,
        "Avg_time": sum(time_values) / len(time_values) if time_values else None,
        "token_samples": len(token_values),
        "time_samples": len(time_values),
        "extra_predictions": len(set(prediction_by_id) - gold_ids),
        "prediction_file": str(relative_path),
    }


def _rounded_rows(rows: Iterable[Mapping[str, Any]], digits: int) -> list[dict[str, Any]]:
    rounded: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = {}
        for key in TABLE_COLUMNS:
            value = row.get(key)
            output[key] = round(value, digits) if isinstance(value, float) else value
        rounded.append(output)
    return rounded


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
        "method",
        "schema_model",
        "total_samples",
        "coverage",
        "LA",
        "EM",
        "Recall",
        "Avg_Ratio",
        "Avg_token",
        "Avg_time",
    )
    if not rows:
        print("No matching non-empty prediction artifacts; nothing to evaluate.")
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
    if args.digits < 0:
        raise SystemExit("--digits must be non-negative.")

    gold_path = (
        resolve_project_path(args.gold_path)
        if args.gold_path
        else DATA_ROOT / args.dataset_name / "gold_sl.json"
    )
    if not gold_path.is_file():
        print(f"Skip: gold schema file does not exist: {gold_path}", file=sys.stderr)
        return 0
    try:
        gold_rows = load_gold(gold_path)
    except ValueError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2

    table_resolver: Spider2TableResolver | None = None
    if (
        args.dataset_name.strip().lower() == "spider2"
        and not args.strict_spider2_table_names
    ):
        db_info_path = (
            resolve_project_path(args.db_info_path)
            if args.db_info_path
            else DATA_ROOT / args.dataset_name / "db_info.json"
        )
        if not db_info_path.is_file():
            print(f"Evaluation failed: missing Spider2 db_info: {db_info_path}", file=sys.stderr)
            return 2
        try:
            table_resolver = load_spider2_table_resolver(db_info_path)
        except ValueError as exc:
            print(f"Evaluation failed: {exc}", file=sys.stderr)
            return 2

    artifacts, warnings = discover_predictions(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        method=args.method,
        database_model_name=args.database_model_name,
        prediction_path=args.prediction_path,
    )
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    for path, payload in artifacts:
        if not payload["predictions"] and not args.include_empty:
            print(f"Skip empty prediction artifact: {path}", file=sys.stderr)
            continue
        rows.append(
            evaluate_payload(
                payload,
                gold_rows,
                path,
                table_resolver=table_resolver,
            )
        )

    rows.sort(key=lambda row: (row["method"], row["schema_model"], row["database_model"]))
    rounded = _rounded_rows(rows, args.digits)
    output_path = (
        resolve_project_path(args.output_path)
        if args.output_path
        else RESULTS_ROOT / "evaluation" / args.dataset_name / "metrics.csv"
    )
    write_csv(output_path, rounded)
    print_table(rounded)
    print(f"\nSaved {len(rounded)} evaluation row(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
