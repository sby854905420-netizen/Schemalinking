"""Offline CLI for converting native MDBlink logs to unified SL artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config import EMBEDDING_MODEL_NAME, dataset_root, resolve_project_path
from Utils.json_utils import load_json_records
from Utils.prediction_adapter import build_prediction_from_native
from Utils.prediction_store import (
    METHOD_NAMES,
    build_prediction_path,
    initialize_prediction_file,
    replace_predictions,
)
from Utils.value_utils import get_row_value, index_records_by_id


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert native MDBlink logs without invoking models or databases."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHOD_NAMES))
    parser.add_argument("--schema-results", type=Path, required=True)
    parser.add_argument("--database-results", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--documents-dir", type=Path, default=None)
    parser.add_argument("--schema-model-name", default=None)
    parser.add_argument("--database-model-name", default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args(argv)


def convert_native_results(
    *,
    schema_records: Sequence[Mapping[str, Any]],
    database_records: Sequence[Mapping[str, Any]],
    dataset_records: Sequence[Mapping[str, Any]],
    dataset_name: str,
    method: str,
    documents_dir: Path,
    output_path: Path,
    database_model_name: str,
    schema_model_name: str,
) -> dict[str, Any]:
    initialize_prediction_file(
        output_path,
        dataset_name=dataset_name,
        method=method,
        database_selection_model_name=database_model_name,
        schema_linking_model_name=schema_model_name,
    )
    schema_index = index_records_by_id(schema_records)
    database_index = index_records_by_id(database_records)
    successes = failures = 0
    missing_ids: list[str] = []
    converted_predictions: list[dict[str, Any]] = []

    for source in dataset_records:
        sample_id = get_row_value(source, "id", "instance_id")
        if sample_id is None:
            missing_ids.append("<missing dataset id>")
            continue
        stable_id = str(sample_id)
        schema_record = schema_index.get(stable_id)
        if schema_record is None:
            missing_ids.append(stable_id)
            continue
        try:
            prediction = build_prediction_from_native(
                schema_record=schema_record,
                database_record=database_index.get(stable_id),
                source_record=source,
                dataset_name=dataset_name,
                method=method,
                documents_dir=documents_dir,
            )
            converted_predictions.append(prediction)
            if prediction["status"] == "success":
                successes += 1
            else:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"Failed to convert id={stable_id}: {type(exc).__name__}: {exc}")

    replace_predictions(output_path, converted_predictions)
    return {
        "successes": successes,
        "failures": failures,
        "missing_ids": missing_ids,
        "output_path": str(output_path),
    }


def _infer_model(records: Sequence[Mapping[str, Any]], fallback: str) -> str:
    for record in records:
        model = record.get("model")
        if isinstance(model, str) and model.strip() and model != "rag_column_retrieval":
            return model.strip()
    return fallback


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    current_dataset_root = dataset_root(args.dataset_name)
    schema_records = load_json_records(resolve_project_path(args.schema_results))
    database_records = (
        load_json_records(resolve_project_path(args.database_results))
        if args.database_results
        else []
    )
    dataset_records = load_json_records(
        resolve_project_path(args.dataset_path)
        if args.dataset_path
        else current_dataset_root / "gold_sl.json"
    )
    documents_dir = (
        resolve_project_path(args.documents_dir)
        if args.documents_dir
        else current_dataset_root / "documents"
    )
    schema_model_name = args.schema_model_name or _infer_model(
        schema_records, EMBEDDING_MODEL_NAME
    )
    database_model_name = args.database_model_name or _infer_model(
        database_records, schema_model_name
    )
    output_path = (
        resolve_project_path(args.output_path)
        if args.output_path
        else build_prediction_path(args.method, args.dataset_name, schema_model_name)
    )
    summary = convert_native_results(
        schema_records=schema_records,
        database_records=database_records,
        dataset_records=dataset_records,
        dataset_name=args.dataset_name,
        method=args.method,
        documents_dir=documents_dir,
        output_path=output_path,
        database_model_name=database_model_name,
        schema_model_name=schema_model_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
