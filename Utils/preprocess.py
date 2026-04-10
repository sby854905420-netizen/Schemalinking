from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from config import *



def build_default_db_info_path(dataset: str) -> Path:
    return PROJECT_ROOT / "Data" / dataset / "db_info.json"


def build_default_output_dir(dataset: str) -> Path:
    return PROJECT_ROOT / "Data" / dataset / "Column_level_schema"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate column-level JSON files from a db_info.json schema file."
    )
    parser.add_argument(
        "--dataset",
        default="MMQA",
        help="Dataset name used to infer default db_info and output paths.",
    )
    parser.add_argument(
        "--db-info-path",
        type=Path,
        default=None,
        help="Path to the source db_info.json file. Defaults to Data/<dataset>/db_info.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where per-database column JSON files will be written. Defaults to Data/<dataset>/Column_level_schema.",
    )
    args = parser.parse_args()
    if args.db_info_path is None:
        args.db_info_path = build_default_db_info_path(args.dataset)
    if args.output_dir is None:
        args.output_dir = build_default_output_dir(args.dataset)
    return args


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sanitize_filename(name: str) -> str:
    return name.replace("/", "|").replace("\\", "|")


def build_sample_values(
    sample_rows: dict[str, list[dict[str, Any]]],
    table_name: str,
    column_name: str,
) -> list[Any]:
    values: list[Any] = []

    for row in sample_rows.get(table_name, []):
        if column_name not in row:
            continue

        value = row[column_name]
        if value is None:
            continue

        values.append(value)

    return values


def build_sample_values_text(sample_values: list[Any]) -> str:
    try:
        return ", ".join(str(value) for value in sample_values)
    except Exception:
        safe_values: list[str] = []
        for value in sample_values:
            try:
                safe_values.append(str(value))
            except Exception:
                safe_values.append("[INVALID_VALUE]")
        return ", ".join(safe_values)


def flatten_primary_keys(primary_keys: list[Any]) -> set[int]:
    primary_key_indices: set[int] = set()
    for entry in primary_keys:
        if isinstance(entry, list):
            primary_key_indices.update(entry)
        else:
            primary_key_indices.add(entry)
    return primary_key_indices


def build_foreign_key_map(foreign_keys: list[list[int]]) -> dict[int, int]:
    foreign_key_map: dict[int, int] = {}
    for source_idx, target_idx in foreign_keys:
        foreign_key_map[source_idx] = target_idx
    return foreign_key_map


def build_column_record(
    db: dict[str, Any],
    column_idx: int,
    foreign_key_map: dict[int, int],
    primary_key_indices: set[int],
) -> dict[str, Any] | None:
    table_idx, column_name = db["column_names"][column_idx]
    if table_idx == -1 or column_name == "*":
        return None

    table_name = db["table_names"][table_idx]
    db_id = db["db_id"]
    column_id = f"{db_id}.{table_name}.{column_name}"
    sample_values = build_sample_values(
        sample_rows=db.get("sample_rows", {}),
        table_name=table_name,
        column_name=column_name,
    )

    record: dict[str, Any] = {
        "column_id": column_id,
        "column_name": column_name,
        "column_description": db["column_descriptions"][column_idx] or "",
        "column_data_type": db["column_types"][column_idx],
        "sample_values": sample_values,
        "sample_values_text": build_sample_values_text(sample_values),
        "is_primary_key": column_idx in primary_key_indices,
        "is_foreign_key": column_idx in foreign_key_map,
        "value_descriptions": "",
        "foreign_key": "",
        "meta_data": {
            "db_id": db_id,
            "table_name": table_name,
        },
    }

    value_descriptions = db.get("value_descriptions", [])
    if column_idx < len(value_descriptions):
        value_description = value_descriptions[column_idx]
        if value_description is not None:
            record["value_descriptions"] = str(value_description)

    if column_idx in foreign_key_map:
        target_idx = foreign_key_map[column_idx]
        target_table_idx, target_column_name = db["column_names"][target_idx]
        if target_table_idx != -1 and target_column_name != "*":
            record["foreign_key"] = f"{db['table_names'][target_table_idx]}.{target_column_name}"

    return record


def write_column_file(output_dir: Path, record: dict[str, Any]) -> Path:
    db_id = record["meta_data"]["db_id"]
    db_dir = output_dir / db_id
    db_dir.mkdir(parents=True, exist_ok=True)

    output_path = db_dir / f"{sanitize_filename(record['column_id'])}.json"
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def generate_column_level_schema(db_info_path: Path, output_dir: Path) -> int:
    databases = load_json(db_info_path)
    written_files = 0

    for db in databases:
        primary_key_indices = flatten_primary_keys(db.get("primary_keys", []))
        foreign_key_map = build_foreign_key_map(db.get("foreign_keys", []))

        for column_idx in range(len(db["column_names"])):
            record = build_column_record(
                db=db,
                column_idx=column_idx,
                foreign_key_map=foreign_key_map,
                primary_key_indices=primary_key_indices,
            )
            if record is None:
                continue

            write_column_file(output_dir=output_dir, record=record)
            written_files += 1

    return written_files


def main() -> None:
    args = parse_args()
    written_files = generate_column_level_schema(
        db_info_path=args.db_info_path,
        output_dir=args.output_dir,
    )
    print(f"Wrote {written_files} column files to {args.output_dir}")


if __name__ == "__main__":
    main()
