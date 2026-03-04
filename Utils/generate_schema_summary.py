#!/usr/bin/env python3
"""Generate schema summary CSV files for Spider SQLite databases."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


DEFAULT_BASE_DIR = Path(__file__).resolve().parent / "data" / "spider"
DEFAULT_SOURCE_DIRS = ("database", "test_database")
DEFAULT_OUTPUT_FILE_NAME = "schema description.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate schema summary CSV files for each SQLite database under "
            "Spider's database and test_database directories."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base Spider data directory. Default: %(default)s",
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=list(DEFAULT_SOURCE_DIRS),
        help="Subdirectories under base-dir to scan. Default: database test_database",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output root directory. If omitted, each CSV is written next to "
            "the SQLite file."
        ),
    )
    parser.add_argument(
        "--output-file-name",
        default=DEFAULT_OUTPUT_FILE_NAME,
        help=(
            "Output CSV file name when writing next to SQLite files. "
            "Default: %(default)s"
        ),
    )
    return parser.parse_args()


def find_sqlite_files(source_dir: Path) -> list[Path]:
    sqlite_files = []
    for db_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        direct_matches = sorted(db_dir.glob("*.sqlite"))
        if not direct_matches:
            continue
        sqlite_files.append(direct_matches[0])
    return sqlite_files


def quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def get_user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [row[0] for row in rows]


def get_foreign_key_map(conn: sqlite3.Connection, table_name: str) -> dict[str, str]:
    fk_rows = conn.execute(
        f"PRAGMA foreign_key_list({quote_identifier(table_name)})"
    ).fetchall()
    foreign_key_map: dict[str, str] = {}
    for row in fk_rows:
        from_column = row[3]
        target_table = row[2]
        target_column = row[4]
        if not target_column:
            continue
        foreign_key_map[from_column] = f"{target_table}.{target_column}"
    return foreign_key_map


def collect_schema_rows(sqlite_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.row_factory = sqlite3.Row
        for table_name in get_user_tables(conn):
            foreign_key_map = get_foreign_key_map(conn, table_name)
            column_rows = conn.execute(
                f"PRAGMA table_info({quote_identifier(table_name)})"
            ).fetchall()
            for column in column_rows:
                column_name = column["name"]
                data_type = column["type"] or ""
                is_primary_key = "1" if column["pk"] else "0"
                foreign_key_value = foreign_key_map.get(column_name, "0")
                rows.append(
                    [
                        table_name,
                        column_name,
                        data_type,
                        is_primary_key,
                        foreign_key_value,
                    ]
                )
    finally:
        conn.close()
    return rows


def write_schema_summary(sqlite_path: Path, output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = collect_schema_rows(sqlite_path)
    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["table_name", "column_name", "data_type", "is_primary_key", "foreign_key"]
        )
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_root = args.output_dir.resolve() if args.output_dir is not None else None

    for source_name in args.source_dirs:
        source_dir = (base_dir / source_name).resolve()
        if not source_dir.exists():
            print(f"Skip missing source directory: {source_dir}")
            continue

        sqlite_files = find_sqlite_files(source_dir)
        if not sqlite_files:
            print(f"No SQLite files found under: {source_dir}")
            continue

        for sqlite_path in sqlite_files:
            if output_root is None:
                output_csv_path = sqlite_path.parent / args.output_file_name
            else:
                target_dir = output_root / source_name
                db_name = sqlite_path.stem
                output_csv_path = target_dir / f"{db_name}.csv"
            write_schema_summary(sqlite_path, output_csv_path)
            print(f"Wrote {output_csv_path}")


if __name__ == "__main__":
    main()
