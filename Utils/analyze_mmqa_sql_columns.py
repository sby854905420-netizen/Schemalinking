#!/usr/bin/env python3
"""Extract SQL column usage from MMQA records and append it to Spider-mapped files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_FILES = (
    "Synthesized_two_table_with_spider_db_id.json",
    "Synthesized_three_table_with_spider_db_id.json",
)
DEFAULT_MMQA_DIR = Path(__file__).resolve().parents[1] / "Data" / "MMQA"
OUTPUT_FIELD = "sql_columns_by_table"

TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)
QUALIFIED_COLUMN_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b"
)
IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
STRING_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")

SQL_KEYWORDS = {
    "ALL",
    "AND",
    "ANY",
    "AS",
    "ASC",
    "AVG",
    "BETWEEN",
    "BY",
    "CASE",
    "COUNT",
    "CURRENT_DATE",
    "CURRENT_TIME",
    "CURRENT_TIMESTAMP",
    "DATE",
    "DELETE",
    "DESC",
    "DISTINCT",
    "ELSE",
    "END",
    "EXCEPT",
    "EXISTS",
    "FROM",
    "GROUP",
    "HAVING",
    "IN",
    "INNER",
    "INSERT",
    "INTERSECT",
    "INTO",
    "IS",
    "JOIN",
    "LEFT",
    "LIKE",
    "LIMIT",
    "MAX",
    "MIN",
    "NOT",
    "NULL",
    "ON",
    "OR",
    "ORDER",
    "OUTER",
    "RIGHT",
    "ROUND",
    "SELECT",
    "SET",
    "SUM",
    "THEN",
    "UNION",
    "UPDATE",
    "VALUES",
    "WHEN",
    "WHERE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MMQA SQL statements, extract used columns by table, and add "
            "the result to the Spider db_id JSON files."
        )
    )
    parser.add_argument(
        "--mmqa-dir",
        type=Path,
        default=DEFAULT_MMQA_DIR,
        help="Directory containing MMQA JSON files. Default: %(default)s",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_FILES),
        help="MMQA files to process. Default: the two Spider db_id files.",
    )
    parser.add_argument(
        "--field-name",
        default=OUTPUT_FIELD,
        help="Field name to append to each record. Default: %(default)s",
    )
    return parser.parse_args()


def strip_string_literals(sql: str) -> str:
    return STRING_LITERAL_RE.sub(" ", sql)


def build_alias_map(sql: str, table_names: list[str]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    known_tables = {table.lower(): table for table in table_names}

    for match in TABLE_REF_RE.finditer(sql):
        table_token, alias_token = match.groups()
        table_name = known_tables.get(table_token.lower(), table_token)
        alias_map[table_name.lower()] = table_name

        if alias_token:
            alias_upper = alias_token.upper()
            if alias_upper not in SQL_KEYWORDS and alias_token.lower() != table_name.lower():
                alias_map[alias_token.lower()] = table_name

    for table_name in table_names:
        alias_map[table_name.lower()] = table_name

    return alias_map


def build_column_lookup(record: dict) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for table_name, table_info in zip(record["table_names"], record["tables"]):
        columns = table_info.get("table_columns", [])
        lookup[table_name] = {column.lower(): column for column in columns}
    return lookup


def add_column(
    collected: dict[str, list[str]],
    seen: dict[str, set[str]],
    table_name: str,
    column_name: str,
) -> None:
    if column_name in seen[table_name]:
        return
    seen[table_name].add(column_name)
    collected[table_name].append(column_name)


def extract_columns_by_table(record: dict) -> dict[str, list[str]]:
    sql = strip_string_literals(record["SQL"])
    table_names = record["table_names"]
    alias_map = build_alias_map(sql, table_names)
    column_lookup = build_column_lookup(record)
    collected = {table_name: [] for table_name in table_names}
    seen = {table_name: set() for table_name in table_names}

    for qualifier, column_token in QUALIFIED_COLUMN_RE.findall(sql):
        table_name = alias_map.get(qualifier.lower())
        if not table_name:
            continue
        canonical_column = column_lookup.get(table_name, {}).get(column_token.lower())
        if canonical_column:
            add_column(collected, seen, table_name, canonical_column)

    sql_without_qualified = QUALIFIED_COLUMN_RE.sub(" ", sql)
    known_table_tokens = {table.lower() for table in table_names}
    known_alias_tokens = set(alias_map.keys())

    for token in IDENTIFIER_RE.findall(sql_without_qualified):
        token_lower = token.lower()
        if token.upper() in SQL_KEYWORDS:
            continue
        if token_lower in known_table_tokens or token_lower in known_alias_tokens:
            continue

        matched_tables = []
        for table_name in table_names:
            canonical_column = column_lookup[table_name].get(token_lower)
            if canonical_column:
                matched_tables.append((table_name, canonical_column))

        for table_name, canonical_column in matched_tables:
            add_column(collected, seen, table_name, canonical_column)

    return {table_name: columns for table_name, columns in collected.items() if columns}


def format_columns_by_table(columns_by_table: dict[str, list[str]], table_names: list[str]) -> str:
    parts = []
    for table_name in table_names:
        columns = columns_by_table.get(table_name)
        if not columns:
            continue
        parts.append(f"{table_name}: [{', '.join(columns)}]")
    return "; ".join(parts)


def process_file(json_path: Path, field_name: str) -> None:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    for record in records:
        columns_by_table = extract_columns_by_table(record)
        record[field_name] = format_columns_by_table(columns_by_table, record["table_names"])

    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Updated {json_path}")


def main() -> None:
    args = parse_args()
    mmqa_dir = args.mmqa_dir.resolve()

    for file_name in args.files:
        json_path = (mmqa_dir / file_name).resolve()
        if not json_path.exists():
            print(f"Skip missing file: {json_path}")
            continue
        process_file(json_path, args.field_name)


if __name__ == "__main__":
    main()
