"""Normalize native table/column predictions for downstream consumers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from Utils.json_utils import normalize_response_text


def normalize_predicted_columns(value: Any) -> dict[str, list[str]]:
    if isinstance(value, str):
        try:
            value = json.loads(normalize_response_text(value))
        except json.JSONDecodeError:
            return {}
    if isinstance(value, Mapping) and "relevant_columns" in value:
        value = value.get("relevant_columns")
        if isinstance(value, str):
            return normalize_predicted_columns(value)
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, list[str]] = {}
    for table_name, column_names in value.items():
        table = str(table_name).strip()
        if not table or not isinstance(column_names, Sequence) or isinstance(
            column_names, (str, bytes)
        ):
            continue
        seen: set[str] = set()
        columns: list[str] = []
        for column_name in column_names:
            column = str(column_name).strip()
            if column and column not in seen:
                seen.add(column)
                columns.append(column)
        if columns:
            normalized[table] = columns
    return normalized


def parse_predicted_columns_from_text(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, str) or not value.strip():
        return {}
    return normalize_predicted_columns(value)


def resolve_predicted_columns(row: Mapping[str, Any]) -> dict[str, list[str]]:
    return normalize_predicted_columns(
        row.get("predict_columns")
    ) or parse_predicted_columns_from_text(row.get("predict_columns_text"))


def normalize_predicted_tables(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(normalize_response_text(value))
        except json.JSONDecodeError:
            return []
    if isinstance(value, Mapping) and "relevant_tables" in value:
        value = value.get("relevant_tables")
        if isinstance(value, str):
            return normalize_predicted_tables(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for table_name in value:
        table = str(table_name).strip()
        if table and table not in seen:
            seen.add(table)
            normalized.append(table)
    return normalized


def resolve_predicted_tables(row: Mapping[str, Any]) -> list[str]:
    return normalize_predicted_tables(
        row.get("predict_tables")
    ) or normalize_predicted_tables(row.get("predict_tables_text"))


__all__ = [
    "normalize_predicted_columns",
    "normalize_predicted_tables",
    "parse_predicted_columns_from_text",
    "resolve_predicted_columns",
    "resolve_predicted_tables",
]
