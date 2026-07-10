"""Scalar and dataset-value helpers without pandas dependencies."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        comparison = value != value
    except Exception:
        return False
    try:
        return bool(comparison)
    except (TypeError, ValueError):
        return True


def get_row_value(source: Any, *keys: str) -> Any:
    if not hasattr(source, "get"):
        return None
    for key in keys:
        value = source.get(key)
        if not is_missing_value(value):
            return value
    return None


def index_records_by_id(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_id = get_row_value(record, "id", "instance_id")
        if sample_id is not None:
            indexed[str(sample_id)] = record
    return indexed


def resolve_external_knowledge(
    source: Any,
    *,
    key: str = "external_knowledge",
    dataset_name: str | None = None,
    documents_dir: Path | None = None,
) -> str | None:
    value = source.get(key) if hasattr(source, "get") else source
    if is_missing_value(value):
        return None
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value).strip()
    if not text:
        return None
    if not dataset_name or dataset_name.lower() != "spider2" or documents_dir is None:
        return text

    documents_root = documents_dir.resolve()
    document_path = (documents_root / text).resolve()
    try:
        document_path.relative_to(documents_root)
    except ValueError:
        return text
    if not document_path.is_file():
        return text
    return document_path.read_text(encoding="utf-8").strip()


__all__ = [
    "get_row_value",
    "index_records_by_id",
    "is_missing_value",
    "resolve_external_knowledge",
]
