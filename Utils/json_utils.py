"""Lightweight JSON I/O and model-response helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json_records(path: Path | str) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    return [record for record in payload if isinstance(record, dict)]


def atomic_write_json(path: Path | str, payload: Any) -> None:
    """Write UTF-8 JSON through a same-directory temporary file and replace."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def upsert_ordered_record(
    records: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
    *,
    key: str = "id",
) -> list[dict[str, Any]]:
    """Replace a keyed record in place, append new keys, and collapse duplicates."""

    target = record[key]
    result: list[dict[str, Any]] = []
    replaced = False
    for existing in records:
        if existing.get(key) == target:
            if not replaced:
                result.append(dict(record))
                replaced = True
            continue
        result.append(dict(existing))
    if not replaced:
        result.append(dict(record))
    return result


def normalize_response_text(response_text: str) -> str:
    """Remove only outer thinking/Markdown JSON wrappers.

    JSON string contents are left untouched; in particular, occurrences of
    the word ``json`` inside identifiers or values are never removed.
    """

    text = str(response_text).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"^\s*json\s+(?=[\[{])", "", text, flags=re.IGNORECASE)
    return text.strip()


__all__ = [
    "atomic_write_json",
    "load_json",
    "load_json_records",
    "normalize_response_text",
    "upsert_ordered_record",
]
