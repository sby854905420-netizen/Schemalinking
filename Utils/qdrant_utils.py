from __future__ import annotations

import json
from pathlib import Path


def resolve_qdrant_collection_name(
    qdrant_path: Path,
    *,
    explicit_name: str | None = None,
    fallback: str | None = None,
) -> str:
    normalized_explicit_name = str(explicit_name or "").strip()
    if normalized_explicit_name:
        return normalized_explicit_name

    meta_path = qdrant_path / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        collections = meta.get("collections")
        if isinstance(collections, dict) and collections:
            return str(next(iter(collections)))

    normalized_fallback = str(fallback or "").strip()
    if normalized_fallback:
        return normalized_fallback

    raise ValueError(
        f"Could not resolve a Qdrant collection name from {meta_path}; "
        "provide --collection-name explicitly."
    )
