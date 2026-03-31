#!/usr/bin/env python3
"""Build a Qdrant column-level vector index from MMQA Column_level_schema."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable
from config import *

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Llm.embedding_model_loader import EmbeddingModelLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR, help="Directory containing column-level JSON files")
    parser.add_argument("--collection-name", default="mmqa_column_level", help="Qdrant collection name")
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH, help="Local Qdrant storage path")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model name")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache", help="Embedding model cache directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--upsert-batch-size", type=int, default=128, help="Qdrant upsert batch size")
    parser.add_argument("--device", default=None, help="Embedding device override, e.g. cpu or cuda")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it already exists")
    return parser.parse_args()


def iter_column_files(schema_dir: Path) -> Iterable[Path]:
    yield from sorted(schema_dir.glob("*/*.json"))


def load_column_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_document(record: dict[str, Any]) -> str:
    meta = record["meta_data"]
    lines = [
        f"Database: {meta['db_id']}",
        f"Table: {meta['table_name']}",
        f"Column: {record['column_name']}",
        f"Type: {record['column_type']}",
        f"Normalized type: {record['normalized_type']}",
        f"Description: {record['column_description']}",
    ]
    sample_values_text = record.get("sample_values_text", "").strip()
    if sample_values_text:
        lines.append(f"Sample values: {sample_values_text}")
    return "\n".join(lines)


def build_payload(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    meta = record["meta_data"]
    return {
        "column_id": record["column_id"],
        "column_name": record["column_name"],
        "column_description": record["column_description"],
        "column_type": record["column_type"],
        "normalized_type": record["normalized_type"],
        "sample_values": record.get("sample_values", []),
        "sample_values_text": record.get("sample_values_text", ""),
        "db_id": meta["db_id"],
        "table_name": meta["table_name"],
        "source_path": str(source_path),
    }


def stable_point_id(column_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, column_id))


def get_qdrant_client(qdrant_path: Path):
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError("Please install `qdrant-client` before building the Qdrant index.") from exc

    return QdrantClient(path=str(qdrant_path))


def ensure_collection(client, collection_name: str, vector_size: int, recreate: bool) -> None:
    from qdrant_client.http import models

    exists = client.collection_exists(collection_name=collection_name)
    if exists and recreate:
        client.delete_collection(collection_name=collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def batched(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def build_index(args: argparse.Namespace) -> None:
    schema_dir = args.schema_dir
    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema directory does not exist: {schema_dir}")

    column_files = list(iter_column_files(schema_dir))
    if not column_files:
        raise FileNotFoundError(f"No column JSON files were found under {schema_dir}")

    records: list[dict[str, Any]] = []
    documents: list[str] = []
    payloads: list[dict[str, Any]] = []

    for column_file in column_files:
        record = load_column_record(column_file)
        records.append(record)
        documents.append(build_document(record))
        payloads.append(build_payload(record, column_file))

    embedder = EmbeddingModelLoader(
        model_name=args.model_name,
        cache_dir=str(args.cache_dir),
        device=args.device,
        trust_remote_code=True,
    )
    embeddings = embedder.batch_encode(documents, batch_size=args.batch_size)
    vector_size = len(embeddings[0])

    client = get_qdrant_client(args.qdrant_path)
    ensure_collection(
        client=client,
        collection_name=args.collection_name,
        vector_size=vector_size,
        recreate=args.recreate,
    )

    from qdrant_client.http import models

    points = [
        models.PointStruct(
            id=stable_point_id(record["column_id"]),
            vector=embedding,
            payload=payload,
        )
        for record, embedding, payload in zip(records, embeddings, payloads)
    ]

    for batch in batched(points, args.upsert_batch_size):
        client.upsert(collection_name=args.collection_name, points=batch, wait=True)

    print(f"Collection: {args.collection_name}")
    print(f"Indexed columns: {len(points)}")
    print(f"Vector size: {vector_size}")
    print(f"Embedding model: {args.model_name}")
    print(f"Qdrant path: {args.qdrant_path}")


def main() -> None:
    args = parse_args()
    build_index(args)


if __name__ == "__main__":
    main()
