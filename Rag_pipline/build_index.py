#!/usr/bin/env python3
"""Build a Qdrant column-level vector index from dataset-specific Column_level_schema files."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

from config import *
from tqdm.auto import tqdm

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Llm.embedding_model_loader import EmbeddingModelLoader

MAX_EMBEDDING_INPUT_TOKENS = 8096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=DEFAULT_SCHEMA_DIR,
        help="Directory containing column-level JSON files.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Qdrant collection name. Defaults to the dataset name and preserves case.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=None,
        help="Local Qdrant storage path. Defaults to <dataset_root>/qdrant_column_index.",
    )
    parser.add_argument(
        "--model-name",
        default=EMBEDDING_MODEL_NAME,
        help="Embedding model name.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "Llm" / "cache",
        help="Embedding model cache directory.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=128,
        help="How many column files to normalize, embed, and upsert at a time.",
    )
    parser.add_argument("--device", default=None, help="Embedding device override, e.g. cpu or cuda.")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it already exists.")
    return parser.parse_args()


def infer_dataset_name(schema_dir: Path) -> str:
    if schema_dir.name == "Column_level_schema" and schema_dir.parent.name:
        return schema_dir.parent.name
    return schema_dir.name or "dataset"


def resolve_collection_name(collection_name: str | None, schema_dir: Path) -> str:
    if collection_name:
        return collection_name
    return infer_dataset_name(schema_dir)


def resolve_qdrant_path(qdrant_path: Path | None, schema_dir: Path) -> Path:
    if qdrant_path is not None:
        return qdrant_path
    if schema_dir.name == "Column_level_schema":
        return schema_dir.parent / "qdrant_column_index"
    return DEFAULT_QDRANT_PATH


def iter_column_files(schema_dir: Path) -> Iterable[Path]:
    yield from sorted(schema_dir.rglob("*.json"))


def load_column_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def normalize_sample_values(record: dict[str, Any]) -> tuple[list[Any], str]:
    raw_values = record.get("sample_values", [])
    if isinstance(raw_values, list):
        sample_values = raw_values
    elif raw_values is None:
        sample_values = []
    else:
        sample_values = [raw_values]

    sample_values_text = clean_text(record.get("sample_values_text"))
    if not sample_values_text and sample_values:
        sample_values_text = ", ".join(clean_text(value) for value in sample_values if clean_text(value))

    return sample_values, sample_values_text


def normalize_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    meta = record["meta_data"]
    column_description = clean_text(record.get("column_description"))
    column_data_type = clean_text(record.get("column_data_type"))
    sample_values, sample_values_text = normalize_sample_values(record)
    value_descriptions = clean_text(record.get("value_descriptions"))
    foreign_key = clean_text(record.get("foreign_key"))

    return {
        "column_id": clean_text(record["column_id"]),
        "column_name": clean_text(record["column_name"]),
        "column_description": column_description,
        "column_data_type": column_data_type,
        "sample_values": sample_values,
        "sample_values_text": sample_values_text,
        "is_primary_key": record.get("is_primary_key"),
        "is_foreign_key": record.get("is_foreign_key"),
        "value_descriptions": value_descriptions,
        "foreign_key": foreign_key,
        "db_id": clean_text(meta["db_id"]),
        "table_name": clean_text(meta["table_name"]),
        "source_path": str(source_path),
    }


def build_document(record: dict[str, Any]) -> str:
    lines = [
        f"Database: {record['db_id']}",
        f"Table: {record['table_name']}",
        f"Column: {record['column_name']}",
        f"Data type: {record['column_data_type']}",
    ]

    description = record.get("column_description", "")
    if description:
        lines.append(f"Description: {description}")

    value_descriptions = record.get("value_descriptions", "")
    if value_descriptions:
        lines.append(f"Value meanings: {value_descriptions}")

    foreign_key = record.get("foreign_key", "")
    if foreign_key:
        lines.append(f"Foreign key: {foreign_key}")

    sample_values_text = record.get("sample_values_text", "")
    if sample_values_text:
        lines.append(f"Sample values: {sample_values_text}")

    return "\n".join(lines)


def build_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "column_id": record["column_id"],
        "column_name": record["column_name"],
        "column_description": record["column_description"],
        "column_data_type": record["column_data_type"],
        "sample_values": record.get("sample_values", []),
        "sample_values_text": record.get("sample_values_text", ""),
        "is_primary_key": record["is_primary_key"],
        "is_foreign_key": record["is_foreign_key"],
        "value_descriptions": record.get("value_descriptions", ""),
        "foreign_key": record.get("foreign_key", ""),
        "db_id": record["db_id"],
        "table_name": record["table_name"],
        "source_path": record["source_path"],
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


def resolve_embedding_tokenizer(embedder: EmbeddingModelLoader):
    if getattr(embedder, "tokenizer", None) is not None:
        return embedder.tokenizer

    model = getattr(embedder, "model", None)
    return getattr(model, "tokenizer", None)


def truncate_text_by_tokens(text: str, tokenizer, max_tokens: int) -> tuple[str, bool]:
    if tokenizer is None or max_tokens <= 0 or not text:
        return text, False

    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(token_ids) <= max_tokens:
        return text, False

    truncated_text = tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
    return truncated_text, True


def build_index(args: argparse.Namespace) -> None:
    schema_dir = args.schema_dir.resolve()
    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema directory does not exist: {schema_dir}")

    column_files = list(iter_column_files(schema_dir))
    if not column_files:
        raise FileNotFoundError(f"No column JSON files were found under {schema_dir}")

    collection_name = resolve_collection_name(args.collection_name, schema_dir)
    qdrant_path = resolve_qdrant_path(args.qdrant_path, schema_dir)

    embedder = EmbeddingModelLoader(
        model_name=args.model_name,
        cache_dir=str(args.cache_dir),
        device=args.device,
        trust_remote_code=True,
    )
    tokenizer = resolve_embedding_tokenizer(embedder)
    client = get_qdrant_client(qdrant_path)

    from qdrant_client.http import models

    vector_size: int | None = None
    total_points = 0
    truncated_documents = 0
    with tqdm(total=len(column_files), desc="Indexing columns", unit="column") as progress_bar:
        for batch_files in batched(column_files, args.upsert_batch_size):
            normalized_records = [
                normalize_record(load_column_record(column_file), column_file)
                for column_file in batch_files
            ]
            documents = []
            for record in normalized_records:
                document, was_truncated = truncate_text_by_tokens(
                    build_document(record),
                    tokenizer=tokenizer,
                    max_tokens=MAX_EMBEDDING_INPUT_TOKENS,
                )
                documents.append(document)
                if was_truncated:
                    truncated_documents += 1
            payloads = [build_payload(record) for record in normalized_records]
            embeddings = embedder.batch_encode(documents, batch_size=args.batch_size)

            if vector_size is None:
                vector_size = len(embeddings[0])
                ensure_collection(
                    client=client,
                    collection_name=collection_name,
                    vector_size=vector_size,
                    recreate=args.recreate,
                )

            points = [
                models.PointStruct(
                    id=stable_point_id(record["column_id"]),
                    vector=embedding,
                    payload=payload,
                )
                for record, embedding, payload in zip(normalized_records, embeddings, payloads)
            ]
            client.upsert(collection_name=collection_name, points=points, wait=True)
            total_points += len(points)
            progress_bar.update(len(batch_files))
            progress_bar.set_postfix(indexed=total_points, truncated=truncated_documents)

    if vector_size is None:
        raise RuntimeError(f"Failed to create embeddings from {schema_dir}")

    print(f"Dataset: {infer_dataset_name(schema_dir)}")
    print(f"Collection: {collection_name}")
    print(f"Indexed columns: {total_points}")
    print(f"Vector size: {vector_size}")
    print(f"Embedding model: {args.model_name}")
    print(f"Truncated documents: {truncated_documents}")
    print(f"Schema dir: {schema_dir}")
    print(f"Qdrant path: {qdrant_path}")


def main() -> None:
    args = parse_args()
    build_index(args)


if __name__ == "__main__":
    main()
