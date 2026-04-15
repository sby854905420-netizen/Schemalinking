#!/usr/bin/env python3
"""Build a Qdrant column-level vector index directly from dataset-specific db_info.json files."""

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
from Utils.render_tools import SchemaTextRenderer

MAX_EMBEDDING_INPUT_TOKENS = 8096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-info-path",
        type=Path,
        default=PROJECT_ROOT / "Data" / DATASET_NAME / "db_info.json",
        help="Path to the dataset db_info.json file.",
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
        help="How many column records to normalize, embed, and upsert at a time.",
    )
    parser.add_argument("--device", default=None, help="Embedding device override, e.g. cpu or cuda.")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it already exists.")
    return parser.parse_args()


def infer_dataset_name(db_info_path: Path) -> str:
    if db_info_path.name == "db_info.json" and db_info_path.parent.name:
        return db_info_path.parent.name
    return db_info_path.stem or db_info_path.parent.name or "dataset"


def resolve_collection_name(collection_name: str | None, db_info_path: Path) -> str:
    if collection_name:
        return collection_name
    return infer_dataset_name(db_info_path)


def resolve_qdrant_path(qdrant_path: Path | None, db_info_path: Path) -> Path:
    if qdrant_path is not None:
        return qdrant_path
    return db_info_path.parent / "qdrant_column_index"


def load_db_infos(db_info_path: Path) -> list[dict[str, Any]]:
    with db_info_path.open("r", encoding="utf-8") as file:
        db_infos = json.load(file)

    if not isinstance(db_infos, list):
        raise TypeError(f"Expected db_info.json to contain a list, got {type(db_infos)!r}")

    return [entry for entry in db_infos if isinstance(entry, dict)]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def get_indexed_value(values: Any, index: int, default: Any = None) -> Any:
    if not isinstance(values, list):
        return default
    if index < 0 or index >= len(values):
        return default
    return values[index]


def build_sample_values(
    sample_rows: Any,
    table_name: str,
    column_name: str,
) -> list[Any]:
    values: list[Any] = []
    if not isinstance(sample_rows, dict):
        return values

    table_rows = sample_rows.get(table_name, [])
    if not isinstance(table_rows, list):
        return values

    for row in table_rows:
        if not isinstance(row, dict):
            continue
        if column_name not in row:
            continue

        value = row[column_name]
        if value is None:
            continue
        values.append(value)

    return values


def flatten_primary_keys(primary_keys: Any) -> set[int]:
    primary_key_indices: set[int] = set()
    if not isinstance(primary_keys, list):
        return primary_key_indices

    for entry in primary_keys:
        if isinstance(entry, list):
            for index in entry:
                if isinstance(index, int):
                    primary_key_indices.add(index)
            continue
        if isinstance(entry, int):
            primary_key_indices.add(entry)

    return primary_key_indices


def build_foreign_key_map(foreign_keys: Any) -> dict[int, int]:
    foreign_key_map: dict[int, int] = {}
    if not isinstance(foreign_keys, list):
        return foreign_key_map

    for entry in foreign_keys:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        source_idx, target_idx = entry
        if not isinstance(source_idx, int) or not isinstance(target_idx, int):
            continue
        foreign_key_map[source_idx] = target_idx

    return foreign_key_map


def build_foreign_key_text(
    db: dict[str, Any],
    column_idx: int,
    foreign_key_map: dict[int, int],
) -> str:
    target_idx = foreign_key_map.get(column_idx)
    if target_idx is None:
        return ""

    target_entry = get_indexed_value(db.get("column_names"), target_idx)
    if not isinstance(target_entry, list) or len(target_entry) != 2:
        return ""

    target_table_idx, target_column_name = target_entry
    if target_table_idx == -1 or target_column_name == "*":
        return ""

    table_names = db.get("table_names")
    target_table_name = get_indexed_value(table_names, target_table_idx, "")
    if not target_table_name:
        return ""

    return f"{target_table_name}.{target_column_name}"


def build_value_descriptions(db: dict[str, Any], column_idx: int) -> str:
    return clean_text(get_indexed_value(db.get("value_descriptions"), column_idx, ""))


def build_column_record(
    db: dict[str, Any],
    column_idx: int,
    primary_key_indices: set[int],
    foreign_key_map: dict[int, int],
) -> dict[str, Any] | None:
    column_entry = get_indexed_value(db.get("column_names"), column_idx)
    if not isinstance(column_entry, list) or len(column_entry) != 2:
        return None

    table_idx, column_name = column_entry
    if table_idx == -1 or column_name == "*":
        return None

    table_name = clean_text(get_indexed_value(db.get("table_names"), table_idx, ""))
    db_id = clean_text(db.get("db_id"))
    normalized_column_name = clean_text(column_name)
    if not db_id or not table_name or not normalized_column_name:
        return None

    return {
        "column_id": f"{db_id}.{table_name}.{normalized_column_name}",
        "column_name": normalized_column_name,
        "column_description": clean_text(get_indexed_value(db.get("column_descriptions"), column_idx, "")),
        "column_data_type": clean_text(get_indexed_value(db.get("column_types"), column_idx, "")),
        "sample_values": build_sample_values(
            sample_rows=db.get("sample_rows", {}),
            table_name=table_name,
            column_name=normalized_column_name,
        ),
        "is_primary_key": column_idx in primary_key_indices,
        "is_foreign_key": column_idx in foreign_key_map,
        "value_descriptions": build_value_descriptions(db, column_idx),
        "foreign_key": clean_text(build_foreign_key_text(db, column_idx, foreign_key_map)),
        "db_id": db_id,
        "table_name": table_name,
    }


def iter_column_records(db_infos: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for db in db_infos:
        primary_key_indices = flatten_primary_keys(db.get("primary_keys", []))
        foreign_key_map = build_foreign_key_map(db.get("foreign_keys", []))
        column_names = db.get("column_names", [])
        if not isinstance(column_names, list):
            continue

        for column_idx in range(len(column_names)):
            record = build_column_record(
                db=db,
                column_idx=column_idx,
                primary_key_indices=primary_key_indices,
                foreign_key_map=foreign_key_map,
            )
            if record is not None:
                yield record


def count_columns(db_infos: Iterable[dict[str, Any]]) -> int:
    total_columns = 0
    for db in db_infos:
        column_names = db.get("column_names", [])
        if not isinstance(column_names, list):
            continue
        for entry in column_names:
            if not isinstance(entry, list) or len(entry) != 2:
                continue
            table_idx, column_name = entry
            if table_idx == -1 or column_name == "*":
                continue
            total_columns += 1
    return total_columns


def build_document(record: dict[str, Any], renderer: SchemaTextRenderer) -> str:
    return renderer.render_index_column_text(record)


def build_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "column_id": record["column_id"],
        "column_name": record["column_name"],
        "column_description": record["column_description"],
        "column_data_type": record["column_data_type"],
        "sample_values": record.get("sample_values", []),
        "is_primary_key": record["is_primary_key"],
        "is_foreign_key": record["is_foreign_key"],
        "value_descriptions": record.get("value_descriptions", ""),
        "foreign_key": record.get("foreign_key", ""),
        "db_id": record["db_id"],
        "table_name": record["table_name"],
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


def batched(values: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


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
    db_info_path = args.db_info_path.resolve()
    if not db_info_path.is_file():
        raise FileNotFoundError(f"db_info.json file does not exist: {db_info_path}")

    db_infos = load_db_infos(db_info_path)
    total_columns = count_columns(db_infos)
    if total_columns <= 0:
        raise FileNotFoundError(f"No valid column records were found in {db_info_path}")

    collection_name = resolve_collection_name(args.collection_name, db_info_path)
    qdrant_path = resolve_qdrant_path(args.qdrant_path, db_info_path)

    embedder = EmbeddingModelLoader(
        model_name=args.model_name,
        cache_dir=str(args.cache_dir),
        device=args.device,
        trust_remote_code=True,
    )
    tokenizer = resolve_embedding_tokenizer(embedder)
    if tokenizer is None:
        raise RuntimeError("Failed to resolve a tokenizer for the embedding model.")
    renderer = SchemaTextRenderer(tokenizer=tokenizer)
    client = get_qdrant_client(qdrant_path)

    from qdrant_client.http import models

    vector_size: int | None = None
    total_points = 0
    truncated_documents = 0
    with tqdm(total=total_columns, desc="Indexing columns", unit="column") as progress_bar:
        for normalized_records in batched(iter_column_records(db_infos), args.upsert_batch_size):
            documents = []
            for record in normalized_records:
                document, was_truncated = truncate_text_by_tokens(
                    build_document(record, renderer=renderer),
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
            progress_bar.update(len(normalized_records))
            progress_bar.set_postfix(indexed=total_points, truncated=truncated_documents)

    if vector_size is None:
        raise RuntimeError(f"Failed to create embeddings from {db_info_path}")

    print(f"Dataset: {infer_dataset_name(db_info_path)}")
    print(f"Collection: {collection_name}")
    print(f"Indexed columns: {total_points}")
    print(f"Vector size: {vector_size}")
    print(f"Embedding model: {args.model_name}")
    print(f"Truncated documents: {truncated_documents}")
    print(f"DB info path: {db_info_path}")
    print(f"Qdrant path: {qdrant_path}")


def main() -> None:
    args = parse_args()
    build_index(args)


if __name__ == "__main__":
    main()
