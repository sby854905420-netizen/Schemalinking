#!/usr/bin/env python3
"""Query top-k results from a local Qdrant collection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from config import *

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Llm.embedding_model_loader import EmbeddingModelLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Query text to search in Qdrant")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=None,
        help="Ratio of the collection size to retrieve, e.g. 0.1 means top 10%%",
    )
    parser.add_argument("--collection-name", default="mmqa_column_level", help="Qdrant collection name")
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH, help="Local Qdrant storage path")
    parser.add_argument("--model-name", default=EMBEDDING_MODEL_NAME, help="Embedding model name")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache", help="Embedding model cache directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--device", default=None, help="Embedding device override, e.g. cpu or cuda")
    parser.add_argument("--db-id", default=None, help="Optional db_id filter")
    parser.add_argument("--table-name", default=None, help="Optional table_name filter")
    parser.add_argument("--with-vector", action="store_true", help="Include vectors in the returned results")
    parser.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help="How to print the retrieved results",
    )
    return parser.parse_args()


def get_qdrant_client(qdrant_path: Path):
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError("Please install `qdrant-client` before querying Qdrant.") from exc

    return QdrantClient(path=str(qdrant_path))


def build_query_filter(db_id: str | None, table_name: str | None):
    if not db_id and not table_name:
        return None

    from qdrant_client.http import models

    conditions = []
    if db_id:
        conditions.append(
            models.FieldCondition(
                key="db_id",
                match=models.MatchValue(value=db_id),
            )
        )
    if table_name:
        conditions.append(
            models.FieldCondition(
                key="table_name",
                match=models.MatchValue(value=table_name),
            )
        )

    return models.Filter(must=conditions)


def extract_points(search_result: Any) -> list[Any]:
    if isinstance(search_result, list):
        return search_result

    if hasattr(search_result, "points"):
        return list(search_result.points)

    raise TypeError(f"Unsupported Qdrant result type: {type(search_result)!r}")


def query_qdrant(client, collection_name: str, query_vector: list[float], top_k: int, query_filter, with_vector: bool):
    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=with_vector,
        )
        return extract_points(result)

    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=with_vector,
    )


def point_to_dict(point: Any) -> dict[str, Any]:
    payload = dict(point.payload or {})
    point_id = getattr(point, "id", None)
    score = getattr(point, "score", None)
    vector = getattr(point, "vector", None)

    return {
        "id": point_id,
        "score": score,
        "payload": payload,
        "vector": vector,
    }


def count_points(client, collection_name: str, query_filter) -> int:
    try:
        from qdrant_client.http import models
    except ImportError as exc:
        raise ImportError("Please install `qdrant-client` before counting Qdrant points.") from exc

    result = client.count(
        collection_name=collection_name,
        count_filter=query_filter,
        exact=True,
    )
    if hasattr(result, "count"):
        return int(result.count)
    if isinstance(result, models.CountResult):
        return int(result.count)
    raise TypeError(f"Unsupported Qdrant count result type: {type(result)!r}")


def resolve_top_k(client, collection_name: str, top_k: int, top_ratio: float | None, query_filter) -> int:
    if top_ratio is None:
        if top_k <= 0:
            raise ValueError("--top-k must be greater than 0")
        return top_k

    if not 0 < top_ratio <= 1:
        raise ValueError("--top-ratio must be within (0, 1]")

    total_points = count_points(client, collection_name, query_filter)
    if total_points <= 0:
        return 0

    return max(1, math.ceil(total_points * top_ratio))


def print_text_results(query: str, results: list[dict[str, Any]]) -> None:
    print(f"Query: {query}")
    print(f"Hits: {len(results)}")

    for index, result in enumerate(results, start=1):
        payload = result["payload"]
        print("-" * 80)
        print(f"Rank: {index}")
        print(f"Score: {result['score']:.6f}" if result["score"] is not None else "Score: None")
        print(f"db_id: {payload.get('db_id')}")
        print(f"table_name: {payload.get('table_name')}")
        print(f"column_name: {payload.get('column_name')}")
        print(f"column_type: {payload.get('column_type')}")
        print(f"normalized_type: {payload.get('normalized_type')}")
        print(f"column_id: {payload.get('column_id')}")
        print(f"description: {payload.get('column_description')}")
        print(f"sample_values_text: {payload.get('sample_values_text')}")
        print(f"source_path: {payload.get('source_path')}")


def main() -> None:
    args = parse_args()

    embedder = EmbeddingModelLoader(
        model_name=args.model_name,
        cache_dir=str(args.cache_dir),
        device=args.device,
        trust_remote_code=True,
    )
    query_vector = embedder.encode(args.query, batch_size=args.batch_size, convert_to_list=True)

    client = get_qdrant_client(args.qdrant_path)
    query_filter = build_query_filter(args.db_id, args.table_name)
    resolved_top_k = resolve_top_k(
        client=client,
        collection_name=args.collection_name,
        top_k=args.top_k,
        top_ratio=args.top_ratio,
        query_filter=query_filter,
    )
    points = query_qdrant(
        client=client,
        collection_name=args.collection_name,
        query_vector=query_vector,
        top_k=resolved_top_k,
        query_filter=query_filter,
        with_vector=args.with_vector,
    )
    results = [point_to_dict(point) for point in points]

    if args.output_format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print_text_results(args.query, results)


if __name__ == "__main__":
    main()
