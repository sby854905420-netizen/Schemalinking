from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_NAME
from Run.logging_utils import log_run_configuration, setup_task_logger
from Utils.tools import (
    build_db_id_filter,
    get_qdrant_client,
    get_row_value,
    query_qdrant,
    resolve_hint,
)


BASELINE_NAME = "rag_column_retrieval"
BASELINE_PROVIDER = "qdrant"
DEFAULT_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TOP_K = 10


@dataclass
class RetrievedColumn:
    column_id: str
    db_id: str
    table_name: str
    column_name: str
    score: float
    rank: int
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pure RAG column-retrieval baseline. Globally retrieve k columns, "
            "choose the database with the most hits, then retrieve k columns "
            "inside that database as the final schema-linking result."
        )
    )
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--input-path", dest="input_path", type=Path, default=None)
    parser.add_argument("--qdrant-path", dest="qdrant_path", type=Path, default=None)
    parser.add_argument("--collection-name", dest="collection_name", type=str, default=None)
    parser.add_argument("--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--global-top-k", dest="global_top_k", type=int, default=None)
    parser.add_argument("--local-top-k", dest="local_top_k", type=int, default=None)
    parser.add_argument("--embedding-model-name", dest="embedding_model_name", type=str, default=None)
    parser.add_argument("--cache-dir", dest="cache_dir", type=Path, default=None)
    parser.add_argument("--device", dest="device", type=str, default=None)
    parser.add_argument(
        "--include-hint-in-query",
        dest="include_hint_in_query",
        action="store_true",
        help="Append external_knowledge/hint text to the embedding query when available.",
    )
    parser.add_argument("--start-index", dest="start_index", type=int, default=0)
    parser.add_argument("--limit", dest="limit", type=int, default=None)
    return parser.parse_args()


def validate_top_k(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Could not find input file at {input_path}.")
    return pd.read_json(input_path)


def load_qdrant_collection_name(qdrant_path: Path, fallback: str) -> str:
    meta_path = qdrant_path / "meta.json"
    if not meta_path.is_file():
        return fallback

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    collections = meta.get("collections")
    if isinstance(collections, dict) and collections:
        return str(next(iter(collections)))
    return fallback


def resolve_output_path(output_path: Path | None, dataset_name: str) -> Path:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = PROJECT_ROOT / "Logs" / BASELINE_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"zero_shot_table2column_{dataset_name}_{run_id}.json"


def point_payload(point: Any) -> dict[str, Any]:
    payload = getattr(point, "payload", None)
    if payload is None and isinstance(point, dict):
        payload = point.get("payload")
    return dict(payload or {})


def point_score(point: Any) -> float:
    score = getattr(point, "score", None)
    if score is None and isinstance(point, dict):
        score = point.get("score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return float("-inf")


def normalize_retrieved_columns(points: Sequence[Any]) -> list[RetrievedColumn]:
    columns: list[RetrievedColumn] = []
    seen_column_ids: set[str] = set()

    for rank, point in enumerate(points, start=1):
        payload = point_payload(point)
        db_id = str(payload.get("db_id", "")).strip()
        table_name = str(payload.get("table_name", "")).strip()
        column_name = str(payload.get("column_name", "")).strip()
        column_id = str(payload.get("column_id") or f"{db_id}.{table_name}.{column_name}").strip()
        if not db_id or not table_name or not column_name or not column_id:
            continue
        if column_id in seen_column_ids:
            continue

        seen_column_ids.add(column_id)
        columns.append(
            RetrievedColumn(
                column_id=column_id,
                db_id=db_id,
                table_name=table_name,
                column_name=column_name,
                score=point_score(point),
                rank=rank,
                payload=payload,
            )
        )
    return columns


def retrieve_columns(
    *,
    client: Any,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    candidate_db_ids: list[str] | None = None,
) -> list[RetrievedColumn]:
    query_filter = build_db_id_filter(candidate_db_ids)
    if candidate_db_ids is not None and query_filter is None:
        return []

    points = query_qdrant(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=top_k,
        query_filter=query_filter,
        with_vectors=False,
    )
    return normalize_retrieved_columns(points)


def rank_databases(columns: Sequence[RetrievedColumn]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for column in columns:
        stats = grouped.setdefault(
            column.db_id,
            {
                "db_id": column.db_id,
                "hit_count": 0,
                "hit_rate": 0.0,
                "score_sum": 0.0,
                "max_score": float("-inf"),
                "first_rank": column.rank,
            },
        )
        stats["hit_count"] += 1
        stats["score_sum"] += column.score
        stats["max_score"] = max(stats["max_score"], column.score)
        stats["first_rank"] = min(stats["first_rank"], column.rank)

    total_hits = max(1, len(columns))
    for stats in grouped.values():
        stats["hit_rate"] = stats["hit_count"] / total_hits
        stats["score_sum"] = round(stats["score_sum"], 8)
        stats["max_score"] = round(stats["max_score"], 8)
        stats["hit_rate"] = round(stats["hit_rate"], 8)

    return sorted(
        grouped.values(),
        key=lambda item: (
            -int(item["hit_count"]),
            -float(item["score_sum"]),
            -float(item["max_score"]),
            int(item["first_rank"]),
            str(item["db_id"]),
        ),
    )


def group_columns_by_table(columns: Sequence[RetrievedColumn]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for column in columns:
        key = (column.table_name, column.column_name)
        if key in seen:
            continue
        seen.add(key)
        grouped[column.table_name].append(column.column_name)
    return dict(grouped)


def resolve_query_text(row: Any, dataset_name: str, documents_dir: Path, include_hint: bool) -> tuple[str, str]:
    question = str(row["question"])
    hint = resolve_hint(row, dataset_name=dataset_name, documents_dir=documents_dir)
    if include_hint and hint != "No hint":
        return f"{question}\n{hint}", hint
    return question, hint


def append_result(
    records: list[dict[str, Any]],
    output_path: Path,
    row: Any,
    *,
    predicted_db_id: str | None,
    global_columns: Sequence[RetrievedColumn],
    local_columns: Sequence[RetrievedColumn],
    db_ranking: Sequence[dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    predict_columns = group_columns_by_table(local_columns)
    predict_tables = list(predict_columns.keys())
    sample_id = get_row_value(row, "id", "instance_id")
    gold_db_id = get_row_value(row, "db_id", "gold_db_id", "spider_db_id")
    question = str(row["question"])

    records.append(
        {
            "model": BASELINE_NAME,
            "provider": BASELINE_PROVIDER,
            "id": None if sample_id is None else str(sample_id),
            "question": question,
            "spider_db_id": gold_db_id,
            "gold_db_id": gold_db_id,
            "predict_db_id": predicted_db_id,
            "predict_tables_text": json.dumps({"relevant_tables": predict_tables}, ensure_ascii=False),
            "predict_columns_text": json.dumps({"relevant_columns": predict_columns}, ensure_ascii=False),
            "predict_tables": predict_tables,
            "predict_columns": predict_columns,
            "retrieval": {
                "global_retrieved_columns": [
                    {
                        "db_id": column.db_id,
                        "table_name": column.table_name,
                        "column_name": column.column_name,
                        "score": column.score,
                        "rank": column.rank,
                    }
                    for column in global_columns
                ],
                "candidate_db_ranking": list(db_ranking),
                "local_retrieved_columns": [
                    {
                        "db_id": column.db_id,
                        "table_name": column.table_name,
                        "column_name": column.column_name,
                        "score": column.score,
                        "rank": column.rank,
                    }
                    for column in local_columns
                ],
            },
            "efficiency": {
                "sample_elapsed_seconds": round(elapsed_seconds, 6),
            },
        }
    )
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def run_baseline(
    *,
    dataset_df: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    documents_dir: Path,
    embedder: Any,
    qdrant_client: Any,
    collection_name: str,
    global_top_k: int,
    local_top_k: int,
    include_hint_in_query: bool,
    start_index: int,
    limit: int | None,
) -> int:
    from tqdm import tqdm

    result_records: list[dict[str, Any]] = []
    selected_df = dataset_df.iloc[max(0, start_index) :]
    if limit is not None:
        selected_df = selected_df.iloc[: max(0, limit)]

    for _, row in tqdm(selected_df.iterrows(), total=len(selected_df)):
        started_at = perf_counter()
        query_text, _ = resolve_query_text(
            row=row,
            dataset_name=dataset_name,
            documents_dir=documents_dir,
            include_hint=include_hint_in_query,
        )
        query_vector = embedder.encode(query_text, convert_to_list=True)
        global_columns = retrieve_columns(
            client=qdrant_client,
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=global_top_k,
        )
        db_ranking = rank_databases(global_columns)
        predicted_db_id = str(db_ranking[0]["db_id"]) if db_ranking else None
        local_columns = (
            retrieve_columns(
                client=qdrant_client,
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=local_top_k,
                candidate_db_ids=[predicted_db_id],
            )
            if predicted_db_id
            else []
        )
        elapsed_seconds = perf_counter() - started_at
        append_result(
            records=result_records,
            output_path=output_path,
            row=row,
            predicted_db_id=predicted_db_id,
            global_columns=global_columns,
            local_columns=local_columns,
            db_ranking=db_ranking,
            elapsed_seconds=elapsed_seconds,
        )

    return len(result_records)


def main() -> None:
    args = parse_args()

    dataset_name = args.dataset_name or DATASET_NAME
    dataset_root = PROJECT_ROOT / "Data" / dataset_name
    input_path = args.input_path or (dataset_root / "gold_sl.json")
    qdrant_path = args.qdrant_path or (dataset_root / "qdrant_column_index")
    documents_dir = dataset_root / "documents"
    output_path = resolve_output_path(args.output_path, dataset_name)
    collection_name = args.collection_name or load_qdrant_collection_name(
        qdrant_path=qdrant_path,
        fallback=dataset_name,
    )
    top_k = validate_top_k(args.top_k, "top-k")
    global_top_k = validate_top_k(
        args.global_top_k if args.global_top_k is not None else top_k,
        "global-top-k",
    )
    local_top_k = validate_top_k(
        args.local_top_k if args.local_top_k is not None else top_k,
        "local-top-k",
    )
    embedding_model_name = args.embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME
    cache_dir = args.cache_dir or (PROJECT_ROOT / "Llm" / "cache")

    dataset_df = load_dataset(input_path)
    logger, logger_path = setup_task_logger(BASELINE_NAME, output_path)
    log_run_configuration(
        logger,
        task_name="RAG Column Retrieval Baseline",
        dataset_name=dataset_name,
        data_count=len(dataset_df),
        model_name=BASELINE_NAME,
        provider=BASELINE_PROVIDER,
        result_path=output_path,
        extra_fields={
            "Input path": input_path,
            "Qdrant path": qdrant_path,
            "Qdrant collection": collection_name,
            "Embedding model": embedding_model_name,
            "Cache dir": cache_dir,
            "Device": args.device or "auto",
            "Global top k": global_top_k,
            "Local top k": local_top_k,
            "Include hint in query": args.include_hint_in_query,
            "Start index": args.start_index,
            "Limit": args.limit,
            "Logger path": logger_path,
        },
    )

    from Llm.embedding_model_loader import EmbeddingModelLoader

    embedder = EmbeddingModelLoader(
        model_name=embedding_model_name,
        cache_dir=str(cache_dir),
        device=args.device,
        trust_remote_code=True,
    )
    qdrant_client = get_qdrant_client(qdrant_path)
    processed_count = run_baseline(
        dataset_df=dataset_df,
        output_path=output_path,
        dataset_name=dataset_name,
        documents_dir=documents_dir,
        embedder=embedder,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        global_top_k=global_top_k,
        local_top_k=local_top_k,
        include_hint_in_query=args.include_hint_in_query,
        start_index=args.start_index,
        limit=args.limit,
    )
    logger.info("Completed %s records.", processed_count)


if __name__ == "__main__":
    main()
