import pandas as pd
import argparse
import math
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from typing import Any,Iterable
from pathlib import Path
from tqdm import tqdm
import json
import torch
from datetime import datetime


from Llm.embedding_model_loader import EmbeddingModelLoader
from config import *
from Llm.llm_loader import LLM
from Run.logging_utils import log_run_configuration, setup_task_logger


def get_qdrant_client(qdrant_path: Path):
    return QdrantClient(path=str(qdrant_path))

def build_db_id_filter(candidate_db_ids: list[str] | None):
    if candidate_db_ids is None:
        return None

    normalized_db_ids = [db_id for db_id in candidate_db_ids if db_id]
    if not normalized_db_ids:
        return None

    try:
        from qdrant_client.http import models
    except ImportError as exc:
        raise ImportError("Please install `qdrant-client` before building Qdrant filters.") from exc

    return models.Filter(
        should=[
            models.FieldCondition(
                key="db_id",
                match=models.MatchValue(value=db_id),
            )
            for db_id in normalized_db_ids
        ]
    )


def count_points(client:QdrantClient, collection_name: str, query_filter=None) -> int:
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


def resolve_top_k(client:QdrantClient, collection_name: str, top_ratio: float, query_filter=None) -> int:
    if not 0 < top_ratio <= 1:
        raise ValueError("--top-ratio must be within (0, 1]")

    total_points = count_points(client, collection_name, query_filter=query_filter)
    if total_points <= 0:
        return 0

    return max(1, math.ceil(total_points * top_ratio))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative coarse database retrieval.")
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--answer-llm-name", dest="answer_llm_name", type=str, default=None)
    parser.add_argument("--provider", dest="provider", type=str, default=None)
    parser.add_argument(
        "--max-input-length",
        dest="max_input_length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-generation-num",
        dest="max_generation_num",
        type=int,
        default=None,
    )
    parser.add_argument("--hrc-top-p", dest="hrc_top_p", type=float, default=None)
    parser.add_argument("--candidate-db-top-k", dest="candidate_db_top_k", type=int, default=None)
    return parser.parse_args()

def query_qdrant(client:QdrantClient, collection_name: str, query_vector: list[float], top_k: int, with_vector: bool, query_filter=None):
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=with_vector,
    )

    if hasattr(response, "points"):
        return list(response.points)
    if isinstance(response, list):
        return response

    raise TypeError(f"Unsupported Qdrant query result type: {type(response)!r}")

def point_to_dict(point: Any) -> dict[str, Any]:
    if isinstance(point, tuple) and len(point) == 2:
        point = point[1]

    payload = getattr(point, "payload", None)
    if payload is None and isinstance(point, dict):
        payload = point.get("payload")

    point_id = getattr(point, "id", None)
    if point_id is None and isinstance(point, dict):
        point_id = point.get("id")

    score = getattr(point, "score", None)
    if score is None and isinstance(point, dict):
        score = point.get("score")

    vector = getattr(point, "vector", None)
    if vector is None and isinstance(point, dict):
        vector = point.get("vector")

    return {
        "id": point_id,
        "score": score,
        "payload": dict(payload or {}),
        "vector": vector,
    }

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

def iter_column_files(schema_dir: Path) -> Iterable[Path]:
    yield from sorted(schema_dir.glob("*.json"))


def build_document(record: dict[str, Any]) -> str:
    meta = record["meta_data"]
    lines = [
        f"Database Name: {meta['db_id']}",
        f"Table Name: {meta['table_name']}",
        f"Column Name: {record['column_name']}",
        f"Data Type: {record['normalized_type']}",
        f"Column Description: {record['column_description']}",
    ]
    sample_values_text = record.get("sample_values_text", "").strip()
    if sample_values_text:
        lines.append(f"Sample values: {sample_values_text}")
    return "\n".join(lines)

def load_column_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
    

def get_Highly_Relevant_Columns(query_vector:list[float], 
                                qdrant_client:QdrantClient, 
                                top_p:float,
                                candidate_db_ids:list[str]=None
                                ) -> list[ScoredPoint]:
    
    query_filter = build_db_id_filter(candidate_db_ids)

    if candidate_db_ids is not None and query_filter is None:
        return []
    
    resolved_top_k = resolve_top_k(
        client=qdrant_client,
        collection_name="mmqa_column_level",
        top_ratio=top_p,
        query_filter=query_filter,
    )

    if resolved_top_k <= 0:
        return []

    Highly_Relevant_Columns = query_qdrant(
        client=qdrant_client,
        collection_name="mmqa_column_level",
        query_vector=query_vector,
        top_k=resolved_top_k,
        with_vector=False,
        query_filter=query_filter,
    )
    return Highly_Relevant_Columns


def database_pruning(HRC_points:list[ScoredPoint],
                      min_hit_count:int, 
                      min_sim_ratio: float
                    ) -> list[str]:
    results = [point_to_dict(point) for point in HRC_points]
    results_df = pd.DataFrame(
        {
            "db_id": r["payload"].get("db_id", "UNKNOWN"),
            "score": r["score"],
        }
        for r in results
    )
    candidate_db_stats = (
        results_df.groupby("db_id", as_index=False)
        .agg(
            hit_count=("db_id", "size"),
            max_score=("score", "max"),
            mean_score=("score", "mean"),
            score_sum=("score", "sum"),
        )
        .sort_values(["max_score", "score_sum", "hit_count"], ascending=False)
        .reset_index(drop=True)
    )
    filtered_db_stats = candidate_db_stats[
        (candidate_db_stats["hit_count"] >= min_hit_count) | (candidate_db_stats["max_score"] >= candidate_db_stats["max_score"].quantile(min_sim_ratio))
    ]
    return list(set(filtered_db_stats["db_id"]))


def collect_single_token_candidates(tokenizer, target_text: str):
    target = target_text.lower()
    matched_ids = []

    for token_id in range(tokenizer.vocab_size):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        normalized = decoded.strip().lower()
        if normalized == target:
            matched_ids.append(token_id)

    if not matched_ids:
        raise ValueError(f"No single-token candidate found for {target_text!r}.")

    return sorted(set(matched_ids))


def compute_yes_probability(next_token_logits:torch.tensor, tokenizer):
    YES_TOKEN_IDS = collect_single_token_candidates(tokenizer, "yes")
    NO_TOKEN_IDS = collect_single_token_candidates(tokenizer, "no")

    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    yes_token_ids = torch.tensor(YES_TOKEN_IDS, device=next_token_probs.device)
    no_token_ids = torch.tensor(NO_TOKEN_IDS, device=next_token_probs.device)

    yes_prob_raw = next_token_probs[0].index_select(0, yes_token_ids).sum()
    no_prob_raw = next_token_probs[0].index_select(0, no_token_ids).sum()

    yes_no_total = yes_prob_raw + no_prob_raw
    yes_prob_binary = yes_prob_raw / yes_no_total

    return yes_prob_binary


def CFCD_rerank_select(query:str, ranking_llm:LLM, 
                       CFCD_db_ids: list[str], prompt_template:str, top_k:float,
                       schema_dir: Path):
    split_line = '\n'+ '-'*80 + '\n'
    yes_scores = []
    for db_id in CFCD_db_ids:
        documents: list[str] = []
        column_files = list(iter_column_files(schema_dir / db_id))
        for column_file in column_files:
            record = load_column_record(column_file)
            document = build_document(record)
            documents.append(document)

        schema_text = split_line.join(documents)
        prompt = (
                prompt_template
                .replace('{DATABASE_SCHEMAS}', schema_text)
                .replace('{QUESTION}', query)
                .replace('{HINT}', 'No hint')
                )
        # prompt_token_count = ranking_llm.count_input_tokens(prompt)
        # print(f"[GlobalCoarse] db_id={db_id} prompt_tokens={prompt_token_count}")
        next_token_logits = ranking_llm._query_transformers(prompt, output_hidden_states=True)
        yes_prob_binary = compute_yes_probability(next_token_logits,ranking_llm.tokenizer)
        yes_scores.append(float(yes_prob_binary.detach()))

        del next_token_logits
        del yes_prob_binary
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    db_score_df = pd.DataFrame({
        "db_id": CFCD_db_ids,
        "yes_score": yes_scores,
    })
    ranked_db_ids = (
        db_score_df.sort_values("yes_score", ascending=False)["db_id"]
        .head(top_k)
        .tolist()
    )
    return ranked_db_ids
    

    

def main() -> None:
    args = parse_args()

    dataset_name = args.dataset_name or DATASET_NAME
    answer_llm_name = args.answer_llm_name or ANSWER_LLM_NAME
    provider = args.provider or PROVIDER
    max_input_length = args.max_input_length or MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or MAX_GENERATEION_NUM
    hrc_top_p = args.hrc_top_p if args.hrc_top_p is not None else HRC_TOP_P
    candidate_db_top_k = args.candidate_db_top_k if args.candidate_db_top_k is not None else CANDIDATE_DB_TOP_K
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_root = PROJECT_ROOT / "Data" / dataset_name
    qdrant_path = dataset_root / "qdrant_column_index"
    schema_dir = dataset_root / "Column_level_schema"

    dataset_df = pd.read_json(dataset_root / "preprocessed_data.json")
    prompt_path = PROJECT_ROOT / "Templates" / "zero_shot" / "binary_classification_database.txt"
    prompt_template = prompt_path.read_text(encoding='utf-8').strip()
    
    logs_dir = PROJECT_ROOT / 'Logs' / answer_llm_name / "Database_Retrival"
    logs_dir.mkdir(parents=True,exist_ok=True)
    log_path = logs_dir / f'iterative_database_retrival_{dataset_name}_{run_id}.json'
    logger, logger_path = setup_task_logger("iterative_database_retrival", log_path)

    log_run_configuration(
        logger,
        task_name="Global Coarse Retrieval",
        dataset_name=dataset_name,
        data_count=len(dataset_df),
        model_name=answer_llm_name,
        provider=provider,
        result_path=log_path,
        extra_fields={
            "Embedding model": EMBEDDING_MODEL_NAME,
            "Prompt template": prompt_path,
            "Qdrant path": qdrant_path,
            "Schema dir": schema_dir,
            "HRC top p": hrc_top_p,
            "Candidate db top k": candidate_db_top_k,
            "Max input length": max_input_length,
            "Max generation num": max_generation_num,
            "Logger path": logger_path,
        },
    )

    embedder = EmbeddingModelLoader(
        model_name=EMBEDDING_MODEL_NAME,
    )
    client = get_qdrant_client(qdrant_path)
    
    ranking_llm = LLM(
        model_name=answer_llm_name,
        provider=provider,
        max_input_length=max_input_length,
        max_generation_num=max_generation_num,
    )

    log_records = []


    for _,row in tqdm(dataset_df.iterrows(),total=len(dataset_df)):
        # First Round for all databases
        # step 1: Global Highly Relevant columns (HRC) Retrival
        query_embedding = embedder.encode(row['question'], convert_to_list=True)
        HRC_points = get_Highly_Relevant_Columns(query_embedding, client, top_p=hrc_top_p)

        # step 2: Support-Based Database Pruning, resulting Coarsely_Filtered Candidate Databases (CFCD)
        CFCD_db_ids  = database_pruning(HRC_points, min_hit_count=2, min_sim_ratio=0.9)

        # step 3：CFCD Reranking & Selection Top-K, resulting in Final Candidate Databases (FCD)
        FCD_db_ids = CFCD_rerank_select(
            row['question'],
            ranking_llm,
            CFCD_db_ids,
            prompt_template,
            top_k=candidate_db_top_k,
            schema_dir=schema_dir,
        )

        # Second Round for Final Candidate Databases
        # step 1: Global Highly Relevant columns (HRC) Retrival for Final Candidate Databases
        Last_HRC_points = get_Highly_Relevant_Columns(
            query_embedding,
            client,
            top_p=hrc_top_p,
            candidate_db_ids=FCD_db_ids,
        )
        # step 2: Support-Based Database Pruning, resulting Coarsely_Filtered Candidate Databases (CFCD)
        CFCD_db_ids  = database_pruning(Last_HRC_points, min_hit_count=2, min_sim_ratio=0.9)
        # step 3：CFCD Reranking & Selection Top-1, resulting in Target Database
        target_db_id = CFCD_rerank_select(
            row['question'],
            ranking_llm,
            CFCD_db_ids,
            prompt_template,
            top_k=1,
            schema_dir=schema_dir,
        )
        # save the predict results
        log_records.append(
            {
                'model': answer_llm_name,
                'provider': provider,
                'id': f"{row['instance_id']}",
                'spider_db_id': row['db_id'],
                'question': row['question'],
                'FCD_ids': ",".join(FCD_db_ids),
                'CFCD_db_ids': ",".join(CFCD_db_ids),
                'predict_db_id': target_db_id[0]
            }
        )
        log_path.write_text(json.dumps(log_records, ensure_ascii=False, indent=2), encoding='utf-8')

    logger.info("Completed %s records.", len(log_records))
    

if __name__ == "__main__":
    main()
