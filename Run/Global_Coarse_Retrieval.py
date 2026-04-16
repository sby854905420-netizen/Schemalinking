import pandas as pd
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from typing import Any
from pathlib import Path
from tqdm import tqdm
import json
import torch
from datetime import datetime
import logging


from Llm.embedding_model_loader import EmbeddingModelLoader
from config import *
from Llm.llm_loader import LLM
from Run.logging_utils import log_run_configuration, setup_task_logger
from Utils.efficiency_utils import SampleEfficiencyTracker
from Utils.render_tools import SchemaTextRenderer
from Utils.schema_selection import (
    DbInfoSchemaStore,
    count_prompt_tokens,
    load_db_counts,
    resolve_schema_text_for_db,
)
from Utils.tools import (
    build_db_id_filter,
    get_qdrant_client,
    load_db_info_index,
    query_qdrant,
    render_prompt,
    resolve_hint,
    resolve_prompt_token_cap,
)

PROMPT_BUDGET_BUFFER = 512
PROMPT_BUDGET_RATIO = 0.9

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()

def resolve_top_kd(column_count: int) -> int:
    if column_count <= 20:
        return 4
    if column_count <= 50:
        return 5
    if column_count <= 100:
        return 6
    if column_count <= 200:
        return 6
    if column_count <= 500:
        return 7
    if column_count <= 1000:
        return 8
    if column_count <= 5000:
        return 10
    return 12


def resolve_top_k(
    db_counts: dict[str, int],
    candidate_db_ids: list[str] | None = None,
    global_cap: int = TOP_KD_CAP,
) -> int:
    if candidate_db_ids is None:
        target_db_ids = list(db_counts.keys())
    else:
        target_db_ids = [db_id for db_id in candidate_db_ids if db_id]

    if not target_db_ids:
        return 0

    resolved_top_k = sum(resolve_top_kd(max(0, db_counts.get(db_id, 0))) for db_id in target_db_ids)
    return max(1, min(resolved_top_k, global_cap))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative coarse database retrieval.")
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--answer-llm-name", dest="answer_llm_name", type=str, default=None)
    parser.add_argument("--provider", dest="provider", type=str, default=None)
    parser.add_argument("--input-path", dest="input_path", type=Path, default=None)
    parser.add_argument("--output-path", dest="output_path", type=Path, default=None)
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
    parser.add_argument("--candidate-db-top-k", dest="candidate_db_top_k", type=int, default=None)
    parser.add_argument(
        "--enable-progress-log",
        dest="enable_progress_log",
        action="store_true",
        help="Enable verbose per-sample progress logs for retrieval steps.",
    )
    return parser.parse_args()


def build_sample_tag(row: pd.Series, sample_index: int, total_samples: int) -> str:
    sample_id = clean_text(row.get("id")) or str(sample_index + 1)
    db_id = clean_text(row.get("db_id")) or "UNKNOWN"
    return f"sample {sample_index + 1}/{total_samples} id={sample_id} gold_db={db_id}"


def log_progress(
    logger: logging.Logger | None,
    enabled: bool,
    sample_tag: str,
    step: str,
    message: str,
    *args: Any,
) -> None:
    if not enabled or logger is None:
        return
    logger.info("[%s] [%s] " + message, sample_tag, step, *args)


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


def resolve_base_prompt_hint(
    ranking_llm: LLM,
    prompt_template: str,
    query: str,
    raw_hint_text: str,
    target_prompt_cap: int,
) -> str:
    prompt_tokens_with_hint = count_prompt_tokens(
        ranking_llm=ranking_llm,
        prompt_template=prompt_template,
        schema_text="",
        query=query,
        hint_text=raw_hint_text,
    )
    if prompt_tokens_with_hint <= target_prompt_cap:
        return raw_hint_text

    return "No hint"


def get_Highly_Relevant_Columns(query_vector:list[float], 
                                qdrant_client:QdrantClient, 
                                db_counts: dict[str, int],
                                collection_name:str,
                                candidate_db_ids:list[str]=None,  
                                ) -> list[ScoredPoint]:
    
    query_filter = build_db_id_filter(candidate_db_ids)

    if candidate_db_ids is not None and query_filter is None:
        return []
    
    resolved_top_k = resolve_top_k(
        db_counts=db_counts,
        candidate_db_ids=candidate_db_ids,
    )

    if resolved_top_k <= 0:
        return []

    Highly_Relevant_Columns = query_qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=resolved_top_k,
        query_filter=query_filter,
        with_vectors=False,
    )
    return Highly_Relevant_Columns


def database_pruning(HRC_points:list[ScoredPoint],
                      min_hit_count:int, 
                      min_sim_ratio: float
                    ) -> list[str]:
    if not HRC_points:
        return []

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
    return filtered_db_stats["db_id"].tolist()


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


def CFCD_rerank_select(query:str, query_vector: list[float], ranking_llm:LLM, 
                       CFCD_db_ids: list[str], prompt_template:str, top_k:float,
                       hint_text: str,
                       schema_store: DbInfoSchemaStore,
                       qdrant_client: QdrantClient,
                       collection_name: str,
                       db_counts: dict[str, int],
                       efficiency_tracker: SampleEfficiencyTracker | None = None,
                       logger: logging.Logger | None = None,
                       enable_progress_log: bool = False,
                       sample_tag: str = ""):
    if not CFCD_db_ids:
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "rerank",
            "Skipped reranking because there are no candidate databases.",
        )
        return []

    yes_scores = []
    target_prompt_cap = resolve_prompt_token_cap(ranking_llm.max_input_length)
    base_hint_text = resolve_base_prompt_hint(
        ranking_llm=ranking_llm,
        prompt_template=prompt_template,
        query=query,
        raw_hint_text=hint_text,
        target_prompt_cap=target_prompt_cap,
    )
    log_progress(
        logger,
        enable_progress_log,
        sample_tag,
        "rerank",
        "Starting rerank for %s candidate databases with top_k=%s and prompt cap=%s.",
        len(CFCD_db_ids),
        top_k,
        target_prompt_cap,
    )
    for db_index, db_id in enumerate(CFCD_db_ids, start=1):
        hint_text = base_hint_text
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "rerank",
            "Evaluating candidate db %s/%s: %s",
            db_index,
            len(CFCD_db_ids),
            db_id,
        )
        schema_text, selected_records = resolve_schema_text_for_db(
            query=query,
            query_vector=query_vector,
            db_id=db_id,
            ranking_llm=ranking_llm,
            prompt_template=prompt_template,
            hint_text=hint_text,
            schema_store=schema_store,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            db_counts=db_counts,
        )
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "rerank",
            "Rendered schema for db=%s with %s selected columns.",
            db_id,
            len(selected_records),
        )

        prompt = render_prompt(
            prompt_template,
            DATABASE_SCHEMAS=schema_text,
            QUESTION=query,
            HINT=hint_text,
        )
        prompt_token_count = ranking_llm.count_input_tokens(prompt)
        if prompt_token_count > target_prompt_cap:
            hint_text = "No hint"
            prompt = render_prompt(
                prompt_template,
                DATABASE_SCHEMAS=schema_text,
                QUESTION=query,
                HINT=hint_text,
            )
            prompt_token_count = ranking_llm.count_input_tokens(prompt)
        if prompt_token_count > target_prompt_cap:
            yes_scores.append(float("-inf"))
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "rerank",
                "Skipped db=%s because prompt tokens=%s still exceed cap=%s after fallback hint handling.",
                db_id,
                prompt_token_count,
                target_prompt_cap,
            )
            continue
        next_token_logits, total_tokens = ranking_llm.query_logits_with_usage(prompt)
        if efficiency_tracker is not None:
            efficiency_tracker.add_llm_total_tokens(total_tokens)
        yes_prob_binary = compute_yes_probability(next_token_logits,ranking_llm.tokenizer)
        yes_scores.append(float(yes_prob_binary.detach()))
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "rerank",
            "Finished db=%s with prompt tokens=%s and yes_score=%.6f.",
            db_id,
            prompt_token_count,
            yes_scores[-1],
        )

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
    log_progress(
        logger,
        enable_progress_log,
        sample_tag,
        "rerank",
        "Rerank completed. Selected %s databases: %s",
        len(ranked_db_ids),
        ",".join(ranked_db_ids) if ranked_db_ids else "NONE",
    )
    return ranked_db_ids


def main() -> None:
    args = parse_args()

    dataset_name = args.dataset_name or DATASET_NAME
    answer_llm_name = args.answer_llm_name or ANSWER_LLM_NAME
    provider = args.provider or PROVIDER
    max_input_length = args.max_input_length or MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or MAX_GENERATEION_NUM
    candidate_db_top_k = args.candidate_db_top_k if args.candidate_db_top_k is not None else CANDIDATE_DB_TOP_K
    enable_progress_log = args.enable_progress_log
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_root = PROJECT_ROOT / "Data" / dataset_name
    qdrant_path = dataset_root / "qdrant_column_index"
    db_info_path = dataset_root / "db_info.json"
    documents_dir = dataset_root / "documents"
    dataset_path = args.input_path or (dataset_root / "gold_sl.json")

    dataset_df = pd.read_json(dataset_path)
    db_info_index = load_db_info_index(db_info_path)
    db_counts = load_db_counts(db_info_index)
    prompt_path = PROJECT_ROOT / "Templates" / "zero_shot" / "binary_classification_database.txt"
    prompt_template = prompt_path.read_text(encoding='utf-8').strip()
    
    if args.output_path is None:
        logs_dir = PROJECT_ROOT / 'Logs' / answer_llm_name / "Database_Retrival"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f'iterative_database_retrival_{dataset_name}_{run_id}.json'
    else:
        log_path = args.output_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
    logger, logger_path = setup_task_logger("iterative_database_retrival", log_path)

    client = get_qdrant_client(qdrant_path)
    collection_name = dataset_name

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
            "Input path": dataset_path,
            "Prompt template": prompt_path,
            "Qdrant path": qdrant_path,
            "Qdrant collection": collection_name,
            "DB info path": db_info_path,
            "Documents dir": documents_dir,
            "Top-k cap": TOP_KD_CAP,
            "Top-k schedule": "<=20:4, <=50:5, <=100:6, <=200:6, <=500:7, <=1000:8, <=5000:10, >5000:12",
            "Prompt budget ratio": PROMPT_BUDGET_RATIO,
            "Prompt budget buffer": PROMPT_BUDGET_BUFFER,
            "Candidate db top k": candidate_db_top_k,
            "Max input length": max_input_length,
            "Max generation num": max_generation_num,
            "Enable progress log": enable_progress_log,
            "Logger path": logger_path,
        },
    )

    embedder = EmbeddingModelLoader(
        model_name=EMBEDDING_MODEL_NAME,
    )
    
    ranking_llm = LLM(
        model_name=answer_llm_name,
        provider=provider,
        max_input_length=max_input_length,
        max_generation_num=max_generation_num,
    )
    renderer = SchemaTextRenderer(tokenizer=ranking_llm.tokenizer)
    schema_store = DbInfoSchemaStore(
        db_info_index=db_info_index,
        renderer=renderer,
    )

    log_records = []


    total_samples = len(dataset_df)
    for sample_index, (_, row) in enumerate(tqdm(dataset_df.iterrows(), total=total_samples)):
        efficiency_tracker = SampleEfficiencyTracker()
        sample_tag = build_sample_tag(row, sample_index, total_samples)
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "start",
            "Starting sample processing.",
        )
        hint_text = resolve_hint(
            row,
            dataset_name=dataset_name,
            documents_dir=documents_dir,
        )
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "prepare",
            "Resolved hint text. hint_available=%s",
            hint_text != "No hint",
        )
        # First Round for all databases
        # step 1: Global Highly Relevant columns (HRC) Retrival
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "round1.step1",
            "Encoding question and retrieving global highly relevant columns.",
        )
        query_embedding = embedder.encode(row['question'], convert_to_list=True)
        HRC_points = get_Highly_Relevant_Columns(
            query_embedding,
            client,
            db_counts=db_counts,
            collection_name=collection_name,
        )
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "round1.step1",
            "Retrieved %s highly relevant column points.",
            len(HRC_points),
        )

        # step 2: Support-Based Database Pruning, resulting Coarsely_Filtered Candidate Databases (CFCD)
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "round1.step2",
            "Pruning databases from first-round HRC results.",
        )
        first_round_cfcd_db_ids = database_pruning(HRC_points, min_hit_count=2, min_sim_ratio=0.8)
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "round1.step2",
            "First-round pruning kept %s candidate databases: %s",
            len(first_round_cfcd_db_ids),
            ",".join(first_round_cfcd_db_ids) if first_round_cfcd_db_ids else "NONE",
        )

        final_cfcd_db_ids = list(first_round_cfcd_db_ids)
        if len(first_round_cfcd_db_ids) <= candidate_db_top_k:
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "round1.step3",
                "Skipping second round because first-round pruning count=%s does not exceed top_k=%s; selecting target database directly.",
                len(first_round_cfcd_db_ids),
                candidate_db_top_k,
            )
            FCD_db_ids = CFCD_rerank_select(
                row['question'],
                query_embedding,
                ranking_llm,
                first_round_cfcd_db_ids,
                prompt_template,
                top_k=1,
                hint_text=hint_text,
                schema_store=schema_store,
                qdrant_client=client,
                collection_name=collection_name,
                db_counts=db_counts,
                efficiency_tracker=efficiency_tracker,
                logger=logger,
                enable_progress_log=enable_progress_log,
                sample_tag=sample_tag,
            )
            target_db_id = FCD_db_ids
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "round1.step3",
                "First-round rerank directly selected target database: %s",
                target_db_id[0] if target_db_id else "NONE",
            )
        else:
            # step 3：CFCD Reranking & Selection Top-K, resulting in Final Candidate Databases (FCD)
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "round1.step3",
                "Starting first-round rerank to select top %s databases.",
                candidate_db_top_k,
            )
            FCD_db_ids = CFCD_rerank_select(
                row['question'],
                query_embedding,
                ranking_llm,
                first_round_cfcd_db_ids,
                prompt_template,
                top_k=candidate_db_top_k,
                hint_text=hint_text,
                schema_store=schema_store,
                qdrant_client=client,
                collection_name=collection_name,
                db_counts=db_counts,
                efficiency_tracker=efficiency_tracker,
                logger=logger,
                enable_progress_log=enable_progress_log,
                sample_tag=sample_tag,
            )
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "round1.step3",
                "First-round rerank selected %s final candidate databases: %s",
                len(FCD_db_ids),
                ",".join(FCD_db_ids) if FCD_db_ids else "NONE",
            )

            # Second Round for Final Candidate Databases
            # step 1: Global Highly Relevant columns (HRC) Retrival for Final Candidate Databases
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "round2.step1",
                "Retrieving highly relevant columns within first-round selected databases.",
            )
            last_hrc_points = get_Highly_Relevant_Columns(
                query_embedding,
                client,
                db_counts=db_counts,
                collection_name=collection_name,
                candidate_db_ids=FCD_db_ids,
            )
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "round2.step1",
                "Retrieved %s second-round highly relevant column points.",
                len(last_hrc_points),
            )
            # step 2: Support-Based Database Pruning, resulting Coarsely_Filtered Candidate Databases (CFCD)
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "round2.step2",
                "Pruning databases from second-round HRC results.",
            )
            second_round_cfcd_db_ids = database_pruning(last_hrc_points, min_hit_count=2, min_sim_ratio=0.8)
            final_cfcd_db_ids = list(second_round_cfcd_db_ids)
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "round2.step2",
                "Second-round pruning kept %s candidate databases: %s",
                len(second_round_cfcd_db_ids),
                ",".join(second_round_cfcd_db_ids) if second_round_cfcd_db_ids else "NONE",
            )
            # step 3：CFCD Reranking & Selection Top-1, resulting in Target Database
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "round2.step3",
                "Starting second-round rerank to select the target database.",
            )
            target_db_id = CFCD_rerank_select(
                row['question'],
                query_embedding,
                ranking_llm,
                second_round_cfcd_db_ids,
                prompt_template,
                top_k=1,
                hint_text=hint_text,
                schema_store=schema_store,
                qdrant_client=client,
                collection_name=collection_name,
                db_counts=db_counts,
                efficiency_tracker=efficiency_tracker,
                logger=logger,
                enable_progress_log=enable_progress_log,
                sample_tag=sample_tag,
            )
            log_progress(
                logger,
                enable_progress_log,
                sample_tag,
                "round2.step3",
                "Second-round rerank selected target database: %s",
                target_db_id[0] if target_db_id else "NONE",
            )
        # save the predict results
        log_records.append(
            {
                'model': answer_llm_name,
                'provider': provider,
                'id': f"{row['id']}",
                'gold_db_id': row['db_id'],
                'question': row['question'],
                'FCD_ids': ",".join(FCD_db_ids),
                'CFCD_db_ids': ",".join(final_cfcd_db_ids),
                'predict_db_id': target_db_id[0] if target_db_id else None,
                'efficiency': efficiency_tracker.finalize(),
            }
        )
        log_path.write_text(json.dumps(log_records, ensure_ascii=False, indent=2), encoding='utf-8')
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "finish",
            "Saved sample result to %s. processed_records=%s",
            log_path,
            len(log_records),
        )

    logger.info("Completed %s records.", len(log_records))
    

if __name__ == "__main__":
    main()
