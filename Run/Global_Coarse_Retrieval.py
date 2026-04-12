import pandas as pd
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from typing import Any, Iterable
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
from Utils.tools import (
    build_db_id_filter,
    get_qdrant_client,
    query_qdrant,
    render_prompt,
    resolve_hint,
    resolve_prompt_token_cap,
)

MAX_COLUMN_DOCUMENT_TOKENS = 2048
# Step 3 in CFCD reranking: cap how many top-ranked columns we retrieve per DB
# before building schema documents and assembling the rerank prompt.
MAX_RERANK_COLUMNS_PER_DB = 200
PROMPT_BUDGET_BUFFER = 512
PROMPT_BUDGET_RATIO = 0.85
SCHEMA_SPLIT_LINE = "\n" + "-" * 80 + "\n"

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


def normalize_record(record: dict[str, Any], source_path: Path | None = None) -> dict[str, Any]:
    meta = record.get("meta_data") or {}
    sample_values, sample_values_text = normalize_sample_values(record)

    return {
        "column_id": clean_text(record.get("column_id")),
        "column_name": clean_text(record.get("column_name")),
        "column_description": clean_text(record.get("column_description")),
        "column_data_type": clean_text(record.get("column_data_type") or record.get("normalized_type")),
        "sample_values": sample_values,
        "sample_values_text": sample_values_text,
        "is_primary_key": record.get("is_primary_key"),
        "is_foreign_key": record.get("is_foreign_key"),
        "value_descriptions": clean_text(record.get("value_descriptions")),
        "references": record.get("references"),
        "db_id": clean_text(record.get("db_id") or meta.get("db_id")),
        "table_name": clean_text(record.get("table_name") or meta.get("table_name")),
        "source_path": str(source_path or record.get("source_path") or ""),
    }
def load_db_counts(db_info_path: Path) -> dict[str, int]:
    with db_info_path.open("r", encoding="utf-8") as file:
        db_infos = json.load(file)

    if not isinstance(db_infos, list):
        raise TypeError(f"Expected db_info.json to contain a list, got {type(db_infos)!r}")

    db_counts: dict[str, int] = {}
    for entry in db_infos:
        if not isinstance(entry, dict):
            continue
        db_id = clean_text(entry.get("db_id"))
        if not db_id:
            continue
        raw_count = entry.get("db_counts", 0)
        try:
            db_counts[db_id] = int(raw_count)
        except (TypeError, ValueError):
            db_counts[db_id] = 0

    return db_counts


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


def resolve_external_knowledge_for_prompt(
    dataset_name: str,
    external_knowledge: Any,
    documents_dir: Path | None = None,
) -> Any:
    if dataset_name.lower() != "spider2":
        return external_knowledge

    if not isinstance(external_knowledge, str):
        return external_knowledge

    document_name = external_knowledge.strip()
    if not document_name or documents_dir is None:
        return external_knowledge

    document_path = documents_dir / document_name
    if not document_path.is_file():
        return external_knowledge

    return document_path.read_text(encoding="utf-8").strip()


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


def iter_column_files(schema_dir: Path) -> Iterable[Path]:
    yield from sorted(schema_dir.rglob("*.json"))


def build_document(
    record: dict[str, Any],
    ranking_llm: LLM | None = None,
    max_document_tokens: int = MAX_COLUMN_DOCUMENT_TOKENS,
) -> str:
    normalized_record = normalize_record(record)
    base_lines = [
        f"Database: {normalized_record['db_id']}",
        f"Table: {normalized_record['table_name']}",
        f"Column: {normalized_record['column_name']}",
        f"Data type: {normalized_record['column_data_type']}",
    ]

    optional_lines: list[tuple[str, str]] = []

    column_description = normalized_record["column_description"]
    if column_description:
        optional_lines.append(("description", f"Description: {column_description}"))

    value_descriptions = normalized_record["value_descriptions"]
    if value_descriptions:
        optional_lines.append(("value_meanings", f"Value meanings: {value_descriptions}"))

    sample_values_text = normalized_record["sample_values_text"]
    if sample_values_text:
        optional_lines.append(("sample_values", f"Sample values: {sample_values_text}"))

    def render_document(lines: list[str]) -> str:
        return "\n".join(lines)

    document_lines = base_lines + [line for _, line in optional_lines]
    document = render_document(document_lines)
    if ranking_llm is None:
        return document

    if ranking_llm.count_input_tokens(document) <= max_document_tokens:
        return document

    sections_to_drop = ["sample_values", "value_meanings", "description"]
    remaining_optional_lines = list(optional_lines)
    for section_name in sections_to_drop:
        remaining_optional_lines = [
            (name, line)
            for name, line in remaining_optional_lines
            if name != section_name
        ]
        document = render_document(base_lines + [line for _, line in remaining_optional_lines])
        if ranking_llm.count_input_tokens(document) <= max_document_tokens:
            return document

    return document


def build_schema_text(documents: list[str]) -> str:
    return SCHEMA_SPLIT_LINE.join(documents)


def count_prompt_tokens(
    ranking_llm: LLM,
    prompt_template: str,
    documents: list[str],
    query: str,
    hint_text: str,
) -> int:
    prompt = render_prompt(
        prompt_template,
        DATABASE_SCHEMAS=build_schema_text(documents),
        QUESTION=query,
        HINT=hint_text,
    )
    return ranking_llm.count_input_tokens(prompt)


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
        documents=[],
        query=query,
        hint_text=raw_hint_text,
    )
    if prompt_tokens_with_hint <= target_prompt_cap:
        return raw_hint_text

    return "No hint"


def load_column_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_db_documents(schema_dir: Path, db_id: str, ranking_llm: LLM | None = None) -> list[str]:
    documents: list[str] = []
    for column_file in iter_column_files(schema_dir / db_id):
        record = load_column_record(column_file)
        documents.append(build_document(record, ranking_llm=ranking_llm))
    return documents


def get_ranked_db_documents(
    query_vector: list[float],
    qdrant_client: QdrantClient,
    collection_name: str,
    db_id: str,
    db_counts: dict[str, int],
    ranking_llm: LLM,
) -> list[dict[str, Any]]:
    query_filter = build_db_id_filter([db_id])
    if query_filter is None:
        return []

    resolved_top_k = min(
        MAX_RERANK_COLUMNS_PER_DB,
        max(1, db_counts.get(db_id, 0)),
    )
    ranked_points = query_qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=resolved_top_k,
        query_filter=query_filter,
        with_vectors=False,
    )

    ranked_documents: list[dict[str, Any]] = []
    seen_column_ids: set[str] = set()
    for point in ranked_points:
        point_dict = point_to_dict(point)
        payload = point_dict["payload"]
        column_id = clean_text(payload.get("column_id"))
        if not column_id:
            column_id = f"{clean_text(payload.get('table_name'))}::{clean_text(payload.get('column_name'))}"
        if column_id in seen_column_ids:
            continue

        seen_column_ids.add(column_id)
        ranked_documents.append(
            {
                "column_id": column_id,
                "table_name": clean_text(payload.get("table_name")),
                "score": float(point_dict["score"]) if point_dict["score"] is not None else float("-inf"),
                "document": build_document(payload, ranking_llm=ranking_llm),
            }
        )

    return ranked_documents


def append_documents_until_budget(
    selected_documents: list[str],
    selected_column_ids: set[str],
    candidate_documents: list[dict[str, Any]],
    ranking_llm: LLM,
    prompt_template: str,
    query: str,
    hint_text: str,
    target_prompt_cap: int,
) -> tuple[list[str], set[str]]:
    documents = list(selected_documents)
    column_ids = set(selected_column_ids)

    remaining_candidates = [
        candidate
        for candidate in candidate_documents
        if candidate["column_id"] not in column_ids
    ]
    if remaining_candidates:
        proposed_documents = documents + [candidate["document"] for candidate in remaining_candidates]
        proposed_tokens = count_prompt_tokens(
            ranking_llm=ranking_llm,
            prompt_template=prompt_template,
            documents=proposed_documents,
            query=query,
            hint_text=hint_text,
        )
        if proposed_tokens <= target_prompt_cap:
            documents.extend(candidate["document"] for candidate in remaining_candidates)
            column_ids.update(candidate["column_id"] for candidate in remaining_candidates)
            return documents, column_ids

    for candidate in candidate_documents:
        column_id = candidate["column_id"]
        if column_id in column_ids:
            continue

        proposed_documents = documents + [candidate["document"]]
        proposed_tokens = count_prompt_tokens(
            ranking_llm=ranking_llm,
            prompt_template=prompt_template,
            documents=proposed_documents,
            query=query,
            hint_text=hint_text,
        )
        if proposed_tokens <= target_prompt_cap:
            documents.append(candidate["document"])
            column_ids.add(column_id)

    return documents, column_ids


def select_relevant_documents_for_db(
    query: str,
    query_vector: list[float],
    db_id: str,
    ranking_llm: LLM,
    prompt_template: str,
    hint_text: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    db_counts: dict[str, int],
) -> list[str]:
    ranked_documents = get_ranked_db_documents(
        query_vector=query_vector,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        db_id=db_id,
        db_counts=db_counts,
        ranking_llm=ranking_llm,
    )
    if not ranked_documents:
        return []

    target_prompt_cap = resolve_prompt_token_cap(ranking_llm.max_input_length)

    table_representatives: list[dict[str, Any]] = []
    remaining_documents: list[dict[str, Any]] = []
    seen_tables: set[str] = set()
    for candidate in ranked_documents:
        table_name = candidate["table_name"]
        if table_name not in seen_tables:
            table_representatives.append(candidate)
            seen_tables.add(table_name)
        else:
            remaining_documents.append(candidate)

    selected_documents, selected_column_ids = append_documents_until_budget(
        selected_documents=[],
        selected_column_ids=set(),
        candidate_documents=table_representatives,
        ranking_llm=ranking_llm,
        prompt_template=prompt_template,
        query=query,
        hint_text=hint_text,
        target_prompt_cap=target_prompt_cap,
    )
    selected_documents, _ = append_documents_until_budget(
        selected_documents=selected_documents,
        selected_column_ids=selected_column_ids,
        candidate_documents=remaining_documents,
        ranking_llm=ranking_llm,
        prompt_template=prompt_template,
        query=query,
        hint_text=hint_text,
        target_prompt_cap=target_prompt_cap,
    )
    return selected_documents


def resolve_schema_documents_for_db(
    query: str,
    query_vector: list[float],
    db_id: str,
    ranking_llm: LLM,
    prompt_template: str,
    hint_text: str,
    schema_dir: Path,
    qdrant_client: QdrantClient,
    collection_name: str,
    db_counts: dict[str, int],
) -> list[str]:
    return select_relevant_documents_for_db(
        query=query,
        query_vector=query_vector,
        db_id=db_id,
        ranking_llm=ranking_llm,
        prompt_template=prompt_template,
        hint_text=hint_text,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        db_counts=db_counts,
    )


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
                       external_knowledge: Any,
                       schema_dir: Path,
                       qdrant_client: QdrantClient,
                       collection_name: str,
                       db_counts: dict[str, int],
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
        raw_hint_text=resolve_hint(external_knowledge),
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
        documents = resolve_schema_documents_for_db(
            query=query,
            query_vector=query_vector,
            db_id=db_id,
            ranking_llm=ranking_llm,
            prompt_template=prompt_template,
            hint_text=hint_text,
            schema_dir=schema_dir,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            db_counts=db_counts,
        )
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "rerank",
            "Collected %s schema documents for db=%s.",
            len(documents),
            db_id,
        )

        prompt = render_prompt(
            prompt_template,
            DATABASE_SCHEMAS=build_schema_text(documents),
            QUESTION=query,
            HINT=hint_text,
        )
        prompt_token_count = ranking_llm.count_input_tokens(prompt)
        if prompt_token_count > target_prompt_cap:
            hint_text = "No hint"
            prompt = render_prompt(
                prompt_template,
                DATABASE_SCHEMAS=build_schema_text(documents),
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
        next_token_logits = ranking_llm._query_transformers(prompt, output_hidden_states=True)
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
    schema_dir = dataset_root / "Column_level_schema"
    documents_dir = dataset_root / "documents"
    dataset_path = args.input_path or (dataset_root / "gold_sl.json")

    dataset_df = pd.read_json(dataset_path)
    db_counts = load_db_counts(dataset_root / "db_info.json")
    prompt_path = PROJECT_ROOT / "Templates" / "zero_shot" / "binary_classification_database.txt"
    prompt_template = prompt_path.read_text(encoding='utf-8').strip()
    
    if args.output_path is None:
        logs_dir = PROJECT_ROOT / 'Logs' / dataset_name / answer_llm_name / "Database_Retrival"
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
            "Schema dir": schema_dir,
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

    log_records = []


    total_samples = len(dataset_df)
    for sample_index, (_, row) in enumerate(tqdm(dataset_df.iterrows(), total=total_samples)):
        sample_tag = build_sample_tag(row, sample_index, total_samples)
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "start",
            "Starting sample processing.",
        )
        external_knowledge = resolve_external_knowledge_for_prompt(
            dataset_name=dataset_name,
            external_knowledge=row.get('external_knowledge'),
            documents_dir=documents_dir,
        )
        external_knowledge_text = resolve_hint(external_knowledge)
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "prepare",
            "Resolved external knowledge. hint_available=%s",
            external_knowledge_text != "No hint",
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
        CFCD_db_ids  = database_pruning(HRC_points, min_hit_count=2, min_sim_ratio=0.8)
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "round1.step2",
            "First-round pruning kept %s candidate databases: %s",
            len(CFCD_db_ids),
            ",".join(CFCD_db_ids) if CFCD_db_ids else "NONE",
        )

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
            CFCD_db_ids,
            prompt_template,
            top_k=candidate_db_top_k,
            external_knowledge=external_knowledge,
            schema_dir=schema_dir,
            qdrant_client=client,
            collection_name=collection_name,
            db_counts=db_counts,
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
        Last_HRC_points = get_Highly_Relevant_Columns(
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
            len(Last_HRC_points),
        )
        # step 2: Support-Based Database Pruning, resulting Coarsely_Filtered Candidate Databases (CFCD)
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "round2.step2",
            "Pruning databases from second-round HRC results.",
        )
        CFCD_db_ids  = database_pruning(Last_HRC_points, min_hit_count=2, min_sim_ratio=0.8)
        log_progress(
            logger,
            enable_progress_log,
            sample_tag,
            "round2.step2",
            "Second-round pruning kept %s candidate databases: %s",
            len(CFCD_db_ids),
            ",".join(CFCD_db_ids) if CFCD_db_ids else "NONE",
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
            CFCD_db_ids,
            prompt_template,
            top_k=1,
            external_knowledge=external_knowledge,
            schema_dir=schema_dir,
            qdrant_client=client,
            collection_name=collection_name,
            db_counts=db_counts,
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
                'spider_db_id': row['db_id'],
                'question': row['question'],
                'FCD_ids': ",".join(FCD_db_ids),
                'CFCD_db_ids': ",".join(CFCD_db_ids),
                'predict_db_id': target_db_id[0] if target_db_id else None
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
