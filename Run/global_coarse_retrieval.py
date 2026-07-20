from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from tqdm import tqdm


from Llm.embedding_model_loader import EmbeddingModelLoader
from config import (
    ANSWER_LLM_NAME,
    DATASET_NAME,
    EMBEDDING_MODEL_NAME,
    MAX_GENERATEION_NUM,
    MAX_INPUT_LENGTH,
    PROVIDER,
    ROUND1_HRC_RETRIEVAL_CAP,
    ROUND2_HRC_RETRIEVAL_CAP,
    TEMPLATES_ROOT,
    dataset_root,
    resolve_project_path,
)
from Llm.llm_loader import LLM, resolve_provider
from Utils.efficiency_utils import SampleEfficiencyTracker
from Utils.database_prediction_store import (
    build_database_prediction,
    build_database_prediction_path,
    initialize_database_prediction_file,
    replace_database_predictions,
    upsert_database_prediction,
)
from Utils.render_tools import SchemaTextRenderer
from Utils.qdrant_utils import resolve_qdrant_collection_name
from Utils.decision_trace import (
    append_event,
    build_trace_paths,
    build_trace_summary,
    finalize_trace,
    initialize_trace,
    load_events,
    mark_trace_failed,
)
from Utils.database_retrieval_logic import (
    PruningResult,
    database_pruning,
    resolve_hrc_budget,
    resolve_round1_rerank_top_k,
    should_run_round2,
)
from Utils.schema_selection import (
    DbInfoSchemaStore,
    count_prompt_tokens,
    load_db_counts,
    resolve_schema_text_for_db,
)
from Utils.tools import (
    build_db_id_filter,
    clean_text,
    get_qdrant_client,
    load_db_info_index,
    query_qdrant,
    render_prompt,
    resolve_hint,
    resolve_prompt_token_cap,
)
from Utils.artifact_paths import require_results_output

SUPPORTED_DB_SELECTION_MODES = {"rerank", "pruning"}


@dataclass(frozen=True)
class RerankResult:
    database_scores: list[dict[str, Any]]
    selected_db_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative coarse database retrieval.")
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--answer-llm-name", dest="answer_llm_name", type=str, default=None)
    parser.add_argument("--provider", dest="provider", type=str, default=None)
    parser.add_argument("--input-path", dest="input_path", type=Path, default=None)
    parser.add_argument("--qdrant-path", dest="qdrant_path", type=Path, default=None)
    parser.add_argument("--collection-name", dest="collection_name", type=str, default=None)
    parser.add_argument("--prediction-path", dest="prediction_path", type=Path, default=None)
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
    parser.add_argument(
        "--db-selection-mode",
        dest="db_selection_mode",
        choices=sorted(SUPPORTED_DB_SELECTION_MODES),
        default="rerank",
        help=(
            "How to select databases after support-based pruning. "
            "'rerank' uses the LLM yes/no reranker; 'pruning' directly keeps the pruning order."
        ),
    )
    return parser.parse_args()


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


def get_highly_relevant_columns(
    query_vector: list[float],
    qdrant_client: QdrantClient,
    *,
    collection_name: str,
    top_k: int,
    candidate_db_ids: list[str] | None = None,
) -> list[ScoredPoint]:
    if top_k <= 0:
        return []
    query_filter = build_db_id_filter(candidate_db_ids)
    if candidate_db_ids is not None and query_filter is None:
        return []
    return query_qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=top_k,
        query_filter=query_filter,
        with_vectors=False,
    )


def collect_single_token_candidates(tokenizer: Any, target_text: str) -> list[int]:
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


def compute_yes_probability(next_token_logits: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    yes_token_ids = collect_single_token_candidates(tokenizer, "yes")
    no_token_ids = collect_single_token_candidates(tokenizer, "no")
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    yes_ids = torch.tensor(yes_token_ids, device=next_token_probs.device)
    no_ids = torch.tensor(no_token_ids, device=next_token_probs.device)
    yes_prob = next_token_probs[0].index_select(0, yes_ids).sum()
    no_prob = next_token_probs[0].index_select(0, no_ids).sum()
    return yes_prob / (yes_prob + no_prob)


def rerank_candidate_databases(
    query: str,
    query_vector: list[float],
    ranking_llm: LLM,
    candidate_db_ids: list[str],
    prompt_template: str,
    top_k: int,
    hint_text: str,
    schema_store: DbInfoSchemaStore,
    qdrant_client: QdrantClient,
    collection_name: str,
    db_counts: dict[str, int],
    efficiency_tracker: SampleEfficiencyTracker | None = None,
) -> RerankResult:
    if not candidate_db_ids:
        return RerankResult([], [])

    yes_scores: list[float | None] = []
    target_prompt_cap = resolve_prompt_token_cap(ranking_llm.max_input_length)
    base_hint_text = resolve_base_prompt_hint(
        ranking_llm=ranking_llm,
        prompt_template=prompt_template,
        query=query,
        raw_hint_text=hint_text,
        target_prompt_cap=target_prompt_cap,
    )
    for db_id in candidate_db_ids:
        hint_text = base_hint_text
        schema_text, _ = resolve_schema_text_for_db(
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
            yes_scores.append(None)
            continue
        next_token_logits, total_tokens = ranking_llm.query_logits_with_usage(prompt)
        if efficiency_tracker is not None:
            efficiency_tracker.add_llm_total_tokens(total_tokens)
        yes_prob_binary = compute_yes_probability(next_token_logits, ranking_llm.tokenizer)
        score = float(yes_prob_binary.detach())
        yes_scores.append(score if math.isfinite(score) else None)
        del next_token_logits
        del yes_prob_binary
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranked_indices = sorted(
        range(len(candidate_db_ids)),
        key=lambda index: (
            yes_scores[index]
            if yes_scores[index] is not None
            else float("-inf")
        ),
        reverse=True,
    )
    return RerankResult(
        database_scores=[
            {
                "db_id": candidate_db_ids[index],
                "yes_score": yes_scores[index],
                "rank": rank,
            }
            for rank, index in enumerate(ranked_indices, 1)
        ],
        selected_db_ids=[candidate_db_ids[index] for index in ranked_indices[:top_k]],
    )


def select_candidate_databases(
    *,
    query: str,
    query_vector: list[float],
    ranking_llm: Optional[LLM],
    candidate_db_ids: list[str],
    prompt_template: str,
    top_k: int,
    hint_text: str,
    schema_store: Optional[DbInfoSchemaStore],
    qdrant_client: QdrantClient,
    collection_name: str,
    db_counts: dict[str, int],
    db_selection_mode: str,
    efficiency_tracker: SampleEfficiencyTracker | None = None,
) -> RerankResult:
    if db_selection_mode == "pruning":
        return RerankResult(
            database_scores=[
                {"db_id": db_id, "yes_score": None, "rank": rank}
                for rank, db_id in enumerate(candidate_db_ids, 1)
            ],
            selected_db_ids=candidate_db_ids[:top_k],
        )

    if ranking_llm is None or schema_store is None:
        raise ValueError("LLM reranking requires ranking_llm and schema_store.")

    return rerank_candidate_databases(
        query,
        query_vector,
        ranking_llm,
        candidate_db_ids,
        prompt_template,
        top_k,
        hint_text,
        schema_store,
        qdrant_client,
        collection_name,
        db_counts,
        efficiency_tracker,
    )


def _gold_hit_count(pruning: PruningResult, gold_db_id: str) -> int:
    for stats in pruning.database_stats:
        if stats["db_id"] == gold_db_id:
            return int(stats["hit_count"])
    return 0


def _append_pruning_event(
    events_path: Path,
    sample_id: str,
    stage: str,
    pruning: PruningResult,
    gold_db_id: str,
) -> None:
    gold_in_hrc = _gold_hit_count(pruning, gold_db_id) > 0
    diagnostics = {
        "gold_in_hrc": gold_in_hrc,
        "gold_after_filter": gold_db_id in pruning.filtered_db_ids,
        "gold_in_selected": gold_db_id in pruning.selected_db_ids,
    }
    if stage == "database.round1.pruning":
        diagnostics["gold_in_top10"] = gold_db_id in pruning.selected_db_ids
    append_event(
        events_path,
        sample_id=sample_id,
        stage=stage,
        output={
            "database_stats": pruning.database_stats,
            "filtered_db_ids": pruning.filtered_db_ids,
            "selected_db_ids": pruning.selected_db_ids,
        },
        diagnostics=diagnostics,
    )


def _append_rerank_event(
    events_path: Path,
    sample_id: str,
    stage: str,
    result: RerankResult,
    input_db_ids: list[str],
    gold_db_id: str,
) -> None:
    append_event(
        events_path,
        sample_id=sample_id,
        stage=stage,
        output={
            "database_scores": result.database_scores,
            "selected_db_ids": result.selected_db_ids,
        },
        diagnostics={
            "gold_in_input": gold_db_id in input_db_ids,
            "gold_in_selected": gold_db_id in result.selected_db_ids,
        },
    )


def _database_first_error_stage(
    *,
    gold_db_id: str,
    round1_pruning: PruningResult,
    round1_selected: list[str],
    round2_pruning: PruningResult | None,
    final_selected: list[str],
) -> str:
    if _gold_hit_count(round1_pruning, gold_db_id) == 0:
        return "database.round1.hrc"
    if gold_db_id not in round1_pruning.filtered_db_ids:
        return "database.round1.pruning.filter"
    if gold_db_id not in round1_pruning.selected_db_ids:
        return "database.round1.pruning.top10"
    if gold_db_id not in round1_selected:
        return "database.round1.rerank"
    if round2_pruning is not None:
        if _gold_hit_count(round2_pruning, gold_db_id) == 0:
            return "database.round2.hrc"
        if gold_db_id not in round2_pruning.selected_db_ids:
            return "database.round2.pruning"
        if gold_db_id not in final_selected:
            return "database.round2.rerank"
    elif gold_db_id not in final_selected:
        return "database.round1.rerank"
    return "correct"


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or DATASET_NAME
    answer_llm_name = args.answer_llm_name or ANSWER_LLM_NAME
    provider = resolve_provider(args.provider or PROVIDER)
    max_input_length = args.max_input_length or MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or MAX_GENERATEION_NUM
    db_selection_mode = args.db_selection_mode

    current_dataset_root = dataset_root(dataset_name)
    qdrant_path = (
        resolve_project_path(args.qdrant_path)
        if args.qdrant_path
        else current_dataset_root / "qdrant_column_index"
    )
    db_info_path = current_dataset_root / "db_info.json"
    documents_dir = current_dataset_root / "documents"
    dataset_path = resolve_project_path(args.input_path) if args.input_path else current_dataset_root / "gold_sl.json"

    dataset_df = pd.read_json(dataset_path)
    db_info_index = load_db_info_index(db_info_path)
    db_counts = load_db_counts(db_info_index)
    prompt_path = TEMPLATES_ROOT / "zero_shot" / "binary_classification_database.txt"
    prompt_template = prompt_path.read_text(encoding='utf-8').strip()
    collection_name = resolve_qdrant_collection_name(
        qdrant_path,
        explicit_name=args.collection_name,
        fallback=dataset_name,
    )

    database_method = f"global_coarse_{db_selection_mode}"
    prediction_path = require_results_output(
        resolve_project_path(args.prediction_path)
        if args.prediction_path
        else build_database_prediction_path(
            database_method, dataset_name, answer_llm_name
        )
    )
    initialize_database_prediction_file(
        prediction_path,
        dataset_name=dataset_name,
        method=database_method,
        model_name=answer_llm_name,
    )
    replace_database_predictions(prediction_path, [])

    trace_paths = build_trace_paths(
        "db", database_method, dataset_name, answer_llm_name
    )
    initialize_trace(
        trace_paths,
        {
            "dataset": dataset_name,
            "method": database_method,
            "ranking_model": answer_llm_name,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "input_path": str(dataset_path),
            "qdrant_path": str(qdrant_path),
            "qdrant_collection": collection_name,
            "total_samples": len(dataset_df),
        },
    )

    try:
        client = get_qdrant_client(qdrant_path)
        embedder = EmbeddingModelLoader(model_name=EMBEDDING_MODEL_NAME)
        ranking_llm = None
        schema_store = None
        if db_selection_mode == "rerank":
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

        for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
            sample_id = clean_text(row.get("id"))
            question = str(row["question"])
            gold_db_id = clean_text(row.get("db_id"))
            efficiency_tracker = SampleEfficiencyTracker()
            append_event(
                trace_paths.events,
                sample_id=sample_id,
                stage="sample.input",
                output={"question": question, "gold_db_id": gold_db_id},
            )
            current_stage = "database.round1.hrc"
            try:
                hint_text = resolve_hint(
                    row,
                    dataset_name=dataset_name,
                    documents_dir=documents_dir,
                )
                query_embedding = embedder.encode(question, convert_to_list=True)

                round1_budget = resolve_hrc_budget(
                    db_counts,
                    candidate_db_ids=None,
                    cap=ROUND1_HRC_RETRIEVAL_CAP,
                )
                round1_points = get_highly_relevant_columns(
                    query_embedding,
                    client,
                    collection_name=collection_name,
                    top_k=round1_budget.top_k,
                )
                round1_pruning = database_pruning(
                    round1_points,
                    min_hit_count=2,
                    min_sim_ratio=0.8,
                )
                round1_gold_hits = _gold_hit_count(round1_pruning, gold_db_id)
                append_event(
                    trace_paths.events,
                    sample_id=sample_id,
                    stage=current_stage,
                    output={
                        "candidate_column_count": round1_budget.candidate_column_count,
                        "top_k": round1_budget.top_k,
                        "returned_count": len(round1_points),
                    },
                    diagnostics={
                        "gold_db_hit_count": round1_gold_hits,
                        "gold_in_output": round1_gold_hits > 0,
                    },
                )

                current_stage = "database.round1.pruning"
                _append_pruning_event(
                    trace_paths.events,
                    sample_id,
                    current_stage,
                    round1_pruning,
                    gold_db_id,
                )
                cd1_db_ids = round1_pruning.selected_db_ids

                current_stage = "database.round1.rerank"
                round1_top_k = resolve_round1_rerank_top_k(len(cd1_db_ids))
                round1_rerank = select_candidate_databases(
                    query=question,
                    query_vector=query_embedding,
                    ranking_llm=ranking_llm,
                    candidate_db_ids=cd1_db_ids,
                    prompt_template=prompt_template,
                    top_k=round1_top_k,
                    hint_text=hint_text,
                    schema_store=schema_store,
                    qdrant_client=client,
                    collection_name=collection_name,
                    db_counts=db_counts,
                    db_selection_mode=db_selection_mode,
                    efficiency_tracker=efficiency_tracker,
                )
                _append_rerank_event(
                    trace_paths.events,
                    sample_id,
                    current_stage,
                    round1_rerank,
                    cd1_db_ids,
                    gold_db_id,
                )

                round2_pruning: PruningResult | None = None
                if not should_run_round2(len(cd1_db_ids)):
                    final_selection = round1_rerank
                    for skipped_stage in (
                        "database.round2.hrc",
                        "database.round2.pruning",
                        "database.round2.rerank",
                    ):
                        current_stage = skipped_stage
                        append_event(
                            trace_paths.events,
                            sample_id=sample_id,
                            stage=skipped_stage,
                            status="skipped",
                            output={
                                "selected_db_ids": final_selection.selected_db_ids
                            },
                        )
                else:
                    cd2_db_ids = round1_rerank.selected_db_ids
                    current_stage = "database.round2.hrc"
                    round2_budget = resolve_hrc_budget(
                        db_counts,
                        candidate_db_ids=cd2_db_ids,
                        cap=ROUND2_HRC_RETRIEVAL_CAP,
                    )
                    round2_points = get_highly_relevant_columns(
                        query_embedding,
                        client,
                        collection_name=collection_name,
                        top_k=round2_budget.top_k,
                        candidate_db_ids=cd2_db_ids,
                    )
                    round2_pruning = database_pruning(
                        round2_points,
                        min_hit_count=2,
                        min_sim_ratio=0.8,
                    )
                    round2_gold_hits = _gold_hit_count(round2_pruning, gold_db_id)
                    append_event(
                        trace_paths.events,
                        sample_id=sample_id,
                        stage=current_stage,
                        output={
                            "candidate_db_ids": cd2_db_ids,
                            "candidate_column_count": (
                                round2_budget.candidate_column_count
                            ),
                            "top_k": round2_budget.top_k,
                            "returned_count": len(round2_points),
                        },
                        diagnostics={
                            "gold_in_input": gold_db_id in cd2_db_ids,
                            "gold_db_hit_count": round2_gold_hits,
                            "gold_in_output": round2_gold_hits > 0,
                        },
                    )

                    current_stage = "database.round2.pruning"
                    _append_pruning_event(
                        trace_paths.events,
                        sample_id,
                        current_stage,
                        round2_pruning,
                        gold_db_id,
                    )
                    cd3_db_ids = round2_pruning.selected_db_ids

                    current_stage = "database.round2.rerank"
                    final_selection = select_candidate_databases(
                        query=question,
                        query_vector=query_embedding,
                        ranking_llm=ranking_llm,
                        candidate_db_ids=cd3_db_ids,
                        prompt_template=prompt_template,
                        top_k=1,
                        hint_text=hint_text,
                        schema_store=schema_store,
                        qdrant_client=client,
                        collection_name=collection_name,
                        db_counts=db_counts,
                        db_selection_mode=db_selection_mode,
                        efficiency_tracker=efficiency_tracker,
                    )
                    _append_rerank_event(
                        trace_paths.events,
                        sample_id,
                        current_stage,
                        final_selection,
                        cd3_db_ids,
                        gold_db_id,
                    )

                current_stage = "database.final"
                predicted_db_id = (
                    final_selection.selected_db_ids[0]
                    if final_selection.selected_db_ids
                    else None
                )
                first_error_stage = _database_first_error_stage(
                    gold_db_id=gold_db_id,
                    round1_pruning=round1_pruning,
                    round1_selected=round1_rerank.selected_db_ids,
                    round2_pruning=round2_pruning,
                    final_selected=final_selection.selected_db_ids,
                )
                prediction = build_database_prediction(
                    sample_id=sample_id,
                    question=question,
                    predicted_db_id=predicted_db_id,
                    efficiency=efficiency_tracker.finalize(),
                )
                final_event_status = "success"
                final_event_output = {"predicted_db_id": predicted_db_id}
                final_event_diagnostics = {
                    "correct": predicted_db_id == gold_db_id,
                    "first_error_stage": first_error_stage,
                }
                final_event_error = None
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                if current_stage != "database.final":
                    append_event(
                        trace_paths.events,
                        sample_id=sample_id,
                        stage=current_stage,
                        status="failed",
                        error=error,
                    )
                prediction = build_database_prediction(
                    sample_id=sample_id,
                    question=question,
                    predicted_db_id=None,
                    efficiency=efficiency_tracker.finalize(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                final_event_status = "failed"
                final_event_output = {"predicted_db_id": None}
                final_event_diagnostics = {
                    "correct": False,
                    "first_error_stage": current_stage,
                }
                final_event_error = error
            try:
                upsert_database_prediction(prediction_path, prediction)
            except Exception as exc:
                persistence_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                try:
                    append_event(
                        trace_paths.events,
                        sample_id=sample_id,
                        stage="database.final",
                        status="failed",
                        output=final_event_output,
                        diagnostics={
                            "correct": False,
                            "first_error_stage": (
                                final_event_diagnostics["first_error_stage"]
                                if final_event_diagnostics["first_error_stage"]
                                != "correct"
                                else "database.persistence"
                            ),
                        },
                        error=persistence_error,
                    )
                except Exception:
                    pass
                raise
            append_event(
                trace_paths.events,
                sample_id=sample_id,
                stage="database.final",
                status=final_event_status,
                output=final_event_output,
                diagnostics=final_event_diagnostics,
                error=final_event_error,
            )

        finalize_trace(
            trace_paths,
            build_trace_summary(load_events(trace_paths.events)),
        )
        print(f"Completed {len(dataset_df)} database prediction records.")
    except BaseException as exc:
        mark_trace_failed(trace_paths, exc)
        raise


if __name__ == "__main__":
    main()
