from __future__ import annotations

from typing import Any

from Utils.render_tools import SchemaTextRenderer
from Utils.tools import (
    build_db_id_filter,
    clean_text,
    load_column_records_from_db_info,
    query_qdrant,
    render_prompt,
    resolve_prompt_token_cap,
)


MAX_RANKED_SCHEMA_CANDIDATES = 256


def count_valid_columns(db_info: dict[str, Any]) -> int:
    column_names = db_info.get("column_names", [])
    if not isinstance(column_names, list):
        return 0

    valid_column_count = 0
    for entry in column_names:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        table_idx, column_name = entry
        if table_idx == -1 or column_name == "*":
            continue
        valid_column_count += 1
    return valid_column_count


def load_db_counts(db_info_index: dict[str, dict[str, Any]]) -> dict[str, int]:
    db_counts: dict[str, int] = {}
    for db_id, entry in db_info_index.items():
        raw_count = entry.get("db_counts")
        if raw_count is None:
            db_counts[db_id] = count_valid_columns(entry)
            continue
        try:
            db_counts[db_id] = int(raw_count)
        except (TypeError, ValueError):
            db_counts[db_id] = count_valid_columns(entry)

    return db_counts


def build_column_id(record: dict[str, Any]) -> str:
    column_id = clean_text(record.get("column_id"))
    if column_id:
        return column_id

    meta = record.get("meta_data") or {}
    db_id = clean_text(record.get("db_id") or meta.get("db_id"))
    table_name = clean_text(record.get("table_name") or meta.get("table_name"))
    column_name = clean_text(record.get("column_name"))

    if db_id and table_name and column_name:
        return f"{db_id}.{table_name}.{column_name}"
    if table_name and column_name:
        return f"{table_name}::{column_name}"
    return column_name


def get_record_table_name(record: dict[str, Any]) -> str:
    meta = record.get("meta_data") or {}
    return clean_text(record.get("table_name") or meta.get("table_name"))


def is_truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def count_prompt_tokens(
    ranking_llm: Any,
    prompt_template: str,
    schema_text: str,
    query: str,
    hint_text: str,
) -> int:
    prompt = render_prompt(
        prompt_template,
        DATABASE_SCHEMAS=schema_text,
        QUESTION=query,
        HINT=hint_text,
    )
    return ranking_llm.count_input_tokens(prompt)


class DbInfoSchemaStore:
    def __init__(
        self,
        db_info_index: dict[str, dict[str, Any]],
        renderer: SchemaTextRenderer,
    ) -> None:
        self.db_info_index = db_info_index
        self.renderer = renderer
        self._column_record_cache: dict[str, list[dict[str, Any]]] = {}
        self._key_metadata_cache: dict[str, tuple[bool, bool]] = {}
        self._full_schema_text_cache: dict[str, str] = {}

    def get_column_records(self, db_id: str) -> list[dict[str, Any]]:
        if db_id not in self._column_record_cache:
            try:
                records = load_column_records_from_db_info(
                    predict_db_id=db_id,
                    db_info_index=self.db_info_index,
                )
            except FileNotFoundError:
                records = []
            self._column_record_cache[db_id] = records
        return list(self._column_record_cache[db_id])

    def get_key_metadata_flags(self, db_id: str) -> tuple[bool, bool]:
        if db_id not in self._key_metadata_cache:
            db_entry = self.db_info_index.get(db_id) or {}
            self._key_metadata_cache[db_id] = (
                "primary_keys" in db_entry,
                "foreign_keys" in db_entry,
            )
        return self._key_metadata_cache[db_id]

    def render_schema_text(
        self,
        db_id: str,
        column_records: list[dict[str, Any]],
    ) -> str:
        primary_keys_available, foreign_keys_available = self.get_key_metadata_flags(db_id)
        return self.renderer.render_selected_database_text(
            db_id=db_id,
            column_records=column_records,
            primary_keys_available=primary_keys_available,
            foreign_keys_available=foreign_keys_available,
        )

    def get_full_schema_text(self, db_id: str) -> str:
        if db_id not in self._full_schema_text_cache:
            self._full_schema_text_cache[db_id] = self.render_schema_text(
                db_id=db_id,
                column_records=self.get_column_records(db_id),
            )
        return self._full_schema_text_cache[db_id]


def get_ranked_db_column_candidates(
    query_vector: list[float],
    qdrant_client: Any,
    collection_name: str,
    db_id: str,
    db_counts: dict[str, int],
    schema_store: DbInfoSchemaStore,
) -> list[dict[str, Any]]:
    query_filter = build_db_id_filter([db_id])
    if query_filter is None:
        return []

    all_records = schema_store.get_column_records(db_id)
    if not all_records:
        return []

    db_column_count = max(db_counts.get(db_id, 0), len(all_records))
    ranked_top_k = max(1, db_column_count if db_column_count > 0 else MAX_RANKED_SCHEMA_CANDIDATES)

    ranked_points = query_qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=ranked_top_k,
        query_filter=query_filter,
        with_vectors=False,
    )

    score_by_column_id: dict[str, float] = {}
    for point in ranked_points:
        payload = getattr(point, "payload", None)
        if payload is None and isinstance(point, dict):
            payload = point.get("payload")
        payload = dict(payload or {})

        score = getattr(point, "score", None)
        if score is None and isinstance(point, dict):
            score = point.get("score")

        column_id = clean_text(payload.get("column_id"))
        if not column_id:
            column_id = f"{clean_text(payload.get('table_name'))}::{clean_text(payload.get('column_name'))}"
        if not column_id or column_id in score_by_column_id:
            continue

        score_by_column_id[column_id] = float(score) if score is not None else float("-inf")

    candidates: list[dict[str, Any]] = []
    for original_index, record in enumerate(all_records):
        column_id = build_column_id(record)
        candidates.append(
            {
                "column_id": column_id,
                "table_name": get_record_table_name(record),
                "score": score_by_column_id.get(column_id, float("-inf")),
                "record": record,
                "original_index": original_index,
            }
        )

    return candidates


def materialize_selected_column_records(
    all_records: list[dict[str, Any]],
    selected_column_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        record
        for record in all_records
        if build_column_id(record) in selected_column_ids
    ]


def evaluate_selected_columns(
    db_id: str,
    all_records: list[dict[str, Any]],
    selected_column_ids: set[str],
    schema_store: DbInfoSchemaStore,
    ranking_llm: Any,
    prompt_template: str,
    query: str,
    hint_text: str,
) -> tuple[list[dict[str, Any]], str, int]:
    selected_records = materialize_selected_column_records(all_records, selected_column_ids)
    schema_text = schema_store.render_schema_text(db_id, selected_records)
    prompt_tokens = count_prompt_tokens(
        ranking_llm=ranking_llm,
        prompt_template=prompt_template,
        schema_text=schema_text,
        query=query,
        hint_text=hint_text,
    )
    return selected_records, schema_text, prompt_tokens


def resolve_table_top_one_column_ids(
    candidates: list[dict[str, Any]],
) -> set[str]:
    best_candidates_by_table: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        table_name = candidate["table_name"]
        if not table_name:
            continue

        current_best = best_candidates_by_table.get(table_name)
        if current_best is None:
            best_candidates_by_table[table_name] = candidate
            continue

        if candidate["score"] > current_best["score"]:
            best_candidates_by_table[table_name] = candidate
            continue

        if (
            candidate["score"] == current_best["score"]
            and candidate["original_index"] < current_best["original_index"]
        ):
            best_candidates_by_table[table_name] = candidate

    return {candidate["column_id"] for candidate in best_candidates_by_table.values()}


def resolve_ranked_candidates_by_table(
    candidates: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    table_order: list[str] = []
    candidates_by_table: dict[str, list[dict[str, Any]]] = {}

    for candidate in candidates:
        table_name = candidate["table_name"]
        if not table_name:
            continue
        if table_name not in candidates_by_table:
            table_order.append(table_name)
            candidates_by_table[table_name] = []
        candidates_by_table[table_name].append(candidate)

    for table_name in table_order:
        candidates_by_table[table_name].sort(
            key=lambda candidate: (-candidate["score"], candidate["original_index"])
        )

    return table_order, candidates_by_table


def prune_key_columns_to_budget(
    db_id: str,
    all_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selected_column_ids: set[str],
    protected_column_ids: set[str],
    schema_store: DbInfoSchemaStore,
    ranking_llm: Any,
    prompt_template: str,
    query: str,
    hint_text: str,
    target_prompt_cap: int,
) -> set[str]:
    pruned_column_ids = set(selected_column_ids)
    _, _, prompt_tokens = evaluate_selected_columns(
        db_id=db_id,
        all_records=all_records,
        selected_column_ids=pruned_column_ids,
        schema_store=schema_store,
        ranking_llm=ranking_llm,
        prompt_template=prompt_template,
        query=query,
        hint_text=hint_text,
    )
    if prompt_tokens <= target_prompt_cap:
        return pruned_column_ids

    removable_foreign_keys = sorted(
        [
            candidate
            for candidate in candidates
            if candidate["column_id"] in pruned_column_ids
            and candidate["column_id"] not in protected_column_ids
            and is_truthy_flag(candidate["record"].get("is_foreign_key"))
        ],
        key=lambda candidate: (candidate["score"], candidate["original_index"]),
    )
    removable_primary_keys = sorted(
        [
            candidate
            for candidate in candidates
            if candidate["column_id"] in pruned_column_ids
            and candidate["column_id"] not in protected_column_ids
            and is_truthy_flag(candidate["record"].get("is_primary_key"))
        ],
        key=lambda candidate: (candidate["score"], candidate["original_index"]),
    )

    for removable_candidates in (removable_foreign_keys, removable_primary_keys):
        for candidate in removable_candidates:
            column_id = candidate["column_id"]
            if column_id not in pruned_column_ids or column_id in protected_column_ids:
                continue

            pruned_column_ids.remove(column_id)
            _, _, prompt_tokens = evaluate_selected_columns(
                db_id=db_id,
                all_records=all_records,
                selected_column_ids=pruned_column_ids,
                schema_store=schema_store,
                ranking_llm=ranking_llm,
                prompt_template=prompt_template,
                query=query,
                hint_text=hint_text,
            )
            if prompt_tokens <= target_prompt_cap:
                return pruned_column_ids

    return pruned_column_ids


def select_relevant_column_records_for_db(
    query: str,
    query_vector: list[float],
    db_id: str,
    ranking_llm: Any,
    prompt_template: str,
    hint_text: str,
    schema_store: DbInfoSchemaStore,
    qdrant_client: Any,
    collection_name: str,
    db_counts: dict[str, int],
) -> list[dict[str, Any]]:
    target_prompt_cap = resolve_prompt_token_cap(ranking_llm.max_input_length)
    all_records = schema_store.get_column_records(db_id)
    if not all_records:
        return []

    candidates = get_ranked_db_column_candidates(
        query_vector=query_vector,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        db_id=db_id,
        db_counts=db_counts,
        schema_store=schema_store,
    )
    if not candidates:
        return all_records

    table_top_one_column_ids = resolve_table_top_one_column_ids(candidates)
    selected_column_ids = set(table_top_one_column_ids)
    for candidate in candidates:
        record = candidate["record"]
        if is_truthy_flag(record.get("is_primary_key")) or is_truthy_flag(record.get("is_foreign_key")):
            selected_column_ids.add(candidate["column_id"])

    selected_column_ids = prune_key_columns_to_budget(
        db_id=db_id,
        all_records=all_records,
        candidates=candidates,
        selected_column_ids=selected_column_ids,
        protected_column_ids=table_top_one_column_ids,
        schema_store=schema_store,
        ranking_llm=ranking_llm,
        prompt_template=prompt_template,
        query=query,
        hint_text=hint_text,
        target_prompt_cap=target_prompt_cap,
    )

    selected_records, _, prompt_tokens = evaluate_selected_columns(
        db_id=db_id,
        all_records=all_records,
        selected_column_ids=selected_column_ids,
        schema_store=schema_store,
        ranking_llm=ranking_llm,
        prompt_template=prompt_template,
        query=query,
        hint_text=hint_text,
    )
    if prompt_tokens > target_prompt_cap:
        return selected_records

    table_order, candidates_by_table = resolve_ranked_candidates_by_table(candidates)
    while True:
        round_column_ids: list[str] = []
        for table_name in table_order:
            for candidate in candidates_by_table[table_name]:
                if candidate["column_id"] in selected_column_ids:
                    continue
                round_column_ids.append(candidate["column_id"])
                break

        if not round_column_ids:
            break

        proposed_column_ids = set(selected_column_ids)
        proposed_column_ids.update(round_column_ids)
        proposed_records, _, proposed_prompt_tokens = evaluate_selected_columns(
            db_id=db_id,
            all_records=all_records,
            selected_column_ids=proposed_column_ids,
            schema_store=schema_store,
            ranking_llm=ranking_llm,
            prompt_template=prompt_template,
            query=query,
            hint_text=hint_text,
        )
        if proposed_prompt_tokens > target_prompt_cap:
            break

        selected_column_ids = proposed_column_ids
        selected_records = proposed_records

    return selected_records


def resolve_schema_text_for_db(
    query: str,
    query_vector: list[float],
    db_id: str,
    ranking_llm: Any,
    prompt_template: str,
    hint_text: str,
    schema_store: DbInfoSchemaStore,
    qdrant_client: Any,
    collection_name: str,
    db_counts: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    target_prompt_cap = resolve_prompt_token_cap(ranking_llm.max_input_length)
    full_schema_text = schema_store.get_full_schema_text(db_id)
    full_records = schema_store.get_column_records(db_id)
    if full_schema_text:
        full_prompt_tokens = count_prompt_tokens(
            ranking_llm=ranking_llm,
            prompt_template=prompt_template,
            schema_text=full_schema_text,
            query=query,
            hint_text=hint_text,
        )
        if full_prompt_tokens <= target_prompt_cap:
            return full_schema_text, full_records

    selected_records = select_relevant_column_records_for_db(
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
    return schema_store.render_schema_text(db_id, selected_records), selected_records


__all__ = [
    "DbInfoSchemaStore",
    "build_column_id",
    "count_prompt_tokens",
    "count_valid_columns",
    "get_ranked_db_column_candidates",
    "get_record_table_name",
    "is_truthy_flag",
    "load_db_counts",
    "materialize_selected_column_records",
    "prune_key_columns_to_budget",
    "resolve_ranked_candidates_by_table",
    "resolve_schema_text_for_db",
    "resolve_table_top_one_column_ids",
    "select_relevant_column_records_for_db",
]
