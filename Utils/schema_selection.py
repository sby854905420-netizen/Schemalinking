from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from Utils.prediction_store import snowflake_identifier_key, split_qualified_identifier
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


def _snowflake_table_candidates(
    requested_table: str,
    *,
    db_id: str,
    full_name_index: dict[tuple[str, ...], set[str]],
    suffix_index: dict[tuple[str, ...], set[str]],
    basename_index: dict[str, set[str]],
) -> set[str]:
    """Resolve one table reference within a selected Spider2 database."""

    parts = split_qualified_identifier(requested_table)
    key = snowflake_identifier_key(requested_table)
    if not parts or not key:
        return set()
    if len(parts) >= 3:
        # Never suffix-match a fully qualified name whose outer database is
        # wrong.  This keeps cross-database predictions fail-closed.
        return set(full_name_index.get(key, set()))
    if len(parts) == 2:
        suffix_matches = set(suffix_index.get(key, set()))
        if suffix_matches:
            return suffix_matches
        db_key = snowflake_identifier_key(db_id)
        if db_key and key[0] == db_key[0]:
            # A bare model prediction is restored as DB.TABLE before db_info
            # lookup; resolve it only when the table basename is unique.
            return set(basename_index.get(key[-1], set()))
        return set()
    return set(basename_index.get(key[0], set()))


def canonicalize_snowflake_schema_predictions(
    *,
    db_id: str,
    predicted_columns: dict[str, list[str]],
    predicted_tables: Sequence[str],
    schema_store: DbInfoSchemaStore,
) -> tuple[dict[str, list[str]], list[str], dict[str, Any]]:
    """Map Spider2 table aliases to canonical db_info three-part names.

    Resolution is restricted to ``db_id`` and only accepts a unique match.
    Fully qualified references are matched semantically with Snowflake's
    quoted/unquoted case rules; partial references use schema/table suffixes or
    a unique basename.  Ambiguous basenames remain unresolved.
    """

    all_records = schema_store.get_column_records(db_id)
    available_tables = list(
        dict.fromkeys(
            table
            for record in all_records
            if (table := get_record_table_name(record))
        )
    )
    available_table_set = set(available_tables)
    full_name_index: dict[tuple[str, ...], set[str]] = {}
    suffix_index: dict[tuple[str, ...], set[str]] = {}
    basename_index: dict[str, set[str]] = {}
    column_name_index: dict[str, dict[str, set[str]]] = {}
    for table in available_tables:
        key = snowflake_identifier_key(table)
        if not key:
            continue
        full_name_index.setdefault(key, set()).add(table)
        if len(key) >= 2:
            suffix_index.setdefault(key[-2:], set()).add(table)
        basename_index.setdefault(key[-1], set()).add(table)
    for record in all_records:
        table = get_record_table_name(record)
        column = clean_text(record.get("column_name"))
        column_key = snowflake_identifier_key(column)
        if table and len(column_key) == 1:
            column_name_index.setdefault(table, {}).setdefault(
                column_key[0], set()
            ).add(column)

    def candidates(table: str) -> set[str]:
        return _snowflake_table_candidates(
            table,
            db_id=db_id,
            full_name_index=full_name_index,
            suffix_index=suffix_index,
            basename_index=basename_index,
        )

    resolved_tables: list[str] = []
    resolved_table_context: set[str] = set()
    resolved_alias_count = 0
    unresolved_tables: list[str] = []
    ambiguous_tables: list[str] = []
    for raw_table in predicted_tables:
        table = str(raw_table).strip()
        matches = candidates(table)
        if len(matches) == 1:
            resolved = next(iter(matches))
            if resolved != table:
                resolved_alias_count += 1
            resolved_table_context.add(resolved)
        else:
            resolved = table
            target = ambiguous_tables if len(matches) > 1 else unresolved_tables
            if table and table not in target:
                target.append(table)
        if resolved and resolved not in resolved_tables:
            resolved_tables.append(resolved)

    resolved_columns: dict[str, list[str]] = {}
    unresolved_column_tables: list[str] = []
    ambiguous_column_tables: list[str] = []
    resolved_column_alias_count = 0
    unresolved_columns: list[str] = []
    ambiguous_columns: list[str] = []
    for raw_table, raw_columns in predicted_columns.items():
        table = str(raw_table).strip()
        matches = candidates(table)
        if len(matches) > 1 and resolved_table_context:
            contextual_matches = matches & resolved_table_context
            if len(contextual_matches) == 1:
                matches = contextual_matches
        if len(matches) == 1:
            resolved = next(iter(matches))
            if resolved != table:
                resolved_alias_count += 1
        else:
            resolved = table
            target = (
                ambiguous_column_tables
                if len(matches) > 1
                else unresolved_column_tables
            )
            if table and table not in target:
                target.append(table)

        columns = resolved_columns.setdefault(resolved, [])
        for raw_column in raw_columns:
            column = str(raw_column).strip()
            canonical_column = column
            if column != "*" and resolved in available_table_set:
                column_key = snowflake_identifier_key(column)
                column_matches = (
                    column_name_index.get(resolved, {}).get(column_key[0], set())
                    if len(column_key) == 1
                    else set()
                )
                if len(column_matches) == 1:
                    canonical_column = next(iter(column_matches))
                    if canonical_column != column:
                        resolved_column_alias_count += 1
                else:
                    identifier = f"{resolved}.{column}"
                    target = (
                        ambiguous_columns
                        if len(column_matches) > 1
                        else unresolved_columns
                    )
                    if identifier not in target:
                        target.append(identifier)
            if canonical_column and canonical_column not in columns:
                columns.append(canonical_column)

    return resolved_columns, resolved_tables, {
        "available_table_count": len(available_table_set),
        "resolved_table_alias_count": resolved_alias_count,
        "resolved_column_alias_count": resolved_column_alias_count,
        "unresolved_tables": unresolved_tables,
        "ambiguous_tables": ambiguous_tables,
        "unresolved_column_tables": unresolved_column_tables,
        "ambiguous_column_tables": ambiguous_column_tables,
        "unresolved_columns": unresolved_columns,
        "ambiguous_columns": ambiguous_columns,
    }


def select_predicted_column_records(
    db_id: str,
    predicted_columns: dict[str, list[str]],
    predicted_tables: Sequence[str],
    schema_store: DbInfoSchemaStore,
    include_key_columns: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select schema records explicitly named by a linking prediction."""

    all_records = schema_store.get_column_records(db_id)
    predicted_table_set = set(predicted_columns)
    predicted_table_set.update(
        str(table).strip() for table in predicted_tables if str(table).strip()
    )
    selected_record_ids: set[str] = set()

    if predicted_columns:
        column_sets = {
            table: set(columns) for table, columns in predicted_columns.items()
        }
        for record in all_records:
            table = str(record.get("table_name", "")).strip()
            column = str(record.get("column_name", "")).strip()
            requested = column_sets.get(table)
            if requested is not None and (column in requested or "*" in requested):
                selected_record_ids.add(build_column_id(record))
    elif predicted_table_set:
        for record in all_records:
            if str(record.get("table_name", "")).strip() in predicted_table_set:
                selected_record_ids.add(build_column_id(record))

    if include_key_columns and predicted_table_set:
        for record in all_records:
            table = str(record.get("table_name", "")).strip()
            if table not in predicted_table_set:
                continue
            if is_truthy_flag(record.get("is_primary_key")) or is_truthy_flag(
                record.get("is_foreign_key")
            ):
                selected_record_ids.add(build_column_id(record))

    selected_records = [
        record for record in all_records if build_column_id(record) in selected_record_ids
    ]
    return selected_records, {
        "available_column_count": len(all_records),
        "selected_column_count": len(selected_records),
        "predicted_table_count": len(predicted_table_set),
    }


__all__ = [
    "DbInfoSchemaStore",
    "build_column_id",
    "canonicalize_snowflake_schema_predictions",
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
    "select_predicted_column_records",
]
