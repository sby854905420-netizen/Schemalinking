from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from Utils.value_utils import resolve_external_knowledge


SCHEMA_DF_COLUMNS = [
    "table_name",
    "column_name",
    "data_type",
    "key_type",
    "referenced_column",
    "column_description",
    "example_values",
]


def resolve_supported_method(
    method: str | None,
    *,
    default: str,
    supported: set[str],
) -> str:
    resolved = method or default
    if resolved not in supported:
        expected = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported method {resolved!r}. Expected one of: {expected}.")
    return resolved


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
    limit: int | None = None,
    deduplicate: bool = False,
) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    if not isinstance(sample_rows, dict):
        return values

    table_rows = sample_rows.get(table_name, [])
    if not isinstance(table_rows, list):
        return values

    for row in table_rows:
        if not isinstance(row, dict) or column_name not in row:
            continue

        value = row[column_name]
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue

        if deduplicate:
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)

        values.append(value)
        if limit is not None and len(values) >= limit:
            break

    return values


def build_sample_values_text(sample_values: list[Any]) -> str:
    safe_values: list[str] = []
    for value in sample_values:
        normalized_value = clean_text(value)
        if normalized_value:
            safe_values.append(normalized_value)
    return ", ".join(safe_values)


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
    db_entry: dict[str, Any],
    column_idx: int,
    foreign_key_map: dict[int, int],
) -> str:
    target_idx = foreign_key_map.get(column_idx)
    if target_idx is None:
        return ""

    target_entry = get_indexed_value(db_entry.get("column_names"), target_idx)
    if not isinstance(target_entry, list) or len(target_entry) != 2:
        return ""

    target_table_idx, target_column_name = target_entry
    if target_table_idx == -1 or target_column_name == "*":
        return ""

    target_table_name = clean_text(get_indexed_value(db_entry.get("table_names"), target_table_idx, ""))
    if not target_table_name:
        return ""

    return f"{target_table_name}.{target_column_name}"


def build_value_descriptions(db_entry: dict[str, Any], column_idx: int) -> str:
    return clean_text(get_indexed_value(db_entry.get("value_descriptions"), column_idx, ""))


def build_column_record_from_db_info(
    db_entry: dict[str, Any],
    column_idx: int,
    primary_key_indices: set[int] | None = None,
    foreign_key_map: dict[int, int] | None = None,
) -> dict[str, Any] | None:
    column_entry = get_indexed_value(db_entry.get("column_names"), column_idx)
    if not isinstance(column_entry, list) or len(column_entry) != 2:
        return None

    table_idx, column_name = column_entry
    if table_idx == -1 or column_name == "*":
        return None

    table_name = clean_text(get_indexed_value(db_entry.get("table_names"), table_idx, ""))
    db_id = clean_text(db_entry.get("db_id"))
    normalized_column_name = clean_text(column_name)
    if not db_id or not table_name or not normalized_column_name:
        return None

    resolved_primary_key_indices = primary_key_indices or set()
    resolved_foreign_key_map = foreign_key_map or {}
    sample_values = build_sample_values(
        sample_rows=db_entry.get("sample_rows", {}),
        table_name=table_name,
        column_name=normalized_column_name,
    )

    return {
        "column_id": f"{db_id}.{table_name}.{normalized_column_name}",
        "column_name": normalized_column_name,
        "column_description": clean_text(get_indexed_value(db_entry.get("column_descriptions"), column_idx, "")),
        "column_data_type": clean_text(get_indexed_value(db_entry.get("column_types"), column_idx, "")),
        "sample_values": sample_values,
        "sample_values_text": build_sample_values_text(sample_values),
        "is_primary_key": column_idx in resolved_primary_key_indices,
        "is_foreign_key": column_idx in resolved_foreign_key_map,
        "value_descriptions": build_value_descriptions(db_entry, column_idx),
        "foreign_key": clean_text(build_foreign_key_text(db_entry, column_idx, resolved_foreign_key_map)),
        "db_id": db_id,
        "table_name": table_name,
        "meta_data": {
            "db_id": db_id,
            "table_name": table_name,
        },
    }


def iter_column_records_from_db_info(db_entry: dict[str, Any]) -> Iterable[dict[str, Any]]:
    primary_key_indices = flatten_primary_keys(db_entry.get("primary_keys", []))
    foreign_key_map = build_foreign_key_map(db_entry.get("foreign_keys", []))
    column_names = db_entry.get("column_names", [])
    if not isinstance(column_names, list):
        return

    for column_idx in range(len(column_names)):
        record = build_column_record_from_db_info(
            db_entry=db_entry,
            column_idx=column_idx,
            primary_key_indices=primary_key_indices,
            foreign_key_map=foreign_key_map,
        )
        if record is not None:
            yield record


def load_db_info_index(db_info_path: Path) -> dict[str, dict[str, Any]]:
    if not db_info_path.is_file():
        raise FileNotFoundError(f"Could not find db_info.json at {db_info_path}.")

    db_info = json.loads(db_info_path.read_text(encoding="utf-8"))
    if not isinstance(db_info, list):
        raise ValueError(f"Expected a list of database entries in {db_info_path}.")

    index: dict[str, dict[str, Any]] = {}
    for entry in db_info:
        if not isinstance(entry, dict):
            continue
        db_id = clean_text(entry.get("db_id"))
        if db_id:
            index[db_id] = entry
    return index


def load_column_records_from_db_info(
    predict_db_id: str,
    db_info_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    db_entry = db_info_index.get(predict_db_id)
    if db_entry is None:
        raise FileNotFoundError(f"Could not find schema for database '{predict_db_id}' in db_info.json.")
    return list(iter_column_records_from_db_info(db_entry))


def resolve_hint(
    source: Any,
    key: str = "external_knowledge",
    dataset_name: str | None = None,
    documents_dir: Path | None = None,
) -> str:
    """Load and normalize hint text for prompting.

    The default `external_knowledge` field is used across datasets, but its
    storage differs by dataset. This helper centralizes the convention so call
    sites can always request final prompt text from one place.
    """
    value = resolve_external_knowledge(
        source,
        key=key,
        dataset_name=dataset_name,
        documents_dir=documents_dir,
    )
    return value if value is not None else "No hint"


def render_prompt(prompt_template: str, **replacements: Any) -> str:
    prompt = prompt_template
    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{key}}}", "" if value is None else str(value))
    return prompt


def has_explicit_key_metadata(dataset_name: str, db_entry: dict[str, Any]) -> bool:
    if dataset_name.lower() == "spider2":
        return False
    return "primary_keys" in db_entry or "foreign_keys" in db_entry


def format_example_values(values: list[Any]) -> str:
    if not values:
        return "NOT_AVAILABLE"
    return ", ".join(str(value) for value in values)


def build_schema_dataframe_from_db_info(db_entry: dict[str, Any], dataset_name: str) -> pd.DataFrame:
    table_names = db_entry.get("table_names", [])
    column_names = db_entry.get("column_names", [])
    column_types = db_entry.get("column_types", [])
    column_descriptions = db_entry.get("column_descriptions", [])
    sample_rows = db_entry.get("sample_rows", {})
    key_metadata_available = has_explicit_key_metadata(dataset_name, db_entry)
    primary_key_indices = flatten_primary_keys(db_entry.get("primary_keys", []))
    foreign_key_map = build_foreign_key_map(db_entry.get("foreign_keys", []))

    records: list[dict[str, Any]] = []
    for column_idx, column_info in enumerate(column_names):
        if not isinstance(column_info, list) or len(column_info) != 2:
            continue

        table_idx, column_name = column_info
        if table_idx == -1 or column_name == "*":
            continue

        table_name = clean_text(get_indexed_value(table_names, table_idx, ""))
        normalized_column_name = clean_text(column_name)
        if not table_name or not normalized_column_name:
            continue

        data_type = clean_text(get_indexed_value(column_types, column_idx, "")) or "NOT_AVAILABLE"
        description = clean_text(get_indexed_value(column_descriptions, column_idx, "")) or "NOT_AVAILABLE"
        example_values = format_example_values(
            build_sample_values(
                sample_rows=sample_rows,
                table_name=table_name,
                column_name=normalized_column_name,
                limit=5,
                deduplicate=True,
            )
        )

        if key_metadata_available:
            if column_idx in primary_key_indices:
                key_type = "PRIMARY KEY"
                referenced_column = "NONE"
            elif column_idx in foreign_key_map:
                key_type = "FOREIGN KEY"
                referenced_column = build_foreign_key_text(db_entry, column_idx, foreign_key_map) or "NONE"
            else:
                key_type = "NONE"
                referenced_column = "NONE"
        else:
            key_type = "NOT_AVAILABLE"
            referenced_column = "NOT_AVAILABLE"

        records.append(
            {
                "table_name": table_name,
                "column_name": normalized_column_name,
                "data_type": data_type,
                "key_type": key_type,
                "referenced_column": referenced_column,
                "column_description": description,
                "example_values": example_values,
            }
        )

    return pd.DataFrame(records, columns=SCHEMA_DF_COLUMNS)


def load_schema_dataframe_from_db_info(
    predict_db_id: str,
    dataset_name: str,
    db_info_index: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    db_entry = db_info_index.get(predict_db_id)
    if db_entry is None:
        raise FileNotFoundError(f"Could not find schema for database '{predict_db_id}' in db_info.json.")
    return build_schema_dataframe_from_db_info(db_entry=db_entry, dataset_name=dataset_name)


def ordered_table_names(schema_df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(str(table_name) for table_name in schema_df["table_name"].tolist()))


def build_column_key(row: pd.Series) -> tuple[str, str]:
    return str(row["table_name"]), str(row["column_name"])


def get_key_columns(schema_df: pd.DataFrame) -> set[tuple[str, str]]:
    key_columns: set[tuple[str, str]] = set()
    for _, row in schema_df.iterrows():
        if row["key_type"] in {"PRIMARY KEY", "FOREIGN KEY"}:
            key_columns.add(build_column_key(row))
    return key_columns


def build_foreign_key_relationship_lines(schema_df: pd.DataFrame) -> list[str]:
    relationships: list[str] = []
    seen: set[str] = set()

    for _, row in schema_df.iterrows():
        referenced_column = row["referenced_column"]
        if referenced_column in {"NONE", "NOT_AVAILABLE"}:
            continue
        relationship = f"{row['table_name']}.{row['column_name']} -> {referenced_column}"
        if relationship in seen:
            continue
        seen.add(relationship)
        relationships.append(relationship)

    return relationships


def build_db_schema_text(
    schema_df: pd.DataFrame,
    db_id: str,
    include_selected_tables: bool = False,
    selected_column_keys: set[tuple[str, str]] | None = None,
    include_empty_tables: bool = False,
) -> str:
    lines = [
        "[DATABASE_ID]",
        db_id,
        "",
    ]

    if include_selected_tables:
        lines.append("[SELECTED_TABLES]")
        for table_name in ordered_table_names(schema_df):
            lines.append(f"- {table_name}")
        lines.append("")

    foreign_key_section_name = (
        "[FOREIGN_KEY_RELATIONSHIPS_WITHIN_SELECTED_TABLES]"
        if include_selected_tables
        else "[FOREIGN_KEY_RELATIONSHIPS]"
    )
    lines.append(foreign_key_section_name)

    relationships = build_foreign_key_relationship_lines(schema_df)
    if relationships:
        for relationship in relationships:
            lines.append(f"- {relationship}")
    elif not schema_df.empty and (schema_df["referenced_column"] == "NOT_AVAILABLE").all():
        lines.append("NOT_AVAILABLE")
    else:
        lines.append("NONE")

    for table_name in ordered_table_names(schema_df):
        table_rows = schema_df[schema_df["table_name"] == table_name]
        if selected_column_keys is not None:
            table_rows = table_rows[
                table_rows.apply(lambda row: build_column_key(row) in selected_column_keys, axis=1)
            ]
        if table_rows.empty and not include_empty_tables:
            continue

        lines.extend(["", f"[TABLE] {table_name}"])
        for _, row in table_rows.iterrows():
            lines.append(
                "- "
                f"{row['column_name']} | {row['data_type']} | {row['key_type']} | "
                f"{row['referenced_column']} | {row['column_description']} | {row['example_values']}"
            )

    return "\n".join(lines)


def get_qdrant_client(qdrant_path: Path) -> Any:
    from qdrant_client import QdrantClient

    if not qdrant_path.is_dir():
        raise FileNotFoundError(f"Could not find Qdrant index directory at {qdrant_path}.")

    return QdrantClient(path=str(qdrant_path))


def build_db_id_filter(candidate_db_ids: list[str] | None) -> Any:
    if candidate_db_ids is None:
        return None

    normalized_db_ids = [db_id for db_id in candidate_db_ids if db_id]
    if not normalized_db_ids:
        return None

    from qdrant_client.http import models

    return models.Filter(
        should=[
            models.FieldCondition(
                key="db_id",
                match=models.MatchValue(value=db_id),
            )
            for db_id in normalized_db_ids
        ]
    )


def query_qdrant(
    client: Any,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    query_filter: Any = None,
    with_vectors: bool = False,
) -> list[Any]:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=with_vectors,
    )

    if hasattr(response, "points"):
        return list(response.points)
    if isinstance(response, list):
        return response

    raise TypeError(f"Unsupported Qdrant query result type: {type(response)!r}")


def resolve_prompt_token_cap(
    max_input_length: int,
    prompt_budget_buffer: int = 512,
    prompt_budget_ratio: float = 0.85,
) -> int:
    hard_cap = max(1, max_input_length - prompt_budget_buffer)
    soft_cap = max(1, int(max_input_length * prompt_budget_ratio))
    return max(1, min(hard_cap, soft_cap))
