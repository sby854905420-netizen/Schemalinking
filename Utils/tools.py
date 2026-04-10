from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_DF_COLUMNS = [
    "table_name",
    "column_name",
    "data_type",
    "key_type",
    "referenced_column",
    "column_description",
    "example_values",
]


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
        db_id = entry.get("db_id")
        if isinstance(db_id, str) and db_id.strip():
            index[db_id] = entry
    return index


def extract_timestamp(path: Path, dataset_name: str, timestamp_pattern_template: str) -> str | None:
    timestamp_pattern = re.compile(
        timestamp_pattern_template.format(dataset_name=re.escape(dataset_name))
    )
    match = timestamp_pattern.search(path.name)
    if match is None:
        return None
    return match.group(1)


def find_model_dir(logs_dir: Path, model_name: str, task_dir_name: str = "Database_Retrival") -> Path:
    direct_path = logs_dir / model_name / task_dir_name
    if direct_path.is_dir():
        return direct_path

    model_leaf_name = Path(model_name).name
    matching_dirs = sorted(
        path
        for path in logs_dir.rglob(task_dir_name)
        if path.is_dir() and path.parent.name == model_leaf_name
    )
    if not matching_dirs:
        raise FileNotFoundError(
            f"Could not find a {task_dir_name} directory for model '{model_name}' under {logs_dir}."
        )
    if len(matching_dirs) > 1:
        matched_paths = "\n".join(str(path) for path in matching_dirs)
        raise ValueError(
            f"Found multiple {task_dir_name} directories for model '{model_name}'. Please disambiguate:\n{matched_paths}"
        )
    return matching_dirs[0]


def find_result_file(
    model_dir: Path,
    dataset_name: str,
    input_file_patterns: tuple[str, ...],
    timestamp_pattern_template: str,
) -> Path:
    candidate_files: list[tuple[str, Path]] = []

    for pattern_template in input_file_patterns:
        pattern = pattern_template.format(dataset_name=dataset_name, timestamp="*")
        for path in model_dir.glob(pattern):
            file_timestamp = extract_timestamp(path, dataset_name, timestamp_pattern_template)
            if file_timestamp is None:
                continue
            candidate_files.append((file_timestamp, path))

    if not candidate_files:
        expected_patterns = ", ".join(
            pattern_template.format(dataset_name=dataset_name, timestamp="*")
            for pattern_template in input_file_patterns
        )
        raise FileNotFoundError(f"Could not find any of [{expected_patterns}] under {model_dir}.")

    candidate_files.sort(key=lambda item: item[0], reverse=True)
    return candidate_files[0][1]


def resolve_input_path(
    input_path: Path | None,
    logs_dir: Path,
    answer_llm_name: str,
    dataset_name: str,
    input_file_patterns: tuple[str, ...],
    timestamp_pattern_template: str,
    task_dir_name: str = "Database_Retrival",
) -> Path:
    if input_path is not None:
        return input_path

    model_dir = find_model_dir(logs_dir=logs_dir, model_name=answer_llm_name, task_dir_name=task_dir_name)
    return find_result_file(
        model_dir=model_dir,
        dataset_name=dataset_name,
        input_file_patterns=input_file_patterns,
        timestamp_pattern_template=timestamp_pattern_template,
    )


def resolve_output_path(
    output_path: Path | None,
    answer_llm_name: str,
    dataset_name: str,
    output_stem: str,
    project_root: Path,
) -> Path:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / "Logs" / answer_llm_name
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"{output_stem}_{dataset_name}_{run_id}.json"


def get_row_value(row: pd.Series, *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if pd.notna(value):
            return value
    return None


def resolve_hint(source: Any, key: str = "external_knowledge") -> str:
    if hasattr(source, "get"):
        value = source.get(key)
    else:
        value = source

    if value is None:
        return "No hint"

    if isinstance(value, str):
        hint_text = value.strip()
    elif isinstance(value, (dict, list)):
        hint_text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        try:
            if pd.isna(value):
                return "No hint"
        except (TypeError, ValueError):
            pass
        hint_text = str(value).strip()

    if hint_text == "":
        return "No hint"
    return hint_text


def render_prompt(prompt_template: str, **replacements: Any) -> str:
    prompt = prompt_template
    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{key}}}", "" if value is None else str(value))
    return prompt


def normalize_response_text(response_text: str) -> str:
    normalized_response = response_text.replace("```", "").replace("json", "").strip()
    if "</think>" in normalized_response:
        normalized_response = normalized_response.split("</think>")[-1].strip()
    return normalized_response


def build_sample_values(
    sample_rows: dict[str, list[dict[str, Any]]],
    table_name: str,
    column_name: str,
    limit: int = 5,
) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()

    for row in sample_rows.get(table_name, []):
        if column_name not in row:
            continue

        value = row[column_name]
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue

        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue

        seen.add(key)
        values.append(value)
        if len(values) >= limit:
            break

    return values


def flatten_primary_keys(primary_keys: list[Any]) -> set[int]:
    primary_key_indices: set[int] = set()
    for entry in primary_keys:
        if isinstance(entry, list):
            primary_key_indices.update(entry)
        else:
            primary_key_indices.add(entry)
    return primary_key_indices


def build_foreign_key_map(foreign_keys: list[list[int]]) -> dict[int, int]:
    foreign_key_map: dict[int, int] = {}
    for entry in foreign_keys:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        source_idx, target_idx = entry
        foreign_key_map[source_idx] = target_idx
    return foreign_key_map


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

        table_name = table_names[table_idx]
        data_type = column_types[column_idx] if column_idx < len(column_types) else "NOT_AVAILABLE"
        description = (
            column_descriptions[column_idx]
            if column_idx < len(column_descriptions) and column_descriptions[column_idx]
            else "NOT_AVAILABLE"
        )
        example_values = format_example_values(
            build_sample_values(sample_rows=sample_rows, table_name=table_name, column_name=column_name)
        )

        if key_metadata_available:
            if column_idx in primary_key_indices:
                key_type = "PRIMARY KEY"
                referenced_column = "NONE"
            elif column_idx in foreign_key_map:
                target_idx = foreign_key_map[column_idx]
                target_table_idx, target_column_name = column_names[target_idx]
                key_type = "FOREIGN KEY"
                referenced_column = f"{table_names[target_table_idx]}.{target_column_name}"
            else:
                key_type = "NONE"
                referenced_column = "NONE"
        else:
            key_type = "NOT_AVAILABLE"
            referenced_column = "NOT_AVAILABLE"

        records.append(
            {
                "table_name": table_name,
                "column_name": column_name,
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
