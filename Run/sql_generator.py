from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from config import *
from Llm.llm_loader import LLM
from Run.logging_utils import log_run_configuration, setup_task_logger
from Utils.efficiency_utils import SampleEfficiencyTracker
from Utils.render_tools import SchemaTextRenderer
from Utils.schema_selection import DbInfoSchemaStore, build_column_id, is_truthy_flag
from Utils.tools import (
    get_row_value,
    load_db_info_index,
    normalize_response_text,
    render_prompt,
    resolve_hint,
    resolve_prompt_token_cap,
)


DEFAULT_SQL_LLM_NAME = "mistralai/Ministral-3-14B-Instruct-2512"
DEFAULT_SQL_PROMPT_PATH = PROJECT_ROOT / "Templates" / "sql_generation.txt"
DEFAULT_SQL_GENERATION_NUM = 4096
SUPPORTED_SCHEMA_METHODS = {"zero_shot", "few_shot"}
SUPPORTED_SCHEMA_TASKS = {"baseline_schema_linking", "table2column"}
SQL_GENERATION_QUERY_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.02,
}
SCHEMA_RESULT_PATTERN_TEMPLATE = (
    r"(?P<method>zero_shot|few_shot)_(?P<task>baseline_schema_linking|table2column)_"
    r"{dataset_name}_(?P<timestamp>\d{{8}}_\d{{6}})\.json$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SQL from schema-linking prediction logs."
    )
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--answer-llm-name", dest="answer_llm_name", type=str, default=None)
    parser.add_argument("--provider", dest="provider", type=str, default=None)
    parser.add_argument("--max-input-length", dest="max_input_length", type=int, default=None)
    parser.add_argument("--max-generation-num", dest="max_generation_num", type=int, default=None)
    parser.add_argument("--input-path", dest="input_path", type=Path, default=None)
    parser.add_argument("--schema-llm-name", dest="schema_llm_name", type=str, default=None)
    parser.add_argument(
        "--schema-method",
        dest="schema_method",
        choices=("auto", *sorted(SUPPORTED_SCHEMA_METHODS)),
        default="auto",
    )
    parser.add_argument(
        "--schema-task",
        dest="schema_task",
        choices=("auto", *sorted(SUPPORTED_SCHEMA_TASKS)),
        default="auto",
    )
    parser.add_argument("--logs-dir", dest="logs_dir", type=Path, default=None)
    parser.add_argument("--db-info-path", dest="db_info_path", type=Path, default=None)
    parser.add_argument("--dataset-path", dest="dataset_path", type=Path, default=None)
    parser.add_argument("--documents-dir", dest="documents_dir", type=Path, default=None)
    parser.add_argument("--prompt-path", dest="prompt_path", type=Path, default=None)
    parser.add_argument("--output-path", dest="output_path", type=Path, default=None)
    parser.add_argument("--sql-dialect", dest="sql_dialect", type=str, default=None)
    parser.add_argument("--start-index", dest="start_index", type=int, default=0)
    parser.add_argument("--limit", dest="limit", type=int, default=None)
    parser.add_argument(
        "--include-key-columns",
        dest="include_key_columns",
        action="store_true",
        help="Add primary-key and foreign-key columns from predicted tables to the rendered schema excerpt.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_log_records(input_path: Path) -> list[dict[str, Any]]:
    records = load_json(input_path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of log records in {input_path}.")
    return [record for record in records if isinstance(record, dict)]


def load_dataset_index(dataset_path: Path) -> dict[str, dict[str, Any]]:
    if not dataset_path.is_file():
        return {}

    rows = load_json(dataset_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of dataset rows in {dataset_path}.")

    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id = get_row_value(row, "id", "instance_id")
        if sample_id is None:
            continue
        index[str(sample_id)] = row
    return index


def resolve_schema_result_regex(dataset_name: str) -> re.Pattern[str]:
    return re.compile(
        SCHEMA_RESULT_PATTERN_TEMPLATE.format(dataset_name=re.escape(dataset_name))
    )


def find_model_log_root(logs_dir: Path, model_name: str) -> Path:
    direct_path = logs_dir / model_name
    if direct_path.is_dir():
        return direct_path

    model_leaf_name = Path(model_name).name
    matching_dirs = sorted(
        path
        for path in logs_dir.rglob(model_leaf_name)
        if path.is_dir() and path.name == model_leaf_name
    )
    if not matching_dirs:
        raise FileNotFoundError(
            f"Could not find logs for model '{model_name}' under {logs_dir}."
        )
    if len(matching_dirs) > 1:
        matched_paths = "\n".join(str(path) for path in matching_dirs)
        raise ValueError(
            f"Found multiple log directories for model '{model_name}'. Please disambiguate:\n{matched_paths}"
        )
    return matching_dirs[0]


def find_latest_schema_result_file(
    logs_dir: Path,
    schema_llm_name: str,
    dataset_name: str,
    schema_method: str,
    schema_task: str,
) -> Path:
    model_root = find_model_log_root(logs_dir=logs_dir, model_name=schema_llm_name)
    result_regex = resolve_schema_result_regex(dataset_name)
    candidates: list[tuple[str, Path]] = []

    for path in model_root.glob(f"*_{dataset_name}_*.json"):
        match = result_regex.match(path.name)
        if match is None:
            continue
        if schema_method != "auto" and match.group("method") != schema_method:
            continue
        if schema_task != "auto" and match.group("task") != schema_task:
            continue
        candidates.append((match.group("timestamp"), path))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find schema prediction logs for dataset={dataset_name}, "
            f"schema_method={schema_method}, schema_task={schema_task} under {model_root}."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def resolve_input_path(
    input_path: Path | None,
    logs_dir: Path,
    schema_llm_name: str,
    dataset_name: str,
    schema_method: str,
    schema_task: str,
) -> Path:
    if input_path is not None:
        return input_path

    return find_latest_schema_result_file(
        logs_dir=logs_dir,
        schema_llm_name=schema_llm_name,
        dataset_name=dataset_name,
        schema_method=schema_method,
        schema_task=schema_task,
    )


def resolve_output_path(
    output_path: Path | None,
    logs_dir: Path,
    dataset_name: str,
) -> Path:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = logs_dir / "sql_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"sql_generation_{dataset_name}_{run_id}.json"


def default_sql_dialect(dataset_name: str) -> str:
    if dataset_name.lower() == "mmqa":
        return (
            "Use SQLite SQL. Do not use Snowflake-only features such as QUALIFY, ILIKE, "
            "TRY_CAST, DATEADD, DATEDIFF, TO_DATE, :: casts, or fully qualified "
            "DATABASE.SCHEMA.TABLE notation unless the table name is explicitly shown that way. "
            "Use SQLite-compatible functions such as strftime/date/datetime when date logic is needed."
        )
    if dataset_name.lower() == "spider2":
        return (
            "Use Snowflake SQL. Preserve fully qualified table names exactly as shown, usually "
            "DATABASE.SCHEMA.TABLE. Snowflake features such as CTEs, QUALIFY, ILIKE, DATEADD, "
            "DATEDIFF, TRY_CAST, TO_DATE, TRUE/FALSE boolean literals, and :: casts are allowed "
            "when useful. Do not write SQLite-specific SQL."
        )
    return "Use the dialect implied by the question, schema, and hint."


def normalize_predicted_columns(value: Any) -> dict[str, list[str]]:
    if isinstance(value, str):
        try:
            value = json.loads(normalize_response_text(value))
        except json.JSONDecodeError:
            return {}

    if isinstance(value, Mapping) and "relevant_columns" in value:
        value = value.get("relevant_columns")

    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, list[str]] = {}
    for table_name, column_names in value.items():
        normalized_table_name = str(table_name).strip()
        if not normalized_table_name:
            continue

        if not isinstance(column_names, Sequence) or isinstance(column_names, (str, bytes)):
            continue

        seen: set[str] = set()
        normalized_columns: list[str] = []
        for column_name in column_names:
            normalized_column_name = str(column_name).strip()
            if not normalized_column_name or normalized_column_name in seen:
                continue
            seen.add(normalized_column_name)
            normalized_columns.append(normalized_column_name)

        if normalized_columns:
            normalized[normalized_table_name] = normalized_columns
    return normalized


def parse_predicted_columns_from_text(response_text: Any) -> dict[str, list[str]]:
    if not isinstance(response_text, str) or not response_text.strip():
        return {}
    return normalize_predicted_columns(response_text)


def resolve_predicted_columns(row: Mapping[str, Any]) -> dict[str, list[str]]:
    predicted_columns = normalize_predicted_columns(row.get("predict_columns"))
    if predicted_columns:
        return predicted_columns
    return parse_predicted_columns_from_text(row.get("predict_columns_text"))


def normalize_predicted_tables(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(normalize_response_text(value))
        except json.JSONDecodeError:
            return []

    if isinstance(value, Mapping) and "relevant_tables" in value:
        value = value.get("relevant_tables")

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for table_name in value:
        normalized_table_name = str(table_name).strip()
        if not normalized_table_name or normalized_table_name in seen:
            continue
        seen.add(normalized_table_name)
        normalized.append(normalized_table_name)
    return normalized


def resolve_predicted_tables(row: Mapping[str, Any]) -> list[str]:
    predicted_tables = normalize_predicted_tables(row.get("predict_tables"))
    if predicted_tables:
        return predicted_tables
    return normalize_predicted_tables(row.get("predict_tables_text"))


def select_predicted_column_records(
    db_id: str,
    predicted_columns: dict[str, list[str]],
    predicted_tables: Sequence[str],
    schema_store: DbInfoSchemaStore,
    include_key_columns: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_records = schema_store.get_column_records(db_id)
    predicted_table_set = set(predicted_columns)
    predicted_table_set.update(str(table_name).strip() for table_name in predicted_tables)
    selected_record_ids: set[str] = set()

    if predicted_columns:
        predicted_column_sets = {
            table_name: set(column_names)
            for table_name, column_names in predicted_columns.items()
        }

        for record in all_records:
            table_name = str(record.get("table_name", "")).strip()
            column_name = str(record.get("column_name", "")).strip()
            if not table_name or not column_name:
                continue
            column_names = predicted_column_sets.get(table_name)
            if column_names is None:
                continue
            if "*" in column_names or column_name in column_names:
                selected_record_ids.add(build_column_id(record))
    elif predicted_table_set:
        for record in all_records:
            table_name = str(record.get("table_name", "")).strip()
            if table_name in predicted_table_set:
                selected_record_ids.add(build_column_id(record))

    if include_key_columns and predicted_table_set:
        for record in all_records:
            table_name = str(record.get("table_name", "")).strip()
            if table_name not in predicted_table_set:
                continue
            if is_truthy_flag(record.get("is_primary_key")) or is_truthy_flag(record.get("is_foreign_key")):
                selected_record_ids.add(build_column_id(record))

    selected_records = [
        record
        for record in all_records
        if build_column_id(record) in selected_record_ids
    ]
    metadata = {
        "available_column_count": len(all_records),
        "selected_column_count": len(selected_records),
        "predicted_table_count": len(predicted_table_set),
    }
    return selected_records, metadata


def render_sql_prompt(
    prompt_template: str,
    schema_text: str,
    question: str,
    hint: str,
    dataset_name: str,
    sql_dialect: str,
) -> str:
    return render_prompt(
        prompt_template,
        DATABASE_SCHEMAS=schema_text,
        QUESTION=question,
        HINT=hint,
        DATASET_NAME=dataset_name,
        SQL_DIALECT=sql_dialect,
    )


def fit_prompt_to_budget(
    prompt_template: str,
    schema_text: str,
    question: str,
    hint: str,
    dataset_name: str,
    sql_dialect: str,
    answer_llm: LLM,
    renderer: SchemaTextRenderer,
) -> tuple[str, str, str, int]:
    target_prompt_cap = resolve_prompt_token_cap(answer_llm.max_input_length)
    prompt = render_sql_prompt(
        prompt_template=prompt_template,
        schema_text=schema_text,
        question=question,
        hint=hint,
        dataset_name=dataset_name,
        sql_dialect=sql_dialect,
    )
    prompt_tokens = answer_llm.count_input_tokens(prompt)
    if prompt_tokens <= target_prompt_cap:
        return prompt, schema_text, hint, prompt_tokens

    fitted_hint = hint
    if hint != "No hint":
        prompt_without_hint = render_sql_prompt(
            prompt_template=prompt_template,
            schema_text=schema_text,
            question=question,
            hint="",
            dataset_name=dataset_name,
            sql_dialect=sql_dialect,
        )
        tokens_without_hint = answer_llm.count_input_tokens(prompt_without_hint)
        hint_budget = max(0, target_prompt_cap - tokens_without_hint)
        fitted_hint = renderer.truncate_to_token_budget(hint, hint_budget)
        prompt = render_sql_prompt(
            prompt_template=prompt_template,
            schema_text=schema_text,
            question=question,
            hint=fitted_hint or "No hint",
            dataset_name=dataset_name,
            sql_dialect=sql_dialect,
        )
        prompt_tokens = answer_llm.count_input_tokens(prompt)
        if prompt_tokens <= target_prompt_cap:
            return prompt, schema_text, fitted_hint or "No hint", prompt_tokens

    prompt_without_schema = render_sql_prompt(
        prompt_template=prompt_template,
        schema_text="",
        question=question,
        hint=fitted_hint,
        dataset_name=dataset_name,
        sql_dialect=sql_dialect,
    )
    tokens_without_schema = answer_llm.count_input_tokens(prompt_without_schema)
    schema_budget = max(0, target_prompt_cap - tokens_without_schema)
    fitted_schema_text = renderer.truncate_to_token_budget(schema_text, schema_budget)
    prompt = render_sql_prompt(
        prompt_template=prompt_template,
        schema_text=fitted_schema_text,
        question=question,
        hint=fitted_hint,
        dataset_name=dataset_name,
        sql_dialect=sql_dialect,
    )
    prompt_tokens = answer_llm.count_input_tokens(prompt)
    return prompt, fitted_schema_text, fitted_hint, prompt_tokens


def normalize_sql_response(response_text: str) -> str:
    text = response_text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    fenced_match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match is not None:
        text = fenced_match.group(1).strip()
    else:
        text = text.replace("```", "").strip()

    try:
        response_json = json.loads(text)
    except json.JSONDecodeError:
        response_json = None
    if isinstance(response_json, Mapping):
        sql_value = response_json.get("sql")
        if isinstance(sql_value, str):
            text = sql_value.strip()

    text = re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def write_result_file(
    output_path: Path,
    run_info: dict[str, Any],
    result_records: list[dict[str, Any]],
) -> None:
    output_path.write_text(
        json.dumps(
            {
                "run_info": run_info,
                "results": result_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def append_log_entry(
    result_records: list[dict[str, Any]],
    output_path: Path,
    *,
    run_info: dict[str, Any],
    row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    predict_db_id: str | None,
    predicted_columns: dict[str, list[str]],
    predicted_tables: list[str],
    sql_response_text: str,
    prompt_tokens: int,
    schema_metadata: dict[str, Any],
    efficiency_tracker: SampleEfficiencyTracker,
) -> None:
    efficiency = efficiency_tracker.finalize()
    sample_id = get_row_value(row, "id", "instance_id")
    if sample_id is None:
        sample_id = get_row_value(source_row, "id", "instance_id")

    normalized_sql = normalize_sql_response(sql_response_text)
    error_message = schema_metadata.get("error")
    result_record: dict[str, Any] = {
        "id": None if sample_id is None else str(sample_id),
        "question": row.get("question") or source_row.get("question"),
        "gold_db_id": source_row.get("db_id") or row.get("gold_db_id") or row.get("spider_db_id"),
        "predict_db_id": predict_db_id,
        "schema_linking": {
            "predict_tables": predicted_tables,
            "predict_columns": predicted_columns,
        },
        "predict_sql": normalized_sql,
        "status": "failed" if error_message else ("success" if normalized_sql else "empty"),
        "efficiency": {
            **efficiency,
            "prompt_tokens": prompt_tokens,
        },
    }
    if error_message:
        result_record["error"] = error_message
    if sql_response_text.strip() and sql_response_text.strip() != normalized_sql:
        result_record["raw_response"] = sql_response_text
    if schema_metadata.get("hint_truncated") or schema_metadata.get("schema_truncated"):
        result_record["truncation"] = {
            "hint_truncated": bool(schema_metadata.get("hint_truncated")),
            "schema_truncated": bool(schema_metadata.get("schema_truncated")),
        }

    result_records.append(result_record)
    write_result_file(output_path, run_info, result_records)


def run_sql_generation(
    schema_log_records: Sequence[dict[str, Any]],
    dataset_index: dict[str, dict[str, Any]],
    dataset_name: str,
    documents_dir: Path,
    prompt_template: str,
    output_path: Path,
    schema_store: DbInfoSchemaStore,
    renderer: SchemaTextRenderer,
    answer_llm: LLM,
    run_info: dict[str, Any],
    sql_dialect: str,
    start_index: int,
    limit: int | None,
    include_key_columns: bool,
) -> int:
    from tqdm import tqdm

    selected_rows = list(schema_log_records[max(0, start_index):])
    if limit is not None:
        selected_rows = selected_rows[: max(0, limit)]

    result_records: list[dict[str, Any]] = []

    for row in tqdm(selected_rows, total=len(selected_rows)):
        efficiency_tracker = SampleEfficiencyTracker()
        sample_id = get_row_value(row, "id", "instance_id")
        source_row = dataset_index.get(str(sample_id), {}) if sample_id is not None else {}
        question = str(row.get("question") or source_row.get("question") or "").strip()
        predict_db_id_value = get_row_value(row, "predict_db_id")
        predict_db_id = None if predict_db_id_value is None else str(predict_db_id_value).strip()
        predicted_columns = resolve_predicted_columns(row)
        predicted_tables = resolve_predicted_tables(row)
        hint = resolve_hint(
            source_row or row,
            dataset_name=dataset_name,
            documents_dir=documents_dir,
        )

        if not question or not predict_db_id:
            append_log_entry(
                result_records=result_records,
                output_path=output_path,
                run_info=run_info,
                row=row,
                source_row=source_row,
                predict_db_id=predict_db_id,
                predicted_columns=predicted_columns,
                predicted_tables=predicted_tables,
                sql_response_text="",
                prompt_tokens=0,
                schema_metadata={"error": "Missing question or predicted database."},
                efficiency_tracker=efficiency_tracker,
            )
            continue

        selected_records, schema_metadata = select_predicted_column_records(
            db_id=predict_db_id,
            predicted_columns=predicted_columns,
            predicted_tables=predicted_tables,
            schema_store=schema_store,
            include_key_columns=include_key_columns,
        )
        if not selected_records:
            append_log_entry(
                result_records=result_records,
                output_path=output_path,
                run_info=run_info,
                row=row,
                source_row=source_row,
                predict_db_id=predict_db_id,
                predicted_columns=predicted_columns,
                predicted_tables=predicted_tables,
                sql_response_text="",
                prompt_tokens=0,
                schema_metadata={**schema_metadata, "error": "No selected schema columns."},
                efficiency_tracker=efficiency_tracker,
            )
            continue

        schema_text = schema_store.render_schema_text(predict_db_id, selected_records)
        prompt, fitted_schema_text, fitted_hint, prompt_tokens = fit_prompt_to_budget(
            prompt_template=prompt_template,
            schema_text=schema_text,
            question=question,
            hint=hint,
            dataset_name=dataset_name,
            sql_dialect=sql_dialect,
            answer_llm=answer_llm,
            renderer=renderer,
        )
        schema_metadata = {
            **schema_metadata,
            "prompt_tokens": prompt_tokens,
            "hint_truncated": fitted_hint != hint,
            "schema_truncated": fitted_schema_text != schema_text,
        }

        sql_response_text, total_tokens = answer_llm.query_with_usage(prompt)
        efficiency_tracker.add_llm_total_tokens(total_tokens)
        append_log_entry(
            result_records=result_records,
            output_path=output_path,
            run_info=run_info,
            row=row,
            source_row=source_row,
            predict_db_id=predict_db_id,
            predicted_columns=predicted_columns,
            predicted_tables=predicted_tables,
            sql_response_text=sql_response_text,
            prompt_tokens=prompt_tokens,
            schema_metadata=schema_metadata,
            efficiency_tracker=efficiency_tracker,
        )

    return len(result_records)


def main() -> None:
    args = parse_args()

    dataset_name = args.dataset_name or DATASET_NAME
    answer_llm_name = args.answer_llm_name or DEFAULT_SQL_LLM_NAME
    provider = args.provider or PROVIDER
    max_input_length = args.max_input_length or MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or max(MAX_GENERATEION_NUM, DEFAULT_SQL_GENERATION_NUM)
    logs_dir = args.logs_dir or (PROJECT_ROOT / "Logs")
    schema_llm_name = args.schema_llm_name or answer_llm_name
    dataset_root = PROJECT_ROOT / "Data" / dataset_name
    dataset_path = args.dataset_path or (dataset_root / "gold_sl.json")
    db_info_path = args.db_info_path or (dataset_root / "db_info.json")
    documents_dir = args.documents_dir or (dataset_root / "documents")
    prompt_path = args.prompt_path or DEFAULT_SQL_PROMPT_PATH
    sql_dialect = args.sql_dialect or default_sql_dialect(dataset_name)

    input_path = resolve_input_path(
        input_path=args.input_path,
        logs_dir=logs_dir,
        schema_llm_name=schema_llm_name,
        dataset_name=dataset_name,
        schema_method=args.schema_method,
        schema_task=args.schema_task,
    )
    output_path = resolve_output_path(
        output_path=args.output_path,
        logs_dir=logs_dir,
        dataset_name=dataset_name,
    )
    logger, logger_path = setup_task_logger("sql_generation", output_path)

    schema_log_records = load_log_records(input_path)
    dataset_index = load_dataset_index(dataset_path)
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    db_info_index = load_db_info_index(db_info_path)
    run_info = {
        "task": "sql_generation",
        "dataset_name": dataset_name,
        "model": answer_llm_name,
        "provider": provider,
        "schema_source_model": schema_llm_name,
        "schema_input_path": str(input_path),
        "prompt_template": str(prompt_path),
        "dataset_path": str(dataset_path),
        "db_info_path": str(db_info_path),
        "documents_dir": str(documents_dir),
        "sql_dialect": sql_dialect,
        "max_input_length": max_input_length,
        "max_generation_num": max_generation_num,
        "start_index": args.start_index,
        "limit": args.limit,
        "include_key_columns": args.include_key_columns,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    log_run_configuration(
        logger,
        task_name="SQL Generation",
        dataset_name=dataset_name,
        data_count=len(schema_log_records),
        model_name=answer_llm_name,
        provider=provider,
        result_path=output_path,
        extra_fields={
            "Schema input path": input_path,
            "Schema source model": schema_llm_name,
            "Schema method": args.schema_method,
            "Schema task": args.schema_task,
            "Prompt template": prompt_path,
            "Dataset path": dataset_path,
            "DB info path": db_info_path,
            "Documents dir": documents_dir,
            "SQL dialect": sql_dialect,
            "Max input length": max_input_length,
            "Max generation num": max_generation_num,
            "Start index": args.start_index,
            "Limit": args.limit,
            "Include key columns": args.include_key_columns,
            "Logger path": logger_path,
        },
    )

    answer_llm = LLM(
        model_name=answer_llm_name,
        provider=provider,
        max_input_length=max_input_length,
        max_generation_num=max_generation_num,
        query_settings=SQL_GENERATION_QUERY_SETTINGS,
    )
    renderer = SchemaTextRenderer(tokenizer=answer_llm.tokenizer)
    schema_store = DbInfoSchemaStore(
        db_info_index=db_info_index,
        renderer=renderer,
    )

    processed_count = run_sql_generation(
        schema_log_records=schema_log_records,
        dataset_index=dataset_index,
        dataset_name=dataset_name,
        documents_dir=documents_dir,
        prompt_template=prompt_template,
        output_path=output_path,
        schema_store=schema_store,
        renderer=renderer,
        answer_llm=answer_llm,
        run_info=run_info,
        sql_dialect=sql_dialect,
        start_index=args.start_index,
        limit=args.limit,
        include_key_columns=args.include_key_columns,
    )
    logger.info("Completed %s records.", processed_count)


if __name__ == "__main__":
    main()
