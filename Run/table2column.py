from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import *
from Llm.llm_loader import LLM
from Run.logging_utils import log_run_configuration, setup_task_logger
from Utils.efficiency_utils import SampleEfficiencyTracker
from Utils.render_tools import SchemaTextRenderer
from Utils.schema_selection import (
    DbInfoSchemaStore,
    count_prompt_tokens as count_schema_prompt_tokens,
    load_db_counts,
    resolve_schema_text_for_db,
)
from Utils.tools import (
    get_qdrant_client,
    get_row_value,
    load_db_info_index,
    normalize_response_text,
    render_prompt,
    resolve_hint,
    resolve_input_path,
    resolve_output_path,
    resolve_prompt_token_cap,
)

SUPPORTED_METHODS = {"zero_shot", "few_shot"}
INPUT_FILE_PATTERNS = (
    "baseline_database_retrival_{dataset_name}_{timestamp}.json",
    "iterative_database_retrival_{dataset_name}_{timestamp}.json",
)
TIMESTAMP_PATTERN_TEMPLATE = (
    r"(?:baseline_database_retrival|iterative_database_retrival)_{dataset_name}_(\d{{8}}_\d{{6}})\.json$"
)
DEFAULT_METHOD = "few_shot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table-to-column schema linking.")
    parser.add_argument("--method", dest="method", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--answer-llm-name", dest="answer_llm_name", type=str, default=None)
    parser.add_argument("--provider", dest="provider", type=str, default=None)
    parser.add_argument("--max-input-length", dest="max_input_length", type=int, default=None)
    parser.add_argument("--max-generation-num", dest="max_generation_num", type=int, default=None)
    parser.add_argument("--input-path", dest="input_path", type=Path, default=None)
    parser.add_argument("--logs-dir", dest="logs_dir", type=Path, default=None)
    parser.add_argument("--db-info-path", dest="db_info_path", type=Path, default=None)
    parser.add_argument("--qdrant-path", dest="qdrant_path", type=Path, default=None)
    parser.add_argument("--output-path", dest="output_path", type=Path, default=None)
    return parser.parse_args()


def resolve_method(method: str | None) -> str:
    resolved_method = method or DEFAULT_METHOD
    if resolved_method not in SUPPORTED_METHODS:
        supported_methods = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(f"Unsupported method '{resolved_method}'. Expected one of: {supported_methods}.")
    return resolved_method


def load_dataset(input_path: Path) -> pd.DataFrame:
    return pd.read_json(input_path)


def load_prompt_templates(method_name: str) -> dict[str, str]:
    prompt_dir = PROJECT_ROOT / "Templates" / method_name
    return {
        "table": (prompt_dir / "extract_relevant_tables.txt").read_text(encoding="utf-8").strip(),
        "column": (prompt_dir / "extract_relevant_columns.txt").read_text(encoding="utf-8").strip(),
    }


def load_qdrant_collection_name(qdrant_path: Path) -> str:
    meta_path = qdrant_path / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Could not find Qdrant metadata file at {meta_path}.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    collections = meta.get("collections")
    if not isinstance(collections, dict) or not collections:
        raise ValueError(f"Could not resolve a collection name from {meta_path}.")

    return next(iter(collections))


def normalize_table_names(table_names: Sequence[Any]) -> list[str]:
    normalized_table_names: list[str] = []
    seen: set[str] = set()

    for table_name in table_names:
        normalized_table_name = str(table_name).strip()
        if not normalized_table_name or normalized_table_name in seen:
            continue
        seen.add(normalized_table_name)
        normalized_table_names.append(normalized_table_name)

    return normalized_table_names


def filter_column_records_by_tables(
    column_records: Sequence[dict[str, Any]],
    table_names: Sequence[str],
) -> list[dict[str, Any]]:
    selected_table_names = set(normalize_table_names(table_names))
    if not selected_table_names:
        return []

    return [
        record
        for record in column_records
        if str(record.get("table_name", "")).strip() in selected_table_names
    ]


def normalize_relevant_tables(
    relevant_table_list: Sequence[Any],
    available_records: Sequence[dict[str, Any]],
) -> list[str]:
    available_table_names = {
        str(record.get("table_name", "")).strip()
        for record in available_records
        if str(record.get("table_name", "")).strip()
    }

    return [
        table_name
        for table_name in normalize_table_names(relevant_table_list)
        if table_name in available_table_names
    ]


def render_schema_prompt(
    prompt_template: str,
    schema_text: str,
    question: str,
    hint: str,
) -> str:
    return render_prompt(
        prompt_template,
        DATABASE_SCHEMAS=schema_text,
        QUESTION=question,
        HINT=hint,
    )


def resolve_table_prompt_schema(
    predict_db_id: str,
    question: str,
    hint: str,
    prompt_template: str,
    answer_llm: Any,
    embedder: Any,
    schema_store: DbInfoSchemaStore,
    qdrant_client: Any,
    qdrant_collection_name: str,
    db_counts: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    query_vector = embedder.encode(question, convert_to_list=True)
    return resolve_schema_text_for_db(
        query=question,
        query_vector=query_vector,
        db_id=predict_db_id,
        ranking_llm=answer_llm,
        prompt_template=prompt_template,
        hint_text=hint,
        schema_store=schema_store,
        qdrant_client=qdrant_client,
        collection_name=qdrant_collection_name,
        db_counts=db_counts,
    )


def resolve_column_prompt_records(
    predict_db_id: str,
    question: str,
    hint: str,
    relevant_table_list: Sequence[str],
    table_prompt_records: Sequence[dict[str, Any]],
    schema_store: DbInfoSchemaStore,
    answer_llm: Any,
    prompt_template: str,
) -> list[dict[str, Any]]:
    if not relevant_table_list:
        return list(table_prompt_records)

    full_db_records = schema_store.get_column_records(predict_db_id)
    selected_table_records = filter_column_records_by_tables(full_db_records, relevant_table_list)
    if not selected_table_records:
        fallback_records = filter_column_records_by_tables(table_prompt_records, relevant_table_list)
        return fallback_records or list(table_prompt_records)

    full_schema_text = schema_store.render_schema_text(predict_db_id, selected_table_records)
    prompt_tokens = count_schema_prompt_tokens(
        ranking_llm=answer_llm,
        prompt_template=prompt_template,
        schema_text=full_schema_text,
        query=question,
        hint_text=hint,
    )
    if prompt_tokens <= resolve_prompt_token_cap(answer_llm.max_input_length):
        return selected_table_records

    fallback_records = filter_column_records_by_tables(table_prompt_records, relevant_table_list)
    return fallback_records or list(table_prompt_records) or selected_table_records


def parse_table_response(response_text: str) -> list:
    nor_response_text = normalize_response_text(response_text)
    try:
        response_json = json.loads(nor_response_text)
    except json.JSONDecodeError:
        return []

    if not response_json:
        return []
    
    try:
        relevant_table_list = response_json["relevant_tables"]
    except KeyError:
        return []

    if not isinstance(relevant_table_list, list):
        return []

    return relevant_table_list

def parse_column_response(response_text: str) -> dict:
    nor_response_text = normalize_response_text(response_text)
    try:
        response_json = json.loads(nor_response_text)
    except json.JSONDecodeError:
        return {}

    if not response_json:
        return {}

    try:
        relevant_columns = response_json['relevant_columns']
    except KeyError:
        return {}

    if not isinstance(relevant_columns, dict):
        return {}

    return relevant_columns


def append_log_entry(
    log_records: list[dict[str, Any]],
    row: Any,
    predict_tables:list,
    predict_columns:dict,
    table_response_text: str,
    column_response_text: str,
    efficiency_tracker: SampleEfficiencyTracker,
    answer_llm_name: str,
    provider: str,
    output_path: Path,
) -> None:
    efficiency = efficiency_tracker.finalize()
    log_records.append(
        {
            "model": answer_llm_name,
            "provider": provider,
            "id": f"{get_row_value(row, 'id')}",
            "question": row["question"],
            "spider_db_id": get_row_value(row, "spider_db_id"),
            "predict_db_id": get_row_value(row, "predict_db_id"),
            "predict_tables_text":table_response_text,
            "predict_columns_text":column_response_text,
            "predict_tables": predict_tables,
            "predict_columns": predict_columns,
            "efficiency": efficiency,
        }
    )
    output_path.write_text(json.dumps(log_records, ensure_ascii=False, indent=2), encoding="utf-8")


def run_table2column(
    dataset_df: pd.DataFrame,
    prompt_templates: dict[str, str],
    output_path: Path,
    dataset_name: str,
    documents_dir: Path,
    schema_store: DbInfoSchemaStore,
    db_counts: dict[str, int],
    embedder: Any,
    qdrant_client: Any,
    qdrant_collection_name: str,
    answer_llm: Any,
    answer_llm_name: str,
    provider: str,
) -> int:
    from tqdm import tqdm

    log_records: list[dict[str, Any]] = []

    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        efficiency_tracker = SampleEfficiencyTracker()
        predict_db_id = get_row_value(row, "predict_db_id")
        if predict_db_id is None or str(predict_db_id).strip() == "":
            append_log_entry(
                log_records=log_records,
                row=row,
                predict_tables=[],
                predict_columns={},
                table_response_text="No Valid Database.",
                column_response_text="No Valid Database.",
                efficiency_tracker=efficiency_tracker,
                answer_llm_name=answer_llm_name,
                provider=provider,
                output_path=output_path,
            )
            continue

        predict_db_id = str(predict_db_id)
        question = str(row["question"])
        hint = resolve_hint(
            row,
            dataset_name=dataset_name,
            documents_dir=documents_dir,
        )
        full_db_records = schema_store.get_column_records(predict_db_id)
        if not full_db_records:
            append_log_entry(
                log_records=log_records,
                row=row,
                predict_tables=[],
                predict_columns={},
                table_response_text="No Valid Database.",
                column_response_text="No Valid Database.",
                efficiency_tracker=efficiency_tracker,
                answer_llm_name=answer_llm_name,
                provider=provider,
                output_path=output_path,
            )
            continue

        table_schema_text, table_prompt_records = resolve_table_prompt_schema(
            predict_db_id=predict_db_id,
            question=question,
            hint=hint,
            prompt_template=prompt_templates["table"],
            answer_llm=answer_llm,
            embedder=embedder,
            schema_store=schema_store,
            qdrant_client=qdrant_client,
            qdrant_collection_name=qdrant_collection_name,
            db_counts=db_counts,
        )
        table_prompt = render_schema_prompt(
            prompt_template=prompt_templates["table"],
            schema_text=table_schema_text,
            question=question,
            hint=hint,
        )
        table_response_text, table_total_tokens = answer_llm.query_with_usage(table_prompt)
        efficiency_tracker.add_llm_total_tokens(table_total_tokens)
        relevant_table_list = normalize_relevant_tables(
            parse_table_response(table_response_text),
            full_db_records,
        )

        column_prompt_records = resolve_column_prompt_records(
            predict_db_id=predict_db_id,
            question=question,
            hint=hint,
            relevant_table_list=relevant_table_list,
            table_prompt_records=table_prompt_records,
            schema_store=schema_store,
            answer_llm=answer_llm,
            prompt_template=prompt_templates["column"],
        )
        column_schema_text = schema_store.render_schema_text(
            predict_db_id,
            column_prompt_records,
        )
        column_prompt = render_schema_prompt(
            prompt_template=prompt_templates["column"],
            schema_text=column_schema_text,
            question=question,
            hint=hint,
        )
        column_response_text, column_total_tokens = answer_llm.query_with_usage(column_prompt)
        efficiency_tracker.add_llm_total_tokens(column_total_tokens)
        predict_columns = parse_column_response(column_response_text)
        append_log_entry(
            log_records=log_records,
            row=row,
            predict_tables=relevant_table_list,
            predict_columns=predict_columns,
            table_response_text=table_response_text,
            column_response_text=column_response_text,
            efficiency_tracker=efficiency_tracker,
            answer_llm_name=answer_llm_name,
            provider=provider,
            output_path=output_path,
        )

    return len(log_records)


def main() -> None:
    args = parse_args()

    method_name = resolve_method(args.method)
    dataset_name = args.dataset_name or DATASET_NAME
    answer_llm_name = args.answer_llm_name or ANSWER_LLM_NAME
    provider = args.provider or PROVIDER
    max_input_length = args.max_input_length or MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or MAX_GENERATEION_NUM

    dataset_root = PROJECT_ROOT / "Data" / dataset_name
    documents_dir = dataset_root / "documents"
    logs_dir = args.logs_dir or (PROJECT_ROOT / "Logs")
    db_info_path = args.db_info_path or (dataset_root / "db_info.json")
    qdrant_path = args.qdrant_path or (dataset_root / "qdrant_column_index")
    db_info_index = load_db_info_index(db_info_path)
    db_counts = load_db_counts(db_info_index)
    qdrant_collection_name = load_qdrant_collection_name(qdrant_path)
    input_path = resolve_input_path(
        input_path=args.input_path,
        logs_dir=logs_dir,
        answer_llm_name=answer_llm_name,
        dataset_name=dataset_name,
        input_file_patterns=INPUT_FILE_PATTERNS,
        timestamp_pattern_template=TIMESTAMP_PATTERN_TEMPLATE,
    )
    output_path = resolve_output_path(
        output_path=args.output_path,
        answer_llm_name=answer_llm_name,
        dataset_name=dataset_name,
        output_stem=f"{method_name}_table2column",
        project_root=PROJECT_ROOT,
    )
    logger, logger_path = setup_task_logger("table2column", output_path)

    dataset_df = load_dataset(input_path)
    prompt_templates = load_prompt_templates(method_name)

    log_run_configuration(
        logger,
        task_name="Table-to-Column Schema Linking",
        dataset_name=dataset_name,
        data_count=len(dataset_df),
        model_name=answer_llm_name,
        provider=provider,
        result_path=output_path,
        extra_fields={
            "Method": method_name,
            "Input path": input_path,
            "Table prompt template": PROJECT_ROOT / "Templates" / method_name / "extract_relevant_tables.txt",
            "Column prompt template": PROJECT_ROOT / "Templates" / method_name / "extract_relevant_columns.txt",
            "DB info path": db_info_path,
            "Qdrant path": qdrant_path,
            "Qdrant collection": qdrant_collection_name,
            "Documents dir": documents_dir,
            "Max input length": max_input_length,
            "Max generation num": max_generation_num,
            "Logger path": logger_path,
        },
    )

    answer_llm = LLM(
        model_name=answer_llm_name,
        provider=provider,
        max_input_length=max_input_length,
        max_generation_num=max_generation_num,
        query_settings=BASELINE_SCHEMA_LINKING_QUERY_SETTINGS,
    )
    from Llm.embedding_model_loader import EmbeddingModelLoader

    embedder = EmbeddingModelLoader()
    qdrant_client = get_qdrant_client(qdrant_path)
    renderer = SchemaTextRenderer(tokenizer=answer_llm.tokenizer)
    schema_store = DbInfoSchemaStore(
        db_info_index=db_info_index,
        renderer=renderer,
    )

    processed_count = run_table2column(
        dataset_df=dataset_df,
        prompt_templates=prompt_templates,
        output_path=output_path,
        dataset_name=dataset_name,
        documents_dir=documents_dir,
        schema_store=schema_store,
        db_counts=db_counts,
        embedder=embedder,
        qdrant_client=qdrant_client,
        qdrant_collection_name=qdrant_collection_name,
        answer_llm=answer_llm,
        answer_llm_name=answer_llm_name,
        provider=provider,
    )
    logger.info("Completed %s records.", processed_count)


if __name__ == "__main__":
    main()
