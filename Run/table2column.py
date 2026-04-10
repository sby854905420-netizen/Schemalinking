from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd

from config import *
from Llm.llm_loader import LLM
from Run.logging_utils import log_run_configuration, setup_task_logger
from Utils.tools import (
    build_column_key,
    build_db_id_filter,
    build_db_schema_text,
    get_key_columns,
    get_qdrant_client,
    get_row_value,
    load_db_info_index,
    load_schema_dataframe_from_db_info,
    normalize_response_text,
    query_qdrant,
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


def resolve_external_knowledge_for_prompt(
    dataset_name: str,
    source: Any,
    documents_dir: Path | None = None,
    key: str = "external_knowledge",
) -> Any:
    if dataset_name.lower() != "spider2":
        return source

    if hasattr(source, "get"):
        external_knowledge = source.get(key)
        if external_knowledge is None:
            return source
    else:
        external_knowledge = source

    if not isinstance(external_knowledge, str):
        return source

    document_name = external_knowledge.strip()
    if not document_name or documents_dir is None:
        return source

    document_path = documents_dir / document_name
    if not document_path.is_file():
        return source

    document_text = document_path.read_text(encoding="utf-8").strip()

    if hasattr(source, "copy"):
        resolved_source = source.copy()
        if hasattr(resolved_source, "loc"):
            resolved_source.loc[key] = document_text
        else:
            resolved_source[key] = document_text
        return resolved_source

    return document_text


def get_point_payload(point: Any) -> dict[str, Any]:
    payload = getattr(point, "payload", None)
    if payload is None and isinstance(point, dict):
        payload = point.get("payload")
    return dict(payload or {})


def count_prompt_tokens(answer_llm: Any, prompt: str) -> int:
    try:
        return answer_llm.count_input_tokens(prompt)
    except (NotImplementedError, AttributeError):
        return max(1, len(prompt) // 4)


def score_relevant_columns_for_db(
    question: str,
    predict_db_id: str,
    schema_df: pd.DataFrame,
    embedder: Any,
    qdrant_client: Any,
    qdrant_collection_name: str,
) -> list[tuple[str, str]]:
    query_vector = embedder.encode(question, convert_to_list=True)
    ranked_points = query_qdrant(
        client=qdrant_client,
        collection_name=qdrant_collection_name,
        query_vector=query_vector,
        top_k=max(1, len(schema_df)),
        query_filter=build_db_id_filter([predict_db_id]),
    )

    valid_column_keys = {build_column_key(row) for _, row in schema_df.iterrows()}
    ranked_column_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for point in ranked_points:
        payload = get_point_payload(point)
        column_key = (
            str(payload.get("table_name", "")).strip(),
            str(payload.get("column_name", "")).strip(),
        )
        if column_key in seen or column_key not in valid_column_keys:
            continue
        seen.add(column_key)
        ranked_column_keys.append(column_key)

    return ranked_column_keys


def select_table_prompt_columns(
    schema_df: pd.DataFrame,
    predict_db_id: str,
    question: str,
    hint: str,
    prompt_template: str,
    answer_llm: Any,
    embedder: Any,
    qdrant_client: Any,
    qdrant_collection_name: str,
) -> set[tuple[str, str]]:
    selected_column_keys = get_key_columns(schema_df)
    prompt_token_cap = resolve_prompt_token_cap(answer_llm.max_input_length)

    ranked_column_keys = score_relevant_columns_for_db(
        question=question,
        predict_db_id=predict_db_id,
        schema_df=schema_df,
        embedder=embedder,
        qdrant_client=qdrant_client,
        qdrant_collection_name=qdrant_collection_name,
    )

    for column_key in ranked_column_keys:
        if column_key in selected_column_keys:
            continue

        proposed_column_keys = set(selected_column_keys)
        proposed_column_keys.add(column_key)
        proposed_schema = build_db_schema_text(
            schema_df=schema_df,
            db_id=predict_db_id,
            include_selected_tables=False,
            selected_column_keys=proposed_column_keys,
            include_empty_tables=True,
        )
        proposed_prompt = render_prompt(
            prompt_template,
            DATABASE_SCHEMA=proposed_schema,
            QUESTION=question,
            HINT=hint,
        )
        proposed_tokens = count_prompt_tokens(answer_llm, proposed_prompt)
        if proposed_tokens <= prompt_token_cap:
            selected_column_keys = proposed_column_keys

    return selected_column_keys


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
    answer_llm_name: str,
    provider: str,
    output_path: Path,
) -> None:
    
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
        }
    )
    output_path.write_text(json.dumps(log_records, ensure_ascii=False, indent=2), encoding="utf-8")


def run_table2column(
    dataset_df: pd.DataFrame,
    prompt_templates: dict[str, str],
    output_path: Path,
    dataset_name: str,
    documents_dir: Path,
    db_info_index: dict[str, dict[str, Any]],
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
        predict_db_id = get_row_value(row, "predict_db_id")
        if predict_db_id is None or str(predict_db_id).strip() == "":
            append_log_entry(
                log_records=log_records,
                row=row,
                predict_tables=[],
                predict_columns= {},
                table_response_text="No Vaild Database.",
                column_response_text="No Vaild Database.",
                answer_llm_name=answer_llm_name,
                provider=provider,
                output_path=output_path,
            )
            continue
        predict_db_id = str(predict_db_id)
        hint_source = resolve_external_knowledge_for_prompt(
            dataset_name=dataset_name,
            source=row,
            documents_dir=documents_dir,
        )
        hint = resolve_hint(hint_source)
        try:
            total_schema_df = load_schema_dataframe_from_db_info(
                predict_db_id=predict_db_id,
                dataset_name=dataset_name,
                db_info_index=db_info_index,
            )
        except FileNotFoundError:
            append_log_entry(
                log_records=log_records,
                row=row,
                predict_tables= [],
                predict_columns= {},
                table_response_text="No Vaild Database.",
                column_response_text="No Vaild Database.",
                answer_llm_name=answer_llm_name,
                provider=provider,
                output_path=output_path,
            )
            continue

        table_prompt_column_keys = select_table_prompt_columns(
            schema_df=total_schema_df,
            predict_db_id=predict_db_id,
            question=row["question"],
            hint=hint,
            prompt_template=prompt_templates["table"],
            answer_llm=answer_llm,
            embedder=embedder,
            qdrant_client=qdrant_client,
            qdrant_collection_name=qdrant_collection_name,
        )
        table_prompt = render_prompt(
            prompt_templates["table"],
            DATABASE_SCHEMA=build_db_schema_text(
                schema_df=total_schema_df,
                db_id=predict_db_id,
                include_selected_tables=False,
                selected_column_keys=table_prompt_column_keys,
                include_empty_tables=True,
            ),
            QUESTION=row["question"],
            HINT=hint,
        )
        table_response_text = answer_llm.query(table_prompt)
        relevant_table_list = parse_table_response(table_response_text)

        if relevant_table_list:
            relevant_schema_df = total_schema_df[total_schema_df["table_name"].isin(relevant_table_list)]
        else:
            relevant_schema_df = total_schema_df

        column_prompt = render_prompt(
            prompt_templates["column"],
            DATABASE_SCHEMA=build_db_schema_text(
                schema_df=relevant_schema_df,
                db_id=predict_db_id,
                include_selected_tables=True,
            ),
            QUESTION=row["question"],
            HINT=hint,
        )
        column_response_text = answer_llm.query(column_prompt)
        predict_columns = parse_column_response(column_response_text)
        append_log_entry(
            log_records=log_records,
            row=row,
            predict_tables= relevant_table_list,
            predict_columns= predict_columns,
            table_response_text=table_response_text,
            column_response_text=column_response_text,
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

    processed_count = run_table2column(
        dataset_df=dataset_df,
        prompt_templates=prompt_templates,
        output_path=output_path,
        dataset_name=dataset_name,
        documents_dir=documents_dir,
        db_info_index=db_info_index,
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
