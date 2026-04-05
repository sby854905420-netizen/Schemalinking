from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd


from config import *
from Llm.llm_loader import LLM

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
    parser = argparse.ArgumentParser(description="Baseline schema linking.")
    parser.add_argument("--method", dest="method", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--answer-llm-name", dest="answer_llm_name", type=str, default=None)
    parser.add_argument("--provider", dest="provider", type=str, default=None)
    parser.add_argument("--max-input-length", dest="max_input_length", type=int, default=None)
    parser.add_argument("--max-generation-num", dest="max_generation_num", type=int, default=None)
    parser.add_argument("--input-path", dest="input_path", type=Path, default=None)
    parser.add_argument("--logs-dir", dest="logs_dir", type=Path, default=None)
    parser.add_argument("--table-schema-dir", dest="table_schema_dir", type=Path, default=None)
    parser.add_argument("--output-path", dest="output_path", type=Path, default=None)
    return parser.parse_args()


def resolve_method(method: str | None) -> str:
    resolved_method = method or DEFAULT_METHOD
    if resolved_method not in SUPPORTED_METHODS:
        supported_methods = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(f"Unsupported method '{resolved_method}'. Expected one of: {supported_methods}.")
    return resolved_method


def load_dataset(input_path: Path):

    return pd.read_json(input_path)


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def extract_timestamp(path: Path, dataset_name: str) -> str | None:
    timestamp_pattern = re.compile(
        TIMESTAMP_PATTERN_TEMPLATE.format(dataset_name=re.escape(dataset_name))
    )
    match = timestamp_pattern.search(path.name)
    if match is None:
        return None
    return match.group(1)


def find_model_dir(logs_dir: Path, model_name: str) -> Path:
    direct_path = logs_dir / model_name / "Database_Retrival"
    if direct_path.is_dir():
        return direct_path

    model_leaf_name = Path(model_name).name
    matching_dirs = sorted(
        path
        for path in logs_dir.rglob("Database_Retrival")
        if path.is_dir() and path.parent.name == model_leaf_name
    )
    if not matching_dirs:
        raise FileNotFoundError(
            f"Could not find a Database_Retrival directory for model '{model_name}' under {logs_dir}."
        )
    if len(matching_dirs) > 1:
        matched_paths = "\n".join(str(path) for path in matching_dirs)
        raise ValueError(
            f"Found multiple Database_Retrival directories for model '{model_name}'. Please disambiguate:\n{matched_paths}"
        )
    return matching_dirs[0]


def find_result_file(model_dir: Path, dataset_name: str) -> Path:
    candidate_files: list[tuple[str, Path]] = []

    for pattern_template in INPUT_FILE_PATTERNS:
        pattern = pattern_template.format(dataset_name=dataset_name, timestamp="*")
        for path in model_dir.glob(pattern):
            file_timestamp = extract_timestamp(path, dataset_name)
            if file_timestamp is None:
                continue
            candidate_files.append((file_timestamp, path))

    if not candidate_files:
        expected_patterns = ", ".join(
            pattern_template.format(dataset_name=dataset_name, timestamp="*")
            for pattern_template in INPUT_FILE_PATTERNS
        )
        raise FileNotFoundError(f"Could not find any of [{expected_patterns}] under {model_dir}.")

    candidate_files.sort(key=lambda item: item[0], reverse=True)
    return candidate_files[0][1]


def resolve_input_path(
    input_path: Path | None,
    logs_dir: Path,
    answer_llm_name: str,
    dataset_name: str,
) -> Path:
    if input_path is not None:
        return input_path

    model_dir = find_model_dir(logs_dir=logs_dir, model_name=answer_llm_name)
    return find_result_file(model_dir=model_dir, dataset_name=dataset_name)


def resolve_output_path(
    output_path: Path | None,
    answer_llm_name: str,
    dataset_name: str,
    method_name: str,
) -> Path:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = PROJECT_ROOT / "Logs" / answer_llm_name
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"{method_name}_baseline_schema_linking_{dataset_name}_{run_id}.json"


def get_row_value(row: pd.Series, *keys: str) -> Any:

    for key in keys:
        value = row.get(key)
        if pd.notna(value):
            return value
    return None


def load_database_schema(table_schema_dir: Path, predict_db_id: str) -> str:
    import pandas as pd

    schema_path = table_schema_dir / f"{predict_db_id}.csv"
    schema_df = pd.read_csv(schema_path)
    return schema_df.to_markdown(index=False)


def build_prompt(prompt_template: str, database_schema: str, question: str) -> str:
    return (
        prompt_template
        .replace("{DATABASE_SCHEMA}", database_schema)
        .replace("{QUESTION}", question)
        .replace("{HINT}", "No hint")
    )


def normalize_response(response_text: str) -> str:
    normalized_response = response_text.replace("```", "").replace("json", "").strip()
    if "</think>" in normalized_response:
        normalized_response = normalized_response.split("</think>")[-1].strip()
    return normalized_response


def append_log_entry(
    log_records: list[dict[str, Any]],
    row: Any,
    response_text: str,
    answer_llm_name: str,
    provider: str,
    output_path: Path,
) -> None:
    try:
        predict_columns = json.loads(response_text)
    except json.JSONDecodeError:
        predict_columns = {}
    log_records.append(
        {
            "model": answer_llm_name,
            "provider": provider,
            "id": f"{get_row_value(row, 'id', 'instance_id')}",
            "question": row["question"],
            "spider_db_id": get_row_value(row, "spider_db_id"),
            "predict_db_id": get_row_value(row, "predict_db_id"),
            "predict_columns_text": response_text,
            "predict_columns": predict_columns,
        }
    )
    output_path.write_text(json.dumps(log_records, ensure_ascii=False, indent=2), encoding="utf-8")


def run_baseline_schema_linking(
    dataset_df: Any,
    prompt_template: str,
    output_path: Path,
    table_schema_dir: Path,
    answer_llm: Any,
    answer_llm_name: str,
    provider: str,
) -> None:
    from tqdm import tqdm

    log_records: list[dict[str, Any]] = []

    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        predict_db_id = get_row_value(row, "predict_db_id")
        if predict_db_id is None or str(predict_db_id).strip() == "":
            append_log_entry(
                log_records=log_records,
                row=row,
                response_text="No Valid Database.",
                answer_llm_name=answer_llm_name,
                provider=provider,
                output_path=output_path,
            )
            continue
        database_schema = load_database_schema(table_schema_dir, str(predict_db_id))
        prompt = build_prompt(prompt_template, database_schema, row["question"])
        response_text = normalize_response(answer_llm.query(prompt))
        append_log_entry(
            log_records=log_records,
            row=row,
            response_text=response_text,
            answer_llm_name=answer_llm_name,
            provider=provider,
            output_path=output_path,
        )


def main() -> None:
    args = parse_args()

    method_name = resolve_method(args.method)
    dataset_name = args.dataset_name or DATASET_NAME
    answer_llm_name = args.answer_llm_name or ANSWER_LLM_NAME
    provider = args.provider or PROVIDER
    max_input_length = args.max_input_length or MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or MAX_GENERATEION_NUM

    dataset_root = PROJECT_ROOT / "Data" / dataset_name
    logs_dir = args.logs_dir or (PROJECT_ROOT / "Logs")
    prompt_path = PROJECT_ROOT / "Templates" / method_name / "baseline_schema_linking.txt"
    table_schema_dir = args.table_schema_dir or (dataset_root / "Table_schema_csv")
    input_path = resolve_input_path(
        input_path=args.input_path,
        logs_dir=logs_dir,
        answer_llm_name=answer_llm_name,
        dataset_name=dataset_name,
    )
    output_path = resolve_output_path(
        output_path=args.output_path,
        answer_llm_name=answer_llm_name,
        dataset_name=dataset_name,
        method_name=method_name,
    )

    dataset_df = load_dataset(input_path)
    prompt_template = load_prompt_template(prompt_path)

    answer_llm = LLM(
        model_name=answer_llm_name,
        provider=provider,
        max_input_length=max_input_length,
        max_generation_num=max_generation_num,
        query_settings=BASELINE_SCHEMA_LINKING_QUERY_SETTINGS,
    )

    run_baseline_schema_linking(
        dataset_df=dataset_df,
        prompt_template=prompt_template,
        output_path=output_path,
        table_schema_dir=table_schema_dir,
        answer_llm=answer_llm,
        answer_llm_name=answer_llm_name,
        provider=provider,
    )


if __name__ == "__main__":
    main()
