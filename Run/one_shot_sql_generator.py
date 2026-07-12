"""Reusable helpers for the one-shot mode of :mod:`Run.sql_generator`.

The executable entry point delegates to the unified SQL generator so this
module cannot read predictions from, or write prediction artifacts to, Logs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config import TEMPLATES_ROOT
from Llm.llm_loader import LLM
from Utils.json_utils import load_json
from Utils.render_tools import SchemaTextRenderer
from Utils.tools import render_prompt, resolve_prompt_token_cap
from Utils.value_utils import get_row_value


DEFAULT_SQL_PROMPT_PATH = TEMPLATES_ROOT / "sql_generation" / "one_shot.txt"
SQL_GENERATION_QUERY_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.02,
}


def sql_generation_query_settings(provider: str) -> dict[str, Any]:
    settings = dict(SQL_GENERATION_QUERY_SETTINGS)
    if provider == "openai":
        settings["response_format"] = {"type": "text"}
    return settings


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
        if sample_id is not None:
            index[str(sample_id)] = row
    return index


def default_sql_dialect(dataset_name: str) -> str:
    if dataset_name.lower() == "mmqa":
        return (
            "Use SQLite SQL for MMQA. Use only SQLite-compatible syntax and functions, "
            "and preserve table and column names exactly as shown in the schema excerpt."
        )
    if dataset_name.lower() == "spider2":
        return (
            "Use Snowflake SQL for Spider2. Preserve fully qualified names exactly and "
            "double-quote each identifier part when case or special characters require it."
        )
    return "Use the dialect implied by the question, schema, and hint."


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
        prompt_template, schema_text, question, hint, dataset_name, sql_dialect
    )
    prompt_tokens = answer_llm.count_input_tokens(prompt)
    if prompt_tokens <= target_prompt_cap:
        return prompt, schema_text, hint, prompt_tokens

    fitted_hint = hint
    if hint != "No hint":
        prompt_without_hint = render_sql_prompt(
            prompt_template, schema_text, question, "", dataset_name, sql_dialect
        )
        hint_budget = max(
            0,
            target_prompt_cap - answer_llm.count_input_tokens(prompt_without_hint),
        )
        fitted_hint = renderer.truncate_to_token_budget(hint, hint_budget)
        prompt = render_sql_prompt(
            prompt_template,
            schema_text,
            question,
            fitted_hint or "No hint",
            dataset_name,
            sql_dialect,
        )
        prompt_tokens = answer_llm.count_input_tokens(prompt)
        if prompt_tokens <= target_prompt_cap:
            return prompt, schema_text, fitted_hint or "No hint", prompt_tokens

    prompt_without_schema = render_sql_prompt(
        prompt_template, "", question, fitted_hint, dataset_name, sql_dialect
    )
    schema_budget = max(
        0,
        target_prompt_cap - answer_llm.count_input_tokens(prompt_without_schema),
    )
    fitted_schema_text = renderer.truncate_to_token_budget(schema_text, schema_budget)
    prompt = render_sql_prompt(
        prompt_template,
        fitted_schema_text,
        question,
        fitted_hint,
        dataset_name,
        sql_dialect,
    )
    return (
        prompt,
        fitted_schema_text,
        fitted_hint,
        answer_llm.count_input_tokens(prompt),
    )


def normalize_sql_response(response_text: str) -> str:
    text = response_text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    fenced_match = re.search(
        r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = fenced_match.group(1).strip() if fenced_match else text.replace("```", "").strip()
    try:
        response_json = json.loads(text)
    except json.JSONDecodeError:
        response_json = None
    if isinstance(response_json, Mapping) and isinstance(response_json.get("sql"), str):
        text = response_json["sql"].strip()
    return re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def main() -> None:
    import sys

    from Run.sql_generator import main as unified_main

    unified_main(["--generator-mode", "one_shot", *sys.argv[1:]])


if __name__ == "__main__":
    main()
