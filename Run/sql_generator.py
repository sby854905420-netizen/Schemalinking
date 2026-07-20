from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config import *
from Llm.llm_loader import FallbackTextTokenizer, LLM, resolve_provider
from Utils.prediction_store import (
    METHOD_NAMES,
    build_prediction_path,
    unified_to_native_schema_records,
    validate_prediction_file,
)
from Run import one_shot_sql_generator as legacy
from Run.spider_agent_tc.agent import SpiderAgentTC
from Run.spider_agent_tc.executors.factory import ExecutorFactory, executor_route_for_dataset
from Run.spider_agent_tc.prompt_builder import PromptBuilder, build_tool_schemas
from Run.spider_agent_tc.schema_adapter import SchemaLinkingAdapter
from Run.spider_agent_tc.transformers_backend import TransformersChatBackend
from Run.spider_agent_tc.trace_store import write_agent_failure_trace
from Utils.render_tools import SchemaTextRenderer
from Utils.schema_selection import DbInfoSchemaStore
from Utils.tools import load_db_info_index
from Utils.value_utils import get_row_value
from Utils.artifact_paths import require_results_output
from Utils.sql_prediction_store import (
    build_sql_prediction_path,
    initialize_sql_prediction_file,
    replace_sql_predictions,
    upsert_sql_prediction,
    validate_sql_prediction_file,
)


DEFAULT_SQL_LLM_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_AGENT_SYSTEM_PROMPT_PATH = (
    TEMPLATES_ROOT / "sql_generation" / "spider_agent_tc_system.txt"
)
DEFAULT_MAX_INPUT_LENGTH = 24576
DEFAULT_SQL_GENERATION_NUM = 4096
COMPLETED_STATUSES = {"success", "empty", "failed"}


def _console_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(
        r"(?i)(password|token|secret|credential)\s*[=:]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        message,
    )
    return message[:1000]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SQL from unified MDB-Link schema-linking predictions."
    )
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument(
        "--generator-mode",
        choices=("spider_agent_tc", "one_shot"),
        default="spider_agent_tc",
    )
    parser.add_argument("--answer-llm-name", type=str, default=DEFAULT_SQL_LLM_NAME)
    parser.add_argument("--provider", type=str, default="transformers")
    parser.add_argument("--credential-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--max-input-length", type=int, default=DEFAULT_MAX_INPUT_LENGTH)
    parser.add_argument("--max-generation-num", type=int, default=DEFAULT_SQL_GENERATION_NUM)
    parser.add_argument("--max-agent-rounds", type=int, default=20)
    parser.add_argument("--max-llm-retries", type=int, default=2)
    parser.add_argument("--rollout-number", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--execution-timeout-seconds", type=float, default=120)
    parser.add_argument("--max-result-rows", type=int, default=20)
    parser.add_argument("--max-observation-chars", type=int, default=12000)
    parser.add_argument("--max-history-tokens", type=int, default=12000)
    parser.add_argument("--snowflake-credential-path", type=Path, default=None)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--schema-llm-name", type=str, default=None)
    parser.add_argument(
        "--sl-method",
        choices=sorted(METHOD_NAMES),
        default="table_to_column",
    )
    parser.add_argument("--db-info-path", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--documents-dir", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=None)
    parser.add_argument("--system-prompt-path", type=Path, default=None)
    parser.add_argument("--sql-prediction-path", type=Path, default=None)
    parser.add_argument("--sql-dialect", type=str, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-key-columns", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build Adapter inputs, route, and prompts without loading a model or database.",
    )
    return parser.parse_args(argv)


def default_sql_dialect(dataset_name: str) -> str:
    if dataset_name.lower() in {"mmqa", "bird"}:
        return (
            f"Use SQLite SQL for {dataset_name}. Preserve all supplied identifiers exactly. "
            "Use SQLite-compatible functions and do not use Snowflake-only syntax."
        )
    if dataset_name.lower() == "spider2":
        return (
            "Use Snowflake SQL for Spider2. Preserve fully qualified three-part names and "
            "double-quote each identifier part when case or special characters require it."
        )
    return legacy.default_sql_dialect(dataset_name)


def minimal_result(
    sample_id: str,
    predict_db_id: str | None,
    predict_sql: str,
    status: str,
) -> dict[str, Any]:
    return {
        "id": str(sample_id),
        "predict_db_id": predict_db_id,
        "predict_sql": predict_sql,
        "status": status if status in COMPLETED_STATUSES else "failed",
    }


def load_resume_records(prediction_path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not prediction_path.is_file():
        return []
    payload = validate_sql_prediction_file(prediction_path)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload["predictions"]:
        sample_id = str(raw.get("id", "")).strip()
        status = str(raw.get("status", "")).strip()
        if not sample_id or status not in COMPLETED_STATUSES or sample_id in seen:
            continue
        seen.add(sample_id)
        records.append(
            minimal_result(
                sample_id,
                None,
                str(raw.get("predicted_sql") or ""),
                status,
            )
        )
    return records


def _selected_rows(
    records: Sequence[dict[str, Any]], start_index: int, limit: int | None
) -> list[tuple[int, dict[str, Any]]]:
    selected = list(enumerate(records))[max(0, start_index):]
    return selected if limit is None else selected[: max(0, limit)]


def _source_row(
    row: Mapping[str, Any], dataset_index: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    sample_id = get_row_value(row, "id", "instance_id")
    return dataset_index.get(str(sample_id), {}) if sample_id is not None else {}


def _failure_record(
    row: Mapping[str, Any], source: Mapping[str, Any], fallback_id: str
) -> dict[str, Any]:
    sample_id = get_row_value(row, "id", "instance_id")
    if sample_id is None:
        sample_id = get_row_value(source, "id", "instance_id")
    predict_db_id = get_row_value(row, "predict_db_id")
    return minimal_result(
        str(sample_id if sample_id is not None else fallback_id),
        None if predict_db_id is None else str(predict_db_id).strip(),
        "",
        "failed",
    )


def load_schema_records(
    path: Path,
    dataset_name: str,
    *,
    gold_records: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = legacy.load_json(path)
    unified = validate_prediction_file(payload)
    return (
        unified_to_native_schema_records(
            unified,
            dataset_name,
            gold_records=gold_records,
        ),
        unified,
    )


def _write_sql_prediction(
    prediction_path: Path,
    record: Mapping[str, Any],
    error: str | None,
) -> None:
    upsert_sql_prediction(
        prediction_path,
        sample_id=record["id"],
        predicted_sql=str(record.get("predict_sql") or "") or None,
        status=str(record.get("status") or "failed"),
        error=error,
    )


def run_one_shot_generation(
    *,
    schema_records: Sequence[dict[str, Any]],
    dataset_index: dict[str, dict[str, Any]],
    adapter: SchemaLinkingAdapter,
    prompt_template: str,
    answer_llm: LLM,
    renderer: SchemaTextRenderer,
    start_index: int,
    limit: int | None,
    resume: bool,
    sql_prediction_path: Path,
) -> int:
    from tqdm import tqdm

    records = load_resume_records(sql_prediction_path, resume)
    completed = {record["id"] for record in records}
    for index, row in tqdm(_selected_rows(schema_records, start_index, limit)):
        fallback_id = f"row:{index}"
        source = _source_row(row, dataset_index)
        raw_id = get_row_value(row, "id", "instance_id")
        stable_id = str(raw_id if raw_id is not None else fallback_id)
        if stable_id in completed:
            continue
        error_message = None
        try:
            agent_input = adapter.adapt(row, source, fallback_sample_id=fallback_id)
            prompt, _, _, _ = legacy.fit_prompt_to_budget(
                prompt_template=prompt_template,
                schema_text=agent_input.schema_text,
                question=agent_input.question,
                hint=agent_input.hint,
                dataset_name=agent_input.dataset_name,
                sql_dialect=agent_input.sql_dialect,
                answer_llm=answer_llm,
                renderer=renderer,
            )
            response, _ = answer_llm.query_with_usage(prompt)
            sql = legacy.normalize_sql_response(response)
            record = minimal_result(
                agent_input.sample_id,
                agent_input.predict_db_id,
                sql,
                "success" if sql else "empty",
            )
        except Exception as exc:
            error_message = _console_error(exc)
            tqdm.write(f"one_shot sample {stable_id} failed: {error_message}")
            record = _failure_record(row, source, fallback_id)
        records.append(record)
        completed.add(record["id"])
        _write_sql_prediction(sql_prediction_path, record, error_message)
    return len(records)


def run_spider_agent_generation(
    *,
    schema_records: Sequence[dict[str, Any]],
    dataset_index: dict[str, dict[str, Any]],
    adapter: SchemaLinkingAdapter,
    executor_factory: ExecutorFactory,
    prompt_builder: PromptBuilder,
    backend: TransformersChatBackend,
    args: argparse.Namespace,
    sql_prediction_path: Path,
) -> int:
    from tqdm import tqdm

    records = load_resume_records(sql_prediction_path, args.resume)
    completed = {record["id"] for record in records}
    for index, row in tqdm(_selected_rows(schema_records, args.start_index, args.limit)):
        fallback_id = f"row:{index}"
        source = _source_row(row, dataset_index)
        raw_id = get_row_value(row, "id", "instance_id")
        stable_id = str(raw_id if raw_id is not None else fallback_id)
        if stable_id in completed:
            continue
        executor = None
        error_message = None
        try:
            agent_input = adapter.adapt(row, source, fallback_sample_id=fallback_id)
            executor = executor_factory.create(agent_input.predict_db_id)
            best = None
            for rollout_index in range(args.rollout_number):
                candidate = SpiderAgentTC(
                    backend=backend,
                    executor=executor,
                    prompt_builder=prompt_builder,
                    max_agent_rounds=args.max_agent_rounds,
                    max_llm_retries=args.max_llm_retries,
                    max_observation_chars=args.max_observation_chars,
                    generation_config={
                        "temperature": args.temperature,
                        "seed_offset": rollout_index,
                    },
                ).run(agent_input)
                candidate_rank = (
                    int(candidate.execution_verified),
                    int(candidate.status == "success"),
                    int(bool(candidate.sql)),
                )
                best_rank = (
                    (
                        int(best.execution_verified),
                        int(best.status == "success"),
                        int(bool(best.sql)),
                    )
                    if best is not None
                    else (-1, -1, -1)
                )
                if best is None or candidate_rank > best_rank:
                    best = candidate
                if candidate.execution_verified:
                    break
            assert best is not None
            if not best.execution_verified:
                trace_error = best.error[:1000] or (
                    f"Agent stopped without a verified terminate call: {best.stop_reason}."
                )
                write_agent_failure_trace(
                    sql_prediction_path,
                    sample_id=stable_id,
                    error=trace_error,
                    trace=best.messages,
                )
            if best.status != "success":
                error_message = best.error[:1000] or f"Agent ended with status {best.status}."
                tqdm.write(
                    f"spider_agent_tc sample {stable_id} ended as {best.status}: "
                    f"{best.error[:1000]}"
                )
            record = minimal_result(
                agent_input.sample_id,
                agent_input.predict_db_id,
                best.sql,
                best.status,
            )
        except Exception as exc:
            error_message = _console_error(exc)
            tqdm.write(f"spider_agent_tc sample {stable_id} failed: {error_message}")
            record = _failure_record(row, source, fallback_id)
        finally:
            if executor is not None:
                try:
                    executor.close()
                except Exception:
                    pass
        records.append(record)
        completed.add(record["id"])
        _write_sql_prediction(sql_prediction_path, record, error_message)
    return len(records)


def run_dry_run(
    *,
    schema_records: Sequence[dict[str, Any]],
    dataset_index: dict[str, dict[str, Any]],
    adapter: SchemaLinkingAdapter,
    executor_factory: ExecutorFactory,
    prompt_builder: PromptBuilder,
    start_index: int,
    limit: int | None,
) -> tuple[int, int]:
    valid = failed = 0
    for index, row in _selected_rows(schema_records, start_index, limit):
        try:
            agent_input = adapter.adapt(
                row,
                _source_row(row, dataset_index),
                fallback_sample_id=f"row:{index}",
            )
            prompt_builder.build_fixed_messages(agent_input)
            executor_factory.describe(agent_input.predict_db_id)
            valid += 1
        except Exception as exc:
            failed += 1
            print(f"DRY RUN record {index} failed: {type(exc).__name__}: {exc}")
    print(
        f"DRY RUN complete: valid={valid}, failed={failed}, "
        "model_loaded=no, database_executed=no"
    )
    return valid, failed


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "max_input_length": args.max_input_length,
        "max_generation_num": args.max_generation_num,
        "max_agent_rounds": args.max_agent_rounds,
        "rollout_number": args.rollout_number,
        "execution_timeout_seconds": args.execution_timeout_seconds,
        "max_result_rows": args.max_result_rows,
        "max_observation_chars": args.max_observation_chars,
        "max_history_tokens": args.max_history_tokens,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError("Arguments must be positive: " + ", ".join(invalid))
    if args.max_llm_retries < 0 or args.temperature < 0:
        raise ValueError("Retries and temperature must be non-negative.")
    if (
        args.generator_mode == "spider_agent_tc"
        and args.max_generation_num >= args.max_input_length
    ):
        raise ValueError(
            "spider_agent_tc requires --max-generation-num to be smaller than "
            "--max-input-length so output tokens can be reserved."
        )
    if args.generator_mode == "spider_agent_tc" and args.provider.lower() != "transformers":
        raise ValueError("spider_agent_tc requires --provider transformers.")
    if (
        args.generator_mode == "spider_agent_tc"
        and args.answer_llm_name != DEFAULT_SQL_LLM_NAME
    ):
        raise ValueError(
            "spider_agent_tc baseline is fixed to "
            f"--answer-llm-name {DEFAULT_SQL_LLM_NAME}."
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    dataset_name = args.dataset_name or DATASET_NAME
    current_dataset_root = dataset_root(dataset_name)
    dataset_path = resolve_project_path(args.dataset_path) if args.dataset_path else current_dataset_root / "gold_sl.json"
    db_info_path = resolve_project_path(args.db_info_path) if args.db_info_path else current_dataset_root / "db_info.json"
    documents_dir = resolve_project_path(args.documents_dir) if args.documents_dir else current_dataset_root / "documents"
    prompt_path = resolve_project_path(args.prompt_path) if args.prompt_path else legacy.DEFAULT_SQL_PROMPT_PATH
    system_prompt_path = resolve_project_path(args.system_prompt_path) if args.system_prompt_path else DEFAULT_AGENT_SYSTEM_PROMPT_PATH
    sql_dialect = args.sql_dialect or default_sql_dialect(dataset_name)
    schema_llm_name = args.schema_llm_name or args.answer_llm_name
    executor_route_for_dataset(dataset_name)
    tool_schemas = build_tool_schemas(dataset_name)

    explicit_input_path = (
        resolve_project_path(args.input_path)
        if args.input_path
        else None
    )
    input_path = explicit_input_path or build_prediction_path(
        args.sl_method, dataset_name, schema_llm_name
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Could not find unified SL prediction: {input_path}")
    gold_records = None
    if dataset_path.is_file():
        loaded_gold_records = legacy.load_json(dataset_path)
        if not isinstance(loaded_gold_records, list):
            raise ValueError(f"Expected a list of dataset rows in {dataset_path}.")
        gold_records = loaded_gold_records
    schema_records, unified_sl_payload = load_schema_records(
        input_path,
        dataset_name,
        gold_records=gold_records,
    )
    sl_method = str(unified_sl_payload["method"])
    schema_llm_name = unified_sl_payload["model_names"]["schema_linking"]
    if sl_method == "autolink" and args.include_key_columns:
        raise ValueError(
            "AutoLink gold-database filtering cannot be combined with "
            "--include-key-columns because unpredicted columns must be discarded."
        )
    dataset_index = legacy.load_dataset_index(dataset_path)
    db_info_index = load_db_info_index(db_info_path)
    executor_factory = ExecutorFactory(
        dataset_name=dataset_name,
        database_root=resolve_project_path(args.database_root) if args.database_root else None,
        snowflake_credential_path=(
            resolve_project_path(args.snowflake_credential_path)
            if args.snowflake_credential_path
            else None
        ),
        timeout_seconds=args.execution_timeout_seconds,
        max_result_rows=args.max_result_rows,
    )

    sql_prediction_path = require_results_output(
        resolve_project_path(args.sql_prediction_path)
        if args.sql_prediction_path
        else build_sql_prediction_path(
            args.generator_mode,
            dataset_name,
            args.answer_llm_name,
            sl_method,
            schema_llm_name,
        )
    )

    if args.dry_run:
        tokenizer = FallbackTextTokenizer()
        renderer = SchemaTextRenderer(tokenizer=tokenizer)
        schema_store = DbInfoSchemaStore(db_info_index=db_info_index, renderer=renderer)
        adapter = SchemaLinkingAdapter(
            dataset_name=dataset_name,
            documents_dir=documents_dir,
            schema_store=schema_store,
            sql_dialect=sql_dialect,
            include_key_columns=args.include_key_columns,
        )
        prompt_builder = PromptBuilder(
            system_template=system_prompt_path.read_text(encoding="utf-8").strip(),
            tokenizer=tokenizer,
            renderer=renderer,
            max_input_length=args.max_input_length,
            max_history_tokens=args.max_history_tokens,
            reserved_output_tokens=args.max_generation_num,
            tools=tool_schemas,
        )
        run_dry_run(
            schema_records=schema_records,
            dataset_index=dataset_index,
            adapter=adapter,
            executor_factory=executor_factory,
            prompt_builder=prompt_builder,
            start_index=args.start_index,
            limit=args.limit,
        )
        return

    initialize_sql_prediction_file(
        sql_prediction_path,
        dataset_name=dataset_name,
        sql_method=args.generator_mode,
        sql_model_name=args.answer_llm_name,
        schema_linking_method=sl_method,
        schema_linking_model_name=schema_llm_name,
    )
    if not args.resume:
        replace_sql_predictions(sql_prediction_path, [])

    if args.generator_mode == "one_shot":
        provider = resolve_provider(args.provider)
        answer_llm = LLM(
            model_name=args.answer_llm_name,
            provider=provider,
            max_input_length=args.max_input_length,
            max_generation_num=args.max_generation_num,
            query_settings=legacy.sql_generation_query_settings(provider),
            credential_path=(
                resolve_project_path(args.credential_path)
                if args.credential_path
                else None
            ),
        )
        renderer = SchemaTextRenderer(tokenizer=answer_llm.tokenizer)
        schema_store = DbInfoSchemaStore(db_info_index=db_info_index, renderer=renderer)
        adapter = SchemaLinkingAdapter(
            dataset_name=dataset_name,
            documents_dir=documents_dir,
            schema_store=schema_store,
            sql_dialect=sql_dialect,
            include_key_columns=args.include_key_columns,
        )
        processed = run_one_shot_generation(
            schema_records=schema_records,
            dataset_index=dataset_index,
            adapter=adapter,
            prompt_template=prompt_path.read_text(encoding="utf-8").strip(),
            answer_llm=answer_llm,
            renderer=renderer,
            start_index=args.start_index,
            limit=args.limit,
            resume=args.resume,
            sql_prediction_path=sql_prediction_path,
        )
    else:
        backend = TransformersChatBackend(
            model_name=args.answer_llm_name,
            device=args.device,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_generation_num,
            temperature=args.temperature,
            random_seed=args.random_seed,
            tools=tool_schemas,
        )
        renderer = SchemaTextRenderer(tokenizer=backend.tokenizer)
        schema_store = DbInfoSchemaStore(db_info_index=db_info_index, renderer=renderer)
        adapter = SchemaLinkingAdapter(
            dataset_name=dataset_name,
            documents_dir=documents_dir,
            schema_store=schema_store,
            sql_dialect=sql_dialect,
            include_key_columns=args.include_key_columns,
        )
        prompt_builder = PromptBuilder(
            system_template=system_prompt_path.read_text(encoding="utf-8").strip(),
            tokenizer=backend.tokenizer,
            renderer=renderer,
            max_input_length=args.max_input_length,
            max_history_tokens=args.max_history_tokens,
            reserved_output_tokens=args.max_generation_num,
            tools=tool_schemas,
        )
        processed = run_spider_agent_generation(
            schema_records=schema_records,
            dataset_index=dataset_index,
            adapter=adapter,
            executor_factory=executor_factory,
            prompt_builder=prompt_builder,
            backend=backend,
            args=args,
            sql_prediction_path=sql_prediction_path,
        )
    print(f"Completed {processed} records: {sql_prediction_path}")


if __name__ == "__main__":
    main()
