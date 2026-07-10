from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config import *
from Llm.llm_loader import FallbackTextTokenizer, LLM, resolve_provider
from Utils.json_utils import atomic_write_json
from Utils.prediction_store import (
    METHOD_NAMES,
    build_prediction_path,
    unified_to_native_schema_records,
    validate_prediction_file,
)
from Run import one_shot_sql_generator as legacy
from Run.spider_agent_tc.agent import SpiderAgentTC
from Run.spider_agent_tc.executors.factory import ExecutorFactory, executor_route_for_dataset
from Run.spider_agent_tc.prompt_builder import PromptBuilder
from Run.spider_agent_tc.schema_adapter import SchemaLinkingAdapter
from Run.spider_agent_tc.transformers_backend import TransformersChatBackend
from Utils.render_tools import SchemaTextRenderer
from Utils.schema_selection import DbInfoSchemaStore
from Utils.tools import load_db_info_index
from Utils.value_utils import get_row_value
from Utils.sql_prediction_store import (
    build_sql_prediction_path,
    initialize_sql_prediction_file,
    upsert_sql_prediction,
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
        description="Generate SQL from MDB-Link schema-linking prediction logs."
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
    parser.add_argument("--max-agent-rounds", type=int, default=10)
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
        choices=("auto", *sorted(METHOD_NAMES)),
        default="auto",
    )
    parser.add_argument(
        "--schema-method",
        choices=("auto", *sorted(legacy.SUPPORTED_SCHEMA_METHODS)),
        default="auto",
    )
    parser.add_argument(
        "--schema-task",
        choices=("auto", *sorted(legacy.SUPPORTED_SCHEMA_TASKS)),
        default="auto",
    )
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--db-info-path", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--documents-dir", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=None)
    parser.add_argument("--system-prompt-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
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


def write_result_file(
    output_path: Path,
    run_info_or_records: dict[str, Any] | list[dict[str, Any]],
    result_records: list[dict[str, Any]] | None = None,
) -> None:
    """Atomically persist only the fields consumed by EX evaluation.

    The legacy three-argument call remains accepted, but run_info is no longer
    written into the prediction artifact.
    """

    records = result_records if result_records is not None else run_info_or_records
    if not isinstance(records, list):
        raise TypeError("result_records must be a list")
    atomic_write_json(output_path, {"results": records})


def load_resume_records(output_path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not output_path.is_file():
        return []
    payload = legacy.load_json(output_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Cannot resume invalid SQL result file: {output_path}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload["results"]:
        if not isinstance(raw, dict):
            continue
        sample_id = str(raw.get("id", "")).strip()
        status = str(raw.get("status", "")).strip()
        if not sample_id or status not in COMPLETED_STATUSES or sample_id in seen:
            continue
        seen.add(sample_id)
        records.append(
            minimal_result(
                sample_id,
                raw.get("predict_db_id"),
                str(raw.get("predict_sql") or ""),
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


def resolve_sl_method(
    requested: str,
    *,
    input_path: Path | None = None,
    schema_task: str = "auto",
) -> str:
    if requested != "auto":
        return requested
    path_text = str(input_path or "").lower()
    if "rag_column_retrieval" in path_text or "rag_baseline" in path_text:
        return "rag_column_retrieval"
    if schema_task == "baseline_schema_linking" or "baseline_schema_linking" in path_text:
        return "prompt_baseline"
    return "table_to_column"


def load_schema_records(path: Path, dataset_name: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    payload = legacy.load_json(path)
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)], None
    unified = validate_prediction_file(payload)
    return unified_to_native_schema_records(unified, dataset_name), unified


def _write_standalone_sql(
    prediction_path: Path | None,
    record: Mapping[str, Any],
    error: str | None,
) -> None:
    if prediction_path is None:
        return
    upsert_sql_prediction(
        prediction_path,
        sample_id=record["id"],
        predicted_sql=str(record.get("predict_sql") or "") or None,
        status=str(record.get("status") or "failed"),
        error=error,
    )


def run_one_shot_generation(
    *,
    schema_log_records: Sequence[dict[str, Any]],
    dataset_index: dict[str, dict[str, Any]],
    adapter: SchemaLinkingAdapter,
    prompt_template: str,
    output_path: Path,
    answer_llm: LLM,
    renderer: SchemaTextRenderer,
    start_index: int,
    limit: int | None,
    resume: bool,
    sql_prediction_path: Path | None = None,
) -> int:
    from tqdm import tqdm

    records = load_resume_records(output_path, resume)
    for record in records:
        _write_standalone_sql(sql_prediction_path, record, None)
    completed = {record["id"] for record in records}
    for index, row in tqdm(_selected_rows(schema_log_records, start_index, limit)):
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
        write_result_file(output_path, records)
        _write_standalone_sql(sql_prediction_path, record, error_message)
    return len(records)


def run_spider_agent_generation(
    *,
    schema_log_records: Sequence[dict[str, Any]],
    dataset_index: dict[str, dict[str, Any]],
    adapter: SchemaLinkingAdapter,
    executor_factory: ExecutorFactory,
    prompt_builder: PromptBuilder,
    backend: TransformersChatBackend,
    output_path: Path,
    args: argparse.Namespace,
    sql_prediction_path: Path | None = None,
) -> int:
    from tqdm import tqdm

    records = load_resume_records(output_path, args.resume)
    for record in records:
        _write_standalone_sql(sql_prediction_path, record, None)
    completed = {record["id"] for record in records}
    for index, row in tqdm(_selected_rows(schema_log_records, args.start_index, args.limit)):
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
            for _ in range(args.rollout_number):
                candidate = SpiderAgentTC(
                    backend=backend,
                    executor=executor,
                    prompt_builder=prompt_builder,
                    max_agent_rounds=args.max_agent_rounds,
                    max_llm_retries=args.max_llm_retries,
                    max_observation_chars=args.max_observation_chars,
                    generation_config={"temperature": args.temperature},
                ).run(agent_input)
                if best is None or candidate.execution_verified:
                    best = candidate
                if candidate.execution_verified:
                    break
            assert best is not None
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
        write_result_file(output_path, records)
        _write_standalone_sql(sql_prediction_path, record, error_message)
    return len(records)


def run_dry_run(
    *,
    schema_log_records: Sequence[dict[str, Any]],
    dataset_index: dict[str, dict[str, Any]],
    adapter: SchemaLinkingAdapter,
    executor_factory: ExecutorFactory,
    prompt_builder: PromptBuilder,
    start_index: int,
    limit: int | None,
) -> tuple[int, int]:
    valid = failed = 0
    for index, row in _selected_rows(schema_log_records, start_index, limit):
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
    logs_dir = resolve_project_path(args.logs_dir) if args.logs_dir else LOGS_ROOT
    current_dataset_root = dataset_root(dataset_name)
    dataset_path = resolve_project_path(args.dataset_path) if args.dataset_path else current_dataset_root / "gold_sl.json"
    db_info_path = resolve_project_path(args.db_info_path) if args.db_info_path else current_dataset_root / "db_info.json"
    documents_dir = resolve_project_path(args.documents_dir) if args.documents_dir else current_dataset_root / "documents"
    prompt_path = resolve_project_path(args.prompt_path) if args.prompt_path else legacy.DEFAULT_SQL_PROMPT_PATH
    system_prompt_path = resolve_project_path(args.system_prompt_path) if args.system_prompt_path else DEFAULT_AGENT_SYSTEM_PROMPT_PATH
    sql_dialect = args.sql_dialect or default_sql_dialect(dataset_name)
    schema_llm_name = args.schema_llm_name or args.answer_llm_name
    executor_route_for_dataset(dataset_name)

    explicit_input_path = resolve_project_path(args.input_path) if args.input_path else None
    if explicit_input_path is not None:
        input_path = explicit_input_path
    elif args.sl_method != "auto":
        input_path = build_prediction_path(args.sl_method, dataset_name, schema_llm_name)
        if not input_path.is_file():
            raise FileNotFoundError(f"Could not find unified SL prediction: {input_path}")
    else:
        input_path = legacy.resolve_input_path(
            None,
            logs_dir,
            schema_llm_name,
            dataset_name,
            args.schema_method,
            args.schema_task,
        )
    output_path = legacy.resolve_output_path(
        resolve_project_path(args.output_path) if args.output_path else None,
        resolve_project_path(args.output_dir) if args.output_dir else None,
        logs_dir,
        dataset_name,
        args.answer_llm_name,
    )
    schema_records, unified_sl_payload = load_schema_records(input_path, dataset_name)
    sl_method = (
        str(unified_sl_payload["method"])
        if unified_sl_payload is not None
        else resolve_sl_method(
            args.sl_method,
            input_path=input_path,
            schema_task=args.schema_task,
        )
    )
    if unified_sl_payload is not None:
        schema_llm_name = unified_sl_payload["model_names"]["schema_linking"]
    elif schema_records:
        native_model = schema_records[0].get("model")
        if isinstance(native_model, str) and native_model.strip():
            schema_llm_name = native_model.strip()
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

    sql_prediction_path = (
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
        )
        run_dry_run(
            schema_log_records=schema_records,
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
            schema_log_records=schema_records,
            dataset_index=dataset_index,
            adapter=adapter,
            prompt_template=prompt_path.read_text(encoding="utf-8").strip(),
            output_path=output_path,
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
        )
        processed = run_spider_agent_generation(
            schema_log_records=schema_records,
            dataset_index=dataset_index,
            adapter=adapter,
            executor_factory=executor_factory,
            prompt_builder=prompt_builder,
            backend=backend,
            output_path=output_path,
            args=args,
            sql_prediction_path=sql_prediction_path,
        )
    print(
        f"Completed {processed} records: native={output_path}, "
        f"standalone={sql_prediction_path}"
    )


if __name__ == "__main__":
    main()
