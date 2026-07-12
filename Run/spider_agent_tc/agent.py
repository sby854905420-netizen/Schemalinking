from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .executors.safety import validate_readonly_sql
from .prompt_builder import PromptBuilder, execute_tool_for_dataset
from .result_types import AgentInput, AgentResult, ExecutionResult
from .tool_call_parser import ToolCallParseError, parse_tool_call


def format_correction(execute_function: str, error: str) -> str:
    return (
        f"FORMAT_ERROR: {error}\n"
        "Reply with exactly one tool call and no suffix. Use one of:\n"
        f"<tool_call>\n<function={execute_function}>\n"
        "<parameter=sql>SELECT ...;</parameter>\n</function>\n</tool_call>\n"
        "or\n"
        "<tool_call>\n<function=terminate>\n"
        "<parameter=answer>SELECT ...;</parameter>\n</function>\n</tool_call>"
    )


def _safe_error(exc: Exception, limit: int = 1000) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(
        r"(?i)(password|token|secret|credential)\s*[=:]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        message,
    )
    return message[:limit]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def format_execution_observation(
    result: ExecutionResult,
    max_chars: int,
) -> str:
    payload = {
        "status": "success",
        "columns": list(result.columns),
        "rows": [[_json_safe(value) for value in row] for row in result.rows],
        "preview_row_count": result.returned_row_count,
        "truncated": result.truncated,
    }
    text = "EXECUTION_RESULT: " + json.dumps(payload, ensure_ascii=False)
    if len(text) > max_chars:
        return text[:max_chars] + "...[observation truncated]"
    return text


class SpiderAgentTC:
    def __init__(
        self,
        *,
        backend: Any,
        executor: Any,
        prompt_builder: PromptBuilder,
        max_agent_rounds: int = 10,
        max_llm_retries: int = 2,
        max_observation_chars: int = 12000,
        generation_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.executor = executor
        self.prompt_builder = prompt_builder
        self.max_agent_rounds = max_agent_rounds
        self.max_llm_retries = max_llm_retries
        self.max_observation_chars = max_observation_chars
        self.generation_config = dict(generation_config or {})

    def _generate_with_retries(self, messages: list[dict[str, Any]]) -> str:
        last_error: Exception | None = None
        for _ in range(self.max_llm_retries + 1):
            try:
                return self.backend.generate(messages, self.generation_config)
            except Exception as exc:  # a temporary model failure is sample-local
                last_error = exc
        assert last_error is not None
        raise RuntimeError(
            f"LLM failed after {self.max_llm_retries + 1} attempts: {_safe_error(last_error)}"
        ) from last_error

    def run(self, agent_input: AgentInput) -> AgentResult:
        history: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        last_legal_sql = ""
        last_error = ""
        execute_function = execute_tool_for_dataset(agent_input.dataset_name)

        for round_number in range(1, self.max_agent_rounds + 1):
            messages = self.prompt_builder.build_messages(agent_input, history)
            try:
                response = self._generate_with_retries(messages)
            except Exception as exc:
                last_error = _safe_error(exc)
                trace.append(
                    {"round": round_number, "outcome": "llm_error", "error": last_error}
                )
                if last_legal_sql:
                    return AgentResult(
                        sql=last_legal_sql,
                        status="success",
                        stop_reason="llm_error",
                        execution_verified=False,
                        rounds=round_number,
                        error=last_error,
                        messages=tuple(trace),
                    )
                return AgentResult(
                    sql="",
                    status="failed",
                    stop_reason="llm_error",
                    execution_verified=False,
                    rounds=round_number,
                    error=last_error,
                    messages=tuple(trace),
                )

            round_trace: dict[str, Any] = {
                "round": round_number,
                "response": response,
            }
            try:
                tool_call = parse_tool_call(response, execute_function=execute_function)
            except ToolCallParseError as exc:
                last_error = _safe_error(exc)
                round_trace.update({"outcome": "format_error", "error": last_error})
                trace.append(round_trace)
                history.append({"role": "assistant", "content": response})
                history.append(
                    {
                        "role": "user",
                        "content": format_correction(execute_function, str(exc)),
                    }
                )
                continue

            history.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool_call.function,
                                "arguments": {
                                    tool_call.parameter_name: tool_call.value,
                                },
                            },
                        }
                    ],
                }
            )
            round_trace["tool_call"] = {
                "function": tool_call.function,
                "parameter": tool_call.parameter_name,
                "value": tool_call.value,
            }

            try:
                candidate_sql = validate_readonly_sql(tool_call.value)
                last_legal_sql = candidate_sql
            except Exception as exc:
                last_error = _safe_error(exc)
                round_trace.update({"outcome": "sql_safety_error", "error": last_error})
                trace.append(round_trace)
                history.append(
                    {
                        "role": "tool",
                        "content": f"SQL_SAFETY_ERROR: {last_error}",
                    }
                )
                continue

            try:
                execution_result = self.executor.execute(candidate_sql)
            except Exception as exc:
                last_error = _safe_error(exc)
                round_trace.update({"outcome": "execution_error", "error": last_error})
                trace.append(round_trace)
                history.append(
                    {
                        "role": "tool",
                        "content": f"EXECUTION_ERROR: {last_error}",
                    }
                )
                continue

            if tool_call.function == "terminate":
                round_trace["outcome"] = "terminate"
                trace.append(round_trace)
                return AgentResult(
                    sql=candidate_sql,
                    status="success",
                    stop_reason="terminate",
                    execution_verified=True,
                    rounds=round_number,
                    messages=tuple(trace),
                )

            observation = format_execution_observation(
                execution_result, self.max_observation_chars
            )
            round_trace.update(
                {
                    "outcome": "execution_success",
                    "observation": observation,
                }
            )
            trace.append(round_trace)
            history.append(
                {
                    "role": "tool",
                    "content": observation,
                }
            )

        if last_legal_sql:
            return AgentResult(
                sql=last_legal_sql,
                status="success",
                stop_reason="max_rounds",
                execution_verified=False,
                rounds=self.max_agent_rounds,
                error=last_error,
                messages=tuple(trace),
            )
        return AgentResult(
            sql="",
            status="empty" if not last_error else "failed",
            stop_reason="max_rounds",
            execution_verified=False,
            rounds=self.max_agent_rounds,
            error=last_error,
            messages=tuple(trace),
        )


__all__ = [
    "SpiderAgentTC",
    "format_correction",
    "format_execution_observation",
]
