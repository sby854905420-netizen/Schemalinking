from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Run.spider_agent_tc.agent import SpiderAgentTC
from Run.spider_agent_tc.prompt_builder import PromptBuilder, build_tool_schemas
from Run.spider_agent_tc.result_types import AgentInput, ExecutionResult
from Run.spider_agent_tc.trace_store import write_agent_failure_trace
from Run.spider_agent_tc.transformers_backend import TransformersChatBackend


class FakeTokenizer:
    def __init__(self) -> None:
        self.tool_calls = []

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        self.tool_calls.append(tools)
        return list(range(10 + len(messages)))


class FakeTensor:
    shape = (1, 4)

    def to(self, device):
        return self


class BackendTokenizer:
    def __init__(self):
        self.tools = None

    def apply_chat_template(self, messages, *, tools, **kwargs):
        self.tools = tools
        return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}


class FakeRenderer:
    @staticmethod
    def count_tokens(text):
        return len(str(text).split())

    @staticmethod
    def truncate_to_token_budget(text, budget):
        return " ".join(str(text).split()[:budget])


class FakePromptBuilder:
    @staticmethod
    def build_messages(agent_input, history):
        return [{"role": "system", "content": "strict"}, *history]


class FakeBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def generate(self, messages, generation_config):
        self.messages.append(messages)
        return next(self.responses)


class FakeExecutor:
    def __init__(self):
        self.sql = []

    def execute(self, sql):
        self.sql.append(sql)
        return ExecutionResult(columns=("value",), rows=((1,),))


def agent_input() -> AgentInput:
    return AgentInput(
        sample_id="1",
        dataset_name="MMQA",
        question="Return one.",
        hint="",
        predict_db_id="db",
        predict_tables=("t",),
        predict_columns={"t": ("value",)},
        selected_column_records=(
            {"table_name": "t", "column_name": "value", "column_data_type": "INTEGER"},
        ),
        schema_text="Table: t\n- value (INTEGER)",
        sql_dialect="SQLite",
    )


class SpiderAgentProtocolTests(unittest.TestCase):
    def test_qwen_template_receives_registered_tools_during_token_count(self):
        tokenizer = FakeTokenizer()
        tools = build_tool_schemas("MMQA")
        builder = PromptBuilder(
            system_template="Use {EXECUTE_TOOL} with {SQL_DIALECT}.",
            tokenizer=tokenizer,
            renderer=FakeRenderer(),
            max_input_length=1000,
            max_history_tokens=100,
            reserved_output_tokens=100,
            tools=tools,
        )

        builder.build_messages(agent_input(), [])

        self.assertTrue(tokenizer.tool_calls)
        names = [tool["function"]["name"] for tool in tokenizer.tool_calls[-1]]
        self.assertEqual(names, ["execute_sqlite_sql", "terminate"])

    def test_backend_generation_inputs_receive_the_same_registered_tools(self):
        tools = build_tool_schemas("MMQA")
        tokenizer = BackendTokenizer()
        backend = object.__new__(TransformersChatBackend)
        backend.tokenizer = tokenizer
        backend.tools = tools
        backend.device = "cpu"

        inputs = backend._model_inputs([{"role": "user", "content": "question"}])

        self.assertIn("input_ids", inputs)
        names = [tool["function"]["name"] for tool in tokenizer.tools]
        self.assertEqual(names, ["execute_sqlite_sql", "terminate"])

    def test_valid_calls_use_structured_assistant_and_tool_history(self):
        backend = FakeBackend(
            [
                "<tool_call><function=execute_sqlite_sql><parameter=sql>"
                "SELECT 1;</parameter></function></tool_call>",
                "<tool_call><function=terminate><parameter=answer>"
                "SELECT 1;</parameter></function></tool_call>",
            ]
        )
        executor = FakeExecutor()
        with patch(
            "Run.spider_agent_tc.agent.validate_readonly_sql",
            side_effect=lambda sql: sql.rstrip(";"),
        ):
            result = SpiderAgentTC(
                backend=backend,
                executor=executor,
                prompt_builder=FakePromptBuilder(),
                max_agent_rounds=2,
            ).run(agent_input())

        self.assertTrue(result.execution_verified)
        second_round = backend.messages[1]
        self.assertIn("tool_calls", second_round[1])
        self.assertEqual(second_round[2]["role"], "tool")
        self.assertEqual(executor.sql, ["SELECT 1", "SELECT 1"])

    def test_direct_sql_is_never_accepted(self):
        backend = FakeBackend(["SELECT 1;", "```sql\nSELECT 1;\n```"])
        executor = FakeExecutor()
        result = SpiderAgentTC(
            backend=backend,
            executor=executor,
            prompt_builder=FakePromptBuilder(),
            max_agent_rounds=2,
        ).run(agent_input())

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.sql, "")
        self.assertEqual(executor.sql, [])
        self.assertEqual(len(result.messages), 2)
        self.assertIn("<function=execute_sqlite_sql>", backend.messages[1][-1]["content"])

    def test_failure_trace_is_sidecar_and_upserted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.json"
            write_agent_failure_trace(
                output,
                sample_id="1",
                error="first",
                trace=({"round": 1, "response": "plain SQL"},),
            )
            write_agent_failure_trace(
                output,
                sample_id="1",
                error="updated",
                trace=({"round": 2, "response": "still plain SQL"},),
            )

            trace_path = Path(directory) / "predictions_agent_failures.json"
            payload = __import__("json").loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["failures"]), 1)
            self.assertEqual(payload["failures"][0]["error"], "updated")


if __name__ == "__main__":
    unittest.main()
