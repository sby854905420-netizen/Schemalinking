from __future__ import annotations

import html
import re

from .result_types import ToolCall


class ToolCallParseError(ValueError):
    pass


TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*"
    r"<function=(?P<function>[A-Za-z_][A-Za-z0-9_]*)>\s*"
    r"<parameter=(?P<parameter>[A-Za-z_][A-Za-z0-9_]*)>"
    r"(?P<value>.*?)"
    r"</parameter>\s*</function>\s*</tool_call>",
    flags=re.DOTALL,
)


def parse_tool_call(response_text: str, execute_function: str) -> ToolCall:
    text = "" if response_text is None else str(response_text)
    opening_count = len(re.findall(r"<tool_call\b", text, flags=re.IGNORECASE))
    if opening_count != 1:
        if opening_count > 1:
            raise ToolCallParseError("Exactly one tool call is allowed per round.")
        raise ToolCallParseError("Missing tool call.")

    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if len(matches) != 1:
        raise ToolCallParseError("Malformed tool call. Use the required XML protocol.")

    match = matches[0]
    trailing = text[match.end():].strip()
    if "<tool_call" in trailing.lower():
        raise ToolCallParseError("Exactly one tool call is allowed per round.")

    function = match.group("function")
    parameter = match.group("parameter")
    value = html.unescape(match.group("value")).strip()
    allowed = {execute_function: "sql", "terminate": "answer"}
    if function not in allowed:
        raise ToolCallParseError(f"Unknown tool function: {function}.")
    if parameter != allowed[function]:
        raise ToolCallParseError(
            f"Function {function} requires parameter '{allowed[function]}'."
        )
    if not value:
        raise ToolCallParseError("Tool call SQL must not be empty.")

    return ToolCall(function=function, parameter_name=parameter, value=value)


__all__ = ["ToolCallParseError", "parse_tool_call"]
