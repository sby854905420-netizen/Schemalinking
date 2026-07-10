"""Spider-Agent-TC SQL generation components for MDB-Link.

Exports are loaded lazily so lightweight helpers and executors do not require
the schema-linking/dataframe dependency chain at import time.
"""

from importlib import import_module

__all__ = [
    "AgentInput",
    "AgentResult",
    "ExecutionResult",
    "SchemaLinkingAdapter",
    "SpiderAgentTC",
    "ToolCall",
]

_EXPORT_MODULES = {
    "SpiderAgentTC": ".agent",
    "AgentInput": ".result_types",
    "AgentResult": ".result_types",
    "ExecutionResult": ".result_types",
    "ToolCall": ".result_types",
    "SchemaLinkingAdapter": ".schema_adapter",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
