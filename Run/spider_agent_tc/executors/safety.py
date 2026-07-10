from __future__ import annotations

from typing import Any

try:
    import sqlparse
    from sqlparse import tokens as T
except ImportError as exc:  # pragma: no cover - incomplete environments only
    sqlparse = None
    T = None
    _SQLPARSE_IMPORT_ERROR: Exception | None = exc
else:
    _SQLPARSE_IMPORT_ERROR = None


class SQLSafetyError(ValueError):
    pass


FORBIDDEN_KEYWORDS = {
    "ALTER",
    "ATTACH",
    "CALL",
    "COPY",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "EXEC",
    "EXECUTE",
    "GET",
    "GRANT",
    "INSERT",
    "INTO",
    "MERGE",
    "PRAGMA",
    "PUT",
    "REMOVE",
    "REPLACE",
    "REVOKE",
    "SHOW",
    "TRUNCATE",
    "UPDATE",
    "USE",
}

# These endpoints are read-only but can reveal schema that was not selected by
# MDB-Link. Blocking them is part of the Adapter's no-full-schema boundary.
FORBIDDEN_METADATA_IDENTIFIERS = {
    "ACCOUNT_USAGE",
    "INFORMATION_SCHEMA",
    "ORGANIZATION_USAGE",
    "SQLITE_MASTER",
    "SQLITE_SCHEMA",
}
FORBIDDEN_METADATA_PREFIXES = ("PRAGMA_", "SYSTEM$")
FORBIDDEN_METADATA_FUNCTIONS = {"GET_DDL"}


def _require_sqlparse() -> None:
    if sqlparse is None:
        raise ImportError(
            "SQL safety validation requires sqlparse==0.5.3. "
            "Install the project requirements before running the agent."
        ) from _SQLPARSE_IMPORT_ERROR


def _meaningful_tokens(statement: Any) -> list[Any]:
    assert T is not None
    meaningful: list[Any] = []
    for token in statement.flatten():
        if token.is_whitespace or token.ttype in T.Comment:
            continue
        meaningful.append(token)
    return meaningful


def validate_readonly_sql(sql: str) -> str:
    """Return one normalized read-only query or raise a safety error."""

    _require_sqlparse()
    text = "" if sql is None else str(sql).strip()
    if not text:
        raise SQLSafetyError("Empty SQL query.")

    statements = [statement for statement in sqlparse.parse(text) if str(statement).strip()]
    if len(statements) != 1:
        raise SQLSafetyError("Only one SQL statement is allowed.")

    statement = statements[0]
    meaningful = _meaningful_tokens(statement)
    while meaningful and meaningful[-1].value == ";":
        meaningful.pop()
    if not meaningful:
        raise SQLSafetyError("Empty SQL query.")
    if any(token.value == ";" for token in meaningful):
        raise SQLSafetyError("Only one SQL statement is allowed.")

    first_keyword = meaningful[0].normalized.upper()
    statement_type = statement.get_type().upper()
    if first_keyword not in {"SELECT", "WITH"} or statement_type != "SELECT":
        raise SQLSafetyError("Only SELECT or WITH ... SELECT queries are allowed.")

    assert T is not None
    for token in meaningful:
        if token.ttype in T.Literal.String:
            continue
        normalized = token.normalized.upper()
        if normalized in FORBIDDEN_KEYWORDS:
            raise SQLSafetyError(f"Forbidden SQL keyword: {normalized}.")

    for token in meaningful:
        # Single-quoted literals are data. Double-quoted symbols are identifiers
        # and must still be checked for metadata object names.
        if token.ttype in T.Literal.String.Single:
            continue
        normalized = token.value.strip('"`[]').upper()
        if (
            normalized in FORBIDDEN_METADATA_IDENTIFIERS
            or normalized in FORBIDDEN_METADATA_FUNCTIONS
            or normalized.startswith(FORBIDDEN_METADATA_PREFIXES)
        ):
            raise SQLSafetyError(f"Metadata access is forbidden: {normalized}.")

    normalized_sql = str(statement).strip()
    if normalized_sql.endswith(";"):
        normalized_sql = normalized_sql[:-1].rstrip()
    return normalized_sql + ";"


__all__ = [
    "FORBIDDEN_KEYWORDS",
    "FORBIDDEN_METADATA_IDENTIFIERS",
    "SQLSafetyError",
    "validate_readonly_sql",
]
