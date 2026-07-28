from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from Utils.schema_selection import (
    DbInfoSchemaStore,
    canonicalize_snowflake_schema_predictions,
    select_predicted_column_records,
)
from Utils.schema_prediction_utils import (
    normalize_predicted_columns,
    normalize_predicted_tables,
)
from Utils.tools import resolve_hint
from Utils.value_utils import choose_external_knowledge_source

from .result_types import AgentInput


class AdapterError(ValueError):
    pass


class SchemaLinkingAdapter:
    """Turn a validated schema prediction into a typed agent input."""

    def __init__(
        self,
        *,
        dataset_name: str,
        documents_dir: Any,
        schema_store: DbInfoSchemaStore,
        sql_dialect: str,
        include_key_columns: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.documents_dir = documents_dir
        self.schema_store = schema_store
        self.sql_dialect = sql_dialect
        self.include_key_columns = include_key_columns

    def adapt(
        self,
        row: Mapping[str, Any],
        source_row: Mapping[str, Any] | None = None,
    ) -> AgentInput:
        source = source_row or {}
        schema_input_error = str(row.get("schema_input_error") or "").strip()
        if schema_input_error:
            raise AdapterError(schema_input_error)
        raw_sample_id = row.get("id")
        raw_question = row.get("question")
        raw_predict_db_id = row.get("predict_db_id")
        sample_id = raw_sample_id.strip() if isinstance(raw_sample_id, str) else ""
        question = raw_question.strip() if isinstance(raw_question, str) else ""
        predict_db_id = (
            raw_predict_db_id.strip() if isinstance(raw_predict_db_id, str) else ""
        )
        if not sample_id:
            raise AdapterError("Missing stable sample id.")
        if not question:
            raise AdapterError("Missing question.")
        if not predict_db_id:
            raise AdapterError("Missing predicted database.")

        raw_predicted_columns = row.get("predict_columns")
        raw_predicted_tables = row.get("predict_tables")
        if not isinstance(raw_predicted_columns, Mapping):
            raise AdapterError("Predicted columns must be a mapping.")
        if any(
            not isinstance(table, str)
            or not isinstance(columns, list)
            or any(not isinstance(column, str) for column in columns)
            for table, columns in raw_predicted_columns.items()
        ):
            raise AdapterError(
                "Predicted columns must map table names to lists of column names.",
            )
        if not isinstance(raw_predicted_tables, list) or any(
            not isinstance(table, str) for table in raw_predicted_tables
        ):
            raise AdapterError(
                "Predicted tables must be a list of table names.",
            )
        predicted_columns = normalize_predicted_columns(raw_predicted_columns)
        predicted_tables = normalize_predicted_tables(raw_predicted_tables)
        name_diagnostics: dict[str, Any] = {}
        if self.dataset_name.strip().lower() == "spider2":
            (
                predicted_columns,
                predicted_tables,
                name_diagnostics,
            ) = canonicalize_snowflake_schema_predictions(
                db_id=predict_db_id,
                predicted_columns=predicted_columns,
                predicted_tables=predicted_tables,
                schema_store=self.schema_store,
            )
        selected_records, _ = select_predicted_column_records(
            db_id=predict_db_id,
            predicted_columns=predicted_columns,
            predicted_tables=predicted_tables,
            schema_store=self.schema_store,
            include_key_columns=self.include_key_columns,
        )
        if not selected_records:
            if predict_db_id not in self.schema_store.db_info_index:
                raise AdapterError(
                    f"Predicted database '{predict_db_id}' is absent from db_info.json.",
                )
            unresolved = [
                *name_diagnostics.get("unresolved_column_tables", []),
                *name_diagnostics.get("ambiguous_column_tables", []),
            ]
            detail = (
                f" Unresolved or ambiguous table identifiers: {unresolved!r}."
                if unresolved
                else ""
            )
            raise AdapterError(
                f"No selected schema columns.{detail}",
            )

        # Method-specific preparation (including AutoLink's explicit
        # oracle-database filtering) has already been materialized in the row.
        # The source row is used only as the authoritative hint source.
        hint_source = choose_external_knowledge_source(source, row)
        hint = resolve_hint(
            hint_source,
            dataset_name=self.dataset_name,
            documents_dir=self.documents_dir,
        )
        schema_text = self.schema_store.render_schema_text(predict_db_id, selected_records)
        frozen_columns = {
            table: tuple(columns) for table, columns in predicted_columns.items()
        }
        return AgentInput(
            sample_id=sample_id,
            dataset_name=self.dataset_name,
            question=question,
            hint=hint,
            predict_db_id=predict_db_id,
            predict_tables=tuple(predicted_tables),
            predict_columns=frozen_columns,
            selected_column_records=tuple(dict(record) for record in selected_records),
            schema_text=schema_text,
            sql_dialect=self.sql_dialect,
        )


__all__ = [
    "AdapterError",
    "SchemaLinkingAdapter",
]
