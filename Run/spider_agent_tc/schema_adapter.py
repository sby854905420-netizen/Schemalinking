from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from Utils.schema_selection import (
    DbInfoSchemaStore,
    canonicalize_snowflake_schema_predictions,
    select_predicted_column_records,
)
from Utils.schema_prediction_utils import resolve_predicted_columns, resolve_predicted_tables
from Utils.tools import resolve_hint
from Utils.value_utils import choose_external_knowledge_source, get_row_value

from .result_types import AgentInput


class AdapterError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SchemaLinkingAdapter:
    """Turn legacy MDB-Link logs into a typed, prediction-only agent input."""

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
        *,
        fallback_sample_id: str = "",
    ) -> AgentInput:
        source = source_row or {}
        schema_input_error = str(row.get("schema_input_error") or "").strip()
        if schema_input_error:
            raise AdapterError("schema_input_error", schema_input_error)
        sample_id_value = get_row_value(row, "id", "instance_id") or get_row_value(
            source, "id", "instance_id"
        )
        sample_id = str(sample_id_value or fallback_sample_id).strip()
        question = str(row.get("question") or source.get("question") or "").strip()
        predict_db_id = str(get_row_value(row, "predict_db_id") or "").strip()
        if not sample_id:
            raise AdapterError("missing_id", "Missing stable sample id.")
        if not question:
            raise AdapterError("missing_question", "Missing question.")
        if not predict_db_id:
            raise AdapterError("missing_predict_db_id", "Missing predicted database.")

        predicted_columns = resolve_predicted_columns(row)
        predicted_tables = resolve_predicted_tables(row)
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
                    "invalid_predict_db_id",
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
                "empty_predicted_schema",
                f"No selected schema columns.{detail}",
            )

        # Only question/hint/id are read from source data here. Any method-specific
        # preparation (including AutoLink's explicit oracle-database filtering) has
        # already been materialized in the prediction row.
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
