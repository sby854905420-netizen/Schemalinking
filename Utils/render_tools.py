from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ColumnRenderingConfig:
    sample_values_max_tokens: int = 256
    value_descriptions_max_tokens: int = 1024
    missing_value_text: str = "NOT_AVAILABLE"
    truncation_suffix: str = "..."
    sample_value_separator: str = "; "


class SchemaTextRenderer:
    """Render schema text documents for downstream retrieval and prompting.

    This renderer implements column-, table-, and database-level rendering.
    Higher-level renderers reuse lower-level renderers and preserve the
    caller-provided schema order.
    """

    def __init__(self, tokenizer: Any, config: ColumnRenderingConfig | None = None):
        if tokenizer is None:
            raise ValueError("tokenizer must be provided for token-based rendering.")

        self.tokenizer = tokenizer
        self.config = config or ColumnRenderingConfig()

    def render_column_fields(self, record: Mapping[str, Any]) -> dict[str, str]:
        """Build normalized, rendered field values for a single column record."""

        column_name = self._require_text(record, ("column_name",))
        column_type = self._optional_text(
            record,
            ("column_data_type", "column_type", "data_type"),
        )
        description = self._optional_text(
            record,
            ("column_description", "description"),
        )
        sample_values = self._render_sample_values(record)
        value_descriptions = self._render_value_descriptions(record)

        return {
            "Column": column_name,
            "Data type": column_type or self.config.missing_value_text,
            "Description": description or self.config.missing_value_text,
            "Sample values": sample_values,
            "Value descriptions": value_descriptions,
        }

    def render_column_text(self, record: Mapping[str, Any]) -> str:
        """Render a single column document using the agreed base template."""

        fields = self.render_column_fields(record)
        return "\n".join(f"{label}: {value}" for label, value in fields.items())

    def render_index_column_fields(self, record: Mapping[str, Any]) -> dict[str, str]:
        """Build sparse fields for embedding-based global column retrieval.

        This index-facing template keeps the most semantically stable fields:
        table name, column name, optional description, and optional value
        descriptions. Missing optional fields are omitted entirely.
        """

        table_name = self._require_text(
            record,
            ("table_name",),
            meta_keys=("table_name",),
        )
        column_name = self._require_text(record, ("column_name",))
        description = self._optional_text(
            record,
            ("column_description", "description"),
        )
        value_descriptions = self._optional_text(record, ("value_descriptions",))

        fields = {
            "Table": table_name,
            "Column": column_name,
        }
        if description:
            fields["Description"] = description
        if value_descriptions:
            fields["Value descriptions"] = value_descriptions
        return fields

    def render_index_column_text(self, record: Mapping[str, Any]) -> str:
        """Render the sparse column template used by the vector index."""

        fields = self.render_index_column_fields(record)
        return "\n".join(f"{label}: {value}" for label, value in fields.items())

    def render_table_text(
        self,
        table_name: Any,
        column_records: Sequence[Mapping[str, Any]],
    ) -> str:
        """Render a table block from an ordered sequence of column records."""

        normalized_table_name = self._normalize_scalar(table_name)
        if not normalized_table_name:
            raise ValueError("table_name is required for table-level rendering.")

        rendered_columns = [self.render_column_text(record) for record in column_records]
        lines = [f"Table: {normalized_table_name}"]
        if rendered_columns:
            lines.extend(["", "\n\n".join(rendered_columns)])
        return "\n".join(lines)

    def render_database_text(
        self,
        db_id: Any,
        tables: Mapping[str, Sequence[Mapping[str, Any]]] | Iterable[tuple[str, Sequence[Mapping[str, Any]]]],
    ) -> str:
        """Render a database block from ordered table data.

        Parameters
        ----------
        db_id:
            Database identifier shown once at the top of the rendered block.
        tables:
            Either an ordered mapping of table_name -> ordered column records,
            or an iterable of (table_name, ordered column records) pairs.
        """

        normalized_db_id = self._normalize_scalar(db_id)
        if not normalized_db_id:
            raise ValueError("db_id is required for database-level rendering.")

        rendered_tables = [
            self.render_table_text(table_name, column_records)
            for table_name, column_records in self._iter_tables(tables)
        ]

        lines = [f"Database: {normalized_db_id}"]
        if rendered_tables:
            lines.extend(["", "\n\n".join(rendered_tables)])
        return "\n".join(lines)

    def render_selected_database_text(
        self,
        db_id: Any,
        column_records: Sequence[Mapping[str, Any]],
        primary_keys_available: bool | None = None,
        foreign_keys_available: bool | None = None,
    ) -> str:
        """Render a full database block from an ordered set of selected columns.

        This is the coarse-retrieval facing entry point. It keeps the input
        column order within each table, renders primary/foreign-key text above
        the table blocks, and avoids repeating database/table names at the
        column level.
        """

        normalized_db_id = self._normalize_scalar(db_id)
        if not normalized_db_id:
            raise ValueError("db_id is required for database-level rendering.")

        rendered_tables = [
            self.render_table_text(table_name, table_records)
            for table_name, table_records in self._group_column_records_by_table(column_records)
        ]
        key_relationship_text = self.render_key_relationship_text(
            column_records,
            primary_keys_available=primary_keys_available,
            foreign_keys_available=foreign_keys_available,
        )

        lines = [f"Database: {normalized_db_id}"]
        if key_relationship_text:
            lines.extend(["", key_relationship_text])
        if rendered_tables:
            lines.extend(["", "\n\n".join(rendered_tables)])
        return "\n".join(lines)

    def render_key_relationship_text(
        self,
        column_records: Sequence[Mapping[str, Any]],
        primary_keys_available: bool | None = None,
        foreign_keys_available: bool | None = None,
    ) -> str:
        """Render primary-key and foreign-key relationship text from selected columns.

        Parameters
        ----------
        column_records:
            The currently selected column records. The output is filtered against
            this set so that key relationship lines stay consistent with the
            rendered schema body.
        primary_keys_available:
            Optional override describing whether primary-key metadata is
            available for the current database. When False, the section renders
            as NOT_AVAILABLE.
        foreign_keys_available:
            Optional override describing whether foreign-key metadata is
            available for the current database. When False, the section renders
            as NOT_AVAILABLE.
        """

        if primary_keys_available is None:
            primary_keys_available = any("is_primary_key" in record for record in column_records)
        if foreign_keys_available is None:
            foreign_keys_available = any(
                "is_foreign_key" in record or "foreign_key" in record
                for record in column_records
            )

        selected_column_paths: set[str] = set()
        primary_key_columns_by_table: dict[str, list[str]] = {}
        foreign_key_relationships: list[str] = []
        seen_relationships: set[str] = set()

        for record in column_records:
            table_name = self._optional_text(
                record,
                ("table_name",),
                meta_keys=("table_name",),
            )
            column_name = self._optional_text(record, ("column_name",))
            if not table_name or not column_name:
                continue

            source_column_path = f"{table_name}.{column_name}"
            selected_column_paths.add(source_column_path)

            if self._is_truthy_flag(record.get("is_primary_key")):
                primary_key_columns_by_table.setdefault(table_name, [])
                if column_name not in primary_key_columns_by_table[table_name]:
                    primary_key_columns_by_table[table_name].append(column_name)

        for record in column_records:
            table_name = self._optional_text(
                record,
                ("table_name",),
                meta_keys=("table_name",),
            )
            column_name = self._optional_text(record, ("column_name",))
            if not table_name or not column_name:
                continue

            if not self._is_truthy_flag(record.get("is_foreign_key")):
                continue

            target_column_path = self._optional_text(record, ("foreign_key",))
            if not target_column_path:
                continue
            if target_column_path not in selected_column_paths:
                continue

            relationship = f"{table_name}.{column_name} -> {target_column_path}"
            if relationship in seen_relationships:
                continue
            seen_relationships.add(relationship)
            foreign_key_relationships.append(relationship)

        lines = ["Primary keys:"]
        if not primary_keys_available:
            lines.append(self.config.missing_value_text)
        elif primary_key_columns_by_table:
            for table_name, column_names in primary_key_columns_by_table.items():
                lines.append(f"- {table_name}: {', '.join(column_names)}")
        else:
            lines.append("NONE")

        lines.extend(["", "Foreign key relationships:"])
        if not foreign_keys_available:
            lines.append(self.config.missing_value_text)
        elif foreign_key_relationships:
            lines.extend(f"- {relationship}" for relationship in foreign_key_relationships)
        else:
            lines.append("NONE")

        return "\n".join(lines)

    def count_tokens(self, text: str) -> int:
        return len(self._encode_text(text))

    def truncate_to_token_budget(
        self,
        text: str,
        max_tokens: int,
        suffix: str | None = None,
    ) -> str:
        """Truncate text by token budget while reserving room for the suffix."""

        normalized_text = self._normalize_scalar(text)
        if not normalized_text or max_tokens <= 0:
            return ""

        token_ids = self._encode_text(normalized_text)
        if len(token_ids) <= max_tokens:
            return normalized_text

        suffix_text = self.config.truncation_suffix if suffix is None else suffix
        suffix_token_ids = self._encode_text(suffix_text) if suffix_text else []
        available_tokens = max_tokens - len(suffix_token_ids)
        if available_tokens <= 0:
            available_tokens = max_tokens
            suffix_text = ""

        kept_token_ids = token_ids[:available_tokens]
        candidate = self._decode_tokens(kept_token_ids).rstrip()
        if suffix_text:
            candidate = f"{candidate}{suffix_text}"

        while candidate and self.count_tokens(candidate) > max_tokens and kept_token_ids:
            kept_token_ids = kept_token_ids[:-1]
            candidate = self._decode_tokens(kept_token_ids).rstrip()
            if suffix_text:
                candidate = f"{candidate}{suffix_text}"

        if candidate:
            return candidate

        if not suffix_text:
            return ""

        suffix_only = suffix_text
        while suffix_only and self.count_tokens(suffix_only) > max_tokens:
            suffix_only = suffix_only[:-1]
        return suffix_only

    def _render_sample_values(self, record: Mapping[str, Any]) -> str:
        sample_values = self._extract_sample_values(record)
        if not sample_values:
            return self.config.missing_value_text

        joined_text = self.config.sample_value_separator.join(sample_values)
        if self.count_tokens(joined_text) <= self.config.sample_values_max_tokens:
            return joined_text

        first_value = sample_values[0]
        if self.count_tokens(first_value) <= self.config.sample_values_max_tokens:
            return first_value

        return self.truncate_to_token_budget(
            first_value,
            max_tokens=self.config.sample_values_max_tokens,
        )

    def _render_value_descriptions(self, record: Mapping[str, Any]) -> str:
        value_descriptions = self._optional_text(record, ("value_descriptions",))
        if not value_descriptions:
            return self.config.missing_value_text

        if self.count_tokens(value_descriptions) <= self.config.value_descriptions_max_tokens:
            return value_descriptions

        return self.truncate_to_token_budget(
            value_descriptions,
            max_tokens=self.config.value_descriptions_max_tokens,
        )

    def _extract_sample_values(self, record: Mapping[str, Any]) -> list[str]:
        raw_sample_values = record.get("sample_values", [])
        sample_values: list[str] = []

        if isinstance(raw_sample_values, list):
            candidates = raw_sample_values
        elif raw_sample_values is None:
            candidates = []
        else:
            candidates = [raw_sample_values]

        for value in candidates:
            normalized_value = self._normalize_scalar(value)
            if normalized_value:
                sample_values.append(normalized_value)

        if sample_values:
            return sample_values

        fallback_text = self._optional_text(record, ("sample_values_text",))
        if fallback_text:
            return [fallback_text]

        return []

    def _require_text(
        self,
        record: Mapping[str, Any],
        keys: tuple[str, ...],
        meta_keys: tuple[str, ...] = (),
    ) -> str:
        value = self._optional_text(record, keys, meta_keys=meta_keys)
        if value:
            return value
        if meta_keys:
            raise ValueError(f"Missing required field among keys: {keys} and meta keys: {meta_keys}")
        raise ValueError(f"Missing required field among keys: {keys}")

    def _optional_text(
        self,
        record: Mapping[str, Any],
        keys: tuple[str, ...],
        meta_keys: tuple[str, ...] = (),
    ) -> str:
        for key in keys:
            if key in record:
                return self._normalize_scalar(record.get(key))
        meta = record.get("meta_data")
        if isinstance(meta, Mapping):
            for key in meta_keys:
                if key in meta:
                    return self._normalize_scalar(meta.get(key))
        return ""

    def _iter_tables(
        self,
        tables: Mapping[str, Sequence[Mapping[str, Any]]] | Iterable[tuple[str, Sequence[Mapping[str, Any]]]],
    ) -> Iterable[tuple[str, Sequence[Mapping[str, Any]]]]:
        if isinstance(tables, Mapping):
            yield from tables.items()
            return

        yield from tables

    def _group_column_records_by_table(
        self,
        column_records: Sequence[Mapping[str, Any]],
    ) -> Iterable[tuple[str, list[Mapping[str, Any]]]]:
        grouped_records: dict[str, list[Mapping[str, Any]]] = {}
        for record in column_records:
            table_name = self._optional_text(
                record,
                ("table_name",),
                meta_keys=("table_name",),
            )
            if not table_name:
                continue
            grouped_records.setdefault(table_name, []).append(record)
        yield from grouped_records.items()

    def _normalize_scalar(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False).strip()
        return str(value).strip()

    def _encode_text(self, text: str) -> list[Any]:
        if hasattr(self.tokenizer, "encode"):
            try:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            except TypeError:
                try:
                    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                except TypeError:
                    token_ids = self.tokenizer.encode(text)
        elif callable(self.tokenizer):
            token_ids = self.tokenizer(text, add_special_tokens=False, truncation=False)
        else:
            raise TypeError("Tokenizer must provide an encode method or be callable.")

        if isinstance(token_ids, Mapping):
            token_ids = token_ids.get("input_ids", [])

        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        if isinstance(token_ids, tuple):
            token_ids = list(token_ids)

        if not isinstance(token_ids, list):
            raise TypeError("Tokenizer.encode returned an unsupported token container.")

        return token_ids

    def _decode_tokens(self, token_ids: list[Any]) -> str:
        if not hasattr(self.tokenizer, "decode"):
            raise TypeError("Tokenizer must provide a decode method for token-based truncation.")

        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            return self.tokenizer.decode(token_ids)

    def _is_truthy_flag(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return False


__all__ = ["ColumnRenderingConfig", "SchemaTextRenderer"]
