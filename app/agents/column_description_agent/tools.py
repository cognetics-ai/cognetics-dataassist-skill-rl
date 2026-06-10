from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from app.agents.common.runtime_context import AgentDependencies
from app.observability import get_logger

_logger = get_logger(__name__)

def build_tools(deps: AgentDependencies) -> list:
    """Build tools for column-description synthesis."""

    async def sample_column_values(
        catalog: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        column_names: list[str] | None = None,
        sample_size: int | None = None,
        engine: str | None = None,
        tool_context: Any | None = None,
    ) -> dict[str, Any]:
        """Fetch distinct sampled values for selected backend columns.

        Args:
            catalog: Backend catalog name.
            schema_name: Backend schema name.
            table_name: Backend table name.
            column_names: Column names to project from the table.
            sample_size: Maximum sample rows to return.
            engine: Engine adapter name.
            tool_context: ADK context used to persist sample payload.

        Returns:
            Payload with one distinct non-null sample set per selected column.
        """
        _logger.info(f"Agent calling tool: sample_column_values with catalog={catalog}, "
                     f"schema={schema_name}, table={table_name}")
        catalog = str(catalog or _state_value(tool_context, "catalog") or "").strip()
        schema_name = str(schema_name or _state_value(tool_context, "schema_name") or "").strip()
        table_name = str(table_name or _state_value(tool_context, "table_name") or "").strip()
        engine = str(engine or _state_value(tool_context, "engine") or "starburst").strip()
        sample_size = sample_size or _state_value(tool_context, "sample_size") or 5
        if not catalog or not schema_name or not table_name:
            raise ValueError(
                "catalog, schema_name, and table_name are required for column sampling"
            )

        limit = max(1, min(int(sample_size or 5), 25))
        selected_columns = _column_names(column_names, tool_context)
        _logger.debug(f"Selected columns: {selected_columns}")
        if not selected_columns:
            _logger.error("No columns selected")
            raise ValueError("At least one column name is required for column sampling")

        adapter = deps.engines.get(engine)
        payload = await _fetch_distinct_column_samples(
            adapter,
            catalog=catalog,
            schema_name=schema_name,
            table_name=table_name,
            column_names=selected_columns,
            limit=limit,
        )
        actual_columns = [item["column_name"] for item in payload["column_samples"]]
        unresolved_columns = [
            column
            for column in selected_columns
            if _column_key(column) not in {_column_key(actual) for actual in actual_columns}
        ]
        if unresolved_columns:
            payload["unresolved_columns"] = unresolved_columns

        payload.update(
            {
                "engine": engine,
                "catalog": catalog,
                "schema_name": schema_name,
                "table_name": table_name,
                "column_names": selected_columns,
                "sample_size": limit,
            }
        )

        if tool_context:
            tool_context.state["sampled_column_values_json"] = json.dumps(payload, default=str)
        return payload

    return [sample_column_values]


def _state_value(tool_context: Any | None, key: str) -> Any:
    if not tool_context:
        return None
    return tool_context.state.get(key)


async def _fetch_first_success(adapter: Any, sqls: list[str]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    last_error = ""
    for sql in sqls:
        try:
            payload = await _fetch_sample(adapter, sql)
            if payload["rows"] or sql == sqls[-1]:
                return payload, warnings
            warnings.append("Query returned no rows")
        except Exception as exc:
            last_error = str(exc)
            warnings.append(f"Query failed: {_short_text(last_error)}")
    raise RuntimeError(f"All sample query attempts failed. Last error: {_short_text(last_error)}")


async def _fetch_distinct_column_samples(
    adapter: Any,
    *,
    catalog: str,
    schema_name: str,
    table_name: str,
    column_names: list[str],
    limit: int,
) -> dict[str, Any]:
    table_ref = ".".join(_quote_ident(part) for part in (catalog, schema_name, table_name))
    column_samples: list[dict[str, Any]] = []
    rows_by_column: dict[str, list[Any]] = {}
    warnings: list[str] = []
    for column in column_names:
        _logger.debug(f"Fetching samples for {column}")
        quoted = _quote_ident(column)
        safe = _safe_ident(column)
        sqls = _unique_sqls(
            [
                (
                    f"SELECT DISTINCT {quoted} AS {quoted} "
                    f"FROM {table_ref} WHERE {quoted} IS NOT NULL LIMIT {limit}"
                ),
                f"SELECT DISTINCT {quoted} AS {quoted} FROM {table_ref} LIMIT {limit}",
                (
                    f"SELECT DISTINCT {safe} AS {safe} "
                    f"FROM {table_ref} WHERE {safe} IS NOT NULL LIMIT {limit}"
                ),
                f"SELECT DISTINCT {safe} AS {safe} FROM {table_ref} LIMIT {limit}",
            ]
        )
        try:
            sample_payload, sampling_warnings = await _fetch_first_success(adapter, sqls)
            values = _sample_values(sample_payload, column)
            warnings.extend(f"{column}: {warning}" for warning in sampling_warnings)
            column_samples.append(
                {
                    "column_name": column,
                    "sample_values": values,
                    "row_count": len(values),
                    "sql": sample_payload.get("sql", ""),
                    "schema": sample_payload.get("schema", []),
                }
            )
            rows_by_column[column] = values
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{column}: {_short_text(exc)}")
            column_samples.append(
                {
                    "column_name": column,
                    "sample_values": [],
                    "row_count": 0,
                    "sql": "",
                    "schema": [],
                    "error": _short_text(exc),
                }
            )
            rows_by_column[column] = []

    max_rows = max((len(values) for values in rows_by_column.values()), default=0)
    rows = [
        {
            column: _jsonable(values[idx]) if idx < len(values) else None
            for column, values in rows_by_column.items()
        }
        for idx in range(max_rows)
    ]
    payload: dict[str, Any] = {
        "column_samples": column_samples,
        "schema": [{"name": column, "type": ""} for column in column_names],
        "rows": rows,
        "row_count": max_rows,
    }
    if warnings:
        payload["sampling_warning"] = " | ".join(warnings)
    return payload


async def _fetch_sample(adapter: Any, sql: str) -> dict[str, Any]:
    handle = await adapter.execute_async(sql)
    results = await adapter.fetch_results(handle)
    error = (handle.raw.get("lastPayload") or {}).get("error")
    if error:
        raise RuntimeError(f"Sample query failed: {error.get('message') or error}")

    columns = _result_columns(results.schema, results.rows)
    rows = [
        {
            columns[idx]: _jsonable(value)
            for idx, value in enumerate(row[: len(columns)])
        }
        for row in results.rows
    ]
    return {
        "sql": sql,
        "schema": results.schema,
        "rows": _truncate_rows(rows),
        "row_count": len(rows),
    }


def _sample_values(payload: dict[str, Any], requested_column: str) -> list[Any]:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    if not rows:
        return []
    payload_columns = _payload_columns(payload)
    requested_key = _column_key(requested_column)
    matching_key = next(
        (column for column in payload_columns if _column_key(column) == requested_key),
        payload_columns[0] if payload_columns else requested_column,
    )
    values: list[Any] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_key = next(
            (key for key in row if _column_key(key) == _column_key(matching_key)),
            matching_key,
        )
        value = row.get(row_key)
        if value is None:
            continue
        key = json.dumps(_jsonable(value), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        values.append(_jsonable(value))
    return values


def _result_columns(schema: list[dict[str, Any]], rows: list[list[Any]]) -> list[str]:
    columns = [str(col.get("name") or f"col_{idx}") for idx, col in enumerate(schema)]
    if columns:
        return columns
    width = max((len(row) for row in rows), default=0)
    return [f"col_{idx}" for idx in range(width)]


def _payload_columns(payload: dict[str, Any]) -> list[str]:
    schema = payload.get("schema") if isinstance(payload.get("schema"), list) else []
    columns = [
        str(col.get("name") or "")
        for col in schema
        if isinstance(col, dict) and col.get("name")
    ]
    if columns:
        return columns
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    keys: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            keys.extend(str(key) for key in row)
    return keys


def _column_key(value: Any) -> str:
    return str(value or "").strip().strip('"').strip("`").lower()


def _unique_sqls(sqls: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for sql in sqls:
        if sql in seen:
            continue
        seen.add(sql)
        unique.append(sql)
    return unique


def _column_names(column_names: Any, tool_context: Any | None) -> list[str]:
    if isinstance(column_names, str):
        parsed = _load_json(column_names, default=None)
        values = parsed if isinstance(parsed, list) else column_names.split(",")
    else:
        values = column_names or []

    names = [str(column).strip() for column in values if str(column).strip()]
    if names:
        return names

    if not tool_context:
        return []

    raw_metadata = tool_context.state.get("column_metadata_json")
    metadata = _load_json(raw_metadata, default=[])
    if not isinstance(metadata, list):
        return []
    return [
        str(item.get("column_name") or "").strip()
        for item in metadata
        if str(item.get("column_name") or "").strip()
    ]


def _load_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _safe_ident(identifier: str) -> str:
    text = str(identifier)
    if text.replace("_", "a").isalnum() and (text[:1].isalpha() or text[:1] == "_"):
        return text
    return _quote_ident(text)


def _short_text(value: Any, max_chars: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _truncate_rows(rows: list[dict[str, Any]], max_cell_chars: int = 500) -> list[dict[str, Any]]:
    truncated: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, str) and len(value) > max_cell_chars:
                item[key] = value[:max_cell_chars] + "..."
            else:
                item[key] = value
        truncated.append(item)
    return truncated
