from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from app.agents.common.runtime_context import AgentDependencies
from app.observability import get_logger

_logger = get_logger(__name__)

def build_tools(deps: AgentDependencies) -> list:
    """Build tools for table-description synthesis."""

    async def sample_table_rows(
        catalog: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        sample_size: int | None = None,
        engine: str | None = None,
        tool_context: Any | None = None,
    ) -> dict[str, Any]:
        """Fetch randomly sampled rows from a backend table.

        Args:
            catalog: Backend catalog name.
            schema_name: Backend schema name.
            table_name: Backend table name.
            sample_size: Maximum sample rows to return.
            engine: Engine adapter name.
            tool_context: ADK context used to persist sample payload.

        Returns:
            Payload with SQL, result schema, row dictionaries, and row count.
        """
        _logger.info("In Agent tools to sample table rows")
        catalog = str(catalog or _state_value(tool_context, "catalog") or "").strip()
        schema_name = str(schema_name or _state_value(tool_context, "schema_name") or "").strip()
        table_name = str(table_name or _state_value(tool_context, "table_name") or "").strip()
        engine = str(engine or _state_value(tool_context, "engine") or "starburst").strip()
        sample_size = sample_size or _state_value(tool_context, "sample_size") or 5
        if not catalog or not schema_name or not table_name:
            raise ValueError("catalog, schema_name, and table_name are required for table sampling")

        limit = max(1, min(int(sample_size or 5), 25))
        table_ref = ".".join(_quote_ident(part) for part in (catalog, schema_name, table_name))
        sample_sql = f"SELECT * FROM {table_ref} TABLESAMPLE BERNOULLI (10) LIMIT {limit}"
        fallback_sql = f"SELECT * FROM {table_ref} LIMIT {limit}"

        adapter = deps.engines.get(engine)
        sampling_warning = ""
        try:
            payload = await _fetch_sample(adapter, sample_sql)
            _logger.debug(f"Returned sample payload: {payload}")
        except Exception as exc:
            _logger.error("Failed to fetch sample payload. Falling back to fallback_sql", exc_info=exc)
            sampling_warning = f"TABLESAMPLE query failed; used LIMIT fallback. Error: {exc}"
            payload = await _fetch_sample(adapter, fallback_sql)
        if not payload["rows"] and payload["sql"] != fallback_sql:
            payload = await _fetch_sample(adapter, fallback_sql)
        if sampling_warning:
            payload["sampling_warning"] = sampling_warning

        payload.update(
            {
                "engine": engine,
                "catalog": catalog,
                "schema_name": schema_name,
                "table_name": table_name,
                "sample_size": limit,
            }
        )

        if tool_context:
            tool_context.state["sampled_table_rows_json"] = json.dumps(payload, default=str)
        return payload

    return [sample_table_rows]


def _state_value(tool_context: Any | None, key: str) -> Any:
    if not tool_context:
        return None
    return tool_context.state.get(key)


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


def _result_columns(schema: list[dict[str, Any]], rows: list[list[Any]]) -> list[str]:
    columns = [str(col.get("name") or f"col_{idx}") for idx, col in enumerate(schema)]
    if columns:
        return columns
    width = max((len(row) for row in rows), default=0)
    return [f"col_{idx}" for idx in range(width)]


def _quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


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
