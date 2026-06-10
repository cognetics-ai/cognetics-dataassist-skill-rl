from __future__ import annotations

import json
import logging
from typing import Any

from app.agents.common.runtime_context import AgentDependencies
from app.observability import get_logger

_logger = get_logger(__name__)

def build_tools(deps: AgentDependencies) -> list:
    """Build tools for query-to-NLP synthesis."""

    async def load_query_history_context(
        query: str | None = None,
        raw_history_id: int | None = None,
        engine: str | None = None,
        tool_context: Any | None = None,
    ) -> dict[str, Any]:
        """Load one raw query-history row and related table/column descriptions.

        Args:
            query: Raw SQL text used to find one query-history row.
            raw_history_id: Query-history row ID used to find an exact row.
            engine: Optional engine filter.
            tool_context: ADK context used to read and persist state.

        Returns:
            Query history row plus table and column metadata context.
        """
        _logger.debug(f"Tool call to load query history context with "
                      f"raw history id: {raw_history_id}, engine: {engine}")
        raw_sql = str(_state_value(tool_context, "raw_sql") or query or "").strip()
        history_id = int(_state_value(tool_context, "raw_history_id") or raw_history_id or 0)
        engine_name = str(_state_value(tool_context, "engine") or engine or "").strip()
        if not raw_sql and history_id <= 0:
            raise ValueError("raw_history_id or query/raw_sql is required to load query history context")

        source_id = str(_state_value(tool_context, "source_id") or "").strip()
        _logger.debug(f"Source ID: {source_id}")
        if source_id:
            if not deps.has_catalog or deps.skillsql_resources is None:
                raise RuntimeError("Catalog repository is required for source_id-scoped query history")
            payload = deps.skillsql_resources.repo.get_query_history_context_by_raw_sql(
                source_id=source_id,
                raw_sql=raw_sql or None,
                raw_history_id=history_id or None,
                engine=engine_name or None,
            )
        else:
            payload = await deps.store.get_backend_query_history_context_by_raw_sql(
                raw_sql=raw_sql or None,
                raw_history_id=history_id or None,
                engine=engine_name or None,
            )
        if not payload:
            table_name = "CATALOG_QUERY_HISTORY" if source_id else "BACKEND_QUERY_HISTORY_RAW"
            raise ValueError(f"No {table_name} row matched the provided raw SQL")

        payload = _truncate_context(payload)
        if tool_context:
            tool_context.state["query_history_context_json"] = json.dumps(payload, default=str)
        return payload

    return [load_query_history_context]


def _state_value(tool_context: Any | None, key: str) -> Any:
    if not tool_context:
        return None
    return tool_context.state.get(key)


def _truncate_context(payload: dict[str, Any]) -> dict[str, Any]:
    _logger.debug(f"Payload in tools...: {payload}")
    tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []
    _logger.debug(f"Tables in tools...: {tables}")
    compact_tables: list[dict[str, Any]] = []
    for table in tables[:20]:
        if not isinstance(table, dict):
            continue
        columns = table.get("columns") if isinstance(table.get("columns"), list) else []
        compact_tables.append(
            {
                "name": table.get("name"),
                "catalog": table.get("catalog"),
                "schema_name": table.get("schema_name"),
                "table_name": table.get("table_name"),
                "description": _truncate_text(table.get("description"), 600),
                "columns": [
                    {
                        "column_name": column.get("column_name"),
                        "data_type": column.get("data_type"),
                        "description": _truncate_text(column.get("description"), 400),
                    }
                    for column in columns[:120]
                    if isinstance(column, dict)
                ],
            }
        )
    return {
        "history": payload.get("history") or {},
        "raw_sql": payload.get("raw_sql") or "",
        "query_state": payload.get("history")["QUERY_STATE"] or {},
        "error_exception_message": payload.get("history")["METRICS_JSON"]["error_exception_message"] or {},
        "tables_json": payload.get("tables_json") or [],
        "tables": compact_tables,
    }


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
