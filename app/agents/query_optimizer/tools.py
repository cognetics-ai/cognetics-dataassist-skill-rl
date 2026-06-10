from __future__ import annotations

import json
from typing import Any

from app.agents.common.runtime_context import AgentDependencies
from app.agents.common.sql_quality import optimize_sql_with_guardrails


def build_tools(deps: AgentDependencies) -> list:
    """Build SQL optimizer tools.

    Args:
        deps: Shared dependencies used for optimizer settings.

    Returns:
        List containing optimizer tool callable.
    """

    async def optimize_sql(sql: str | None = None, tool_context: Any | None = None) -> dict[str, Any]:
        """Apply deterministic optimizer rules to final SQL candidate.

        Args:
            sql: SQL candidate produced by upstream workflow stages.
            tool_context: ADK context used to persist optimized SQL and change list.

        Returns:
            Optimization payload including final SQL, changes, and referenced tables.
        """

        sql_text = (sql or "").strip()
        if not sql_text and tool_context:
            sql_text = str(tool_context.state.get("submitted_sql") or tool_context.state.get("final_sql") or "").strip()
            if not sql_text:
                draft_json = _state_json(tool_context, "draft_package_json", default={})
                sql_text = str(draft_json.get("draft_sql") or "").strip()
        payload = optimize_sql_with_guardrails(deps.settings.default_limit, sql_text)
        if tool_context:
            tool_context.state["final_sql"] = payload["optimized_sql"]
            tool_context.state["submitted_sql"] = payload["optimized_sql"]
            tool_context.state["optimization_payload_json"] = json.dumps(payload)
        return payload

    return [optimize_sql]


def _state_json(tool_context: Any | None, key: str, default: Any) -> Any:
    if not tool_context:
        return default
    raw = tool_context.state.get(key)
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default
    return default
