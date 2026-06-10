from __future__ import annotations

import json
from typing import Any

from app.agents.common.runtime_context import AgentDependencies
from app.agents.common.sql_quality import optimize_sql_with_guardrails


def build_tools(deps: AgentDependencies) -> list:
    """Build tools for SQL refiner loop.

    Args:
        deps: Shared runtime dependencies for optimization settings.

    Returns:
        List of tools used to apply recommendations, optimize SQL, and exit the loop.
    """

    async def apply_critic_recommendations(
        sql: str | None = None,
        recommendations: list[str] | None = None,
        tool_context: Any | None = None,
    ) -> dict[str, Any]:
        """Apply deterministic recommendation-based refinements to SQL.

        Args:
            sql: Current SQL candidate.
            recommendations: Critic-provided recommendation list.
            tool_context: ADK context used to persist refined SQL candidate.

        Returns:
            Payload containing refined SQL and a list of deterministic fixes applied.
        """

        sql_text = (sql or "").strip()
        if not sql_text and tool_context:
            sql_text = str(tool_context.state.get("submitted_sql") or tool_context.state.get("generated_sql") or "").strip()
            if not sql_text:
                draft_json = _state_json(tool_context, "draft_package_json", default={})
                sql_text = str(draft_json.get("draft_sql") or "").strip()
        rewritten = sql_text.rstrip(";")
        recs = recommendations or []
        applied: list[str] = []

        for rec in recs:
            lower = rec.lower()
            if "select *" in lower and "*" in rewritten:
                rewritten = rewritten.replace("SELECT *", "SELECT id", 1).replace("select *", "select id", 1)
                applied.append("Replaced SELECT * with SELECT id")

        payload = {
            "refined_sql": rewritten,
            "applied_recommendations": applied,
        }

        if tool_context:
            tool_context.state["refined_sql"] = rewritten
            tool_context.state["submitted_sql"] = rewritten
            tool_context.state["refinement_rules_json"] = json.dumps(payload)
        return payload

    async def optimize_sql(sql: str | None = None, tool_context: Any | None = None) -> dict[str, Any]:
        """Apply deterministic optimizer guardrails to refined SQL.

        Args:
            sql: SQL candidate to optimize.
            tool_context: ADK context where optimized SQL is persisted.

        Returns:
            Optimization payload including rewritten SQL and applied changes.
        """

        sql_text = (sql or "").strip()
        if not sql_text and tool_context:
            sql_text = str(tool_context.state.get("refined_sql") or tool_context.state.get("submitted_sql") or "").strip()
            if not sql_text:
                draft_json = _state_json(tool_context, "draft_package_json", default={})
                sql_text = str(draft_json.get("draft_sql") or "").strip()
        payload = optimize_sql_with_guardrails(deps.settings.default_limit, sql_text)
        if tool_context:
            tool_context.state["final_sql"] = payload["optimized_sql"]
            tool_context.state["submitted_sql"] = payload["optimized_sql"]
            tool_context.state["optimized_payload_json"] = json.dumps(payload)
        return payload

    async def exit_loop(tool_context: Any) -> dict[str, str]:
        """Signal ADK LoopAgent to stop iterating when quality gate is satisfied.

        Args:
            tool_context: ADK tool context that carries loop action hooks.

        Returns:
            Confirmation payload indicating loop termination request.
        """

        actions = getattr(tool_context, "actions", None)
        if actions is not None and hasattr(actions, "escalate"):
            actions.escalate = True
        return {"status": "loop_exit_requested"}

    return [apply_critic_recommendations, optimize_sql, exit_loop]


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
