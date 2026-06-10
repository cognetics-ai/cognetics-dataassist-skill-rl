from __future__ import annotations

import json
from typing import Any

from app.agents.common.runtime_context import AgentDependencies
from app.agents.common.sql_quality import evaluate_sql_quality


def build_tools(deps: AgentDependencies) -> list:
    """Build tools used by the SQL critic agent.

    Args:
        deps: Shared runtime dependencies for policy and explain checks.

    Returns:
        List containing the critic assessment tool.
    """

    async def assess_sql_quality(
        sql: str | None = None,
        engine: str = "starburst",
        role_id: str | None = None,
        prompt: str | None = None,
        tool_context: Any | None = None,
    ) -> dict[str, Any]:
        """Assess SQL quality against policy, explain-plan, and shape risks.

        Args:
            sql: Candidate SQL text to critique.
            engine: Target engine used for EXPLAIN checks.
            role_id: Role identifier used for policy enforcement.
            prompt: Natural-language intent for context in critique output.
            tool_context: ADK tool context used to persist critic state.

        Returns:
            Quality assessment payload with approval flag, issues, and recommendations.
        """

        sql_text = (sql or "").strip()
        if not sql_text and tool_context:
            sql_text = str(tool_context.state.get("submitted_sql") or tool_context.state.get("generated_sql") or "").strip()
            if not sql_text:
                draft_json = _state_json(tool_context, "draft_package_json", default={})
                sql_text = str(draft_json.get("draft_sql") or "").strip()
        if not sql_text:
            raise ValueError("No SQL candidate provided to assess_sql_quality")

        effective_role = role_id or (tool_context.state.get("role_id") if tool_context else "analyst")
        report = await evaluate_sql_quality(
            deps=deps,
            sql=sql_text,
            engine=engine,
            role_id=str(effective_role),
            prompt=prompt or "",
        )

        payload = {
            "approved": report["approved"],
            "risk_score": report["risk_score"],
            "issues": report["issues"],
            "recommendations": report["recommendations"],
            "summary": "Approved" if report["approved"] else "Requires refinement",
        }

        if tool_context:
            tool_context.state["critic_assessment_json"] = json.dumps(payload)
            tool_context.state["critic_report_json"] = json.dumps(report)
        return payload

    return [assess_sql_quality]


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
