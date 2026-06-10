from __future__ import annotations

import json
from typing import Any

from app.agents.common.runtime_context import AgentDependencies
from app.agents.common.sql_quality import evaluate_sql_quality


def build_tools(deps: AgentDependencies) -> list:
    """Build SQL validator tools.

    Args:
        deps: Shared runtime dependencies for policy and engine access.

    Returns:
        List of validation tools for policy checks, explain checks, and unified quality assessment.
    """

    async def validate_sql_policy(sql: str | None = None, role_id: str | None = None, tool_context: Any | None = None) -> dict[str, Any]:
        """Run role-based policy validation over SQL.

        Args:
            sql: SQL statement to validate.
            role_id: Effective role used for policy enforcement.
            tool_context: ADK context used to persist policy findings.

        Returns:
            Policy result with validity, findings, risk score, and fixes.
        """

        sql_text = (sql or "").strip()
        if not sql_text and tool_context:
            sql_text = str(tool_context.state.get("submitted_sql") or tool_context.state.get("final_sql") or "").strip()
            if not sql_text:
                draft_json = _state_json(tool_context, "draft_package_json", default={})
                sql_text = str(draft_json.get("draft_sql") or "").strip()
        if not sql_text:
            raise ValueError("No SQL provided to validate_sql_policy")

        effective_role = role_id or (tool_context.state.get("role_id") if tool_context else "analyst")
        result = deps.policy_checker.check(sql_text, str(effective_role))
        payload = {
            "is_valid": result.is_valid,
            "policy_findings": result.findings,
            "risk_score": result.risk_score,
            "fixes": result.fixes,
        }
        if tool_context:
            tool_context.state["policy_validation_json"] = json.dumps(payload)
        return payload

    async def explain_sql(sql: str | None = None, engine: str = "starburst", tool_context: Any | None = None) -> dict[str, Any]:
        """Execute EXPLAIN on target engine to verify plan viability.

        Args:
            sql: SQL statement to explain.
            engine: Target engine key (for adapter lookup).
            tool_context: ADK context used to persist explain summary.

        Returns:
            Explain payload with success flag, summary, and engine name.
        """

        sql_text = (sql or "").strip()
        if not sql_text and tool_context:
            sql_text = str(tool_context.state.get("submitted_sql") or tool_context.state.get("final_sql") or "").strip()
            if not sql_text:
                draft_json = _state_json(tool_context, "draft_package_json", default={})
                sql_text = str(draft_json.get("draft_sql") or "").strip()
        if not sql_text:
            raise ValueError("No SQL provided to explain_sql")

        explain = await deps.engines.get(engine).explain(sql_text)
        payload = {
            "ok": explain.ok,
            "summary": explain.summary,
            "engine": engine,
        }
        if tool_context:
            tool_context.state["explain_result_json"] = json.dumps(payload)
        return payload

    async def assess_sql_quality(
        sql: str | None = None,
        engine: str = "starburst",
        role_id: str | None = None,
        prompt: str | None = None,
        tool_context: Any | None = None,
    ) -> dict[str, Any]:
        """Run unified quality assessment for final validation gate.

        Args:
            sql: SQL text to assess.
            engine: Target execution engine.
            role_id: Effective role identifier for policy checks.
            prompt: Natural language intent associated with SQL.
            tool_context: ADK context for storing validation report.

        Returns:
            Unified quality report including approval decision and explain summary.
        """

        sql_text = (sql or "").strip()
        if not sql_text and tool_context:
            sql_text = str(tool_context.state.get("submitted_sql") or tool_context.state.get("final_sql") or "").strip()
            if not sql_text:
                draft_json = _state_json(tool_context, "draft_package_json", default={})
                sql_text = str(draft_json.get("draft_sql") or "").strip()
        if not sql_text:
            raise ValueError("No SQL provided to assess_sql_quality")

        effective_role = role_id or (tool_context.state.get("role_id") if tool_context else "analyst")
        report = await evaluate_sql_quality(
            deps=deps,
            sql=sql_text,
            engine=engine,
            role_id=str(effective_role),
            prompt=prompt or "",
        )
        if tool_context:
            tool_context.state["validator_quality_json"] = json.dumps(report)
        return report

    async def compute_verifier_reward(
        sql: str | None = None,
        prompt: str | None = None,
        tool_context: Any | None = None,
    ) -> dict[str, Any]:
        """Compute the composite verifier reward R(τ) from Equation 12.

        This runs the formal static-lattice (Safe, Parse, Bind, Scope, Join) plus
        the obligation-satisfaction score and — when the catalog connector is
        available — bounded execution. The reward total maps directly to the
        training signal: a value >= 0.99 means the query passes all checks and
        matches the gold result; negative values indicate specific failure stages.

        Args:
            sql:          SQL candidate to score.
            prompt:       Natural-language question for obligation extraction.
            tool_context: ADK context used to persist the reward breakdown.

        Returns:
            Reward breakdown dict: total, stage, obligation_score, gate_report,
            formal_gates_passed (bool), and a human-readable summary.
        """
        if not deps.has_catalog:
            return {
                "total": 0.0,
                "stage": "skipped",
                "summary": "Catalog not available; reward scoring skipped.",
                "formal_gates_passed": None,
            }

        sql_text = (sql or "").strip()
        if not sql_text and tool_context:
            sql_text = str(
                tool_context.state.get("submitted_sql")
                or tool_context.state.get("final_sql")
                or ""
            ).strip()
            if not sql_text:
                draft_json = _state_json(tool_context, "draft_package_json", default={})
                sql_text = str(draft_json.get("draft_sql") or "").strip()

        question = (prompt or "").strip()
        if not question and tool_context:
            question = str(tool_context.state.get("user_prompt") or "").strip()

        try:
            from skillsql.verification.reward import RewardConfig, compute_reward

            res = deps.skillsql_resources
            breakdown = await compute_reward(
                question=question,
                sql=sql_text,
                connector=res.connector,
                config=RewardConfig(),
            )
            gate_ok = (
                breakdown.gate_report.passed_all if breakdown.gate_report else True
            )
            payload = {
                "total": round(breakdown.total, 4),
                "stage": breakdown.stage,
                "obligation_score": round(breakdown.obligation_score, 4),
                "equivalent": breakdown.equivalent,
                "formal_gates_passed": gate_ok,
                "gate_messages": (
                    breakdown.gate_report.messages if breakdown.gate_report else []
                ),
                "summary": (
                    f"Reward {breakdown.total:.3f} at stage '{breakdown.stage}'"
                    f" (obligation ω={breakdown.obligation_score:.2f}, gates_ok={gate_ok})"
                ),
            }
        except Exception as e:  # noqa: BLE001 — reward errors must not abort validation
            payload = {
                "total": 0.0,
                "stage": "error",
                "formal_gates_passed": None,
                "summary": f"Reward computation failed: {e}",
            }

        if tool_context:
            tool_context.state["verifier_reward_json"] = json.dumps(payload, default=str)
        return payload

    return [validate_sql_policy, explain_sql, assess_sql_quality, compute_verifier_reward]


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
