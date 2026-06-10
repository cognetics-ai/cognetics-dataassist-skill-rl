from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import ValidationPackageOutput
from app.agents.common.runtime_context import AgentDependencies
from app.agents.query_validator.tools import build_tools


def build_agent(model: Any, deps: AgentDependencies):
    """Create the ADK SQL validator agent.

    Args:
        model: ADK model name or model wrapper.
        deps: Shared dependencies bound to validation tools.

    Returns:
        Configured `LlmAgent` that enforces validation gate for SQL.
    """

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="QueryValidatorAgent",
        model=model,
        description=(
            "Validates SQL against policy, EXPLAIN, formal static-lattice gates, "
            "and the SkillSQL-RL composite verifier reward (Equation 12)."
        ),
        instruction=(
            "Validate SQL in {submitted_sql} for production-readiness using two steps. "
            "Step 1 — Always call assess_sql_quality(sql, engine, role_id, prompt) for the full "
            "quality report including policy, EXPLAIN, formal static gates, and obligation score. "
            "Step 2 — If the catalog is available, also call compute_verifier_reward(sql, prompt) "
            "to get the composite verifier reward R(τ). A total >= 0.0 means the SQL executes; "
            "a total >= 0.99 means it matches the reference result. Include the reward total in "
            "your assessment: add a policy finding if it is below -0.25 (exec failure). "
            "Return strict JSON with keys is_valid, policy_findings, explain_summary, risk_score, fixes. "
            "Response must exactly match the configured output schema."
        ),
        tools=build_tools(deps),
        output_schema=ValidationPackageOutput,
        output_key="validation_package_json",
    )
