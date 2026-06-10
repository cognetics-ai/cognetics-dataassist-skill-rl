from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import CriticPackageOutput
from app.agents.common.runtime_context import AgentDependencies
from app.agents.query_critic.tools import build_tools


def build_agent(model: Any, deps: AgentDependencies):
    """Create the ADK critic agent.

    Args:
        model: ADK model name or model wrapper.
        deps: Shared dependencies for critic assessment tools.

    Returns:
        Configured `LlmAgent` that critiques SQL and recommends refinements.
    """

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="QueryCriticAgent",
        model=model,
        description="Critiques SQL quality and determines approval for loop exit.",
        instruction=(
            "Critique SQL quality for safety and correctness. SQL is {submitted_sql}. "
            "Always call assess_sql_quality(sql, engine, role_id, prompt). "
            "Return strict JSON with keys approved, risk_score, issues, recommendations, summary. "
            "Response must exactly match the configured output schema. Risk score needs to be bounded between 0.0 and 1.0"
        ),
        tools=build_tools(deps),
        output_schema=CriticPackageOutput,
        output_key="critic_package_json",
    )
