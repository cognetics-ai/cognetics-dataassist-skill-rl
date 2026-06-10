from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import OptimizationPackageOutput
from app.agents.common.runtime_context import AgentDependencies
from app.agents.query_optimizer.tools import build_tools


def build_agent(model: Any, deps: AgentDependencies):
    """Create the ADK optimizer agent.

    Args:
        model: ADK model name or model wrapper.
        deps: Shared dependencies for optimizer tools.

    Returns:
        Configured `LlmAgent` that outputs final optimized SQL package.
    """

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="QueryOptimizerAgent",
        model=model,
        description="Applies deterministic optimization guardrails to SQL.",
        instruction=(
            "Optimize SQL from {submitted_sql}. "
            "Always call optimize_sql. "
            "Return strict JSON with keys final_sql and changes. "
            "Response must exactly match the configured output schema."
        ),
        tools=build_tools(deps),
        output_schema=OptimizationPackageOutput,
        output_key="optimization_package_json",
    )
