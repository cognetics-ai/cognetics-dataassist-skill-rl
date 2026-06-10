from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import RouteDecisionOutput
from app.agents.input_router.tools import build_tools


def build_agent(model: Any):
    """Create the ADK input router agent.

    Args:
        model: ADK model name or model wrapper.

    Returns:
        Configured `LlmAgent` that routes input into SQL-direct or NL workflow path.
    """

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="InputRouterAgent",
        model=model,
        description="Routes request to direct SQL execution or Text2SQL workflow.",
        instruction=(
            "Decide whether this request should execute direct SQL or use natural-language Text2SQL generation. "
            "Always call detect_input_mode(submitted_sql, submitted_prompt, input_mode_hint). "
            "Return strict JSON with keys mode and reason. Mode must be either 'sql' or 'nl'. "
            "Response must exactly match the configured output schema."
        ),
        tools=build_tools(),
        output_schema=RouteDecisionOutput,
        output_key="route_decision_json",
    )
