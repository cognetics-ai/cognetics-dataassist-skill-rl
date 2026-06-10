from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import DirectoryAgentOutput
from app.agents.common.runtime_context import AgentDependencies
from app.agents.directory_agent.tools import build_tools


def build_agent(model: Any, deps: AgentDependencies):
    """Create directory bootstrap agent.

    This agent is the first stage in the Text2SQL workflow and populates user
    directory context into shared state for downstream agents.

    Args:
        model: ADK model name or model wrapper.
        deps: Runtime dependencies including directory service.

    Returns:
        Configured `LlmAgent` that fetches and summarizes user directory profile.
    """

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="DirectoryAgent",
        model=model,
        description="Loads user directory profile and initializes shared identity state.",
        instruction=(
            "Irrespective of user's query, perform this mandatory step to fetch the user directory profile for user with employee id '{soeid}' and summarize it for downstream agents. "
            "Always call the tool get_user_directory_info with employee id '{soeid}'. "
            "Return strict JSON with keys UserDirectoryInformation and directory_summary. "
            "Response must exactly match the configured output schema."
        ),
        tools=build_tools(deps),
        output_schema=DirectoryAgentOutput,
        output_key="directory_agent_output_json",
    )
