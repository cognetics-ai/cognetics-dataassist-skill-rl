from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import TableDescriptionOutput
from app.agents.common.runtime_context import AgentDependencies
from app.agents.table_description_agent.tools import build_tools
from app.observability import get_logger

_logger = get_logger(__name__)

def build_agent(model: Any, deps: AgentDependencies):
    """Create the ADK table-description agent."""
    _logger.info(f"Building table-description agent for {model}")
    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="TableDescriptionAgent",
        model=model,
        description="Synthesizes backend table descriptions from sampled table contents.",
        instruction=(
            "Generate a concise, business-friendly table description in an assertive tone. "
            "Always call sample_table_rows before writing the description. "
            "Use the catalog - {catalog}, schema - {schema_name}, table - {table_name} with backend "
            "engine {engine}, and sample size supplied in the current request/session state. "
            "Use the sampled_table_rows_json tool output as evidence. "
            "Describe what the table appears to represent, the likely row grain, and"
            " important entities or measures. "
            "Do not claim certainty beyond the sample. If rows are sparse or ambiguous, "
            "include caveats. Return **strict JSON** with keys table_name, description, confidence, "
            "observed_entities (needs to be a JSON array or list), likely_grain, caveats (needs to be JSON array/list)."
            "Response must **exactly** match the configured output schema. "
            "Confidence needs to be a numeric value bet 0.0 and 1.0 (very high confidence)"
        ),
        tools=build_tools(deps),
        output_schema=TableDescriptionOutput,
        output_key="table_description_json",
    )
