from __future__ import annotations

from typing import Any

from app.agents.column_description_agent.tools import build_tools
from app.agents.common.output_schemas import ColumnDescriptionOutput
from app.agents.common.runtime_context import AgentDependencies


def build_agent(model: Any, deps: AgentDependencies):
    """Create the ADK column-description agent."""

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="ColumnDescriptionAgent",
        model=model,
        description="Synthesizes backend column descriptions from sampled column values.",
        instruction=(
            "Generate concise, business-friendly column descriptions for backend metadata. "
            "Always call sample_column_values before writing descriptions. "
            "Use the catalog: {catalog}, schema: {schema_name}, table: {table_name}, "
            "engine: {engine}, sample size, and target column list: {column_names} "
            "supplied in the current request/session state. "
            "Use column_metadata_json for column names, data types, nullability, and "
            "ordinal positions. "
            "Use sampled_column_values_json as evidence; it contains distinct non-null "
            "sample_values for each target column. "
            "If a single column name is supplied, return exactly that column. Otherwise, "
            "return one item for every column listed in column_metadata_json. "
            "Do not omit columns because samples are sparse or null; include caveats "
            "when evidence is weak. "
            "Return strict JSON with keys table_name (string), columns (list), caveats (list). "
            "Each columns item must include column_name, description, confidence, semantic_type, sample_values, caveats. "
            "Response must exactly match the configured output schema."
            "Confidence needs to be a numeric value bet 0.0 and 1.0 (very high confidence)"
        ),
        tools=build_tools(deps),
        output_schema=ColumnDescriptionOutput,
        output_key="column_description_json",
    )
