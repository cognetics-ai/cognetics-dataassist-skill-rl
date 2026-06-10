from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import QueryToNlpOutput
from app.agents.common.runtime_context import AgentDependencies
from app.agents.query_to_nlp_agent.tools import build_tools


def build_agent(model: Any, deps: AgentDependencies):
    """Create the ADK query-to-NLP agent."""

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="QueryToNlpAgent",
        model=model,
        description="Converts raw SQL query history into concise analyst-style natural language.",
        instruction=(
            "Convert one raw SQL query into a short natural-language request a business analyst would ask in non-technical way. "
            "Always call load_query_history_context before writing the answer. "
            "Use query_history_context_json for raw SQL, tables_json, table descriptions, and column descriptions. "
            "Keep query_nlp concise and specific, ideally between 2 to 4 sentences. "
            "Mention the main entities, measures, filters, grouping, ordering, date windows, and limits when present. "
            "Use metadata descriptions to translate technical table and column names into business meaning. "
            "Do not explain SQL syntax, JOIN conditions or mention implementation details, or include markdown. "
            "Return strict JSON with keys query_nlp, tables, caveats. Please note keys, tables and caveats need to be strictly **JSON arrays** or list."
            "Key query_nlp needs to be a string. "
            " Response must exactly match the configured output schema. "
            "If query_state is FAILED, query_nlp must include the analyst intent plus the failure cause, "
            "a detailed explanation, and a suggested fix using error_code_name, error_code_category, "
            "and error_exception_message from query_history_context_json when present. "
        ),
        tools=build_tools(deps),
        output_schema=QueryToNlpOutput,
        output_key="query_to_nlp_json",
    )
