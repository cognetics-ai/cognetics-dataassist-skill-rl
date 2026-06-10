from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import ContextBuilderOutput
from app.agents.common.runtime_context import AgentDependencies
from app.agents.context_builder.tools import build_tools


def build_agent(model: Any, deps: AgentDependencies):
    """Create the ADK context-builder agent.

    Args:
        model: ADK model name or model wrapper.
        deps: Shared service dependencies for context tool calls.

    Returns:
        Configured `LlmAgent` for role context assembly.
    """

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="ContextBuilderAgent",
        model=model,
        description=(
            "Builds backend metadata, query-history context, and SkillBank SQL skills "
            "for Text2SQL generation."
        ),
        instruction=(
            "Build a complete context bundle for SQL generation. "
            "Step 1 — Always call build_backend_context once with the natural-language question "
            "from '{user_prompt}'. This returns backend tables, columns, and query examples from "
            "the live data warehouse; use these identifiers exactly — do not invent table names. "
            "Step 2 — Also call retrieve_skill_context with the same question to fetch relevant "
            "SQL skills (general patterns, Snowflake dialect heuristics, repair rules) and "
            "semantic schema documentation from the SkillSQL catalog. If the catalog is "
            "unavailable, retrieve_skill_context returns empty strings and you may skip it. "
            "Merge both results: the backend context from step 1 provides live table/column "
            "evidence; the skill context from step 2 provides SQL strategy knowledge that "
            "should guide the downstream query generator. "
            "Persist the skillbank_context_json from step 2 in state for the generator to read. "
            "Return strict JSON with keys role, business_title, segment_scope, queries, examples, "
            "tables, table_context, similar_queries, metadata, metadata_summary, context_pack, "
            "context_text, backend_search. "
            "Response must exactly match the configured output schema."
        ),
        tools=build_tools(deps),
        output_schema=ContextBuilderOutput,
        output_key="context_bundle_json",
    )
