from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import DraftPackageOutput
from app.agents.common.runtime_context import AgentDependencies
from app.agents.query_generator.tools import build_tools


def build_agent(model: Any, deps: AgentDependencies, *, use_tools: bool = True):
    """Create the ADK query generator agent.

    Args:
        model: ADK model name or model wrapper.
        deps: Shared dependencies used by generation helper tools.
        use_tools: Whether the model can call ADK tools.

    Returns:
        Configured `LlmAgent` that drafts SQL from context and prompt.
    """

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="QueryGeneratorAgent",
        model=model,
        description="Generates initial SQL candidate from natural-language prompt and context.",
        instruction=_build_instruction(default_limit=deps.settings.default_limit, use_tools=use_tools),
        tools=build_tools(deps) if use_tools else [],
        output_schema=DraftPackageOutput,
        output_key="draft_package_json",
    )


def _build_instruction(*, default_limit: int, use_tools: bool) -> str:
    skill_guidance = (
        "If {skillbank_context_json} is present in state, read it to extract skills_text "
        "and catalog_text. Apply the SQL skills and dialect heuristics listed there — they encode "
        "patterns proven to work and common failure modes to avoid (e.g. use QUALIFY not WHERE for "
        "Snowflake window filters; build a date spine for periods with zero activity). "
    )
    base = (
        "Create a read-only SQL draft for the current user question. "
        "Use context from {context_bundle_json}, especially context_text and context_pack. "
        "Prefer the backend tables and columns listed there; use example query SQL only as a pattern. "
        + skill_guidance
        + "Return strict JSON with keys draft_sql, explanation, confidence, assumptions, warnings. "
        "Response must exactly match the configured output schema. "
        "The confidence value must be strictly between 0.0 and 1.0. "
    )
    if use_tools:
        return (
            base
            + "Always call generation_constraints before writing SQL and call normalize_generated_sql after writing SQL. "
        )
    return (
        base
        + "You are running as a completion-only Text2SQL model. Do not call tools or functions. "
        + "Do not emit markdown, comments outside JSON, reasoning traces, <think> blocks, or extra keys. "
        + "Constraints: exactly one read-only SQL statement; SELECT or WITH only; "
        + "never DDL/DML/GRANT/CALL; no SELECT *; explicit joins; exact table/column names from context; "
        + f"add LIMIT {default_limit} when returning rows and no stricter limit is implied; "
        + "put uncertainty in assumptions or warnings, never invent identifiers. "
        + "Downstream agents will validate and optimize — focus on a syntactically valid, well-structured first draft. "
    )
