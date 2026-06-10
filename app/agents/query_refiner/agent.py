from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import RefinementPackageOutput
from app.agents.common.runtime_context import AgentDependencies


def build_agent(model: Any, deps: AgentDependencies):
    """Create the ADK refiner agent used inside LoopAgent.

    Args:
        model: ADK model name or model wrapper.
        deps: Shared dependencies used by refinement tools.

    Returns:
        Configured `LlmAgent` that refines SQL and optionally exits loop.
    """

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="QueryRefinerAgent",
        model=model,
        description="Refines SQL from critic recommendations and decides when to exit loop.",
        instruction=(
            "Refine the current SQL without calling tools or functions. "
            "Use: question from {user_prompt}, current SQL from {submitted_sql}, "
            "draft from {draft_package_json}, critic feedback from {critic_package_json}, "
            "and context from {context_bundle_json} (table_context, context_text, examples, similar_queries). "
            "If {skillbank_context_json} is present, check its skills_text for failure-repair rules that "
            "directly address the issues raised by the critic — apply matching repair patterns first. "
            "For example, if the critic flags an invalid identifier error, consult the "
            "'Repair: copy exact identifiers from schema evidence' skill; if it flags a missing GROUP BY, "
            "consult the 'Declare output grain before aggregating' skill. "
            "If critic approved=true, preserve the SQL exactly, set applied_recommendations=[], "
            "exit_loop=true, exit_reason='critic_approved'. "
            "If critic approved=false, rewrite only to address critic issues using evidence from context. "
            "Preserve exact table/column names. Keep SQL read-only (SELECT/WITH only); no SELECT *; "
            "use explicit joins; "
            f"include LIMIT {deps.settings.default_limit} when returning rows and no stricter limit applies. "
            "Never invent identifiers; put uncertainty in rationale. "
            "One refinement pass only — always set exit_loop=true after producing refined_sql. "
            "Return strict JSON: refined_sql, applied_recommendations, rationale, exit_loop, exit_reason. "
            "No markdown, reasoning traces, <think> blocks, or extra keys."
        ),
        tools=[],
        output_schema=RefinementPackageOutput,
        output_key="refinement_package_json",
    )
