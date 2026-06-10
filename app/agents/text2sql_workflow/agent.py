"""Unified Text2SQL workflow (ADK 2.0 SequentialAgent + LoopAgent).

Stages
------
  1. DirectoryAgent              -- user role / identity context
  2. ContextBuilderAgent         -- Starburst discovery context + SkillBank retrieval
  3. QueryGeneratorAgent         -- SQL draft (Arctic-7B or Gemini; consumes skills)
  4. SqlRefinementLoop           -- LoopAgent: Critic → Refiner → LoopExitController
  5. QueryValidatorAgent         -- policy + EXPLAIN + formal gates + reward R(τ)
  6. QueryOptimizerAgent         -- deterministic guardrail rewrites (LIMIT, etc.)
  7. SkillDistillationAgent      -- distill SqlSkillBank skill from this trajectory

The SkillSQL-RL framework is integrated at the tool level:
  - ContextBuilderAgent calls retrieve_skill_context → skills + pgvector catalog docs
  - QueryGeneratorAgent consumes {skillbank_context_json} from state
  - QueryCriticAgent.assess_sql_quality now runs formal static-lattice gates first
  - QueryRefinerAgent instruction includes skill-based repair hints
  - QueryValidatorAgent calls compute_verifier_reward (Equation 12)
  - SkillDistillationAgent saves a new skill (status='candidate') from each trajectory

The SkillSQL-RL workflow (skillsql/workflow/) is the *training* path (GRPO rollouts,
multi-candidate sampling, benchmark runner). This file is the *inference* path
(single best answer per question, with the full production workflow).
"""

from __future__ import annotations

from typing import Any

from app.adk.model_provider import query_generator_uses_tools
from app.agents.common.runtime_context import AgentDependencies
from app.agents.context_builder.agent import build_agent as build_context_builder_agent
from app.agents.directory_agent.agent import build_agent as build_directory_agent
from app.agents.query_critic.agent import build_agent as build_query_critic_agent
from app.agents.query_generator.agent import build_agent as build_query_generator_agent
from app.agents.query_optimizer.agent import build_agent as build_query_optimizer_agent
from app.agents.query_refiner.agent import build_agent as build_query_refiner_agent
from app.agents.query_validator.agent import build_agent as build_query_validator_agent
from app.agents.skill_distillation.agent import build_agent as build_skill_distillation_agent
from app.agents.text2sql_workflow.loop_control import RefinementLoopExitAgent


def build_agent(
    model: Any,
    deps: AgentDependencies,
    max_refinement_iterations: int,
    query_generator_model: Any | None = None,
    enable_distillation: bool = True,
):
    """Create the unified Text2SQL ADK workflow agent.

    Args:
        model:                    ADK model for all agents except the SQL generator.
        query_generator_model:    Optional model override for SQL draft generation.
                                  Defaults to ``model`` when not provided.
        deps:                     Shared runtime dependencies for all agent tools.
        max_refinement_iterations: Upper bound for the critic/refiner loop.
                                   The current workflow caps to one pass via
                                   RefinementLoopExitAgent; increase for deeper refinement.
        enable_distillation:      Whether to include the SkillDistillationAgent at the end.
                                  Set False in benchmarking runs to avoid catalog writes.

    Returns:
        Configured ``SequentialAgent`` composing the full unified Text2SQL workflow.
    """

    from google.adk.agents import LoopAgent, SequentialAgent

    # ── Refinement loop: Critic → Refiner → deterministic exit controller ─────
    critic = build_query_critic_agent(model, deps)
    refiner = build_query_refiner_agent(model, deps)
    loop_exit = RefinementLoopExitAgent(
        name="RefinementLoopExitAgent",
        description="Exits refinement when the critic approves or the pass limit is reached.",
        max_refinement_passes=max(1, max_refinement_iterations),
    )
    refinement_loop = LoopAgent(
        name="SqlRefinementLoop",
        max_iterations=max(1, max_refinement_iterations),
        sub_agents=[critic, refiner, loop_exit],
    )

    # ── Core sequential stages ────────────────────────────────────────────────
    sub_agents: list[Any] = [
        build_directory_agent(model, deps),
        build_context_builder_agent(model, deps),
        build_query_generator_agent(
            query_generator_model or model,
            deps,
            use_tools=query_generator_uses_tools(deps.settings),
        ),
        refinement_loop,
        build_query_validator_agent(model, deps),
        build_query_optimizer_agent(model, deps),
    ]

    # ── Optional skill distillation (best-effort; never blocks) ──────────────
    if enable_distillation and deps.has_catalog:
        sub_agents.append(build_skill_distillation_agent(model, deps))

    return SequentialAgent(
        name="Text2SqlWorkflow",
        sub_agents=sub_agents,
        description=(
            "Unified Text2SQL workflow: user context + SkillBank → draft → "
            "critic/refiner loop → formal validation → optimization → skill distillation."
        ),
    )
