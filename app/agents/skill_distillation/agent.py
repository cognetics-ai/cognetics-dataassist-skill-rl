"""SkillDistillationAgent — final stage of the Text2SQL workflow.

After the optimizer produces the final SQL, this agent inspects the trajectory
(reward score, critic findings, validator findings) and calls the distillation
tool to optionally persist a new SqlSkillBank entry. The agent is explicitly
non-blocking: it always produces a valid output and never causes the workflow to
fail. A failed distillation is logged and surfaced in the output's ``reason``
field.

In the GRPO training loop, the evolution step (scripts/train_grpo.py) handles
bulk distillation from many trajectories at once. This agent handles the
single-trajectory, live-inference case — useful for incremental SkillBank
growth as the system is used in production.

Skills are persisted with status='candidate' in the live workflow so they can be
reviewed before being promoted to the main retrieval pool. In the training loop
(skillsql/rl/evolution.py), skills are auto-promoted after deduplication checks.
"""

from __future__ import annotations

from typing import Any

from app.agents.common.output_schemas import DistillationOutput
from app.agents.common.runtime_context import AgentDependencies
from app.agents.skill_distillation.tools import build_tools


def build_agent(model: Any, deps: AgentDependencies):
    """Create the SkillDistillation ADK agent.

    Args:
        model: ADK model name or model wrapper (tool-capable model recommended).
        deps:  Shared runtime dependencies including skillsql_resources.

    Returns:
        Configured ``LlmAgent`` that distills a skill from the completed trajectory,
        or gracefully returns a non-distilled result when conditions are not met.
    """

    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="SkillDistillationAgent",
        model=model,
        description=(
            "Distills a new SqlSkillBank skill from the completed trajectory — "
            "success pattern or failure repair rule. Best-effort; never blocks the workflow."
        ),
        instruction=(
            "Inspect the completed Text2SQL trajectory and distill a skill if warranted. "
            "Always call distill_trajectory_skill(sql, prompt) once. "
            "Use sql from {final_sql} or {submitted_sql} (whichever is present), and "
            "prompt from {user_prompt}. "
            "The tool reads reward and critic/validator findings from state automatically. "
            "It will skip distillation if the reward is in the mid-range (ambiguous evidence), "
            "if the catalog is unavailable, or if a duplicate skill already exists. "
            "Return strict JSON with keys distilled, skill_title, skill_scope, skill_id, reason. "
            "Response must exactly match the configured output schema. "
            "This step is informational — the final SQL has already been produced by the optimizer."
        ),
        tools=build_tools(deps),
        output_schema=DistillationOutput,
        output_key="distillation_output_json",
    )
