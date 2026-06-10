"""Tools for the SkillDistillationAgent.

Called after the full Text2SQL workflow completes — win or lose — to distill new
or refined skills from this trajectory into the SqlSkillBank. All operations are
best-effort: failures are logged and returned as a non-error payload so they never
abort the workflow.

The tool distinguishes three trajectory outcomes:
  - High quality (reward >= tau_success ≈ 0.99): distill a success skill (schema-specific
    pattern or general strategy that produced a correct query).
  - Low quality (reward <= tau_fail ≈ -0.20): distill a failure skill (repair rule derived
    from the error class and critic/validator feedback).
  - Mid-range: no distillation (the trajectory is ambiguous evidence).

Skills are embedded, deduplicated against existing skills, and persisted with
status='promoted' (auto-promotion in the training loop; human review recommended
in production — set status='candidate' and add a review queue).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agents.common.runtime_context import AgentDependencies

log = logging.getLogger(__name__)

# Reward thresholds matching RewardConfig.tau_success / tau_fail
_TAU_SUCCESS = 0.99
_TAU_FAIL = -0.20


def build_tools(deps: AgentDependencies) -> list:
    """Build the skill distillation tool.

    Args:
        deps: Runtime dependencies; uses deps.skillsql_resources for catalog access.

    Returns:
        List containing the distill_trajectory_skill tool.
    """

    async def distill_trajectory_skill(
        sql: str | None = None,
        prompt: str | None = None,
        tool_context: Any | None = None,
    ) -> dict[str, Any]:
        """Distill a new or refined skill from this workflow trajectory.

        Reads the reward and critic/validator outputs from session state, determines
        whether this was a success or failure trajectory, and calls the teacher to
        produce a structured skill record. The skill is embedded and persisted to
        the SqlSkillBank (pgvector catalog).

        Args:
            sql:          The final SQL from this trajectory (falls back to state).
            prompt:       The original natural-language question (falls back to state).
            tool_context: ADK context providing the full workflow state.

        Returns:
            DistillationOutput-compatible dict: distilled, skill_title, skill_scope,
            skill_id, reason.
        """
        if not deps.has_catalog:
            return {
                "distilled": False,
                "skill_title": "",
                "skill_scope": "",
                "skill_id": None,
                "reason": "Catalog not available; distillation skipped.",
            }

        # ── Resolve SQL and question from state fallbacks ─────────────────────
        sql_text = (sql or "").strip()
        if not sql_text and tool_context:
            sql_text = str(
                tool_context.state.get("final_sql")
                or tool_context.state.get("submitted_sql")
                or ""
            ).strip()

        question = (prompt or "").strip()
        if not question and tool_context:
            question = str(tool_context.state.get("user_prompt") or "").strip()

        if not sql_text:
            return {
                "distilled": False,
                "skill_title": "",
                "skill_scope": "",
                "skill_id": None,
                "reason": "No SQL found in state; skipping distillation.",
            }

        # ── Determine trajectory outcome from reward ──────────────────────────
        reward_total = _reward_from_state(tool_context)
        is_success = reward_total >= _TAU_SUCCESS
        is_failure = reward_total <= _TAU_FAIL

        if not is_success and not is_failure:
            return {
                "distilled": False,
                "skill_title": "",
                "skill_scope": "",
                "skill_id": None,
                "reason": f"Mid-range reward ({reward_total:.3f}) — no distillation.",
            }

        # ── Build a minimal trajectory for the teacher ────────────────────────
        try:
            from skillsql.rl.distillation import RuleBasedTeacher
            from skillsql.rl.rollout import Trajectory
            from skillsql.verification.reward import RewardBreakdown

            class _MinBreakdown:
                total: float = reward_total
                stage: str = "matched" if is_success else "exec_fail"
                gate_report = None
                exec_result = None

                @property
                def is_success(self):
                    return is_success

            traj = Trajectory(
                task_id="live",
                question=question,
                prompt=question,
                sql=sql_text,
                reward=_MinBreakdown(),  # type: ignore[arg-type]
            )

            teacher = RuleBasedTeacher()
            skill = (
                teacher.distill_success(traj) if is_success else teacher.distill_failure(traj)
            )

            if skill is None:
                return {
                    "distilled": False,
                    "skill_title": "",
                    "skill_scope": "",
                    "skill_id": None,
                    "reason": "Teacher returned no skill for this trajectory.",
                }

            # ── Embed + persist ───────────────────────────────────────────────
            res = deps.skillsql_resources
            text = f"{skill.title}. {skill.principle}"
            embedding = res.embedder([text])[0]

            # Dedup check: skip if very similar skill already exists
            existing = res.repo.search_specific_skills(embedding, k=1, threshold=1.0 - 0.15)
            if existing:
                return {
                    "distilled": False,
                    "skill_title": skill.title,
                    "skill_scope": skill.scope,
                    "skill_id": None,
                    "reason": f"Duplicate skill exists: '{existing[0].title}'",
                }

            skill_id = res.repo.add_skill(
                scope=skill.scope,
                skill_type="failure_repair" if is_failure else "strategy",
                title=skill.title,
                principle=skill.principle,
                when_to_apply=skill.when_to_apply,
                negative_example=skill.negative_example if is_failure else None,
                positive_example=sql_text if is_success else None,
                provenance={
                    "source": "live_distillation",
                    "reward": reward_total,
                    "question_preview": question[:80],
                },
                embedding=embedding,
                status="candidate",  # requires human review in production
            )

            if tool_context:
                tool_context.state["distillation_result_json"] = json.dumps(
                    {"distilled": True, "skill_id": str(skill_id), "title": skill.title}
                )

            return {
                "distilled": True,
                "skill_title": skill.title,
                "skill_scope": skill.scope,
                "skill_id": str(skill_id),
                "reason": f"{'Success' if is_success else 'Failure'} skill distilled.",
            }

        except Exception as e:  # noqa: BLE001 — never abort the workflow
            log.warning("distillation_failed", error=str(e))
            return {
                "distilled": False,
                "skill_title": "",
                "skill_scope": "",
                "skill_id": None,
                "reason": f"Distillation error (non-fatal): {e}",
            }

    return [distill_trajectory_skill]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reward_from_state(tool_context: Any | None) -> float:
    """Read the verifier reward total from session state. Defaults to 0.0."""
    if not tool_context:
        return 0.0
    raw = tool_context.state.get("verifier_reward_json")
    if not raw:
        return 0.0
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return float(data.get("total", 0.0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
