"""Bridge production query runs into SkillBank candidate skills.

This module deliberately does not run GRPO or production inference. It converts
live NL->SQL run records into trajectory evidence that can be distilled into
reviewable SkillBank candidates.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from ..catalog.embeddings import Embedder, get_embedder
from ..catalog.repository import CatalogRepository
from ..connectors.base import ExecResult
from ..verification.reward import RewardBreakdown, RewardConfig
from .distillation import LLMTeacher, RuleBasedTeacher, distill_trajectories
from .rollout import Trajectory

_DEDUP_DISTANCE = 0.15


@dataclass
class LiveFeedbackSkillSyncResult:
    runs_seen: int = 0
    trajectories_used: int = 0
    skills_proposed: int = 0
    skills_deduped: int = 0
    skills_inserted: int = 0
    inserted_skill_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def query_run_to_trajectory(run: Any) -> Trajectory | None:
    """Convert a production QueryRun-like object into a distillation trajectory."""
    question = str(
        getattr(run, "natural_language_query", None)
        or getattr(run, "submitted_prompt", None)
        or ""
    ).strip()
    sql = str(getattr(run, "final_sql", None) or getattr(run, "submitted_sql", None) or "").strip()
    if not question or not sql:
        return None

    reward = _reward_from_query_run(run)
    source_id = _uuid_or_none(getattr(run, "source_id", None))
    return Trajectory(
        task_id=str(getattr(run, "run_id", "") or "live"),
        question=question,
        prompt=question,
        sql=sql,
        reward=reward,
        source_id=source_id,
    )


def distill_live_query_runs(
    runs: list[Any],
    repo: CatalogRepository,
    *,
    embedder: Embedder | None = None,
    teacher: RuleBasedTeacher | LLMTeacher | None = None,
    promote: bool = False,
    dedup_distance: float = _DEDUP_DISTANCE,
) -> LiveFeedbackSkillSyncResult:
    """Distill production query runs into SkillBank skills.

    Live feedback defaults to ``candidate`` status. Promotion remains a separate
    review or evaluation step so production traffic does not immediately alter
    benchmark/training behavior.
    """
    result = LiveFeedbackSkillSyncResult(runs_seen=len(runs))
    embedder = embedder or get_embedder(repo.settings)
    teacher = teacher or RuleBasedTeacher()
    cfg = RewardConfig()

    trajectories = [traj for run in runs if (traj := query_run_to_trajectory(run)) is not None]
    result.trajectories_used = len(trajectories)
    successes = [traj for traj in trajectories if traj.reward.total >= cfg.tau_success]
    failures = [traj for traj in trajectories if traj.reward.total <= cfg.tau_fail]
    if not successes and not failures:
        result.warnings.append("No high-confidence success/failure trajectories found.")
        return result

    skills = distill_trajectories(successes, failures, teacher=teacher, config=cfg)
    result.skills_proposed = len(skills)
    status = "promoted" if promote else "candidate"

    for skill in skills:
        text = f"{skill.title}. {skill.principle}"
        embedding = embedder([text])[0]
        if _is_duplicate(repo, embedding, dedup_distance):
            result.skills_deduped += 1
            continue
        skill_id = repo.add_skill(
            scope=skill.scope,
            skill_type=getattr(skill, "skill_type", None) or _skill_type(skill.scope),
            title=skill.title,
            principle=skill.principle,
            when_to_apply=skill.when_to_apply,
            positive_example=skill.positive_example,
            negative_example=skill.negative_example,
            provenance={**skill.provenance, "source": "live_query_runs"},
            dialect=skill.dialect,
            source_id=_uuid_or_none(skill.source_id),
            embedding=embedding,
            status=status,
        )
        result.skills_inserted += 1
        result.inserted_skill_ids.append(str(skill_id))

    return result


def _reward_from_query_run(run: Any) -> RewardBreakdown:
    raw = getattr(run, "reward_json", None)
    reward_json = raw if isinstance(raw, dict) else {}
    status = str(getattr(run, "status", "") or "").lower()
    error_message = str(getattr(run, "error_message", "") or "").strip()

    if reward_json:
        total = _float_value(reward_json.get("total"), default=0.0)
        stage = str(reward_json.get("stage") or ("exec_fail" if total <= -0.2 else "exec_nogold"))
    elif status == "succeeded":
        total = 0.10
        stage = "exec_nogold"
    else:
        total = -0.25
        stage = "exec_fail"

    exec_result = None
    if error_message:
        exec_result = ExecResult(error=error_message, dialect=str(getattr(run, "engine", "") or ""))
    return RewardBreakdown(total=total, stage=stage, exec_result=exec_result)


def _skill_type(scope: str) -> str:
    return "failure_repair" if scope == "failure_repair" else "strategy"


def _is_duplicate(
    repo: CatalogRepository,
    embedding: list[float],
    dedup_distance: float,
) -> bool:
    try:
        return bool(repo.search_specific_skills(embedding, k=1, threshold=1.0 - dedup_distance))
    except Exception:  # noqa: BLE001
        return False


def _uuid_or_none(value: Any) -> uuid.UUID | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except ValueError:
        return None


def _float_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
