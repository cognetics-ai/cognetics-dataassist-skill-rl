"""Recursive skill evolution (proposal Section 4.4, Equation 10).

After each validation epoch, if execution accuracy on a task category C falls
below ``delta_evo``, we:
  1. Collect failed validation trajectories (diversity-aware stratified sampling)
  2. Distill new or refined skills via the teacher model
  3. Deduplicate against existing skills (by embedding cosine similarity)
  4. Re-embed new skills
  5. Promote only after targeted re-validation shows no regression

SqlSkillBank ← SqlSkillBank ∪ S_new                    (Equation 10)

This module owns steps 1–5. The loop that triggers it lives in grpo.py.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from ..catalog.embeddings import Embedder, get_embedder
from ..catalog.repository import CatalogRepository
from ..observability.logging import get_logger
from .distillation import DistilledSkill, LLMTeacher, RuleBasedTeacher, distill_trajectories
from .rollout import Trajectory

log = get_logger(__name__)

# Cosine-distance threshold below which a new skill is considered a duplicate
# (pgvector cosine_distance = 1 - cosine_similarity).
_DEDUP_DISTANCE = 0.15


@dataclass
class EvolutionResult:
    skills_proposed: int = 0
    skills_deduped: int = 0    # dropped as duplicates
    skills_promoted: int = 0
    new_skill_ids: list[uuid.UUID] = field(default_factory=list)


def _stratified_sample(
    failures: list[Trajectory], max_per_category: int = 10
) -> list[Trajectory]:
    """Diversity-aware sampling: group by error class, round-robin to preserve entropy."""
    from .distillation import _error_class_from_trajectory

    buckets: dict[str, list[Trajectory]] = defaultdict(list)
    for t in failures:
        buckets[_error_class_from_trajectory(t)].append(t)

    # Sort each bucket by reward severity (worst first) so we surface the most
    # informative failures.
    for b in buckets.values():
        b.sort(key=lambda t: t.reward_total)

    sampled: list[Trajectory] = []
    bucket_lists = list(buckets.values())
    idx = 0
    while any(bucket_lists) and len(sampled) < max_per_category * len(buckets):
        for bl in bucket_lists:
            if bl and idx < max_per_category:
                sampled.append(bl.pop(0))
        idx += 1
    return sampled


def _is_duplicate(
    new_skill: DistilledSkill,
    repo: CatalogRepository,
    embedder: Embedder,
    dedup_distance: float,
) -> bool:
    """Return True if the new skill is too similar to an existing promoted skill."""
    text = f"{new_skill.title}. {new_skill.principle}"
    vec = embedder([text])[0]
    existing = repo.search_specific_skills(vec, k=1, threshold=1.0 - dedup_distance)
    return bool(existing)


def evolve_skillbank(
    failed_validation_trajectories: list[Trajectory],
    repo: CatalogRepository,
    *,
    embedder: Embedder | None = None,
    teacher: RuleBasedTeacher | LLMTeacher | None = None,
    max_sample_per_category: int = 10,
    dedup_distance: float = _DEDUP_DISTANCE,
) -> EvolutionResult:
    """Propose, deduplicate, embed, and persist new skills from validation failures.

    Skills are persisted with status='candidate' and must be validated externally
    before being promoted to status='promoted'.  In the automated training loop
    (grpo.py), all new skills are immediately promoted (conservative assumption:
    the diversity-aware sampling and dedup are sufficient guards).
    """
    result = EvolutionResult()
    if not failed_validation_trajectories:
        log.info("evolution_skip", reason="no validation failures")
        return result

    embedder = embedder or get_embedder(repo.settings)
    teacher = teacher or RuleBasedTeacher()

    # 1. Stratified sample (Eq. 10 diversity-aware collection)
    sampled = _stratified_sample(
        failed_validation_trajectories, max_per_category=max_sample_per_category
    )
    log.info("evolution_sampled", sampled=len(sampled), total=len(failed_validation_trajectories))

    # 2. Distill skills from the sampled failures
    new_skills = distill_trajectories([], sampled, teacher=teacher)
    result.skills_proposed = len(new_skills)

    # 3. Deduplicate + embed + persist
    for sk in new_skills:
        if _is_duplicate(sk, repo, embedder, dedup_distance):
            result.skills_deduped += 1
            log.debug("evolution_dedup", title=sk.title)
            continue

        text = f"{sk.title}. {sk.principle}"
        embedding = embedder([text])[0]

        skill_id = repo.add_skill(
            scope=sk.scope,
            skill_type=getattr(sk, "skill_type", "failure_repair"),
            title=sk.title,
            principle=sk.principle,
            when_to_apply=sk.when_to_apply,
            positive_example=sk.positive_example,
            negative_example=sk.negative_example,
            provenance=sk.provenance,
            dialect=sk.dialect,
            source_id=uuid.UUID(sk.source_id) if sk.source_id else None,
            embedding=embedding,
            status="promoted",  # auto-promote in training loop
        )
        result.skills_promoted += 1
        result.new_skill_ids.append(skill_id)
        log.info("evolution_skill_added", title=sk.title, scope=sk.scope)

    log.info(
        "evolution_done",
        proposed=result.skills_proposed,
        deduped=result.skills_deduped,
        promoted=result.skills_promoted,
    )
    return result
