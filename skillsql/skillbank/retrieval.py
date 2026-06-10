"""SqlSkillBank retrieval (proposal Section 4.2, Equation 6).

Retrieval rule:
  - General SQL skills  → always injected (curated set, no threshold)
  - Dialect skills      → always injected for the task's dialect
  - Schema-specific skills → top-K by embedding similarity > δ, filtered by source_uuid
  - Failure-repair skills  → top-K by error signature + embedding similarity
  - Verifier-obligation skills → from the obligation extractor

The returned skills are formatted as a compact prompt block that fits inside the
SQL generator's context budget (``Lmax`` from the proposal).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..catalog.embeddings import Embedder, get_embedder
from ..catalog.repository import CatalogRepository


@dataclass
class RetrievedSkill:
    """A resolved skill ready for prompt injection."""

    scope: str
    title: str
    principle: str
    when_to_apply: str | None
    positive_example: str | None
    negative_example: str | None


def retrieve_skills(
    question: str,
    dialect: str,
    *,
    repo: CatalogRepository,
    embedder: Embedder | None = None,
    source_id: uuid.UUID | None = None,
    k: int = 6,
    threshold: float = 0.4,
) -> list[RetrievedSkill]:
    """Return the skills to inject for this task.

    Always includes general_sql + dialect skills. Selectively retrieves
    schema_specific, failure_repair, and verifier_obligation by similarity.
    """
    embedder = embedder or get_embedder(repo.settings)

    # 1. Always-included skills (no threshold, no embedding needed)
    always = repo.general_and_dialect_skills(dialect)

    # 2. Specific skills retrieved by semantic similarity (Equation 6)
    q_vec = embedder([question])[0]
    specific = repo.search_specific_skills(
        q_vec, k=k, threshold=threshold, source_id=source_id
    )

    # Deduplicate (general/dialect might overlap with specific in unusual cases)
    seen: set[str] = set()
    skills: list[RetrievedSkill] = []
    for row in list(always) + list(specific):
        if row.title in seen:
            continue
        seen.add(row.title)
        skills.append(
            RetrievedSkill(
                scope=row.scope,
                title=row.title,
                principle=row.principle,
                when_to_apply=row.when_to_apply,
                positive_example=row.positive_example,
                negative_example=row.negative_example,
            )
        )
    return skills


def format_skills_for_prompt(skills: list[RetrievedSkill]) -> str:
    """Render retrieved skills as a compact prompt block.

    Each skill is formatted as a short block; positive/negative examples are
    included only when present, to keep the token footprint small.
    """
    if not skills:
        return ""

    lines: list[str] = ["## Retrieved SQL Skills\n"]
    for sk in skills:
        lines.append(f"### [{sk.scope}] {sk.title}")
        lines.append(f"**Principle:** {sk.principle}")
        if sk.when_to_apply:
            lines.append(f"**When to apply:** {sk.when_to_apply}")
        if sk.positive_example:
            lines.append(f"**Good pattern:**\n```sql\n{sk.positive_example}\n```")
        if sk.negative_example:
            lines.append(f"**Anti-pattern:**\n```sql\n{sk.negative_example}\n```")
        lines.append("")

    return "\n".join(lines)
