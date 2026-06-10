"""Context builder: assembles the full prompt context for the SQL generator.

The context has three sections (proposal Sections 4.2, 6.2):
  1. Schema context  -- top-K catalog docs retrieved by embedding similarity
  2. Skill context   -- general/dialect skills always + specific skills by similarity
  3. Task            -- the natural-language question + dialect annotation

Total tokens are bounded by the ``max_tokens`` budget (``Lmax`` in the proposal).
We budget by character count (proxy for tokens at ~3.5 chars/token) and trim the
schema context first if needed, since skills are more compressed.
"""

from __future__ import annotations

import uuid

from ..catalog.builder import get_schema_context as _legacy_get_schema_context
from ..catalog.embeddings import Embedder, get_embedder
from ..catalog.repository import CatalogRepository
from ..skillbank.retrieval import RetrievedSkill, format_skills_for_prompt, retrieve_skills

# Conservative chars-per-token ratio.  Set lower to be safe with longer prompts.
_CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def build_context(
    question: str,
    dialect: str,
    *,
    repo: CatalogRepository,
    embedder: Embedder | None = None,
    source_id: uuid.UUID | None = None,
    schema_k: int = 15,
    skill_k: int = 6,
    skill_threshold: float = 0.4,
    max_tokens: int = 8192,
) -> dict[str, str]:
    """Build the context dict for the SQL generator.

    Returns:
        {
            "schema_context": str,   # catalog docs block
            "skill_context":  str,   # skills prompt block
            "full_prompt":    str,   # assembled prompt for the generator
        }
    """
    embedder = embedder or get_embedder(repo.settings)

    # ── Schema context ────────────────────────────────────────────────────────
    schema_text = _generated_schema_context(
        question,
        repo=repo,
        embedder=embedder,
        source_id=source_id,
        schema_k=schema_k,
    )

    # ── Skill context ─────────────────────────────────────────────────────────
    skills: list[RetrievedSkill] = retrieve_skills(
        question,
        dialect,
        repo=repo,
        embedder=embedder,
        source_id=source_id,
        k=skill_k,
        threshold=skill_threshold,
    )
    skill_text = format_skills_for_prompt(skills)

    # ── Token budget enforcement ───────────────────────────────────────────────
    token_budget = max_tokens - 200  # reserve headroom for the question itself
    skill_tokens = _estimate_tokens(skill_text)
    schema_budget = token_budget - skill_tokens

    if schema_budget < 200 and schema_k > 3:
        # Skills consumed too much; fallback to a smaller schema context.
        schema_text = _generated_schema_context(
            question,
            repo=repo,
            embedder=embedder,
            source_id=source_id,
            schema_k=3,
        )
        schema_budget = token_budget - skill_tokens

    # Hard-truncate schema text if still over budget.
    max_schema_chars = int(schema_budget * _CHARS_PER_TOKEN)
    if len(schema_text) > max_schema_chars:
        schema_text = schema_text[:max_schema_chars] + "\n[schema truncated]"

    # ── Assemble full prompt ───────────────────────────────────────────────────
    sections: list[str] = []
    if schema_text:
        sections.append("## Schema Context\n" + schema_text)
    if skill_text:
        sections.append(skill_text)
    sections.append(f"## Task\nDialect: {dialect}\nQuestion: {question}")
    sections.append(
        "## Output\nWrite a single SQL query that answers the question above. "
        "Output only the SQL statement, no explanation."
    )

    full_prompt = "\n\n".join(sections)
    return {
        "schema_context": schema_text,
        "skill_context": skill_text,
        "full_prompt": full_prompt,
    }


def _generated_schema_context(
    question: str,
    *,
    repo: CatalogRepository,
    embedder: Embedder,
    source_id: uuid.UUID | None,
    schema_k: int,
) -> str:
    """Return the current markdown context, falling back to legacy schema docs."""
    try:
        from app.services.catalog import generate_context

        context = generate_context(
            question,
            source_id=str(source_id) if source_id else None,
            schema_k=schema_k,
            query_k=5,
        )
        text = str(context.get("context") or context.get("schema_context") or "").strip()
        if text:
            return text
    except Exception:  # noqa: BLE001
        pass
    return _legacy_get_schema_context(
        repo, question, embedder=embedder, k=schema_k, source_id=source_id
    )
