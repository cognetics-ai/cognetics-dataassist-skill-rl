"""Text-to-SQL workflow nodes (ADK 2.0 dynamic workflow).

Deterministic steps (schema retrieval, verification, selection) are plain
``@node`` functions; SQL generation runs the Arctic agent as a node. State flows
as a plain dict so the graph stays easy to reason about and test.

State keys: ``question``, ``dialect``, ``source_id``, ``schema_context``,
``skills``, ``candidates`` (list[str]), ``best_sql``, ``best_reward``, ``diagnostics``.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from ..agents.sql_generator.agent import build_prompt, get_agent
from ..config.settings import get_settings
from ..resources import get_resources
from ..verification.reward import RewardConfig, compute_reward
from ._adk import clean_sql, get_node_decorator, run_agent_node

node = get_node_decorator()


def _known_tables(source_id: uuid.UUID | None) -> set[str] | None:
    """Lowercased table identifiers for the bind gate (None -> skip binding)."""
    res = get_resources()
    try:
        from ..catalog.models import CatalogTable

        with res.repo.session() as s:
            q = s.query(CatalogTable)
            if source_id:
                q = q.filter(CatalogTable.source_id == source_id)
            rows = q.all()
        known: set[str] = set()
        for r in rows:
            known.add(r.name.lower())
            known.add(r.fqn.lower())
        return known or None
    except Exception:  # noqa: BLE001 -- catalog not ready -> skip binding
        return None


@node(name="retrieve_schema")
def retrieve_schema_node(node_input: dict[str, Any]) -> dict[str, Any]:
    """Retrieve schema/query-history context (+ general/dialect skills)."""
    if not isinstance(node_input, dict):
        raise ValueError("retrieve_schema requires a dict node_input state")
    state = node_input
    res = get_resources()
    from app.services.catalog import generate_context

    sid = state.get("source_id")
    sid_uuid = uuid.UUID(sid) if isinstance(sid, str) else sid
    context_result = generate_context(
        state["question"],
        source_id=str(sid_uuid) if sid_uuid else None,
        engine=state.get("engine"),
        catalog=state.get("catalog") or state.get("catalog_name"),
        database_name=state.get("database_name"),
        schema_name=state.get("schema_name") or state.get("db_schema"),
        schema_k=state.get("top_k", 15),
        query_k=state.get("query_k", 5),
    )
    schema_context = str(
        context_result.get("context") or context_result.get("schema_context") or ""
    )
    skills = ""
    try:
        promoted = res.repo.general_and_dialect_skills(state["dialect"])
        skills = "\n".join(f"- {sk.title}: {sk.principle}" for sk in promoted)
    except Exception:  # noqa: BLE001
        skills = ""
    resolved_source_id = context_result.get("source_id") or state.get("source_id")
    return {
        **state,
        "source_id": resolved_source_id,
        "schema_context": schema_context,
        "schema_context_stats": {
            "docs_retrieved": context_result.get("docs_retrieved", 0),
            "query_examples_retrieved": context_result.get("query_examples_retrieved", 0),
            "tables": len(context_result.get("tables") or []),
        },
        "skills": skills,
    }


async def generate_one(ctx: Any, state: dict[str, Any]) -> str:
    """Sample one candidate SQL from the Arctic agent (driven by the workflow node,
    which holds ``ctx``). Returns the cleaned SQL string."""
    agent = get_agent()
    prompt = build_prompt(
        question=state["question"],
        dialect=state["dialect"],
        schema_context=state.get("schema_context", ""),
        skills=state.get("skills", ""),
    )
    raw = await run_agent_node(ctx, agent, prompt)
    return clean_sql(raw)


@node(name="verify_and_select")
async def verify_and_select_node(node_input: dict[str, Any]) -> dict[str, Any]:
    """Score every candidate with the verifier and select the best.

    With a gold result, "best" maximizes the staged reward (exact-match dominates).
    Without gold, selection falls back to execution-success + self-consistency
    (EX voting across candidates).
    """
    if not isinstance(node_input, dict):
        raise ValueError("verify_and_select requires a dict node_input state")
    state = node_input
    res = get_resources()
    candidates = state.get("candidates", [])
    if not candidates:
        return {**state, "best_sql": None, "best_reward": None, "diagnostics": "no candidates"}

    sid = state.get("source_id")
    sid_uuid = uuid.UUID(sid) if isinstance(sid, str) else sid
    known = _known_tables(sid_uuid)
    gold = state.get("gold")  # ExecResult | None

    # Pre-execute candidates once for the self-consistency pool.
    s = get_settings()
    group_results = await asyncio.gather(*[
        res.connector.execute(
            c, read_only=True, timeout_s=s.SQL_STATEMENT_TIMEOUT_S, row_cap=s.SQL_ROW_CAP
        )
        for c in candidates
    ])

    best_sql, best_reward, best_break = None, float("-inf"), None
    for cand, _gr in zip(candidates, group_results, strict=True):
        rb = await compute_reward(
            question=state["question"],
            sql=cand,
            connector=res.connector,
            gold=gold,
            known_tables=known,
            group_results=group_results,
            timeout_s=s.SQL_STATEMENT_TIMEOUT_S,
            row_cap=s.SQL_ROW_CAP,
            config=RewardConfig(),
        )
        if rb.total > best_reward:
            best_sql, best_reward, best_break = cand, rb.total, rb

    diagnostics = (
        "; ".join(best_break.gate_report.messages)
        if (best_break and best_break.gate_report)
        else ""
    )
    return {
        **state,
        "best_sql": best_sql,
        "best_reward": best_reward,
        "stage": best_break.stage if best_break else None,
        "equivalent": best_break.equivalent if best_break else None,
        "diagnostics": diagnostics,
    }
