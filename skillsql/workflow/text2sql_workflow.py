"""The Text-to-SQL dynamic workflow graph (ADK 2.0).

The root workflow node receives the natural-language question as input, retrieves
schema context, samples ``G`` candidates from the Arctic agent (a loop -- trivial
in a dynamic workflow), verifies/selects the best, and returns a JSON result
string (so the value carried by the terminal ADK Event is unambiguous).

Dialect and active source come from process resources/config rather than being
threaded through the message, which keeps the ADK input a plain string.
"""

from __future__ import annotations

import json
from typing import Any

from ..config.settings import get_settings
from ..resources import get_active_source, get_resources
from ._adk import get_node_decorator, make_workflow
from .nodes import generate_one, retrieve_schema_node, verify_and_select_node

node = get_node_decorator()


def _initial_state(question: str) -> dict[str, Any]:
    res = get_resources()
    src = get_active_source()
    return {
        "question": question,
        "dialect": res.connector.dialect,
        "source_id": str(src) if src else None,
    }


def _public_result(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": state.get("question"),
        "dialect": state.get("dialect"),
        "source_id": state.get("source_id"),
        "sql": state.get("best_sql"),
        "reward": state.get("best_reward"),
        "stage": state.get("stage"),
        "equivalent": state.get("equivalent"),
        "diagnostics": state.get("diagnostics"),
        "schema_context_stats": state.get("schema_context_stats"),
        "candidates": state.get("candidates", []),
    }


@node(name="text2sql", rerun_on_resume=True)
async def text2sql_workflow(ctx: Any, node_input: Any) -> str:
    """End-to-end orchestration for one question. ``node_input`` is the question
    text (a plain string); returns a JSON-encoded result."""
    question = node_input if isinstance(node_input, str) else str(node_input)
    group_size = int(get_settings().BENCH_GROUP_SIZE)

    state = _initial_state(question)
    state = await ctx.run_node(retrieve_schema_node, node_input=state)

    candidates: list[str] = []
    for _ in range(max(1, group_size)):
        candidates.append(await generate_one(ctx, state))
    state["candidates"] = candidates

    state = await ctx.run_node(verify_and_select_node, node_input=state)
    return json.dumps(_public_result(state), default=str)


def build_root_workflow() -> Any:
    """Build the root ``Workflow`` (single-entry graph)."""
    return make_workflow(name="skillsql_text2sql", edges=[("START", text2sql_workflow)])
