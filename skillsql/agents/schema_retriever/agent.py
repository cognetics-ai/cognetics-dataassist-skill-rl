"""Schema-retrieval agent (optional).

Vector retrieval over the catalog is deterministic and lives in the workflow;
this agent exists for *ambiguous* questions where an LLM can rerank or expand the
retrieved schema before generation. Tool-capable model (configurable per .env).
"""

from __future__ import annotations

from typing import Any

from .._adk import agent_for_role

INSTRUCTION = (
    "You help select the most relevant tables and columns for a SQL question. "
    "Given candidate schema snippets and the question, return a concise shortlist "
    "of the tables/columns needed, with one-line justifications."
)


def get_agent() -> Any:
    return agent_for_role(
        role="schema_retriever",
        name="schema_retriever",
        instruction=INSTRUCTION,
        description="Reranks/expands retrieved schema for ambiguous questions.",
    )
