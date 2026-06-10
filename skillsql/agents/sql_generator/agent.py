"""SQL generation agent.

Backed by ``a-kore/Arctic-Text2SQL-R1-7B`` (a GRPO-trained Text-to-SQL model)
served via Ollama. It is **tool-free**: it receives schema context + retrieved
skills + the question and emits a single SQL statement. Retrieval, execution, and
verification are handled by the workflow and other agents.
"""

from __future__ import annotations

from typing import Any

from .._adk import agent_for_role

INSTRUCTION = (
    "You are a senior data engineer and Text-to-SQL expert. "
    "Given a database schema, optional reusable SQL skills, and a natural-language "
    "question, output exactly ONE syntactically valid, read-only SQL query that "
    "answers the question for the specified dialect. Use only tables and columns "
    "present in the provided schema. Do not invent identifiers. Return only the SQL "
    "(no prose, no markdown fences)."
)


def build_prompt(question: str, dialect: str, schema_context: str, skills: str = "") -> str:
    """Assemble the generation prompt from retrieved context."""
    skills_block = f"\n# Reusable SQL skills\n{skills}\n" if skills.strip() else ""
    return (
        f"# Dialect\n{dialect}\n"
        f"# Database schema (retrieved)\n{schema_context}\n"
        f"{skills_block}"
        f"# Question\n{question}\n"
        f"# SQL\n"
    )


def get_agent() -> Any:
    """Build (and cache) the Arctic-backed SQL generation agent."""
    return agent_for_role(
        role="sql_generator",
        name="sql_generator",
        instruction=INSTRUCTION,
        description="Generates a single read-only SQL query from schema + question (Arctic).",
    )
