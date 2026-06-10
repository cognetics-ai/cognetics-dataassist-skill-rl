"""Verifier-assist agent (optional).

The authoritative verifier is deterministic (static lattice + execution + reward
in skillsql.verification). This agent turns a verifier failure (e.g. an
'invalid identifier' or a failed obligation) into a natural-language repair hint
that can seed the next generation attempt or a failure-repair skill.
"""

from __future__ import annotations

from typing import Any

from .._adk import agent_for_role

INSTRUCTION = (
    "You are a SQL repair assistant. Given a failed SQL query, the dialect, and the "
    "verifier's diagnostics (gate failures, execution error, or unmet obligations), "
    "explain the likely root cause in one or two sentences and propose a concrete fix."
)


def get_agent() -> Any:
    return agent_for_role(
        role="verifier",
        name="verifier_agent",
        instruction=INSTRUCTION,
        description="Converts verifier diagnostics into actionable repair hints.",
    )
