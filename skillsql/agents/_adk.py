"""ADK adapter -- the *only* place ``google.adk.Agent`` is constructed.

Centralizing this keeps every agent module free of direct SDK imports, so an ADK
API change is a one-file edit. The import is lazy so the package imports without
``google-adk`` present (unit tests of the pure layers don't need it).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from ..models.registry import resolve_role


def build_agent(
    *,
    name: str,
    model: Any,
    instruction: str,
    description: str = "",
    tools: list[Callable] | None = None,
) -> Any:
    """Construct an ADK ``Agent``. ``model`` is a LiteLlm instance or model string."""
    from google.adk import Agent  # lazy import (ADK 2.0)

    return Agent(
        name=name,
        model=model,
        instruction=instruction,
        description=description,
        tools=tools or [],
    )


@lru_cache(maxsize=None)
def agent_for_role(
    role: str, name: str, instruction: str, description: str = ""
) -> Any:
    """Resolve a role's env-configured model and build a (cached) tool-free agent.

    Tools are intentionally empty here: the SQL generator (Arctic) is a
    completion model with no tool-calling; retrieval/execution/verification run in
    the workflow and other agents.
    """
    resolved = resolve_role(role)
    return build_agent(
        name=name, model=resolved.model, instruction=instruction, description=description
    )
