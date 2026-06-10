"""ADK agents. Each sub-agent ships its own ``.env`` selecting its model."""

from ._adk import agent_for_role, build_agent

__all__ = ["build_agent", "agent_for_role"]
