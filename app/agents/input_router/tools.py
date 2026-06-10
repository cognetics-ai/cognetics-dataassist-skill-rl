from __future__ import annotations

import json
from typing import Any

from app.core.sql_utils import statement_kind


def build_tools() -> list:
    """Build tools for the input router agent.

    Returns:
        A list of callable tools that classify whether input should follow direct
        SQL execution or natural-language Text2SQL workflow.
    """

    async def detect_input_mode(
        submitted_sql: str | None = None,
        submitted_prompt: str | None = None,
        input_mode_hint: str = "auto",
        tool_context: Any | None = None,
    ) -> dict[str, str]:
        """Detect the execution route for user input.

        Args:
            submitted_sql: SQL text provided by the user or UI, if present.
            submitted_prompt: Natural language prompt provided by the user, if present.
            input_mode_hint: Preferred mode (`auto`, `sql`, or `nl`) from the API payload.
            tool_context: ADK tool context used to persist routing decisions in session state.

        Returns:
            Dictionary with `mode` (`sql` or `nl`) and `reason` describing the decision.
        """

        hint = (input_mode_hint or "auto").strip().lower()
        sql_text = (submitted_sql or "").strip()
        prompt_text = (submitted_prompt or "").strip()

        if hint in {"sql", "nl"}:
            decision = {"mode": hint, "reason": f"input_mode_hint explicitly set to '{hint}'"}
        elif sql_text:
            decision = {"mode": "sql", "reason": "Explicit SQL payload provided"}
        elif _looks_like_sql(prompt_text):
            decision = {"mode": "sql", "reason": "Prompt appears to be SQL text"}
        else:
            decision = {"mode": "nl", "reason": "Prompt appears to be natural language"}

        if tool_context:
            tool_context.state["route_decision_json"] = json.dumps(decision)
        return decision

    return [detect_input_mode]


def _looks_like_sql(value: str) -> bool:
    """Heuristically detect if a string looks like SQL.

    Args:
        value: Raw user-provided text.

    Returns:
        `True` when SQL-like statement kind is detected; otherwise `False`.
    """

    if not value:
        return False
    kind = statement_kind(value)
    return kind in {"SELECT", "WITH", "UNION", "INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "DROP", "ALTER"}
