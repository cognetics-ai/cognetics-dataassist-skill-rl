"""ADK workflow adapter -- isolates the ADK 2.0 orchestration primitives.

Everything the workflow needs from the SDK (the ``@node`` decorator, ``Workflow``,
``Context``, ``Event``, and "run an agent as a node") is funneled through here, so
an ADK API change is contained to this file. Imports are lazy where possible.
"""

from __future__ import annotations

import re
from typing import Any


def get_node_decorator():
    """Return ADK's ``@node`` decorator (ADK 2.0: ``google.adk.workflow.node``)."""
    from google.adk.workflow import node  # lazy

    return node


def make_workflow(name: str, edges: list) -> Any:
    """Construct an ADK ``Workflow`` graph from an edges array."""
    from google.adk import Workflow  # lazy

    return Workflow(name=name, edges=edges)


async def run_agent_node(ctx: Any, agent: Any, prompt: str) -> str:
    """Run an ADK agent inside a dynamic workflow and return its text output.

    ADK dynamic workflows execute nodes (which may be Agents) via
    ``ctx.run_node(node, node_input=...)``. We pass the prompt as the node input
    and normalize the result to a string. Output shape extraction is localized
    here so a different ADK return contract is a one-line change.
    """
    result = await ctx.run_node(agent, node_input=prompt)
    return _extract_text(result)


def _extract_text(result: Any) -> str:
    """Best-effort extraction of text from an ADK node/agent result or Event."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    # ADK Event with .content.parts[].text
    content = getattr(result, "content", None)
    if content is not None:
        parts = getattr(content, "parts", None)
        if parts:
            texts = [getattr(p, "text", "") or "" for p in parts]
            joined = "".join(texts).strip()
            if joined:
                return joined
    output = getattr(result, "output", None)
    if isinstance(output, str):
        return output
    return str(result)


_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def clean_sql(text: str) -> str:
    """Strip markdown fences / reasoning preamble and return a single SQL statement.

    Arctic emits reasoning then SQL; we take the fenced block if present, else the
    text from the first SQL keyword, and trim a trailing semicolon.
    """
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    else:
        m2 = re.search(r"\b(WITH|SELECT)\b", text, re.IGNORECASE)
        if m2:
            text = text[m2.start():].strip()
    # keep only the first statement
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    return text
