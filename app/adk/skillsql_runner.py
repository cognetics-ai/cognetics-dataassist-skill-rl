"""SkillSQL-RL ADK runner — the training-path workflow entry point.

This module provides the ADK ``Runner`` wiring for the *training* workflow
(``skillsql/workflow/text2sql_workflow.py``) — the @node-based dynamic graph
that samples G candidates per question for GRPO and benchmark runs.

It is intentionally separate from the production inference runner
(``app/adk/runtime.py``) which uses the full SequentialAgent workflow
with the Critic/Refiner/Distillation loop.

When to use which runner
------------------------
``skillsql.workflow``   + ``skillsql_runner.py``
    GRPO training, Spider-2.0-Snow benchmark, reward scoring per candidate.

``app.agents.text2sql_workflow``  + ``app.adk.runtime.AdkDataAssistRuntime``
    Production API (single best answer, full refinement loop, distillation).

All ADK imports are lazy so the module loads without ``google-adk`` installed
(useful for pure catalog / verification tests).
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from skillsql.config.settings import get_settings

from app.observability.logging import get_logger
from app.adk.retry import AdkRetryConfig, adk_retry

log = get_logger(__name__)


def build_session_service(*, use_memory: bool | None = None) -> Any:
    """Construct the ADK session service for the SkillSQL workflow.

    Uses ``DatabaseSessionService`` (ADK_SESSION_DSN) in staging/prod;
    falls back to ``InMemorySessionService`` when dev and no DSN is set.
    """
    s = get_settings()
    use_mem = (
        use_memory if use_memory is not None
        else (s.SKILLSQL_ENV == "dev" and not s.ADK_SESSION_DSN)
    )
    if use_mem:
        from google.adk.sessions import InMemorySessionService
        return InMemorySessionService()

    from google.adk.sessions import DatabaseSessionService
    url = _adk_state_db_url(dsn=s.ADK_SESSION_DSN, schema=s.ADK_SESSION_SCHEMA)
    return DatabaseSessionService(
        url,
        connect_args=_adk_state_connect_args(s.ADK_SESSION_SCHEMA),
    )


def build_runner(
    root: Any | None = None,
    session_service: Any | None = None,
) -> Any:
    """Build an ADK ``Runner`` bound to the SkillSQL training-path workflow."""
    from google.adk.runners import Runner
    from skillsql.workflow.text2sql_workflow import build_root_workflow

    s = get_settings()
    return Runner(
        agent=root or build_root_workflow(),
        app_name=s.ADK_APP_NAME,
        session_service=session_service or build_session_service(),
    )


async def run_text2sql(
    question: str,
    *,
    runner: Any | None = None,
    user_id: str = "local",
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run one question through the SkillSQL training-path workflow.

    Returns the JSON-encoded result dict from the workflow's terminal event,
    including: sql, reward, stage, equivalent, diagnostics, candidates.
    """
    runner = runner or build_runner()
    session_id = session_id or f"t2s-{uuid.uuid4().hex[:12]}"
    s = get_settings()
    await runner.session_service.create_session(
        app_name=s.ADK_APP_NAME, user_id=user_id, session_id=session_id
    )

    final_text = ""

    @adk_retry(_adk_retry_config(includes_query_generator=True), label="skillsql_text2sql")
    async def _consume() -> None:
        nonlocal final_text
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=_user_message(question),
        ):
            if hasattr(event, "is_final_response") and event.is_final_response():
                content = getattr(event, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    final_text = "".join(getattr(p, "text", "") or "" for p in parts).strip()

    await _consume()

    try:
        return json.loads(final_text)
    except (json.JSONDecodeError, TypeError):
        log.warning("non_json_final_event", preview=final_text[:200])
        return {"sql": None, "raw": final_text, "session_id": session_id}


def run_text2sql_sync(question: str, **kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper (scripts / CLI)."""
    import asyncio
    return asyncio.run(run_text2sql(question, **kwargs))


async def run_agent_once(agent: Any, prompt: str) -> str:
    """Run a single agent turn (ephemeral in-memory session).

    Used for standalone generation and LLM-backed catalog description.
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    s = get_settings()
    runner = Runner(
        agent=agent,
        app_name=s.ADK_APP_NAME,
        session_service=InMemorySessionService(),
    )
    sid = f"once-{uuid.uuid4().hex[:12]}"
    await runner.session_service.create_session(
        app_name=s.ADK_APP_NAME, user_id="local", session_id=sid
    )
    out = ""

    @adk_retry(_adk_retry_config(), label="skillsql_agent_once")
    async def _consume() -> None:
        nonlocal out
        async for event in runner.run_async(
            user_id="local", session_id=sid, new_message=_user_message(prompt)
        ):
            if hasattr(event, "is_final_response") and event.is_final_response():
                content = getattr(event, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    out = "".join(getattr(p, "text", "") or "" for p in parts).strip()

    await _consume()
    return out


# ── Internal ───────────────────────────────────────────────────────────────────

def _adk_retry_config(*, includes_query_generator: bool = False) -> AdkRetryConfig:
    s = get_settings()
    max_retries = s.ADK_MODEL_MAX_RETRIES
    backoff_initial = s.ADK_MODEL_RETRY_BACKOFF_INITIAL_SECONDS
    backoff_max = s.ADK_MODEL_RETRY_BACKOFF_MAX_SECONDS

    if includes_query_generator:
        max_retries = max(max_retries, s.QUERY_GENERATOR_MODEL_MAX_RETRIES)
        backoff_initial = _min_positive(
            backoff_initial,
            s.QUERY_GENERATOR_MODEL_RETRY_BACKOFF_INITIAL_SECONDS,
        )
        backoff_max = max(backoff_max, s.QUERY_GENERATOR_MODEL_RETRY_BACKOFF_MAX_SECONDS)

    return AdkRetryConfig(
        max_retries=max_retries,
        backoff_initial_seconds=backoff_initial,
        backoff_max_seconds=backoff_max,
    )


def _min_positive(left: float, right: float) -> float:
    values = [value for value in (float(left), float(right)) if value > 0]
    return min(values) if values else 0.0


def _user_message(text: str) -> Any:
    from google.genai import types
    return types.Content(role="user", parts=[types.Part(text=text)])


def _adk_state_db_url(dsn: str, schema: str) -> str:
    """Build a SQLAlchemy async URL for ADK's DatabaseSessionService.

    Always uses ``postgresql+asyncpg://`` (required by ADK). The search path is
    passed separately through asyncpg ``server_settings`` because libpq-style
    ``options`` URL parameters are not accepted by asyncpg.

    Accepts any of:
      - ``postgresql://user:pw@host/db``          (libpq — most common in .env)
      - ``postgresql+asyncpg://user:pw@host/db``   (already correct)
      - ``postgresql+psycopg://user:pw@host/db``   (normalised to asyncpg)
    """
    raw = (dsn or "").strip()
    if not raw:
        raise ValueError(
            "ADK_SESSION_DSN is required for session persistence. "
            "Set it in .env: ADK_SESSION_DSN=postgresql://user:pw@host/db"
        )
    parsed = urlsplit(raw)
    if not parsed.scheme.startswith("postgresql"):
        raise ValueError(
            f"ADK_SESSION_DSN must be a PostgreSQL URL (got scheme={parsed.scheme!r}). "
            "Example: ADK_SESSION_DSN=postgresql://user:pw@127.0.0.1:5432/adk_demo_db"
        )
    # ADK requires asyncpg — normalise regardless of what was in .env
    scheme = "postgresql+asyncpg"
    params = parse_qsl(parsed.query, keep_blank_values=True)
    rest = [(k, v) for k, v in params if k != "options"]
    return urlunsplit(
        (
            scheme,
            parsed.netloc,
            parsed.path,
            urlencode(rest, doseq=True),
            parsed.fragment,
        )
    )


def _adk_state_connect_args(schema: str) -> dict[str, Any]:
    return {"server_settings": {"search_path": f"{schema},public"}}
