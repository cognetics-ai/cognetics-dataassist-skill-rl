"""Model registry.

Maps an agent *role* to the model it should use, honoring that agent's own
``.env`` (loaded with override precedence) before falling back to the shared
default. The SQL-generation role resolves to the Arctic Text-to-SQL model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..config.env_loader import load_agent_env
from ..config.settings import get_settings, refresh_settings
from .providers import build_model


@dataclass(frozen=True)
class RoleConfig:
    agent_dir: str
    model_envs: tuple[str, ...]
    provider_envs: tuple[str, ...]
    api_base_envs: tuple[str, ...] = ()
    api_key_envs: tuple[str, ...] = ()
    api_version_envs: tuple[str, ...] = ()
    timeout_envs: tuple[str, ...] = ()
    max_retries_envs: tuple[str, ...] = ()
    timeout_default_attr: str = "ADK_MODEL_TIMEOUT_SECONDS"
    max_retries_default_attr: str = "ADK_MODEL_MAX_RETRIES"
    hard_default: str | None = None


# Role-specific envs win, then the app-level ADK envs. The SQL generator also
# honors QUERY_GENERATOR_* so the SkillSQL path matches the app workflow.
_ROLE_TABLE: dict[str, RoleConfig] = {
    "sql_generator": RoleConfig(
        agent_dir="sql_generator",
        model_envs=("SQL_GEN_MODEL", "QUERY_GENERATOR_MODEL", "OLLAMA_MODEL"),
        provider_envs=(
            "SQL_GEN_MODEL_PROVIDER",
            "QUERY_GENERATOR_MODEL_PROVIDER",
            "ADK_MODEL_PROVIDER",
        ),
        api_base_envs=("SQL_GEN_MODEL_API_BASE", "QUERY_GENERATOR_MODEL_API_BASE"),
        api_key_envs=("SQL_GEN_MODEL_API_KEY", "QUERY_GENERATOR_MODEL_API_KEY"),
        api_version_envs=("SQL_GEN_MODEL_API_VERSION", "QUERY_GENERATOR_MODEL_API_VERSION"),
        timeout_envs=(
            "SQL_GEN_MODEL_TIMEOUT_SECONDS",
            "QUERY_GENERATOR_MODEL_TIMEOUT_SECONDS",
            "ADK_MODEL_TIMEOUT_SECONDS",
        ),
        max_retries_envs=(
            "SQL_GEN_MODEL_MAX_RETRIES",
            "QUERY_GENERATOR_MODEL_MAX_RETRIES",
            "ADK_MODEL_MAX_RETRIES",
        ),
        timeout_default_attr="QUERY_GENERATOR_MODEL_TIMEOUT_SECONDS",
        max_retries_default_attr="QUERY_GENERATOR_MODEL_MAX_RETRIES",
        hard_default="ollama_chat/a-kore/Arctic-Text2SQL-R1-7B:latest",
    ),
    "schema_retriever": RoleConfig(
        agent_dir="schema_retriever",
        model_envs=("SCHEMA_RETRIEVER_MODEL", "ADK_MODEL", "DEFAULT_CHAT_MODEL"),
        provider_envs=("SCHEMA_RETRIEVER_MODEL_PROVIDER", "ADK_MODEL_PROVIDER"),
        api_base_envs=("SCHEMA_RETRIEVER_MODEL_API_BASE", "ADK_MODEL_API_BASE"),
        api_key_envs=("SCHEMA_RETRIEVER_MODEL_API_KEY", "ADK_MODEL_API_KEY"),
        api_version_envs=("SCHEMA_RETRIEVER_MODEL_API_VERSION", "ADK_MODEL_API_VERSION"),
        timeout_envs=("SCHEMA_RETRIEVER_MODEL_TIMEOUT_SECONDS", "ADK_MODEL_TIMEOUT_SECONDS"),
        max_retries_envs=("SCHEMA_RETRIEVER_MODEL_MAX_RETRIES", "ADK_MODEL_MAX_RETRIES"),
    ),
    "verifier": RoleConfig(
        agent_dir="verifier_agent",
        model_envs=("VERIFIER_MODEL", "ADK_MODEL", "DEFAULT_CHAT_MODEL"),
        provider_envs=("VERIFIER_MODEL_PROVIDER", "ADK_MODEL_PROVIDER"),
        api_base_envs=("VERIFIER_MODEL_API_BASE", "ADK_MODEL_API_BASE"),
        api_key_envs=("VERIFIER_MODEL_API_KEY", "ADK_MODEL_API_KEY"),
        api_version_envs=("VERIFIER_MODEL_API_VERSION", "ADK_MODEL_API_VERSION"),
        timeout_envs=("VERIFIER_MODEL_TIMEOUT_SECONDS", "ADK_MODEL_TIMEOUT_SECONDS"),
        max_retries_envs=("VERIFIER_MODEL_MAX_RETRIES", "ADK_MODEL_MAX_RETRIES"),
    ),
}


@dataclass
class ResolvedModel:
    role: str
    model_spec: str
    provider: str
    model: Any  # ADK model object or string


def resolve_role(role: str) -> ResolvedModel:
    """Load the agent's env, read its model spec, and build the ADK model object."""
    if role not in _ROLE_TABLE:
        raise ValueError(f"unknown agent role '{role}'. known: {sorted(_ROLE_TABLE)}")
    config = _ROLE_TABLE[role]
    load_agent_env(config.agent_dir)
    settings = refresh_settings()
    spec = _first_env(config.model_envs) or config.hard_default or settings.DEFAULT_CHAT_MODEL
    provider = _first_env(config.provider_envs) or ""
    timeout_seconds = _first_float_env(config.timeout_envs) if config.timeout_envs else None
    if timeout_seconds is None:
        timeout_seconds = float(getattr(settings, config.timeout_default_attr))
    max_retries = _first_int_env(config.max_retries_envs) if config.max_retries_envs else None
    if max_retries is None:
        max_retries = int(getattr(settings, config.max_retries_default_attr))
    model = build_model(
        spec,
        provider=provider,
        api_base=_first_env(config.api_base_envs),
        api_key=_first_env(config.api_key_envs),
        api_version=_first_env(config.api_version_envs),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    return ResolvedModel(role=role, model_spec=spec, provider=provider, model=model)


def model_spec_for(role: str) -> str:
    """Return just the configured model spec for a role (no ADK import)."""
    config = _ROLE_TABLE[role]
    load_agent_env(config.agent_dir)
    refresh_settings()
    return _first_env(config.model_envs) or config.hard_default or get_settings().DEFAULT_CHAT_MODEL


def _first_env(keys: tuple[str, ...]) -> str:
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return ""


def _first_float_env(keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip():
            return float(value.strip())
    return None


def _first_int_env(keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip():
            return int(value.strip())
    return None
