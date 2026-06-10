"""Layered environment loading.

Precedence (lowest to highest):
    1. Process environment already present.
    2. Repository-root ``.env`` (shared defaults).
    3. A specific agent's ``.env`` (``src/skillsql/agents/<name>/.env``) -- loaded with
       ``override=True`` so an agent can pin its *own* model / timeouts without
       affecting other agents.

This is what lets, e.g., the ``sql_generator`` agent pin
``ollama_chat/a-kore/Arctic-Text2SQL-R1-7B`` while verifier-assist style agents
use a different chat model.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# skillsql/config/env_loader.py is 2 levels below the repo root:
#   <repo>/skillsql/config/env_loader.py
#     parents[0] = <repo>/skillsql/config
#     parents[1] = <repo>/skillsql          ← PACKAGE_ROOT
#     parents[2] = <repo>                   ← REPO_ROOT
_THIS = Path(__file__).resolve()
PACKAGE_ROOT = _THIS.parents[1]          # skillsql/
REPO_ROOT = _THIS.parents[2]             # repository root (cognetics-dataassist-skill-rl/)
AGENTS_DIR = PACKAGE_ROOT / "agents"


@lru_cache(maxsize=1)
def load_root_env() -> None:
    """Load the repository-root ``.env`` once (no override of real process env)."""
    root_env = REPO_ROOT / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)


def load_agent_env(agent_name: str) -> None:
    """Apply an agent's ``.env`` on top of the root env, overriding shared values.

    Safe to call repeatedly; the agent file always wins for keys it defines.
    """
    load_root_env()
    agent_env = AGENTS_DIR / agent_name / ".env"
    if agent_env.exists():
        load_dotenv(agent_env, override=True)


def agent_env_path(agent_name: str) -> Path:
    return AGENTS_DIR / agent_name / ".env"


def get_env(key: str, default: str | None = None) -> str | None:
    load_root_env()
    return os.environ.get(key, default)
