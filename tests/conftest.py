"""Pytest configuration and shared fixtures for SkillSQL-RL tests."""

from __future__ import annotations

import os

import pytest

# ── Environment: point at in-memory / SQLite backends for tests ───────────────
os.environ.setdefault("SKILLSQL_ENV", "dev")
os.environ.setdefault("DATASOURCE_TYPE", "postgres")
os.environ.setdefault(
    "APP_CATALOG_DSN",
    "postgresql+psycopg://skillsql:skillsql@localhost:5432/skillsql_catalog",
)
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Clear the lru_cache on Settings between tests so env overrides take effect."""
    from skillsql.config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
