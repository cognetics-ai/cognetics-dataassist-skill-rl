"""Typed application settings.

All configuration flows through :class:`Settings`, populated from environment
variables (after :func:`env_loader.load_root_env` / ``load_agent_env`` have run).
Sub-configs (``SnowflakeConfig`` etc.) are derived on demand so a connector never
reads ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .env_loader import load_root_env

DatasourceType = Literal["snowflake", "postgres", "starburst", "oracle"]
EmbeddingProvider = Literal["ollama", "openai", "vertex"]


class Settings(BaseSettings):
    """Global settings. Field names map 1:1 to the env vars in ``.env.example``."""

    model_config = SettingsConfigDict(
        env_file=None,  # env is pre-loaded by env_loader (supports per-agent override)
        case_sensitive=True,
        extra="ignore",
    )

    SKILLSQL_ENV: Literal["dev", "staging", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    # --- Two DSNs ---
    APP_CATALOG_DSN: str = "postgresql+psycopg://skillsql:skillsql@localhost:5432/skillsql_catalog"
    APP_CATALOG_SCHEMA: str = "skillsql_catalog"
    ADK_SESSION_DSN: str = "postgresql+asyncpg://skillsql:skillsql@localhost:5432/skillsql_adk"
    ADK_SESSION_SCHEMA: str = "adk_store"
    ADK_APP_NAME: str = "skillsql_rl"

    # --- Datasource selection ---
    DATASOURCE_TYPE: DatasourceType = "snowflake"

    # --- Snowflake ---
    SNOWFLAKE_ACCOUNT: str | None = None
    SNOWFLAKE_USER: str | None = None
    SNOWFLAKE_PASSWORD: SecretStr | None = None
    SNOWFLAKE_ROLE: str = "SKILLSQL_RO"
    SNOWFLAKE_WAREHOUSE: str | None = None
    SNOWFLAKE_DATABASE: str | None = None
    SNOWFLAKE_SCHEMA: str | None = None
    SNOWFLAKE_AUTHENTICATOR: str = "snowflake"
    SNOWFLAKE_PRIVATE_KEY_PATH: str | None = None
    SNOWFLAKE_QUERY_TAG: str = "skillsql_rl"

    # --- Starburst Galaxy / Trino ---
    STARBURST_URL: str | None = None
    STARBURST_TRINO_URL: str | None = None
    STARBURST_API_URL: str | None = None
    STARBURST_HOST: str | None = None
    STARBURST_TRINO_HOST: str | None = None
    STARBURST_PORT: int = 443
    STARBURST_USER: str | None = None
    STARBURST_PASSWORD: SecretStr | None = None
    STARBURST_ROLE: str | None = None
    STARBURST_CATALOG: str | None = None
    STARBURST_SCHEMA: str | None = None
    STARBURST_CLIENT_ID: str | None = None
    STARBURST_CLIENT_SECRET: SecretStr | None = None
    STARBURST_VERIFY_SSL: bool = True
    STARBURST_TIMEOUT_MS: int = 300_000
    STARBURST_SOURCE: str = "skillsql"
    STARBURST_QUERY_TAG: str = "skillsql_rl"
    STARBURST_QUERY_HISTORY_TRINO_URL: str | None = None
    STARBURST_QUERY_HISTORY_TRINO_HOST: str | None = None
    STARBURST_QUERY_HISTORY_USER: str | None = None
    STARBURST_QUERY_HISTORY_PASSWORD: SecretStr | None = None
    STARBURST_QUERY_HISTORY_ROLE: str | None = None
    STARBURST_QUERY_HISTORY_CATALOG: str = "galaxy_telemetry"
    STARBURST_QUERY_HISTORY_SCHEMA: str = "public"
    STARBURST_QUERY_HISTORY_TABLE: str = "query_history"
    STARBURST_QUERY_HISTORY_SOURCE: str = "skillsql-qh"

    # --- Execution safety ---
    SQL_STATEMENT_TIMEOUT_S: int = 60
    SQL_ROW_CAP: int = 5000
    SQL_READ_ONLY: bool = True

    # --- Models ---
    OLLAMA_API_BASE: str = "http://localhost:11434"
    DEFAULT_CHAT_MODEL: str = "ollama_chat/llama3.1:8b"
    ADK_MODEL_TIMEOUT_SECONDS: float = Field(default=1_800.0, ge=0.0)
    ADK_MODEL_MAX_RETRIES: int = Field(default=1, ge=0)
    ADK_MODEL_RETRY_BACKOFF_INITIAL_SECONDS: float = Field(default=2.0, ge=0.0)
    ADK_MODEL_RETRY_BACKOFF_MAX_SECONDS: float = Field(default=30.0, ge=0.0)
    QUERY_GENERATOR_MODEL_TIMEOUT_SECONDS: float = Field(default=1_800.0, ge=0.0)
    QUERY_GENERATOR_MODEL_MAX_RETRIES: int = Field(default=1, ge=0)
    QUERY_GENERATOR_MODEL_RETRY_BACKOFF_INITIAL_SECONDS: float = Field(default=2.0, ge=0.0)
    QUERY_GENERATOR_MODEL_RETRY_BACKOFF_MAX_SECONDS: float = Field(default=30.0, ge=0.0)
    EMBEDDING_PROVIDER: EmbeddingProvider = "ollama"
    EMBEDDING_MODEL: str = "snowflake-arctic-embed:l"
    EMBEDDING_DIM: int = Field(default=1024, ge=1)

    # --- Benchmark ---
    SPIDER2_SNOW_JSONL: str = "./data/spider2-snow.jsonl"
    BENCH_GROUP_SIZE: int = Field(default=8, ge=1)
    BENCH_ORACLE_TABLES: bool = False
    GRPO_POLICY_BACKEND: Literal["noop", "verl"] = "noop"

    # --- Logging ---
    DISABLE_NOISY_MODULES: str = "aiohttp,asyncio,google.adk,snowflake.connector,urllib3"
    ENABLE_NOISY_MODULES: str = ""
    ENABLE_NOISY_MODULES_LOG_LEVEL: str = "DEBUG"
    SQLALCHEMY_LOG_SQL: bool = False
    SQLALCHEMY_LOG_LEVEL: str = "INFO"
    SQLALCHEMY_HIDE_PARAMETERS: bool = True

    # ---- derived helpers ----
    @property
    def is_prod(self) -> bool:
        return self.SKILLSQL_ENV == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings. Call ``get_settings.cache_clear()`` after an agent
    env override if you need values re-read within the same process."""
    load_root_env()
    return Settings()


def refresh_settings() -> Settings:
    """Clear the cache and re-read settings (used after ``load_agent_env``)."""
    get_settings.cache_clear()
    return get_settings()
