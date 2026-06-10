from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file's directory so the path is correct
# whether the server is launched from the project root, app/, or anywhere else.
_ENV_FILE = str(Path(__file__).parent.parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        # Also accept variables already present in the process environment
        # (docker, systemd, CI) without requiring a .env file.
        env_file_override=False,
        populate_by_name=True,
    )

    app_name: str = "Data Assist"
    default_engine: str = Field(default="mock", alias="DATA_ASSIST_DEFAULT_ENGINE")
    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173", alias="DATA_ASSIST_CORS_ORIGINS")
    db_backend: str = Field(default="sqlite", alias="DATA_ASSIST_DB_BACKEND")
    sqlite_path: str = Field(default="data/data_assist.db", alias="DATA_ASSIST_SQLITE_PATH")
    postgres_dsn: str = Field(default="postgresql://localhost:5432/atlast_db", alias="DATA_ASSIST_POSTGRES_DSN")
    postgres_schema: str = Field(default="public", alias="DATA_ASSIST_POSTGRES_SCHEMA")

    # ── ADK session / memory Postgres (separate DB from app catalog) ──────────
    # Primary env var names (used in .env):   ADK_SESSION_DSN, ADK_SESSION_SCHEMA
    # Legacy aliases kept for backward compat: DATA_ASSIST_ADK_POSTGRES_DSN, etc.
    # pydantic-settings picks up the first alias that is present in env/file.
    adk_postgres_dsn: str = Field(
        default="postgresql://localhost:5432/adk_demo_db",
        validation_alias="ADK_SESSION_DSN",
    )
    adk_state_schema: str = Field(default="adk_store", validation_alias="ADK_SESSION_SCHEMA")
    adk_memory_schema: str = Field(default="adk_store", validation_alias="ADK_SESSION_MEMORY_SCHEMA")
    adk_app_name: str = Field(default="skillsql_rl", alias="ADK_APP_NAME")

    # Guardrails
    default_limit: int = Field(default=1000, alias="DATA_ASSIST_DEFAULT_LIMIT")
    max_runtime_seconds: int = Field(default=600, alias="DATA_ASSIST_MAX_RUNTIME_SECONDS")
    blocked_keywords: str = Field(default="insert,update,delete,merge,create,drop,alter", alias="DATA_ASSIST_BLOCKED_KEYWORDS")

    # Logging controls
    disable_noisy_modules: str = Field(
        default="aiohttp,asyncio,google.adk,snowflake.connector,urllib3",
        alias="DISABLE_NOISY_MODULES",
    )
    enable_noisy_modules: str = Field(default="", alias="ENABLE_NOISY_MODULES")
    enable_noisy_modules_log_level: str = Field(default="DEBUG", alias="ENABLE_NOISY_MODULES_LOG_LEVEL")
    sqlalchemy_log_sql: bool = Field(default=False, alias="SQLALCHEMY_LOG_SQL")
    sqlalchemy_log_level: str = Field(default="INFO", alias="SQLALCHEMY_LOG_LEVEL")
    sqlalchemy_hide_parameters: bool = Field(default=True, alias="SQLALCHEMY_HIDE_PARAMETERS")

    # Starburst/Trino adapter
    starburst_url: str = Field(default="", alias="STARBURST_URL")
    starburst_trino_url: str = Field(default="", alias="STARBURST_TRINO_URL")
    starburst_api_url: str = Field(default="", alias="STARBURST_API_URL")
    starburst_host: str = Field(default="", alias="STARBURST_HOST")
    starburst_trino_host: str = Field(default="", alias="STARBURST_TRINO_HOST")
    starburst_port: int = Field(default=443, alias="STARBURST_PORT")
    starburst_user: str = Field(default="", alias="STARBURST_USER")
    starburst_password: str = Field(default="", alias="STARBURST_PASSWORD")
    starburst_use_jwt: bool = Field(default=False, alias="STARBURST_USE_JWT")
    starburst_jwt_token: str = Field(default="", alias="STARBURST_JWT_TOKEN")
    starburst_catalog: str = Field(default="", alias="STARBURST_CATALOG")
    starburst_schema: str = Field(default="", alias="STARBURST_SCHEMA")
    starburst_role: str = Field(default="", alias="STARBURST_ROLE")
    starburst_source: str = Field(default="cognetics-ai", alias="STARBURST_SOURCE")
    starburst_result_page_size: int = Field(default=1000, alias="STARBURST_RESULT_PAGE_SIZE")
    starburst_verify_ssl: bool = Field(default=True, alias="STARBURST_VERIFY_SSL")
    starburst_timeout_ms: int = Field(default=300_000, alias="STARBURST_TIMEOUT_MS")
    starburst_client_id: str = Field(default="", alias="STARBURST_CLIENT_ID")
    starburst_client_secret: str = Field(default="", alias="STARBURST_CLIENT_SECRET")
    starburst_query_history_catalog: str = Field(default="galaxy_telemetry", alias="STARBURST_QUERY_HISTORY_CATALOG")
    starburst_query_history_schema: str = Field(default="public", alias="STARBURST_QUERY_HISTORY_SCHEMA")
    starburst_query_history_table: str = Field(default="query_history", alias="STARBURST_QUERY_HISTORY_TABLE")
    starburst_query_history_trino_url: str = Field(default="", alias="STARBURST_QUERY_HISTORY_TRINO_URL")
    starburst_query_history_trino_host: str = Field(default="", alias="STARBURST_QUERY_HISTORY_TRINO_HOST")
    starburst_query_history_user: str = Field(default="", alias="STARBURST_QUERY_HISTORY_USER")
    starburst_query_history_password: str = Field(default="", alias="STARBURST_QUERY_HISTORY_PASSWORD")
    starburst_query_history_role: str = Field(default="", alias="STARBURST_QUERY_HISTORY_ROLE")
    starburst_query_history_source: str = Field(default="cognetics-ai-query-history", alias="STARBURST_QUERY_HISTORY_SOURCE")

    # ADK model selection. Local defaults use Ollama for all workflow agents.
    # Enterprise deployments can point ADK_MODEL_PROVIDER/ADK_MODEL at Vertex,
    # OpenAI, Azure OpenAI, Anthropic, Bedrock, or any LiteLLM model spec.
    adk_model_provider: str = Field(default="ollama", alias="ADK_MODEL_PROVIDER")
    adk_model: str = Field(
        default="ollama_chat/llama3.1:8b",
        validation_alias=AliasChoices("ADK_MODEL", "DEFAULT_CHAT_MODEL"),
    )
    adk_model_api_base: str = Field(default="", alias="ADK_MODEL_API_BASE")
    adk_model_api_key: str = Field(default="", alias="ADK_MODEL_API_KEY")
    adk_model_api_version: str = Field(default="", alias="ADK_MODEL_API_VERSION")
    adk_model_timeout_seconds: float = Field(
        default=1_800.0,
        ge=0.0,
        alias="ADK_MODEL_TIMEOUT_SECONDS",
    )
    adk_model_max_retries: int = Field(default=1, ge=0, alias="ADK_MODEL_MAX_RETRIES")
    adk_model_retry_backoff_initial_seconds: float = Field(
        default=2.0,
        ge=0.0,
        alias="ADK_MODEL_RETRY_BACKOFF_INITIAL_SECONDS",
    )
    adk_model_retry_backoff_max_seconds: float = Field(
        default=30.0,
        ge=0.0,
        alias="ADK_MODEL_RETRY_BACKOFF_MAX_SECONDS",
    )

    # Query generator can use a completion-only Text2SQL model while the rest of
    # the ADK workflow remains on the default chat model.
    query_generator_model_provider: str = Field(
        default="ollama",
        alias="QUERY_GENERATOR_MODEL_PROVIDER",
    )
    query_generator_model: str = Field(
        default="a-kore/Arctic-Text2SQL-R1-7B:latest",
        validation_alias=AliasChoices(
            "QUERY_GENERATOR_MODEL",
            "SQL_GEN_MODEL",
            "OLLAMA_MODEL",
        ),
    )
    query_generator_model_api_base: str = Field(
        default="",
        alias="QUERY_GENERATOR_MODEL_API_BASE",
    )
    query_generator_model_api_key: str = Field(
        default="",
        alias="QUERY_GENERATOR_MODEL_API_KEY",
    )
    query_generator_model_api_version: str = Field(
        default="",
        alias="QUERY_GENERATOR_MODEL_API_VERSION",
    )
    query_generator_model_timeout_seconds: float = Field(
        default=1_800.0,
        ge=0.0,
        alias="QUERY_GENERATOR_MODEL_TIMEOUT_SECONDS",
    )
    query_generator_model_max_retries: int = Field(
        default=1,
        ge=0,
        alias="QUERY_GENERATOR_MODEL_MAX_RETRIES",
    )
    query_generator_model_retry_backoff_initial_seconds: float = Field(
        default=2.0,
        ge=0.0,
        alias="QUERY_GENERATOR_MODEL_RETRY_BACKOFF_INITIAL_SECONDS",
    )
    query_generator_model_retry_backoff_max_seconds: float = Field(
        default=30.0,
        ge=0.0,
        alias="QUERY_GENERATOR_MODEL_RETRY_BACKOFF_MAX_SECONDS",
    )
    query_generator_use_tools: bool | None = Field(
        default=None,
        alias="QUERY_GENERATOR_USE_TOOLS",
    )

    # Ollama model host for ADK LiteLLM integration. OLLAMA_MODEL is kept as a
    # legacy fallback; prefer ADK_MODEL and QUERY_GENERATOR_MODEL for new config.
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")
    ollama_api_base: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_API_BASE")

    # Embeddings. Local defaults use Ollama; enterprise deployments can use
    # OpenAI-compatible endpoints, Vertex, or LiteLLM-supported providers.
    embedding_provider: str = Field(default="ollama", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=0, alias="EMBEDDING_DIM")
    embedding_api_base: str = Field(default="", alias="EMBEDDING_API_BASE")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")

    # Google ADK + Vertex Gemini
    vertex_project_id: str = Field(default="", alias="VERTEX_PROJECT_ID")
    vertex_location: str = Field(default="us-central1", alias="VERTEX_LOCATION")
    vertex_model: str = Field(default="gemini-2.5-pro", alias="VERTEX_MODEL")
    vertex_context_max_tokens: int = Field(default=1_000_000, alias="VERTEX_CONTEXT_MAX_TOKENS")
    vertex_embedding_model: str = Field(default="text-embedding-005", alias="VERTEX_EMBEDDING_MODEL")
    vertex_embedding_dimension: int = Field(default=0, alias="VERTEX_EMBEDDING_DIMENSION")
    vertex_embedding_rps: float = Field(default=4.0, alias="VERTEX_EMBEDDING_RPS")
    vertex_embedding_max_retries: int = Field(default=6, alias="VERTEX_EMBEDDING_MAX_RETRIES")
    vertex_embedding_backoff_initial_ms: int = Field(default=1_000, alias="VERTEX_EMBEDDING_BACKOFF_INITIAL_MS")
    vertex_embedding_backoff_max_ms: int = Field(default=60_000, alias="VERTEX_EMBEDDING_BACKOFF_MAX_MS")
    vertex_embedding_request_timeout_ms: int = Field(default=60_000, alias="VERTEX_EMBEDDING_REQUEST_TIMEOUT_MS")
    critic_refiner_max_iterations: int = Field(default=1, alias="DATA_ASSIST_CRITIC_MAX_ITERATIONS")

    # Directory lookup (ECS people search)
    directory_people_search_url_template: str = Field(
        default=(
            "https://ecs.search.citi.net/search/api/search/people"
            "?q.fields=soeid&people.fields=*&q={soeid}"
        ),
        alias="DIRECTORY_PEOPLE_SEARCH_URL_TEMPLATE",
    )
    directory_people_search_timeout_ms: int = Field(default=30_000, alias="DIRECTORY_PEOPLE_SEARCH_TIMEOUT_MS")
    directory_people_search_verify_ssl: bool = Field(default=True, alias="DIRECTORY_PEOPLE_SEARCH_VERIFY_SSL")
    directory_people_search_extra_headers_json: str = Field(default="", alias="DIRECTORY_PEOPLE_SEARCH_EXTRA_HEADERS_JSON")
    directory_default_role: str = Field(default="analyst", alias="DIRECTORY_DEFAULT_ROLE")

    # Discovery Catalog ingestion endpoints
    discovery_step1_url: str = Field(
        default="https://discovery.search.apps.namicgmwd58p.ecs.dyn.nsroot.net/discovery/search/getNewFlowSearchResult",
        alias="DISCOVERY_STEP1_URL",
    )
    discovery_step2_url: str = Field(
        default="https://discovery.search.apps.namicgmwd58p.ecs.dyn.nsroot.net/discovery/api/getUsageDataOverviewCommonQueries",
        alias="DISCOVERY_STEP2_URL",
    )
    discovery_step3_url: str = Field(
        default="https://discovery.search.apps.namicgmwd58p.ecs.dyn.nsroot.net/discovery/api/getColumndetails",
        alias="DISCOVERY_STEP3_URL",
    )
    discovery_timeout_ms: int = Field(default=120_000, alias="DISCOVERY_TIMEOUT_MS")
    discovery_api_token: str = Field(default="", alias="DISCOVERY_API_TOKEN")
    discovery_api_token_header: str = Field(default="Authorization", alias="DISCOVERY_API_TOKEN_HEADER")
    discovery_extra_headers_json: str = Field(default="", alias="DISCOVERY_EXTRA_HEADERS_JSON")
    discovery_verify_ssl: bool = Field(default=True, alias="DISCOVERY_VERIFY_SSL")

    # Data usage SQL->NLP enrichment
    data_usage_common_query_select_sql: str = Field(
        default=(
            "SELECT SOEID, EMAIL, NAME, QUERY, TOOL, SCHEMA_TABLE, ALL_QUERY_TABLES "
            "FROM DATA_USAGE_COMMON_QUERIES "
            "WHERE TOOL='Starburst' AND QUERY NOT LIKE '%Query Truncated%'"
        ),
        alias="DATA_USAGE_COMMON_QUERY_SELECT_SQL",
    )
    people_search_url_template: str = Field(
        default=(
            "https://ecs.search.citi.net/search/api/search/people"
            "?q.fields=email&people.fields=soeid,managedsegmenthierarchy,ql_businesscardtitle&q={email}"
        ),
        alias="PEOPLE_SEARCH_URL_TEMPLATE",
    )
    people_search_timeout_ms: int = Field(default=30_000, alias="PEOPLE_SEARCH_TIMEOUT_MS")
    people_search_verify_ssl: bool = Field(default=True, alias="PEOPLE_SEARCH_VERIFY_SSL")
    people_search_extra_headers_json: str = Field(default="", alias="PEOPLE_SEARCH_EXTRA_HEADERS_JSON")
    data_usage_nlp_concurrency: int = Field(default=8, alias="DATA_USAGE_NLP_CONCURRENCY")
    data_usage_nlp_batch_size: int = Field(default=500, alias="DATA_USAGE_NLP_BATCH_SIZE")
    data_usage_nlp_llm_rps: float = Field(default=2.0, alias="DATA_USAGE_NLP_LLM_RPS")
    data_usage_nlp_max_retries: int = Field(default=5, alias="DATA_USAGE_NLP_MAX_RETRIES")
    data_usage_nlp_backoff_initial_ms: int = Field(default=1_000, alias="DATA_USAGE_NLP_BACKOFF_INITIAL_MS")
    data_usage_nlp_backoff_max_ms: int = Field(default=30_000, alias="DATA_USAGE_NLP_BACKOFF_MAX_MS")
    backend_query_nlp_validate_with_explain: bool = Field(default=False, alias="BACKEND_QUERY_NLP_VALIDATE_WITH_EXPLAIN")
    discovery_similar_embedding_top_k: int = Field(default=400, alias="DISCOVERY_SIMILAR_EMBEDDING_TOP_K")
    discovery_similar_business_title_top_k: int = Field(default=120, alias="DISCOVERY_SIMILAR_BUSINESS_TITLE_TOP_K")
    discovery_similar_segment_top_k: int = Field(default=120, alias="DISCOVERY_SIMILAR_SEGMENT_TOP_K")
    discovery_similar_lexical_top_k: int = Field(default=200, alias="DISCOVERY_SIMILAR_LEXICAL_TOP_K")
    discovery_similar_rrf_k: int = Field(default=60, alias="DISCOVERY_SIMILAR_RRF_K")

    @property
    def active_embedding_model(self) -> str:
        model = (self.embedding_model or "").strip()
        if model:
            return model

        provider = (self.embedding_provider or "").strip().lower().replace("-", "_")
        if provider in {"vertex", "gemini", "google", "google_vertex"}:
            return (self.vertex_embedding_model or "text-embedding-005").strip()
        return "snowflake-arctic-embed:l"

    @property
    def active_embedding_dimension(self) -> int:
        if self.embedding_dimension > 0:
            return self.embedding_dimension

        provider = (self.embedding_provider or "").strip().lower().replace("-", "_")
        if (
            provider in {"vertex", "gemini", "google", "google_vertex"}
            and self.vertex_embedding_dimension > 0
        ):
            return self.vertex_embedding_dimension

        model = self.active_embedding_model.strip().lower()
        known = {
            "snowflake-arctic-embed:l": 1024,
            "mxbai-embed-large": 1024,
            "nomic-embed-text": 768,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "text-embedding-005": 768,
            "text-embedding-004": 768,
            "textembedding-gecko": 768,
            "textembedding-gecko@003": 768,
            "multilingual-embedding-002": 768,
            "gemini-embedding-001": 3072,
        }
        return known.get(model, 768)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
