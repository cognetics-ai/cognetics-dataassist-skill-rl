from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AuthMeResponse(BaseModel):
    soeid: str
    role: str
    display_name: str
    email: str
    job_title: str = ""


class AuthLoginRequest(BaseModel):
    soeid: str = Field(min_length=1)
    password: str


class DraftRequest(BaseModel):
    soeid: str
    prompt: str
    engine_preference: str | None = None


class DraftResponse(BaseModel):
    draft_sql: str
    explanation: str
    warnings: list[str]
    context_refs: list[dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0)
    assumptions: list[str]


class ValidateRequest(BaseModel):
    soeid: str
    sql: str
    engine: str = "starburst"


class ValidateResponse(BaseModel):
    is_valid: bool
    policy_findings: list[dict[str, Any]]
    explain_summary: dict[str, Any]
    risk_score: float = Field(ge=0.0, le=1.0)
    fixes: list[str]


class RunInputMode(StrEnum):
    auto = "auto"
    sql = "sql"
    nl = "nl"


class RunRequest(BaseModel):
    soeid: str
    run_id: str | None = None
    sql: str | None = None
    prompt: str | None = None
    engine: str = "starburst"
    input_mode: RunInputMode = RunInputMode.auto
    source_id: str | None = None

    @model_validator(mode="after")
    def validate_input_payload(self) -> "RunRequest":
        has_sql = bool((self.sql or "").strip())
        has_prompt = bool((self.prompt or "").strip())
        if not has_sql and not has_prompt:
            raise ValueError("Either 'sql' or 'prompt' must be provided.")
        return self


class RunResponse(BaseModel):
    run_id: str


class CancelRequest(BaseModel):
    soeid: str
    run_id: str


class CancelResponse(BaseModel):
    run_id: str
    cancelled: bool


class ResultsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str
    status: str
    result_schema: list[dict[str, Any]] = Field(alias="schema")
    rows: list[list[Any]]
    next_page_token: str | None = None
    error_message: str | None = None


class QueryRunHistoryItem(BaseModel):
    run_id: str
    soeid: str
    engine: str
    input_mode: str
    route_mode: str | None = None
    submitted_text: str
    submitted_sql: str | None = None
    submitted_prompt: str | None = None
    natural_language_query: str | None = None
    source_id: str | None = None
    reward_total: float | None = None
    reward_stage: str | None = None
    final_sql: str
    status: str
    query_start_time: datetime | None = None
    query_end_time: datetime | None = None
    created_at: datetime
    error_message: str | None = None
    row_count: int = 0


class QueryRunHistoryResponse(BaseModel):
    soeid: str
    runs: list[QueryRunHistoryItem]


class DiscoveryQuery(BaseModel):
    query_id: str
    sql_text: str
    sql2text: str
    tables: list[str]
    created_at: datetime
    engine: str


class DiscoveryRoleContextResponse(BaseModel):
    soeid: str
    role: str
    queries: list[DiscoveryQuery]
    metadata: dict[str, Any]


class DiscoverySimilarResponse(BaseModel):
    soeid: str
    prompt: str
    queries: list[DiscoveryQuery]


class BackendTableSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    engine: str | None = "starburst"
    catalog: str | None = None
    database_name: str | None = None
    schema_name: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    semantic_top_k: int | None = Field(default=None, ge=1, le=5000)
    lexical_top_k: int | None = Field(default=None, ge=1, le=5000)


class BackendTableSearchItem(BaseModel):
    engine: str
    catalog: str
    database_name: str = ""
    schema_name: str
    table_name: str
    table_type: str = ""
    description: str = ""
    catalog_id: str = ""
    schema_id: str = ""
    table_id: str = ""
    rrf_score: float = 0.0
    cosine_similarity: float = 0.0
    fts_score: float = 0.0
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    updated_at: Any | None = None


class BackendTableSearchResponse(BaseModel):
    query: str
    engine: str = ""
    catalog: str = ""
    database_name: str = ""
    schema_name: str = ""
    top_k: int
    semantic_candidate_count: int
    lexical_candidate_count: int
    embedding_generated: bool = False
    embedding_retries: int = 0
    tables: list[BackendTableSearchItem]


class BackendColumnSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    engine: str | None = "starburst"
    catalog: str | None = None
    database_name: str | None = None
    schema_name: str | None = None
    table_name: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    semantic_top_k: int | None = Field(default=None, ge=1, le=5000)
    lexical_top_k: int | None = Field(default=None, ge=1, le=5000)
    matched_columns_limit: int = Field(default=10, ge=1, le=50)


class BackendColumnSearchColumnItem(BaseModel):
    column_name: str
    column_id: str = ""
    data_type: str = ""
    description: str = ""
    ordinal_position: int | None = None
    nullable: bool | None = None
    rrf_score: float = 0.0
    cosine_similarity: float = 0.0
    fts_score: float = 0.0
    semantic_rank: int | None = None
    lexical_rank: int | None = None


class BackendColumnSearchTableItem(BaseModel):
    engine: str
    catalog: str
    database_name: str = ""
    schema_name: str
    table_name: str
    catalog_id: str = ""
    schema_id: str = ""
    table_id: str = ""
    rrf_score: float = 0.0
    cosine_similarity: float = 0.0
    fts_score: float = 0.0
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    matched_column_count: int = 0
    matched_columns: list[BackendColumnSearchColumnItem] = Field(default_factory=list)
    updated_at: Any | None = None


class BackendColumnSearchResponse(BaseModel):
    query: str
    engine: str = ""
    catalog: str = ""
    database_name: str = ""
    schema_name: str = ""
    table_name: str = ""
    top_k: int
    semantic_candidate_count: int
    lexical_candidate_count: int
    embedding_generated: bool = False
    embedding_retries: int = 0
    tables: list[BackendColumnSearchTableItem]


class DiscoveryCatalogSyncRequest(BaseModel):
    max_assets: int | None = Field(default=None, ge=1, le=5000)
    concurrency: int = Field(default=8, ge=1, le=50)


class DiscoveryCatalogSyncResponse(BaseModel):
    synced_at: datetime
    master_row_count: int
    common_query_row_count: int
    column_detail_row_count: int
    step2_failures: int
    step3_failures: int
    asset_count_processed: int


class BackendCatalogMetadataSyncRequest(BaseModel):
    engine: str = "starburst"
    catalog: str = Field(min_length=1)
    database_name: str | None = None
    include_columns: bool = True
    batch_size: int = Field(default=500, ge=1, le=5000)


class BackendSchemaMetadataSyncRequest(BaseModel):
    engine: str = "starburst"
    catalog: str = Field(min_length=1)
    database_name: str | None = None
    schema_name: str = Field(min_length=1)
    include_columns: bool = True
    batch_size: int = Field(default=500, ge=1, le=5000)


class BackendTableMetadataSyncRequest(BaseModel):
    engine: str = "starburst"
    catalog: str = Field(min_length=1)
    database_name: str | None = None
    schema_name: str = Field(min_length=1)
    table_name: str = Field(min_length=1)
    include_columns: bool = True
    batch_size: int = Field(default=500, ge=1, le=5000)


class BackendTableDescriptionSyncRequest(BaseModel):
    engine: str = "starburst"
    catalog: str | None = None
    database_name: str | None = None
    schema_name: str | None = None
    table_name: str | None = None
    missing_only: bool = True
    limit: int = Field(default=50, ge=1, le=500)
    sample_size: int = Field(default=5, ge=1, le=25)


class BackendTableDescriptionItem(BaseModel):
    engine: str
    catalog: str
    database_name: str = ""
    schema_name: str
    table_name: str
    description: str
    confidence: float = 0.0
    observed_entities: list[str] = Field(default_factory=list)
    likely_grain: str = ""
    embedding_generated: bool = False
    embedding_retries: int = 0
    updated: bool = False
    caveats: list[str] = Field(default_factory=list)


class BackendTableDescriptionSyncResponse(BaseModel):
    synced_at: datetime
    candidate_count: int
    updated_count: int
    embeddings_generated: int = 0
    embedding_retries: int = 0
    skipped_count: int
    items: list[BackendTableDescriptionItem]


class BackendColumnDescriptionSyncRequest(BaseModel):
    engine: str = "starburst"
    catalog: str = Field(min_length=1)
    database_name: str | None = None
    schema_name: str = Field(min_length=1)
    table_name: str = Field(min_length=1)
    column_name: str | None = None
    column_names: list[str] | None = None
    missing_only: bool = True
    limit: int = Field(default=500, ge=1, le=2000)
    sample_size: int = Field(default=5, ge=1, le=25)

    @model_validator(mode="after")
    def _normalize_columns(self) -> "BackendColumnDescriptionSyncRequest":
        if self.column_names is not None:
            self.column_names = [
                str(item).strip() for item in self.column_names if str(item).strip()
            ]
        return self


class BackendColumnDescriptionItem(BaseModel):
    engine: str
    catalog: str
    database_name: str = ""
    schema_name: str
    table_name: str
    column_name: str
    data_type: str = ""
    description: str
    confidence: float = 0.0
    semantic_type: str = ""
    sample_values: list[str] = Field(default_factory=list)
    embedding_generated: bool = False
    embedding_retries: int = 0
    updated: bool = False
    caveats: list[str] = Field(default_factory=list)


class BackendColumnDescriptionSyncResponse(BaseModel):
    synced_at: datetime
    candidate_count: int
    updated_count: int
    embeddings_generated: int = 0
    embedding_retries: int = 0
    skipped_count: int
    items: list[BackendColumnDescriptionItem]


class BackendMetadataSyncResponse(BaseModel):
    synced_at: datetime
    engine: str
    scope_type: str
    scope: dict[str, Any]
    catalog_rows: int
    schema_rows: int
    table_rows: int
    column_rows: int
    batches_processed: int


class CatalogTableDescriptionSyncRequest(BaseModel):
    engine: str = "starburst"
    source_name: str | None = None
    catalog: str = Field(min_length=1)
    database_name: str | None = None
    schema_name: str = Field(min_length=1)
    table_name: str | None = None
    missing_only: bool = True
    limit: int = Field(default=50, ge=1, le=500)
    sample_size: int = Field(default=5, ge=1, le=25)


class CatalogColumnDescriptionSyncRequest(BaseModel):
    engine: str = "starburst"
    source_name: str | None = None
    catalog: str = Field(min_length=1)
    database_name: str | None = None
    schema_name: str = Field(min_length=1)
    table_name: str | None = None
    column_name: str | None = None
    column_names: list[str] | None = None
    missing_only: bool = True
    limit: int = Field(default=500, ge=1, le=2000)
    sample_size: int = Field(default=5, ge=1, le=25)

    @model_validator(mode="after")
    def _normalize_scope(self) -> "CatalogColumnDescriptionSyncRequest":
        if self.table_name is not None:
            table_name = self.table_name.strip()
            self.table_name = table_name or None
        if self.column_names is not None:
            self.column_names = [
                str(item).strip() for item in self.column_names if str(item).strip()
            ]
        if (self.column_name or self.column_names) and not self.table_name:
            raise ValueError("table_name is required when column_name or column_names are supplied")
        return self


class BackendQueryHistorySyncRequest(BaseModel):
    engine: str = "starburst"
    source_name: str | None = None
    source_group_id: str | None = None
    source_group_name: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    catalog: str | None = None
    database_name: str | None = None
    schema_name: str | None = None
    table_name: str | None = None
    limit: int | None = Field(default=None, ge=1, le=1_000_000)
    page_size: int = Field(default=1000, ge=1, le=5000)
    batch_size: int = Field(default=500, ge=1, le=5000)


class BackendQueryHistorySyncResponse(BaseModel):
    synced_at: datetime
    engine: str
    source_id: str | None = None
    scope: dict[str, Any]
    query_history_rows: int


class DataUsageNlpSyncRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1, le=10000)
    concurrency: int | None = Field(default=None, ge=1, le=100)
    batch_size: int | None = Field(default=None, ge=10, le=5000)


class DataUsageNlpSyncByIdRequest(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=10000)
    concurrency: int | None = Field(default=None, ge=1, le=100)
    batch_size: int | None = Field(default=None, ge=10, le=5000)


class BackendQueryHistoryNlpSyncRequest(BaseModel):
    engine: str = "starburst"
    limit: int | None = Field(default=None, ge=1, le=10000)
    concurrency: int | None = Field(default=None, ge=1, le=100)
    batch_size: int | None = Field(default=None, ge=10, le=5000)
    validate_with_explain: bool | None = None


class BackendQueryNlpHistorySyncRequest(BaseModel):
    engine: str = "starburst"
    source_id: str | None = None
    source_name: str | None = None
    source_group_id: str | None = None
    source_group_name: str | None = None
    catalog: str | None = None
    database_name: str | None = None
    schema_name: str | None = None
    ids: list[int] | None = Field(default=None, max_length=5000)
    raw_sql: str | None = None
    limit: int = Field(default=100, ge=1, le=5000)
    missing_only: bool = True


class BackendQueryNlpHistoryItem(BaseModel):
    raw_query_history_id: int = 0
    source_id: str | None = None
    engine: str
    query_id: str
    raw_sql: str
    query_nlp: str
    nlp_text: str = ""
    embedding_generated: bool = False
    embedding_retries: int = 0
    inserted: bool = False
    caveats: list[str] = Field(default_factory=list)


class BackendQueryNlpHistorySyncResponse(BaseModel):
    synced_at: datetime
    source_rows: int
    inserted_rows: int
    embeddings_generated: int = 0
    embedding_retries: int = 0
    rows_failed: int
    items: list[BackendQueryNlpHistoryItem]


class DataUsageNlpSyncResponse(BaseModel):
    synced_at: datetime
    source_rows: int
    explain_passed: int
    explain_failed: int
    people_enriched: int
    nlp_generated: int
    inserted_rows: int
    embeddings_generated: int = 0
    batches_processed: int = 0
    rows_failed: int = 0
    rate_limit_retries: int = 0
    embedding_retries: int = 0
