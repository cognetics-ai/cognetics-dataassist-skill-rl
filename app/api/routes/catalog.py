"""Catalog, SkillBank, verification, and SQL generation API routes.

This module consolidates all SkillSQL-RL HTTP endpoints under the main
FastAPI application — no separate SkillSQL FastAPI instance.

Routers
-------
``catalog_router``   (/catalog/*)   -- catalog build, retrieval, liveness
``skillbank_router`` (/skillbank/*) -- skill seed and list
``sql_router``       (/sql/*)       -- verify, score, generate
``bench_router``     (/admin/*)     -- DB init

All business logic lives in ``app.services.catalog`` (and calls down to
``skillsql.*`` framework modules).  Routes are thin HTTP adaptors only.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import app.services.catalog as svc
from app.dependencies import AppContext, get_ctx
from app.observability.logging import get_logger
from app.schemas import (
    BackendColumnDescriptionSyncResponse,
    BackendColumnSearchRequest,
    BackendColumnSearchResponse,
    BackendQueryHistorySyncRequest,
    BackendQueryHistorySyncResponse,
    BackendQueryNlpHistorySyncRequest,
    BackendQueryNlpHistorySyncResponse,
    BackendTableDescriptionSyncResponse,
    BackendTableSearchRequest,
    BackendTableSearchResponse,
    CatalogColumnDescriptionSyncRequest,
    CatalogTableDescriptionSyncRequest,
)

_logger = get_logger(__name__)
catalog_router  = APIRouter(prefix="/catalog",  tags=["catalog"])
skillbank_router = APIRouter(prefix="/skillbank", tags=["skillbank"])
sql_router      = APIRouter(prefix="/sql",       tags=["sql"])
admin_router    = APIRouter(prefix="/admin",     tags=["admin"])

Ctx = Annotated[AppContext, Depends(get_ctx)]


# ── Request / response schemas ─────────────────────────────────────────────────

class CatalogBuildRequest(BaseModel):
    source_type: str | None = None
    source_name: str = "default"
    source_group_id: str | None = None
    source_group_name: str | None = None
    catalog: str | None = None
    catalog_names: list[str] | None = None   # Starburst: specific catalogs (None = all)
    database_name: str | None = None
    db_schema: str | None = None
    profile: bool = True
    describe: bool = True
    sample_size: int = Field(default=5, ge=1, le=25)


class CatalogMetadataSyncRequest(BaseModel):
    engine: str = "starburst"
    source_name: str | None = None
    source_group_id: str | None = None
    source_group_name: str | None = None
    catalog: str = Field(min_length=1)
    database_name: str | None = None
    include_columns: bool = True
    describe: bool = False
    doc_batch_size: int = Field(default=128, ge=1, le=2000)


class SchemaMetadataSyncRequest(CatalogMetadataSyncRequest):
    schema_name: str = Field(min_length=1)


class TableMetadataSyncRequest(SchemaMetadataSyncRequest):
    table_name: str = Field(min_length=1)


class ColumnMetadataSyncRequest(TableMetadataSyncRequest):
    column_name: str = Field(min_length=1)


class SchemaContextRequest(BaseModel):
    question: str
    source_id: str | None = None
    source_group_id: str | None = None
    engine: str | None = None
    catalog: str | None = None
    database_name: str | None = None
    schema_name: str | None = None
    k: int = 15
    query_k: int = 5


class LiveFeedbackSkillSyncRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=1000)
    statuses: list[str] = Field(default_factory=lambda: ["succeeded", "failed"])
    promote: bool = False


class ColumnSearchRequest(BaseModel):
    query: str
    source_id: str | None = None
    source_group_id: str | None = None
    k: int = 10


class VerifyRequest(BaseModel):
    sql: str
    source_id: str | None = None
    execute: bool = True


class ScoreRequest(BaseModel):
    question: str
    sql: str
    gold_sql: str | None = None
    source_id: str | None = None


class GenerateRequest(BaseModel):
    question: str
    source_id: str | None = None
    source_group_id: str | None = None


# ── /admin routes ──────────────────────────────────────────────────────────────

@admin_router.post("/init-db",
    summary="Create catalog schema + pgvector extension",
    description="Idempotent. Creates the Postgres schema, pgvector extension, "
                "and all catalog + SkillBank tables.")
def init_db() -> dict:
    return svc.init_db()


# ── /catalog routes ────────────────────────────────────────────────────────────

@catalog_router.get("/healthz",
    summary="Connector + catalog liveness",
    description="Returns status=ok when the datasource connector and Postgres "
                "catalog are both reachable.")
async def catalog_health() -> dict:
    return await svc.health_check()


@catalog_router.get("/datasources",
    summary="List datasource engines available for catalog sync",
    description="Returns the engine names registered in the shared EngineAdapter factory. "
                "Use these values as the engine field for metadata sync requests.")
def catalog_datasources(ctx: Ctx) -> dict:
    return svc.available_datasources(ctx.engines)


@catalog_router.post("/build",
    summary="Discover + persist a datasource catalog",
    description="Discovers schema hierarchy (catalog→schema→table→column), "
                "persists it to the semantic catalog, then generates table and "
                "column descriptions from sampled backend data when describe=true. "
                "For describe=true, provide catalog or database_name plus db_schema.")
async def catalog_build(req: CatalogBuildRequest, ctx: Ctx) -> dict:
    _logger.debug(f"Incoming FastAPI request: {req}")
    try:
        return await svc.build_catalog(
            ctx.engines,
            source_type=req.source_type,
            source_name=req.source_name,
            source_group_id=req.source_group_id,
            source_group_name=req.source_group_name,
            catalog=req.catalog,
            catalog_names=req.catalog_names,
            database_name=req.database_name,
            db_schema=req.db_schema,
            profile=req.profile,
            describe=req.describe,
            description_runtime=ctx.adk_runtime,
            sample_size=req.sample_size,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.error(f"Failed building catalog: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@catalog_router.post("/metadata/catalog/sync",
    summary="Stream-sync one datasource catalog into the semantic catalog",
    description="Uses the shared EngineAdapter factory. For Starburst this can stream a "
                "Galaxy catalog. For Snowflake/Postgres, a configured default schema is "
                "required unless using the schema/table/column endpoints.")
async def sync_catalog_metadata(
    req: CatalogMetadataSyncRequest,
    ctx: Ctx,
) -> dict:
    try:
        return await svc.sync_catalog_metadata(
            ctx.engines,
            engine=req.engine,
            source_name=req.source_name,
            source_group_id=req.source_group_id,
            source_group_name=req.source_group_name,
            catalog=req.catalog,
            database_name=req.database_name,
            include_columns=req.include_columns,
            describe=req.describe,
            doc_batch_size=req.doc_batch_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@catalog_router.post("/metadata/schema/sync",
    summary="Stream-sync one schema into the semantic catalog")
async def sync_schema_metadata(
    req: SchemaMetadataSyncRequest,
    ctx: Ctx,
) -> dict:
    _logger.info(f"In sync_schema_metadata with request: {req}")
    try:
        return await svc.sync_schema_metadata(
            ctx.engines,
            engine=req.engine,
            source_name=req.source_name,
            source_group_id=req.source_group_id,
            source_group_name=req.source_group_name,
            catalog=req.catalog,
            database_name=req.database_name,
            schema_name=req.schema_name,
            include_columns=req.include_columns,
            describe=req.describe,
            doc_batch_size=req.doc_batch_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        _logger.error(f"Failed sync schema metadata: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _logger.error(f"Failed sync schema metadata. Generic Exception. {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@catalog_router.post("/metadata/table/sync",
    summary="Stream-sync one table into the semantic catalog")
async def sync_table_metadata(
    req: TableMetadataSyncRequest,
    ctx: Ctx,
) -> dict:
    _logger.info(f"In sync_table_metadata with request: {req}")
    try:
        return await svc.sync_table_metadata(
            ctx.engines,
            engine=req.engine,
            source_name=req.source_name,
            source_group_id=req.source_group_id,
            source_group_name=req.source_group_name,
            catalog=req.catalog,
            database_name=req.database_name,
            schema_name=req.schema_name,
            table_name=req.table_name,
            include_columns=req.include_columns,
            describe=req.describe,
            doc_batch_size=req.doc_batch_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        _logger.error(
            f"Failed building catalog for table {req.table_name}, catalog: {req.catalog}, "
            f"schema: {req.schema_name}: {exc}"
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _logger.error(f"Failed with generic exception while syncing catalog for"
                      f" table: {req.table_name}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@catalog_router.post("/metadata/column/sync",
    summary="Stream-sync one column into the semantic catalog")
async def sync_column_metadata(
    req: ColumnMetadataSyncRequest,
    ctx: Ctx,
) -> dict:
    try:
        return await svc.sync_column_metadata(
            ctx.engines,
            engine=req.engine,
            source_name=req.source_name,
            source_group_id=req.source_group_id,
            source_group_name=req.source_group_name,
            catalog=req.catalog,
            database_name=req.database_name,
            schema_name=req.schema_name,
            table_name=req.table_name,
            column_name=req.column_name,
            describe=req.describe,
            doc_batch_size=req.doc_batch_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@catalog_router.post(
    "/metadata/table-description/sync",
    response_model=BackendTableDescriptionSyncResponse,
    summary="Generate table descriptions in the semantic catalog",
    description="Given catalog/database/schema and an optional table, generates "
                "table descriptions and refreshes the retrievable schema docs. "
                "When table_name is omitted, all matching tables in the schema "
                "are processed up to the request limit.")
async def sync_catalog_table_descriptions(
    req: CatalogTableDescriptionSyncRequest,
    ctx: Ctx,
) -> BackendTableDescriptionSyncResponse:
    try:
        result = await svc.sync_table_descriptions(
            ctx.engines,
            ctx.adk_runtime,
            engine=req.engine,
            source_name=req.source_name,
            catalog=req.catalog,
            database_name=req.database_name,
            schema_name=req.schema_name,
            table_name=req.table_name,
            missing_only=req.missing_only,
            limit=req.limit,
            sample_size=req.sample_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return BackendTableDescriptionSyncResponse(**result)


@catalog_router.post(
    "/metadata/column-description/sync",
    response_model=BackendColumnDescriptionSyncResponse,
    summary="Generate column descriptions in the semantic catalog",
    description="Given catalog/database/schema and optional table, generates "
                "column descriptions and refreshes the retrievable schema docs. "
                "When table_name is supplied, only that table's columns are "
                "processed. When table_name is omitted, all tables in the schema "
                "are processed up to the request limit. column_name/column_names "
                "require table_name.")
async def sync_catalog_column_descriptions(
    req: CatalogColumnDescriptionSyncRequest,
    ctx: Ctx,
) -> BackendColumnDescriptionSyncResponse:
    try:
        result = await svc.sync_column_descriptions(
            ctx.engines,
            ctx.adk_runtime,
            engine=req.engine,
            source_name=req.source_name,
            catalog=req.catalog,
            database_name=req.database_name,
            schema_name=req.schema_name,
            table_name=req.table_name,
            column_name=req.column_name,
            column_names=req.column_names,
            missing_only=req.missing_only,
            limit=req.limit,
            sample_size=req.sample_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return BackendColumnDescriptionSyncResponse(**result)


@catalog_router.get("/sources",
    summary="List registered catalog sources")
def list_sources() -> dict:
    return svc.list_sources()


@catalog_router.post(
    "/query-history/sync",
    response_model=BackendQueryHistorySyncResponse,
    summary="Load datasource query history into the semantic catalog",
)
async def sync_catalog_query_history(
    req: BackendQueryHistorySyncRequest,
    ctx: Ctx,
) -> BackendQueryHistorySyncResponse:
    try:
        result = await svc.sync_query_history(
            ctx.engines,
            ctx.store,
            engine=req.engine,
            source_name=req.source_name,
            source_group_id=req.source_group_id,
            source_group_name=req.source_group_name,
            start_time=req.start_time,
            end_time=req.end_time,
            catalog=req.catalog,
            database_name=req.database_name,
            schema_name=req.schema_name,
            table_name=req.table_name,
            limit=req.limit,
            page_size=req.page_size,
            batch_size=req.batch_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendQueryHistorySyncResponse(**result)


@catalog_router.post(
    "/query-history/nlp-history/sync",
    response_model=BackendQueryNlpHistorySyncResponse,
    summary="Generate NLP text and embeddings for catalog query history",
)
async def sync_catalog_query_nlp_history(
    req: BackendQueryNlpHistorySyncRequest,
    ctx: Ctx,
) -> BackendQueryNlpHistorySyncResponse:
    _logger.debug(f"Building NLP text and embeddings for catalog query "
                  f"history for incoming req {req}")
    try:
        result = await svc.sync_query_history_nlp(
            ctx.store,
            ctx.adk_runtime,
            ctx.embeddings,
            engine=req.engine,
            source_id=req.source_id,
            source_name=req.source_name,
            source_group_id=req.source_group_id,
            source_group_name=req.source_group_name,
            catalog=req.catalog,
            database_name=req.database_name,
            schema_name=req.schema_name,
            ids=req.ids,
            raw_sql=req.raw_sql,
            limit=req.limit,
            missing_only=req.missing_only,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        _logger.error(f"Failed to build NLP text and embeddings for catalog query with"
                      f" exception {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendQueryNlpHistorySyncResponse(**result)


@catalog_router.post("/schema-context",
    summary="Retrieve top-k schema docs for a question",
    description="Returns the top-k schema documents ranked by cosine similarity "
                "to the question embedding. Used by the context builder to assemble "
                "the SQL generator's prompt.")
def schema_context(req: SchemaContextRequest) -> dict:
    return svc.get_schema_context(
        req.question,
        source_id=req.source_id,
        source_group_id=req.source_group_id,
        engine=req.engine,
        catalog=req.catalog,
        database_name=req.database_name,
        schema_name=req.schema_name,
        k=req.k,
        query_k=req.query_k,
    )


@catalog_router.post("/live-feedback/skills/sync",
    summary="Distill recent production query runs into SkillBank candidate skills",
    description="Reads recent NL-backed query_runs, converts high-confidence success "
                "and failure trajectories into SkillBank skills, and stores them as "
                "candidate skills by default. Set promote=true only for controlled "
                "experiments.")
async def sync_live_feedback_skills(
    req: LiveFeedbackSkillSyncRequest,
    ctx: Ctx,
) -> dict:
    runs = await ctx.store.list_runs_for_skill_evolution(
        limit=req.limit,
        statuses=req.statuses,
    )
    return svc.sync_live_feedback_skills(runs, promote=req.promote)


@catalog_router.post("/search-columns",
    summary="Vector search over column descriptions")
def search_columns(req: ColumnSearchRequest) -> dict:
    return svc.search_columns(
        req.query,
        source_id=req.source_id,
        source_group_id=req.source_group_id,
        k=req.k,
    )


@catalog_router.post(
    "/search/tables",
    response_model=BackendTableSearchResponse,
    summary="Hybrid table search over the semantic catalog")
def search_catalog_tables(req: BackendTableSearchRequest, ctx: Ctx) -> BackendTableSearchResponse:
    try:
        result = svc.search_tables(
            ctx.engines,
            query=req.query,
            engine=req.engine,
            catalog=req.catalog,
            database_name=req.database_name,
            schema_name=req.schema_name,
            top_k=req.top_k,
            semantic_top_k=req.semantic_top_k,
            lexical_top_k=req.lexical_top_k,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendTableSearchResponse(**result)


@catalog_router.post(
    "/search/columns",
    response_model=BackendColumnSearchResponse,
    summary="Hybrid column search over the semantic catalog")
def search_catalog_columns(
    req: BackendColumnSearchRequest,
    ctx: Ctx,
) -> BackendColumnSearchResponse:
    try:
        result = svc.search_columns_hybrid(
            ctx.engines,
            query=req.query,
            engine=req.engine,
            catalog=req.catalog,
            database_name=req.database_name,
            schema_name=req.schema_name,
            table_name=req.table_name,
            top_k=req.top_k,
            semantic_top_k=req.semantic_top_k,
            lexical_top_k=req.lexical_top_k,
            matched_columns_limit=req.matched_columns_limit,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendColumnSearchResponse(**result)


# ── /skillbank routes ──────────────────────────────────────────────────────────

@skillbank_router.post("/seed",
    summary="Insert curated general_sql and dialect seed skills",
    description="Inserts the built-in general SQL and Snowflake dialect skills "
                "into the SqlSkillBank. Idempotent — existing titles are skipped.")
def seed_skills() -> dict:
    return svc.seed_skillbank()


@skillbank_router.get("/skills",
    summary="List promoted SqlSkillBank skills")
def list_skills(scope: str | None = None, dialect: str | None = None) -> dict:
    return svc.list_skills(scope=scope, dialect=dialect)


# ── /sql routes ────────────────────────────────────────────────────────────────

@sql_router.post("/verify",
    summary="Run static-lattice gates (and optionally execute) a SQL candidate",
    description="Runs the formal static verification lattice (Safe→Parse→Bind→Scope→Join) "
                "and returns gate results. When execute=true and all gates pass, the SQL "
                "is also executed against the datasource.")
def verify_sql(req: VerifyRequest) -> dict:
    return svc.verify_sql(req.sql, source_id=req.source_id, execute=req.execute)


@sql_router.post("/score",
    summary="Compute composite verifier reward R(τ)",
    description="Scores a candidate SQL with the full reward cascade (Equation 12): "
                "static gates, obligation satisfaction ω, and (if gold_sql is provided) "
                "execution equivalence. The reward total maps directly to the GRPO "
                "training signal.")
async def score_sql(req: ScoreRequest) -> dict:
    return await svc.score_sql(
        req.question, req.sql,
        gold_sql=req.gold_sql, source_id=req.source_id,
    )


@sql_router.post("/generate",
    summary="Single-shot SQL generation (Arctic-7B via Ollama)",
    description="Generates one SQL candidate using the SkillSQL-RL training-path "
                "workflow (schema retrieval + skill context, no critic/refiner loop). "
                "For the production workflow with refinement use POST /text2sql/run.")
async def generate_sql(req: GenerateRequest) -> dict:
    try:
        return await svc.generate_sql(
            req.question,
            source_id=req.source_id,
            source_group_id=req.source_group_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
