import logging

from fastapi import APIRouter, Depends, HTTPException, Query

import app.services.catalog as catalog_svc
from app.dependencies import AppContext, get_ctx
from app.schemas import (
    BackendCatalogMetadataSyncRequest,
    BackendColumnDescriptionSyncRequest,
    BackendColumnDescriptionSyncResponse,
    BackendColumnSearchRequest,
    BackendColumnSearchResponse,
    BackendMetadataSyncResponse,
    BackendQueryHistoryNlpSyncRequest,
    BackendQueryHistorySyncRequest,
    BackendQueryHistorySyncResponse,
    BackendQueryNlpHistorySyncRequest,
    BackendQueryNlpHistorySyncResponse,
    BackendSchemaMetadataSyncRequest,
    BackendTableDescriptionSyncRequest,
    BackendTableDescriptionSyncResponse,
    BackendTableMetadataSyncRequest,
    BackendTableSearchRequest,
    BackendTableSearchResponse,
    DataUsageNlpSyncByIdRequest,
    DataUsageNlpSyncRequest,
    DataUsageNlpSyncResponse,
    DiscoveryCatalogSyncRequest,
    DiscoveryCatalogSyncResponse,
    DiscoveryQuery,
    DiscoveryRoleContextResponse,
    DiscoverySimilarResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/discovery", tags=["discovery"])


@router.get("/role-context", response_model=DiscoveryRoleContextResponse)
async def role_context(
    soeid: str = Query(...),
    limit: int = Query(default=10, ge=1, le=50),
    ctx: AppContext = Depends(get_ctx),
) -> DiscoveryRoleContextResponse:
    segment_scope: list[str] = []
    try:
        payload = await ctx.directory_service.get_user_directory_information(soeid)
        info = payload.get("UserDirectoryInformation", {})
        work = info.get("UsersWorkInformation", {}) if isinstance(info, dict) else {}
        raw = work.get("ManagedSegmentLastTwoLevels", []) if isinstance(work, dict) else []
        if isinstance(raw, list):
            segment_scope = [str(item).strip() for item in raw if str(item).strip()]
    except Exception:
        segment_scope = []

    role, queries, metadata = await ctx.discovery_service.role_context(
        soeid,
        limit=limit,
        segment_values=segment_scope,
    )
    mapped = [
        DiscoveryQuery(
            query_id=item.query_id,
            sql_text=item.sql_text,
            sql2text=item.sql2text,
            tables=item.tables,
            created_at=item.created_at,
            engine=item.engine,
        )
        for item in queries
    ]
    return DiscoveryRoleContextResponse(soeid=soeid, role=role, queries=mapped, metadata=metadata)


@router.get("/similar", response_model=DiscoverySimilarResponse)
async def similar(
    soeid: str = Query(...),
    q: str = Query(..., min_length=3),
    limit: int = Query(default=5, ge=1, le=500),
    ctx: AppContext = Depends(get_ctx),
) -> DiscoverySimilarResponse:
    segment_scope: list[str] = []
    try:
        payload = await ctx.directory_service.get_user_directory_information(soeid)
        info = payload.get("UserDirectoryInformation", {})
        work = info.get("UsersWorkInformation", {}) if isinstance(info, dict) else {}
        raw = work.get("ManagedSegmentLastTwoLevels", []) if isinstance(work, dict) else []
        if isinstance(raw, list):
            segment_scope = [str(item).strip() for item in raw if str(item).strip()]
    except Exception:
        segment_scope = []

    queries = await ctx.discovery_service.similar_queries(
        soeid,
        q,
        limit=limit,
        segment_values=segment_scope,
    )
    mapped = [
        DiscoveryQuery(
            query_id=item.query_id,
            sql_text=item.sql_text,
            sql2text=item.sql2text,
            tables=item.tables,
            created_at=item.created_at,
            engine=item.engine,
        )
        for item in queries
    ]
    return DiscoverySimilarResponse(soeid=soeid, prompt=q, queries=mapped)


@router.post("/catalog/sync", response_model=DiscoveryCatalogSyncResponse)
async def sync_catalog(
    payload: DiscoveryCatalogSyncRequest,
    ctx: AppContext = Depends(get_ctx),
) -> DiscoveryCatalogSyncResponse:
    result = await ctx.discovery_catalog_loader.sync_catalog(
        max_assets=payload.max_assets,
        concurrency=payload.concurrency,
    )
    return DiscoveryCatalogSyncResponse(**result)


@router.post("/backend-metadata/tables/search", response_model=BackendTableSearchResponse)
async def search_backend_metadata_tables(
    payload: BackendTableSearchRequest,
    ctx: AppContext = Depends(get_ctx),
) -> BackendTableSearchResponse:
    try:
        result = catalog_svc.search_tables(
            ctx.engines,
            query=payload.query,
            engine=payload.engine,
            catalog=payload.catalog,
            database_name=payload.database_name,
            schema_name=payload.schema_name,
            top_k=payload.top_k,
            semantic_top_k=payload.semantic_top_k,
            lexical_top_k=payload.lexical_top_k,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendTableSearchResponse(**result)


@router.post("/backend-metadata/columns/search", response_model=BackendColumnSearchResponse)
async def search_backend_metadata_columns(
    payload: BackendColumnSearchRequest,
    ctx: AppContext = Depends(get_ctx),
) -> BackendColumnSearchResponse:
    try:
        result = catalog_svc.search_columns_hybrid(
            ctx.engines,
            query=payload.query,
            engine=payload.engine,
            catalog=payload.catalog,
            database_name=payload.database_name,
            schema_name=payload.schema_name,
            table_name=payload.table_name,
            top_k=payload.top_k,
            semantic_top_k=payload.semantic_top_k,
            lexical_top_k=payload.lexical_top_k,
            matched_columns_limit=payload.matched_columns_limit,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendColumnSearchResponse(**result)


@router.post("/backend-metadata/catalog/sync", response_model=BackendMetadataSyncResponse)
async def sync_backend_catalog_metadata(
    payload: BackendCatalogMetadataSyncRequest,
    ctx: AppContext = Depends(get_ctx),
) -> BackendMetadataSyncResponse:
    try:
        result = await ctx.backend_metadata_sync_service.sync_catalog(
            engine=payload.engine,
            catalog=payload.catalog,
            database_name=payload.database_name,
            include_columns=payload.include_columns,
            batch_size=payload.batch_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendMetadataSyncResponse(**result)


@router.post("/backend-metadata/schema/sync", response_model=BackendMetadataSyncResponse)
async def sync_backend_schema_metadata(
    payload: BackendSchemaMetadataSyncRequest,
    ctx: AppContext = Depends(get_ctx),
) -> BackendMetadataSyncResponse:
    try:
        result = await ctx.backend_metadata_sync_service.sync_schema(
            engine=payload.engine,
            catalog=payload.catalog,
            database_name=payload.database_name,
            schema=payload.schema_name,
            include_columns=payload.include_columns,
            batch_size=payload.batch_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendMetadataSyncResponse(**result)


@router.post("/backend-metadata/table/sync", response_model=BackendMetadataSyncResponse)
async def sync_backend_table_metadata(
    payload: BackendTableMetadataSyncRequest,
    ctx: AppContext = Depends(get_ctx),
) -> BackendMetadataSyncResponse:
    try:
        logger.debug(f"In sync_backend_table_metadata: {payload}")
        result = await ctx.backend_metadata_sync_service.sync_table(
            engine=payload.engine,
            catalog=payload.catalog,
            database_name=payload.database_name,
            schema=payload.schema_name,
            table=payload.table_name,
            include_columns=payload.include_columns,
            batch_size=payload.batch_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendMetadataSyncResponse(**result)


@router.post("/backend-metadata/table-description/sync", response_model=BackendTableDescriptionSyncResponse)
async def sync_backend_table_descriptions(
    payload: BackendTableDescriptionSyncRequest,
    ctx: AppContext = Depends(get_ctx),
) -> BackendTableDescriptionSyncResponse:
    if not payload.catalog or not payload.schema_name:
        raise HTTPException(status_code=400, detail="catalog and schema_name are required")
    try:
        logger.debug(f"In sync_backend_table_descriptions: {payload}")
        result = await catalog_svc.sync_table_descriptions(
            ctx.engines,
            ctx.adk_runtime,
            engine=payload.engine,
            catalog=payload.catalog,
            database_name=payload.database_name,
            schema_name=payload.schema_name,
            table_name=payload.table_name,
            missing_only=payload.missing_only,
            limit=payload.limit,
            sample_size=payload.sample_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        logger.error(f"Failed in sync_backend_table_descriptions: {payload}, {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendTableDescriptionSyncResponse(**result)


@router.post("/backend-metadata/column-description/sync", response_model=BackendColumnDescriptionSyncResponse)
async def sync_backend_column_descriptions(
    payload: BackendColumnDescriptionSyncRequest,
    ctx: AppContext = Depends(get_ctx),
) -> BackendColumnDescriptionSyncResponse:
    try:
        logger.debug(f"In sync_backend_column_descriptions: {payload}")
        result = await catalog_svc.sync_column_descriptions(
            ctx.engines,
            ctx.adk_runtime,
            engine=payload.engine,
            catalog=payload.catalog,
            database_name=payload.database_name,
            schema_name=payload.schema_name,
            table_name=payload.table_name,
            column_name=payload.column_name,
            column_names=payload.column_names,
            missing_only=payload.missing_only,
            limit=payload.limit,
            sample_size=payload.sample_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        logger.error(f"Failed in sync_backend_column_descriptions: {payload}, {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendColumnDescriptionSyncResponse(**result)


@router.post("/backend-metadata/query-history/sync", response_model=BackendQueryHistorySyncResponse)
async def sync_backend_query_history(
    payload: BackendQueryHistorySyncRequest,
    ctx: AppContext = Depends(get_ctx),
) -> BackendQueryHistorySyncResponse:
    try:
        logger.debug(f"In sync_backend_query_history: {payload}")
        result = await catalog_svc.sync_query_history(
            ctx.engines,
            ctx.store,
            engine=payload.engine,
            source_name=payload.source_name,
            start_time=payload.start_time,
            end_time=payload.end_time,
            catalog=payload.catalog,
            database_name=payload.database_name,
            schema_name=payload.schema_name,
            table_name=payload.table_name,
            limit=payload.limit,
            page_size=payload.page_size,
            batch_size=payload.batch_size,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendQueryHistorySyncResponse(**result)


@router.post("/backend-metadata/query-history/nlp-history/sync", response_model=BackendQueryNlpHistorySyncResponse)
async def sync_backend_query_nlp_history(
    payload: BackendQueryNlpHistorySyncRequest,
    ctx: AppContext = Depends(get_ctx),
) -> BackendQueryNlpHistorySyncResponse:
    try:
        logger.debug(f"In sync_backend_query_nlp_history: {payload}")
        result = await catalog_svc.sync_query_history_nlp(
            ctx.store,
            ctx.adk_runtime,
            ctx.embeddings,
            engine=payload.engine,
            source_id=payload.source_id,
            source_name=payload.source_name,
            catalog=payload.catalog,
            database_name=payload.database_name,
            schema_name=payload.schema_name,
            ids=payload.ids,
            raw_sql=payload.raw_sql,
            limit=payload.limit,
            missing_only=payload.missing_only,
        )
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BackendQueryNlpHistorySyncResponse(**result)


@router.post("/data-usage/nlp-sync", response_model=DataUsageNlpSyncResponse)
async def sync_data_usage_nlp(
    payload: DataUsageNlpSyncRequest,
    ctx: AppContext = Depends(get_ctx),
) -> DataUsageNlpSyncResponse:
    result = await ctx.data_usage_nlp_service.sync(
        limit=payload.limit,
        concurrency=payload.concurrency,
        batch_size=payload.batch_size,
    )
    return DataUsageNlpSyncResponse(**result)


@router.post("/backend-metadata/query-history/nlp-sync", response_model=DataUsageNlpSyncResponse)
async def sync_backend_query_history_nlp(
    payload: BackendQueryHistoryNlpSyncRequest,
    ctx: AppContext = Depends(get_ctx),
) -> DataUsageNlpSyncResponse:
    result = await ctx.data_usage_nlp_service.sync_backend_query_history(
        engine=payload.engine,
        limit=payload.limit,
        concurrency=payload.concurrency,
        batch_size=payload.batch_size,
        validate_with_explain=payload.validate_with_explain,
    )
    return DataUsageNlpSyncResponse(**result)


@router.post("/data-usage/nlp-sync-by-id", response_model=DataUsageNlpSyncResponse)
async def sync_data_usage_nlp_by_id(
    payload: DataUsageNlpSyncByIdRequest,
    ctx: AppContext = Depends(get_ctx),
) -> DataUsageNlpSyncResponse:
    result = await ctx.data_usage_nlp_service.sync_by_ids(
        ids=payload.ids,
        concurrency=payload.concurrency,
        batch_size=payload.batch_size,
    )
    return DataUsageNlpSyncResponse(**result)
