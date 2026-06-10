from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from app.adapters.base import BackendMetadataRecord, BackendQueryHistoryRecord
from app.adapters.registry import EngineRegistry
from app.core.store import SQLStore

logger = logging.getLogger(__name__)

@dataclass
class BackendMetadataSyncStats:
    catalog_rows: int = 0
    schema_rows: int = 0
    table_rows: int = 0
    column_rows: int = 0
    batches_processed: int = 0


class BackendMetadataSyncService:
    """Streams backend metadata/query history into the central SQL repository."""

    def __init__(self, store: SQLStore, engines: EngineRegistry):
        self._store = store
        self._engines = engines

    async def sync_catalog(
        self,
        *,
        engine: str,
        catalog: str,
        database_name: str | None = None,
        include_columns: bool = True,
        batch_size: int = 500,
    ) -> dict[str, Any]:
        adapter = self._engines.get(engine)
        stats = await self._sync_metadata_records(
            adapter.iter_catalog_metadata(
                catalog,
                database_name=database_name,
                include_columns=include_columns,
            ),
            batch_size=batch_size,
        )
        return self._metadata_response(
            engine=adapter.name,
            scope_type="catalog",
            scope={"catalog": catalog, "database": database_name},
            stats=stats,
        )

    async def sync_schema(
        self,
        *,
        engine: str,
        catalog: str,
        database_name: str | None = None,
        schema: str,
        include_columns: bool = True,
        batch_size: int = 500,
    ) -> dict[str, Any]:
        adapter = self._engines.get(engine)
        stats = await self._sync_metadata_records(
            adapter.iter_schema_metadata(
                catalog,
                schema,
                database_name=database_name,
                include_columns=include_columns,
            ),
            batch_size=batch_size,
        )
        return self._metadata_response(
            engine=adapter.name,
            scope_type="schema",
            scope={"catalog": catalog, "database": database_name, "schema": schema},
            stats=stats,
        )

    async def sync_table(
        self,
        *,
        engine: str,
        catalog: str,
        database_name: str | None = None,
        schema: str,
        table: str,
        include_columns: bool = True,
        batch_size: int = 500,
    ) -> dict[str, Any]:
        logger.debug(
            "In sync_backend_table_metadata: %s, table: %s, catalog: %s, database: %s, schema: %s",
            engine,
            table,
            catalog,
            database_name,
            schema,
        )
        adapter = self._engines.get(engine)
        stats = await self._sync_metadata_records(
            adapter.iter_table_metadata(
                catalog,
                schema,
                table,
                database_name=database_name,
                include_columns=include_columns,
            ),
            batch_size=batch_size,
        )
        return self._metadata_response(
            engine=adapter.name,
            scope_type="table",
            scope={"catalog": catalog, "database": database_name, "schema": schema, "table": table},
            stats=stats,
        )

    async def sync_query_history(
        self,
        *,
        engine: str,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        table: str | None = None,
        limit: int | None = None,
        page_size: int = 1000,
        batch_size: int = 500,
    ) -> dict[str, Any]:
        logger.debug(
            "In sync_backend_query_history: %s, start_time: %s, end_time: %s, "
            "catalog: %s, schema: %s, table: %s, limit: %s",
            engine,
            start_time,
            end_time,
            catalog,
            schema,
            table,
            limit,
        )
        adapter = self._engines.get(engine)
        rows_synced = await self._sync_query_history_records(
            adapter.iter_query_history(
                start_time=start_time,
                end_time=end_time,
                catalog=catalog,
                schema=schema,
                table=table,
                limit=limit,
                page_size=page_size,
            ),
            batch_size=batch_size,
        )
        return {
            "synced_at": datetime.now(UTC).isoformat(),
            "engine": adapter.name,
            "scope": {
                "start_time": start_time,
                "end_time": end_time,
                "catalog": catalog,
                "schema": schema,
                "table": table,
                "limit": limit,
            },
            "query_history_rows": rows_synced,
        }

    async def _sync_metadata_records(
        self,
        records: AsyncIterator[BackendMetadataRecord],
        *,
        batch_size: int,
    ) -> BackendMetadataSyncStats:
        stats = BackendMetadataSyncStats()
        batch: list[dict[str, Any]] = []

        async for record in records:
            batch.append(asdict(record))
            if len(batch) >= max(1, int(batch_size)):
                self._merge_counts(stats, await self._store.upsert_backend_metadata_records(batch))
                stats.batches_processed += 1
                batch.clear()

        if batch:
            self._merge_counts(stats, await self._store.upsert_backend_metadata_records(batch))
            stats.batches_processed += 1

        return stats

    async def _sync_query_history_records(
        self,
        records: AsyncIterator[BackendQueryHistoryRecord],
        *,
        batch_size: int,
    ) -> int:
        rows_synced = 0
        batch: list[dict[str, Any]] = []

        async for record in records:
            batch.append(asdict(record))
            if len(batch) >= max(1, int(batch_size)):
                rows_synced += await self._store.upsert_backend_query_history_rows(batch)
                batch.clear()

        if batch:
            rows_synced += await self._store.upsert_backend_query_history_rows(batch)

        return rows_synced

    @staticmethod
    def _merge_counts(stats: BackendMetadataSyncStats, counts: dict[str, int]) -> None:
        stats.catalog_rows += int(counts.get("catalogs", 0))
        stats.schema_rows += int(counts.get("schemas", 0))
        stats.table_rows += int(counts.get("tables", 0))
        stats.column_rows += int(counts.get("columns", 0))

    @staticmethod
    def _metadata_response(
        *,
        engine: str,
        scope_type: str,
        scope: dict[str, Any],
        stats: BackendMetadataSyncStats,
    ) -> dict[str, Any]:
        return {
            "synced_at": datetime.now(UTC).isoformat(),
            "engine": engine,
            "scope_type": scope_type,
            "scope": scope,
            "catalog_rows": stats.catalog_rows,
            "schema_rows": stats.schema_rows,
            "table_rows": stats.table_rows,
            "column_rows": stats.column_rows,
            "batches_processed": stats.batches_processed,
        }
