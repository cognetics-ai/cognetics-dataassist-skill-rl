from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from urllib.parse import urlparse

from skillsql.connectors.base import DataSourceConnector, Metadata, SourceConfig, TableMeta
from skillsql.connectors.factory import (
    ConnectorFactory,
    _ensure_registered,
    source_config_from_settings,
)

from app.adapters.base import (
    BackendMetadataRecord,
    EngineAdapter,
    EngineHandle,
    EngineStatus,
    ExplainResult,
    ResultPage,
)
from app.config import Settings
from app.observability.logging import get_logger

_logger = get_logger(__name__)

class ConnectorEngineAdapter(EngineAdapter):
    """EngineAdapter wrapper around a SkillSQL datasource connector.

    The connector owns vendor protocol details. This adapter exposes the
    inherited app's async engine contract, including streaming metadata records
    for semantic catalog sync.
    """

    def __init__(self, name: str, connector: DataSourceConnector, settings: Settings):
        self.name = name
        self._connector = connector
        self._settings = settings

    async def explain(self, sql: str) -> ExplainResult:
        plan = await self._connector.explain_plan(sql)
        return ExplainResult(
            ok=plan.ok,
            summary={
                "engine": self.name,
                "dialect": self._connector.dialect,
                "plan": plan.plan_text,
                "elapsed_ms": plan.elapsed_ms,
                "error": plan.error,
            },
        )

    async def execute_async(self, sql: str) -> EngineHandle:
        result = await self._connector.execute(
            sql,
            read_only=True,
            timeout_s=self._settings.max_runtime_seconds,
            row_cap=self._settings.default_limit,
        )
        handle_id = str(uuid.uuid4())
        return EngineHandle(
            handle_id=handle_id,
            raw={
                "query": sql,
                "done": True,
                "result": result,
                "rows": result.rows,
                "schema": [{"name": name, "type": ""} for name in result.columns],
                "error": result.error,
            },
        )

    async def get_status(self, handle: EngineHandle) -> EngineStatus:
        error = handle.raw.get("error")
        return EngineStatus(
            state="FAILED" if error else "FINISHED",
            done=True,
            progress_percentage=100,
            stats={},
            error={"message": error} if error else None,
        )

    async def fetch_results(
        self,
        handle: EngineHandle,
        page_token: str | None = None,
    ) -> ResultPage:
        return ResultPage(
            schema=list(handle.raw.get("schema") or []),
            rows=list(handle.raw.get("rows") or []),
            next_page_token=None,
        )

    async def cancel(self, handle: EngineHandle) -> bool:
        return True

    async def iter_catalog_metadata(
        self,
        catalog: str,
        *,
        database_name: str | None = None,
        include_columns: bool = True,
    ) -> AsyncIterator[BackendMetadataRecord]:
        _logger.debug(f"Iterating catalog metadata for {catalog} in connector engine adapter")
        concrete_database = self._concrete_database_name(catalog, database_name)
        metadata = await self._connector.get_metadata(
            catalog_name=concrete_database,
            db_schema=self._connector.config.db_schema,
        )
        async for record in self._iter_metadata(
            metadata,
            database_name=concrete_database,
            include_columns=include_columns,
        ):
            yield record

    async def iter_schema_metadata(
        self,
        catalog: str,
        schema: str,
        *,
        database_name: str | None = None,
        include_columns: bool = True,
    ) -> AsyncIterator[BackendMetadataRecord]:
        concrete_database = self._concrete_database_name(catalog, database_name)
        metadata = await self._connector.get_metadata(
            catalog_name=concrete_database,
            db_schema=schema,
        )
        async for record in self._iter_metadata(
            metadata,
            database_name=concrete_database,
            include_columns=include_columns,
        ):
            yield record

    async def iter_table_metadata(
        self,
        catalog: str,
        schema: str,
        table: str,
        *,
        database_name: str | None = None,
        include_columns: bool = True,
    ) -> AsyncIterator[BackendMetadataRecord]:
        concrete_database = self._concrete_database_name(catalog, database_name)
        metadata = await self._connector.get_metadata(
            catalog_name=concrete_database,
            db_schema=schema,
        )
        matching = [item for item in metadata.tables if item.name.lower() == table.lower()]
        if not matching:
            scope = ".".join(part for part in (concrete_database, schema, table) if part)
            raise ValueError(f"{self.name} table not found: {scope}")
        narrowed = Metadata(
            source_type=metadata.source_type,
            catalog_name=metadata.catalog_name,
            db_schema=metadata.db_schema,
            tables=matching,
            collected_at=metadata.collected_at,
        )
        async for record in self._iter_metadata(
            narrowed,
            database_name=concrete_database,
            include_columns=include_columns,
        ):
            yield record

    async def _iter_metadata(
        self,
        metadata: Metadata,
        *,
        database_name: str | None,
        include_columns: bool,
    ) -> AsyncIterator[BackendMetadataRecord]:
        catalog_name = metadata.catalog_name or database_name or self._connector.config.catalog_name
        database = database_name or metadata.catalog_name or self._connector.config.database
        yield BackendMetadataRecord(
            entity_type="catalog",
            engine=self.name,
            catalog_name=catalog_name,
            database_name=database,
        )

        seen_schemas: set[str] = set()
        for table in metadata.tables:
            schema_name = table.db_schema or metadata.db_schema
            if schema_name and schema_name not in seen_schemas:
                seen_schemas.add(schema_name)
                yield BackendMetadataRecord(
                    entity_type="schema",
                    engine=self.name,
                    catalog_name=table.catalog_name or catalog_name,
                    database_name=database,
                    schema_name=schema_name,
                )
            yield self._table_record(
                table,
                catalog_name=catalog_name,
                database_name=database,
                schema_name=schema_name,
            )
            if include_columns:
                for column in table.columns:
                    yield BackendMetadataRecord(
                        entity_type="column",
                        engine=self.name,
                        catalog_name=table.catalog_name or catalog_name,
                        database_name=database,
                        schema_name=schema_name,
                        table_name=table.name,
                        column_name=column.name,
                        ordinal_position=column.ordinal,
                        data_type=column.data_type,
                        nullable=column.nullable,
                        description=column.comment,
                        raw={
                            "sample_values": column.sample_values,
                            "null_fraction": column.null_fraction,
                            "distinct_estimate": column.distinct_estimate,
                            "is_primary_key": column.is_primary_key,
                        },
                    )

    def _table_record(
        self,
        table: TableMeta,
        *,
        catalog_name: str | None,
        database_name: str | None,
        schema_name: str | None,
    ) -> BackendMetadataRecord:
        return BackendMetadataRecord(
            entity_type="table",
            engine=self.name,
            catalog_name=table.catalog_name or catalog_name,
            database_name=database_name,
            schema_name=table.db_schema or schema_name,
            table_name=table.name,
            object_type=table.table_type,
            description=table.comment,
            raw={"row_count": table.row_estimate},
        )

    def _concrete_database_name(
        self,
        catalog: str,
        database_name: str | None,
    ) -> str:
        return (
            database_name
            or self._connector.config.database
            or self._connector.config.catalog_name
            or catalog
        )


def connector_adapter_from_settings(source_type: str, settings: Settings) -> ConnectorEngineAdapter:
    """Create a connector-backed adapter for datasource types not native to app."""
    _ensure_registered()
    key = source_type.lower().strip()
    if key == "sql":
        key = "postgres"
    config = _source_config_from_app_settings(key, settings)
    connector = ConnectorFactory().create(key, config)
    return ConnectorEngineAdapter(key, connector, settings)


def _source_config_from_app_settings(source_type: str, settings: Settings) -> SourceConfig:
    if source_type == "snowflake":
        from skillsql.config.settings import get_settings

        return source_config_from_settings(get_settings(), "snowflake")

    if source_type == "postgres":
        dsn = settings.postgres_dsn
        parsed = urlparse(dsn.replace("postgresql+psycopg://", "postgresql://"))
        database = parsed.path.lstrip("/") or None
        return SourceConfig(
            source_type="postgres",
            name="postgres",
            host=dsn,
            database=database,
            catalog_name=database,
            db_schema=settings.postgres_schema or "public",
            extra={"dsn": dsn},
        )

    raise ValueError(f"Unsupported connector-backed engine: {source_type}")
