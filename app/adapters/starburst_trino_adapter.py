from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from urllib.parse import quote

import aiohttp

from app.adapters.base import (
    BackendMetadataRecord,
    BackendQueryHistoryRecord,
    EngineAdapter,
    EngineHandle,
    EngineStatus,
    ExplainResult,
    ResultPage,
)
from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GalaxyToken:
    access_token: str
    expires_at: float


class StarburstTrinoAdapter(EngineAdapter):
    """Starburst adapter with two explicitly separate auth paths.

    1. Starburst Galaxy API metadata calls use OAuth client credentials against the
       Galaxy account-domain API, and send Authorization: Bearer <access_token>.

    2. Trino query execution uses the cluster-domain Trino REST protocol and sends
       Basic auth. Do not send the Galaxy API bearer token to Trino.
    """

    name = "starburst"

    def __init__(self, settings: Settings):
        self._settings = settings

        # Galaxy account API host. Example: https://cognetics.galaxy.starburst.io
        self._api_base_url = (
            getattr(settings, "starburst_api_url", None)
            or getattr(settings, "starburst_url", None)
            or f"https://{settings.starburst_host}:{settings.starburst_port}"
        ).rstrip("/")

        # Trino cluster host. Example: https://cognetics-free-cluster.trino.galaxy.starburst.io
        self._trino_base_url = (
            getattr(settings, "starburst_trino_url", None)
            or f"https://{settings.starburst_trino_host}:{settings.starburst_port}"
        ).rstrip("/")
        query_history_trino_host = getattr(settings, "starburst_query_history_trino_host", "")
        self._query_history_trino_base_url = (
            getattr(settings, "starburst_query_history_trino_url", None)
            or (f"https://{query_history_trino_host}:{settings.starburst_port}" if query_history_trino_host else "")
            or self._trino_base_url
        ).rstrip("/")

        self._timeout = aiohttp.ClientTimeout(total=settings.starburst_timeout_ms / 1000)
        self._verify_ssl = bool(settings.starburst_verify_ssl)
        self._ssl_context = self._build_ssl_context(self._verify_ssl)
        if not self._verify_ssl:
            logger.warning("Starburst SSL certificate verification is disabled")

        self._galaxy_token: GalaxyToken | None = None
        self._local_cancelled: set[str] = set()

        # Trino session state. These headers are updated from X-Trino-Set-* response headers.
        self._trino_session_props: dict[str, str] = {}
        self._trino_prepared_statements: dict[str, str] = {}
        self._trino_transaction_id: str | None = None
        self._trino_catalog = getattr(settings, "starburst_catalog", None)
        self._trino_schema = getattr(settings, "starburst_schema", None)
        # self._trino_role = getattr(settings, "starburst_role", None)
        role = getattr(settings, "starburst_role", None)
        self._trino_role_header = f"system=ROLE{{{role}}}" if role else None
        query_history_role = getattr(settings, "starburst_query_history_role", None) or role
        self._query_history_role_header = f"system=ROLE{{{query_history_role}}}" if query_history_role else None

    # ---------------------------------------------------------------------
    # Public metadata API
    # ---------------------------------------------------------------------

    async def iter_catalog_metadata(
        self,
        catalog: str,
        *,
        database_name: str | None = None,
        include_columns: bool = True,
    ) -> AsyncIterator[BackendMetadataRecord]:
        logger.debug(f"Iterating catalog metadata for {catalog}")
        catalog_key = self._name_lookup_path_id(catalog)
        logger.debug(f"Catalog Key: {catalog_key}")
        catalog_metadata = await self._galaxy_get_json(f"/public/api/v1/catalog/{catalog_key}/catalogMetadata")
        logger.debug(f"catalog_metadata: {catalog_metadata}")
        yield self._catalog_record(catalog_metadata)

        async for schema in self._galaxy_iter_paginated(f"/public/api/v1/catalog/{catalog_key}/schema"):
            yield self._schema_record(catalog_metadata, schema)
            async for record in self._iter_schema_children(
                catalog_key,
                catalog_metadata,
                schema,
                include_columns=include_columns,
            ):
                logger.debug(f"Record: {record}")
                yield record

    async def iter_schema_metadata(
        self,
        catalog: str,
        schema: str,
        *,
        database_name: str | None = None,
        include_columns: bool = True,
    ) -> AsyncIterator[BackendMetadataRecord]:
        logger.debug(f"Iterating schema metadata for catalog: {catalog}")
        catalog_key = self._name_lookup_path_id(catalog)
        catalog_metadata = await self._galaxy_get_json(f"/public/api/v1/catalog/{catalog_key}/catalogMetadata")
        logger.info(f"Catalog Metadata: {catalog_metadata}")

        schema_metadata = await self._find_schema(catalog_key, schema)
        logger.info(f"Schema Metadata: {schema_metadata}")

        logger.debug("About to return catalog record with catalog metadata")
        yield self._catalog_record(catalog_metadata)
        logger.debug("About to return schema record with schema metadata")
        yield self._schema_record(catalog_metadata, schema_metadata)

        async for record in self._iter_schema_children(
            catalog_key,
            catalog_metadata,
            schema_metadata,
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
        logger.debug(f"Iterating table metadata for {catalog}, table: {table}, schema: {schema}")
        catalog_key = self._name_lookup_path_id(catalog)
        logger.debug(f"Catalog Key: {catalog_key}")
        catalog_metadata = await self._galaxy_get_json(f"/public/api/v1/catalog/{catalog_key}/catalogMetadata")
        schema_metadata = await self._find_schema(catalog_key, schema)
        schema_key = self._metadata_path_id(schema_metadata, "schemaId", "schemaName", "name")
        table_metadata = await self._find_table(catalog_key, schema_key, table)

        logger.debug(f"Returning catalog record: {catalog_metadata}")
        yield self._catalog_record(catalog_metadata)
        logger.debug(f"Returning schema record: {schema_metadata}")
        yield self._schema_record(catalog_metadata, schema_metadata)
        logger.debug(f"Returning table record: {table_metadata}")
        yield self._table_record(catalog_metadata, schema_metadata, table_metadata)

        if include_columns:
            async for record in self._iter_table_columns(
                catalog_key,
                schema_key,
                catalog_metadata,
                schema_metadata,
                table_metadata,
            ):
                logger.debug(f"Returning column record: {record}")
                yield record

    async def iter_query_history(
        self,
        *,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        table: str | None = None,
        limit: int | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[BackendQueryHistoryRecord]:
        logger.debug(f"Iterating query history (starburst engine) for {start_time} to {end_time} with catalog: {catalog}, schema: {schema}, table: {table}")
        effective_page_size = max(1, min(int(page_size or 1000), 5000))
        remaining = int(limit) if limit and limit > 0 else None
        offset = 0

        while True:
            current_limit = effective_page_size if remaining is None else min(effective_page_size, remaining)
            if current_limit <= 0:
                break

            sql = self._query_history_sql(
                start_time=start_time,
                end_time=end_time,
                catalog=catalog,
                schema=schema,
                table=table,
                limit=current_limit,
                offset=offset,
            )
            rows = await self._fetch_query_history_page(sql)
            logger.debug(f"Got {len(rows)} rows")
            if not rows:
                logger.warning("No rows found for query...")
                break

            for row in rows:
                record = self._query_history_record(row)
                if record:
                    yield record

            loaded = len(rows)
            logger.info(f"Loaded {loaded} rows")
            offset += loaded
            if remaining is not None:
                remaining -= loaded
                if remaining <= 0:
                    break
            if loaded < current_limit:
                break

    async def extract_catalog_metadata(self, catalog_id: str) -> dict[str, Any]:
        """Return raw Galaxy metadata and normalized Postgres-ready rows.

        catalog_id can be a raw ID or a lookup expression such as name=tpch.
        """
        catalog_key = self._name_lookup_path_id(catalog_id)

        catalog = await self._galaxy_get_json(
            f"/public/api/v1/catalog/{catalog_key}/catalogMetadata"
        )

        schemas = await self._galaxy_get_paginated(
            f"/public/api/v1/catalog/{catalog_key}/schema"
        )

        catalog_row = self._catalog_row(catalog)
        schema_rows: list[dict[str, Any]] = []
        table_rows: list[dict[str, Any]] = []
        column_rows: list[dict[str, Any]] = []

        for schema in schemas:
            schema_key = self._metadata_path_id(schema, "schemaId", "schemaName", "name")
            schema_rows.append(self._schema_row(catalog, schema))

            tables = await self._galaxy_get_paginated(
                f"/public/api/v1/catalog/{catalog_key}/schema/{schema_key}/table"
            )

            for table in tables:
                table_key = self._metadata_path_id(table, "tableId", "tableName", "name")
                table_row = self._table_row(catalog, schema, table)
                table_rows.append(table_row)

                columns = await self._galaxy_get_paginated(
                    f"/public/api/v1/catalog/{catalog_key}/schema/{schema_key}/table/{table_key}/column"
                )

                for ordinal, column in enumerate(columns, start=1):
                    column_rows.append(
                        self._column_row(catalog, schema, table, column, ordinal)
                    )

        return {
            "raw": {
                "catalog": catalog,
                "schemas": schemas,
            },
            "postgres": {
                "catalogs": [catalog_row],
                "schemas": schema_rows,
                "tables": table_rows,
                "columns": column_rows,
            },
        }

    def postgres_metadata_ddl(self, schema_name: str = "public") -> str:
        """DDL for storing the normalized metadata rows returned above."""
        schema = self._quote_ident(schema_name)
        return f"""
            CREATE TABLE IF NOT EXISTS {schema}.starburst_catalogs (
                catalog_id text PRIMARY KEY,
                catalog_name text,
                description text,
                metadata jsonb NOT NULL,
                extracted_at timestamptz NOT NULL DEFAULT now()
            );
            
            CREATE TABLE IF NOT EXISTS {schema}.starburst_schemas (
                catalog_id text NOT NULL,
                schema_id text NOT NULL,
                schema_name text,
                description text,
                metadata jsonb NOT NULL,
                extracted_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (catalog_id, schema_id)
            );
            
            CREATE TABLE IF NOT EXISTS {schema}.starburst_tables (
                catalog_id text NOT NULL,
                schema_id text NOT NULL,
                table_id text NOT NULL,
                table_name text,
                table_type text,
                description text,
                metadata jsonb NOT NULL,
                extracted_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (catalog_id, schema_id, table_id)
            );
            
            CREATE TABLE IF NOT EXISTS {schema}.starburst_columns (
                catalog_id text NOT NULL,
                schema_id text NOT NULL,
                table_id text NOT NULL,
                column_id text NOT NULL,
                column_name text,
                ordinal_position integer,
                data_type text,
                nullable boolean,
                description text,
                metadata jsonb NOT NULL,
                extracted_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (catalog_id, schema_id, table_id, column_id)
            );
            """.strip()

    # ---------------------------------------------------------------------
    # EngineAdapter: Trino query protocol
    # ---------------------------------------------------------------------

    async def explain(self, sql: str) -> ExplainResult:
        explain_sql = f"EXPLAIN (TYPE DISTRIBUTED) {sql}"
        handle = await self.execute_async(explain_sql)

        while True:
            status = await self.get_status(handle)
            if status.done:
                if status.error:
                    return ExplainResult(
                        ok=False,
                        summary={"engine": self.name, "error": status.error},
                    )
                break

        rows = handle.raw.get("rows", [])
        plan_text = "\n".join(str(col) for row in rows for col in row)
        return ExplainResult(
            ok=True,
            summary={
                "engine": self.name,
                "plan": plan_text[:5000],
                "stats": (handle.raw.get("lastPayload") or {}).get("stats", {}),
                "events": handle.raw.get("events", []),
            },
        )

    async def execute_async(self, sql: str) -> EngineHandle:
        logger.debug(f"Executing {sql}")
        payload = await self._submit_statement(sql)
        handle_id = str(uuid.uuid4())
        handle_raw = {
            "query": sql,
            "nextUri": payload.get("nextUri"),
            "queryId": payload.get("id"),
            "lastPayload": payload,
            "rows": payload.get("data", []),
            "schema": self._schema_from_payload(payload),
            "events": [],
        }
        handle = EngineHandle(handle_id=handle_id, raw=handle_raw)
        self._record_feedback_event(handle, payload)
        return handle

    async def get_status(self, handle: EngineHandle) -> EngineStatus:
        if handle.handle_id in self._local_cancelled:
            return EngineStatus(state="CANCELLED", done=True, progress_percentage=100)

        payload = handle.raw.get("lastPayload") or {}
        next_uri = handle.raw.get("nextUri")
        logger.debug(f"next_uri: {next_uri}")

        if next_uri:
            logger.debug(f"Fetching status for {next_uri}")
            payload = await self._get_next(next_uri)
            logger.debug(f"Payload after _get_next: {payload}")
            handle.raw["lastPayload"] = payload
            handle.raw["nextUri"] = payload.get("nextUri")

            if payload.get("data"):
                handle.raw.setdefault("rows", []).extend(payload["data"])
            if payload.get("columns"):
                handle.raw["schema"] = self._schema_from_payload(payload)

            self._record_feedback_event(handle, payload)

        stats = payload.get("stats", {}) or {}
        error = payload.get("error")
        state = "FAILED" if error else (stats.get("state") or "UNKNOWN").upper()
        if state == 'FAILED':
            logger.error(f"Query failed with error: {error}")

        # Per Trino protocol, completion is determined by absence of nextUri, not a human status string.
        done = not handle.raw.get("nextUri")
        if done and not error and state in {"UNKNOWN", "RUNNING", "QUEUED", "PLANNING", "STARTING", "FINISHING"}:
            state = "FINISHED"

        progress = self._coerce_progress(stats.get("progressPercentage"))
        if state == "FINISHED":
            progress = 100

        return EngineStatus(
            state=state,
            done=done,
            progress_percentage=progress,
            stats=stats,
            error=error,
        )

    async def fetch_results(self, handle: EngineHandle, page_token: str | None = None) -> ResultPage:
        # Drain the query before returning results. If your UI wants true streaming, remove this loop
        # and call get_status() externally while rendering handle.raw["events"].
        while handle.raw.get("nextUri"):
            status = await self.get_status(handle)
            if status.error or status.done:
                break

        schema = handle.raw.get("schema", [])
        rows = handle.raw.get("rows", [])
        page_size = int(getattr(self._settings, "starburst_result_page_size", 1000))
        offset = int(page_token or 0)
        page_rows = rows[offset : offset + page_size]
        next_offset = offset + len(page_rows)
        next_page_token = str(next_offset) if next_offset < len(rows) else None

        return ResultPage(schema=schema, rows=page_rows, next_page_token=next_page_token)

    async def cancel(self, handle: EngineHandle) -> bool:
        self._local_cancelled.add(handle.handle_id)
        next_uri = handle.raw.get("nextUri")
        if not next_uri:
            return True

        connector = self._connector()
        async with aiohttp.ClientSession(
            auth=self._trino_basic_auth(),
            timeout=self._timeout,
            connector=connector,
        ) as session:
            async with session.delete(next_uri, headers=self._trino_headers()) as resp:
                await resp.read()
                return resp.status in {200, 202, 204, 404}

    # ---------------------------------------------------------------------
    # Galaxy API auth and request helpers
    # ---------------------------------------------------------------------

    async def _get_galaxy_access_token(self) -> str:
        if self._galaxy_token and time.time() < self._galaxy_token.expires_at - 60:
            return self._galaxy_token.access_token

        client_id = getattr(self._settings, "starburst_client_id", None)
        client_secret = getattr(self._settings, "starburst_client_secret", None)
        if not client_id or not client_secret:
            raise ValueError(
                "Galaxy metadata API requires STARBURST_CLIENT_ID and STARBURST_CLIENT_SECRET. "
                "These credentials are not used for Trino query execution."
            )

        token_url = f"{self._api_base_url}/oauth/v2/token"
        connector = self._connector()
        async with aiohttp.ClientSession(timeout=self._timeout, connector=connector) as session:
            async with session.post(
                token_url,
                auth=aiohttp.BasicAuth(client_id, client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "client_credentials"},
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(
                        f"Starburst Galaxy token request failed status={resp.status}, url={token_url}, body={text}"
                    )
                payload = json.loads(text)

        access_token = payload.get("access_token")
        if not access_token:
            raise RuntimeError("Starburst Galaxy token response did not include access_token")

        expires_in = int(payload.get("expires_in", 600))
        self._galaxy_token = GalaxyToken(
            access_token=access_token,
            expires_at=time.time() + expires_in,
        )
        return access_token

    async def _galaxy_headers(self) -> dict[str, str]:
        token = await self._get_galaxy_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    async def _galaxy_get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._api_base_url}{path}"
        connector = self._connector()
        async with aiohttp.ClientSession(timeout=self._timeout, connector=connector) as session:
            async with session.get(url, headers=await self._galaxy_headers(), params=params or {}) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.error(f"Starburst Galaxy API GET failed status={resp.status}, url={url}, body={text}")
                    raise RuntimeError(f"Starburst Galaxy API GET failed status={resp.status}, url={url}, body={text}")
                return json.loads(text) if text else {}

    async def _galaxy_get_paginated(self, path: str, page_size: int = 100) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        async for item in self._galaxy_iter_paginated(path, page_size=page_size):
            results.append(item)
        return results

    async def _galaxy_iter_paginated(self, path: str, page_size: int = 100) -> AsyncIterator[dict[str, Any]]:
        page_token: str | None = None

        while True:
            params: dict[str, Any] = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            payload = await self._galaxy_get_json(path, params=params)
            for item in payload.get("result") or []:
                if isinstance(item, dict):
                    yield item

            page_token = payload.get("nextPageToken")
            if not page_token:
                break

    async def _iter_schema_children(
        self,
        catalog_key: str,
        catalog: dict[str, Any],
        schema: dict[str, Any],
        *,
        include_columns: bool,
    ) -> AsyncIterator[BackendMetadataRecord]:
        logger.debug(f"Iterating schema children for catalog key {catalog_key}, schema={schema}, catalog={catalog}")
        schema_key = self._metadata_path_id(schema, "schemaId", "schemaName", "name")
        async for table in self._galaxy_iter_paginated(
            f"/public/api/v1/catalog/{catalog_key}/schema/{schema_key}/table"
        ):
            logger.debug(f"Returning table record: {table}")
            yield self._table_record(catalog, schema, table)
            if include_columns:
                async for column_record in self._iter_table_columns(
                    catalog_key,
                    schema_key,
                    catalog,
                    schema,
                    table,
                ):
                    logger.debug(f"Returning column record: {column_record}")
                    yield column_record

    async def _iter_table_columns(
        self,
        catalog_key: str,
        schema_key: str,
        catalog: dict[str, Any],
        schema: dict[str, Any],
        table: dict[str, Any],
    ) -> AsyncIterator[BackendMetadataRecord]:
        table_key = self._metadata_path_id(table, "tableId", "tableName", "name")
        ordinal = 0
        async for column in self._galaxy_iter_paginated(
            f"/public/api/v1/catalog/{catalog_key}/schema/{schema_key}/table/{table_key}/column"
        ):
            ordinal += 1
            yield self._column_record(catalog, schema, table, column, ordinal)

    async def _find_schema(self, catalog_key: str, schema_ref: str) -> dict[str, Any]:
        async for schema in self._galaxy_iter_paginated(
            f"/public/api/v1/catalog/{catalog_key}/schema"
        ):
            if self._matches_metadata_ref(
                schema_ref,
                schema.get("schemaId"),
                schema.get("schemaName"),
                schema.get("name"),
            ):
                return schema
        raise ValueError(f"Starburst schema not found: {schema_ref}")

    async def _find_table(self, catalog_key: str, schema_key: str, table_ref: str) -> dict[str, Any]:
        async for table in self._galaxy_iter_paginated(
            f"/public/api/v1/catalog/{catalog_key}/schema/{schema_key}/table"
        ):
            if self._matches_metadata_ref(
                table_ref,
                table.get("tableId"),
                table.get("tableName"),
                table.get("name"),
            ):
                return table
        raise ValueError(f"Starburst table not found: {table_ref}")

    # ---------------------------------------------------------------------
    # Trino REST protocol helpers
    # ---------------------------------------------------------------------

    def _trino_basic_auth(self) -> aiohttp.BasicAuth:
        user = getattr(self._settings, "starburst_user", None)
        password = getattr(self._settings, "starburst_password", None)
        if not user or not password:
            raise ValueError("Trino query execution requires STARBURST_USER and STARBURST_PASSWORD")
        return aiohttp.BasicAuth(login=user, password=password)

    def _query_history_basic_auth(self) -> aiohttp.BasicAuth:
        logger.debug(f"In query history basic auth...")
        user = (
            getattr(self._settings, "starburst_query_history_user", None)
            or getattr(self._settings, "starburst_user", None)
        )
        password = (
            getattr(self._settings, "starburst_query_history_password", None)
            or getattr(self._settings, "starburst_password", None)
        )
        logger.debug(f"In query history basic auth, user={user}, password=xxx")
        if not user or not password:
            raise ValueError(
                "Query history sync requires STARBURST_QUERY_HISTORY_USER/PASSWORD "
                "or STARBURST_USER/PASSWORD"
            )
        return aiohttp.BasicAuth(login=user, password=password)

    def _trino_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "text/plain; charset=utf-8",
            "X-Trino-User": self._settings.starburst_user,
            "X-Trino-Source": getattr(self._settings, "starburst_source", "cognetics-ai"),
        }

        if self._trino_catalog:
            headers["X-Trino-Catalog"] = self._trino_catalog
        if self._trino_schema:
            headers["X-Trino-Schema"] = self._trino_schema
        # if self._trino_role:
        #     headers["X-Trino-Role"] = f"system=ROLE{{{self._trino_role}}}"
        if self._trino_role_header:
            headers["X-Trino-Role"] = self._trino_role_header
        if self._trino_session_props:
            headers["X-Trino-Session"] = ",".join(
                f"{key}={value}" for key, value in sorted(self._trino_session_props.items())
            )
        if self._trino_transaction_id:
            headers["X-Trino-Transaction-Id"] = self._trino_transaction_id
        if self._trino_prepared_statements:
            headers["X-Trino-Prepared-Statement"] = ",".join(
                f"{key}={value}" for key, value in sorted(self._trino_prepared_statements.items())
            )

        # Never add the Galaxy API bearer token here. aiohttp BasicAuth adds Authorization: Basic ...
        return headers

    def _query_history_headers(self) -> dict[str, str]:
        user = (
            getattr(self._settings, "starburst_query_history_user", None)
            or getattr(self._settings, "starburst_user", None)
        )
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "text/plain; charset=utf-8",
            "X-Trino-User": user,
            "X-Trino-Source": getattr(
                self._settings,
                "starburst_query_history_source",
                "cognetics-ai-query-history",
            ),
            "X-Trino-Catalog": getattr(self._settings, "starburst_query_history_catalog", "galaxy_telemetry"),
            "X-Trino-Schema": getattr(self._settings, "starburst_query_history_schema", "public"),
        }
        if self._query_history_role_header:
            headers["X-Trino-Role"] = self._query_history_role_header
        return headers

    async def _submit_statement(self, sql: str) -> dict[str, Any]:
        logger.debug(f"Submitting statement {sql}")
        url = f"{self._trino_base_url}/v1/statement"
        return await self._trino_request_json("POST", url, data=sql.encode("utf-8"))

    async def _get_next(self, next_uri: str) -> dict[str, Any]:
        return await self._trino_request_json("GET", next_uri)

    async def _submit_query_history_statement(self, sql: str) -> dict[str, Any]:
        logger.debug("Submitting query-history statement to %s", self._query_history_trino_base_url)
        url = f"{self._query_history_trino_base_url}/v1/statement"
        return await self._trino_request_json(
            "POST",
            url,
            data=sql.encode("utf-8"),
            auth=self._query_history_basic_auth(),
            headers=self._query_history_headers(),
            apply_response_headers=False,
            fail_on_trino_error=True,
        )

    async def _get_query_history_next(self, next_uri: str) -> dict[str, Any]:
        return await self._trino_request_json(
            "GET",
            next_uri,
            auth=self._query_history_basic_auth(),
            headers=self._query_history_headers(),
            apply_response_headers=False,
            fail_on_trino_error=True,
        )

    async def _fetch_query_history_page(self, sql: str) -> list[dict[str, Any]]:
        payload = await self._submit_query_history_statement(sql)
        schema = self._schema_from_payload(payload)
        rows = [self._result_row_to_dict(schema, row) for row in payload.get("data") or []]
        next_uri = payload.get("nextUri")

        while next_uri:
            payload = await self._get_query_history_next(next_uri)
            if payload.get("columns"):
                schema = self._schema_from_payload(payload)
            rows.extend(self._result_row_to_dict(schema, row) for row in payload.get("data") or [])
            next_uri = payload.get("nextUri")
        return rows

    async def _trino_request_json(
        self,
        method: str,
        url: str,
        data: bytes | None = None,
        *,
        auth: aiohttp.BasicAuth | None = None,
        headers: dict[str, str] | None = None,
        apply_response_headers: bool = True,
        fail_on_trino_error: bool = False,
    ) -> dict[str, Any]:
        retry_count = 0
        while True:
            connector = self._connector()
            async with aiohttp.ClientSession(
                auth=auth or self._trino_basic_auth(),
                timeout=self._timeout,
                connector=connector,
            ) as session:
                async with session.request(method, url, data=data, headers=headers or self._trino_headers()) as resp:
                    text = await resp.text()
                    logger.debug(
                        "Trino response status=%s method=%s url=%s bytes=%s",
                        resp.status,
                        method,
                        url,
                        len(text),
                    )
                    if apply_response_headers:
                        self._apply_trino_response_headers(resp.headers)

                    if resp.status in {502, 503, 504} and retry_count < 3:
                        logger.error(f"Trino request failed with status {resp.status} method={method}, url={url}, body={text}")
                        retry_count += 1
                        await asyncio.sleep(0.05 * retry_count)
                        continue

                    if resp.status == 429 and retry_count < 3:
                        logger.error(f"Trino request failed with status {resp.status} method={method}, url={url}, body={text}")
                        retry_count += 1
                        retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        logger.error(f"Trino request failed with status {resp.status} method={method}, url={url}, body={text}")
                        raise RuntimeError(
                            f"Trino request failed status={resp.status}, method={method}, url={url}, body={text}"
                        )

                    try:
                        payload = json.loads(text) if text else {}
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Trino request returned non-JSON response status={resp.status}, method={method}, url={url}, body={text}"
                        ) from exc

                    if fail_on_trino_error:
                        self._raise_for_trino_payload_error(payload, method=method, url=url)

                    return payload

    def _apply_trino_response_headers(self, headers: aiohttp.typedefs.LooseHeaders) -> None:
        # aiohttp returns CIMultiDictProxy at runtime. The helper keeps this easy to test.
        get = headers.get  # type: ignore[attr-defined]
        getall = getattr(headers, "getall", None)

        def values(name: str) -> list[str]:
            if callable(getall):
                return list(getall(name, []))
            value = get(name)  # type: ignore[misc]
            return [value] if value else []

        if catalog := get("X-Trino-Set-Catalog"):  # type: ignore[misc]
            self._trino_catalog = catalog
        if schema := get("X-Trino-Set-Schema"):  # type: ignore[misc]
            self._trino_schema = schema
        # if role := get("X-Trino-Set-Role"):  # type: ignore[misc]
        #     self._trino_role = role

        if role := get("X-Trino-Set-Role"):
            self._trino_role_header = role

        for session_value in values("X-Trino-Set-Session"):
            key, value = self._split_header_assignment(session_value)
            if key:
                self._trino_session_props[key] = value

        for key in values("X-Trino-Clear-Session"):
            self._trino_session_props.pop(key, None)

        if txn := get("X-Trino-Started-Transaction-Id"):  # type: ignore[misc]
            self._trino_transaction_id = txn
        if get("X-Trino-Clear-Transaction-Id"):  # type: ignore[misc]
            self._trino_transaction_id = None

        for prepared in values("X-Trino-Added-Prepare"):
            key, value = self._split_header_assignment(prepared)
            if key:
                self._trino_prepared_statements[key] = value

        for key in values("X-Trino-Deallocated-Prepare"):
            self._trino_prepared_statements.pop(key, None)

    def _record_feedback_event(self, handle: EngineHandle, payload: dict[str, Any]) -> None:
        stats = payload.get("stats") or {}
        event = {
            "ts": time.time(),
            "queryId": payload.get("id") or handle.raw.get("queryId"),
            "state": stats.get("state"),
            "progressPercentage": stats.get("progressPercentage"),
            "processedRows": stats.get("processedRows"),
            "processedBytes": stats.get("processedBytes"),
            "queued": stats.get("queued"),
            "scheduled": stats.get("scheduled"),
            "nodes": stats.get("nodes"),
            "totalSplits": stats.get("totalSplits"),
            "runningSplits": stats.get("runningSplits"),
            "completedSplits": stats.get("completedSplits"),
            "cpuTimeMillis": stats.get("cpuTimeMillis"),
            "wallTimeMillis": stats.get("wallTimeMillis"),
            "queuedTimeMillis": stats.get("queuedTimeMillis"),
            "elapsedTimeMillis": stats.get("elapsedTimeMillis"),
            "warnings": payload.get("warnings") or [],
            "infoUri": payload.get("infoUri"),
            "partialCancelUri": payload.get("partialCancelUri"),
            "updateType": payload.get("updateType"),
            "updateCount": payload.get("updateCount"),
            "error": payload.get("error"),
        }
        handle.raw.setdefault("events", []).append(event)

    # ---------------------------------------------------------------------
    # Normalization helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _catalog_row(catalog: dict[str, Any]) -> dict[str, Any]:
        return {
            "catalog_id": catalog.get("catalogId"),
            "catalog_name": StarburstTrinoAdapter._plain_ref(
                catalog.get("catalogName") or catalog.get("name")
            ),
            "description": StarburstTrinoAdapter._description(catalog),
            "metadata": catalog,
        }

    @staticmethod
    def _schema_row(catalog: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "catalog_id": catalog.get("catalogId"),
            "schema_id": schema.get("schemaId"),
            "schema_name": StarburstTrinoAdapter._plain_ref(
                schema.get("schemaName") or schema.get("name")
            ),
            "description": StarburstTrinoAdapter._description(schema),
            "metadata": schema,
        }

    @staticmethod
    def _table_row(catalog: dict[str, Any], schema: dict[str, Any], table: dict[str, Any]) -> dict[str, Any]:
        return {
            "catalog_id": catalog.get("catalogId"),
            "schema_id": schema.get("schemaId"),
            "table_id": table.get("tableId"),
            "table_name": StarburstTrinoAdapter._plain_ref(
                table.get("tableName") or table.get("name") or table.get("tableId")
            ),
            "table_type": table.get("tableType"),
            "description": StarburstTrinoAdapter._description(table),
            "metadata": table,
        }

    @staticmethod
    def _column_row(
        catalog: dict[str, Any],
        schema: dict[str, Any],
        table: dict[str, Any],
        column: dict[str, Any],
        ordinal: int,
    ) -> dict[str, Any]:
        return {
            "catalog_id": catalog.get("catalogId"),
            "schema_id": schema.get("schemaId"),
            "table_id": table.get("tableId"),
            "column_id": column.get("columnId"),
            "column_name": StarburstTrinoAdapter._plain_ref(
                column.get("columnName") or column.get("name") or column.get("columnId")
            ),
            "ordinal_position": column.get("ordinalPosition") or ordinal,
            "data_type": column.get("dataType") or column.get("type"),
            "nullable": column.get("nullable"),
            "description": StarburstTrinoAdapter._description(column),
            "metadata": column,
        }

    def _catalog_record(self, catalog: dict[str, Any]) -> BackendMetadataRecord:
        row = self._catalog_row(catalog)
        return BackendMetadataRecord(
            entity_type="catalog",
            engine=self.name,
            catalog_id=self._text(row.get("catalog_id")),
            catalog_name=self._text(row.get("catalog_name")),
            description=self._text(row.get("description")),
            raw=row.get("metadata") or {},
        )

    def _schema_record(
        self,
        catalog: dict[str, Any],
        schema: dict[str, Any],
    ) -> BackendMetadataRecord:
        row = self._schema_row(catalog, schema)
        catalog_name = self._plain_ref(catalog.get("catalogName") or catalog.get("name"))
        return BackendMetadataRecord(
            entity_type="schema",
            engine=self.name,
            catalog_id=self._text(row.get("catalog_id")),
            catalog_name=self._text(catalog_name),
            schema_id=self._text(row.get("schema_id")),
            schema_name=self._text(row.get("schema_name")),
            description=self._text(row.get("description")),
            raw=row.get("metadata") or {},
        )

    def _table_record(
        self,
        catalog: dict[str, Any],
        schema: dict[str, Any],
        table: dict[str, Any],
    ) -> BackendMetadataRecord:
        row = self._table_row(catalog, schema, table)
        catalog_name = self._plain_ref(catalog.get("catalogName") or catalog.get("name"))
        schema_name = self._plain_ref(
            schema.get("schemaName") or schema.get("name") or schema.get("schemaId")
        )
        return BackendMetadataRecord(
            entity_type="table",
            engine=self.name,
            catalog_id=self._text(row.get("catalog_id")),
            catalog_name=self._text(catalog_name),
            schema_id=self._text(row.get("schema_id")),
            schema_name=self._text(schema_name),
            table_id=self._text(row.get("table_id")),
            table_name=self._text(row.get("table_name")),
            object_type=self._text(row.get("table_type")),
            description=self._text(row.get("description")),
            raw=row.get("metadata") or {},
        )

    def _column_record(
        self,
        catalog: dict[str, Any],
        schema: dict[str, Any],
        table: dict[str, Any],
        column: dict[str, Any],
        ordinal: int,
    ) -> BackendMetadataRecord:
        row = self._column_row(catalog, schema, table, column, ordinal)
        table_row = self._table_row(catalog, schema, table)
        catalog_name = self._plain_ref(catalog.get("catalogName") or catalog.get("name"))
        schema_name = self._plain_ref(
            schema.get("schemaName") or schema.get("name") or schema.get("schemaId")
        )
        return BackendMetadataRecord(
            entity_type="column",
            engine=self.name,
            catalog_id=self._text(row.get("catalog_id")),
            catalog_name=self._text(catalog_name),
            schema_id=self._text(row.get("schema_id")),
            schema_name=self._text(schema_name),
            table_id=self._text(row.get("table_id")),
            table_name=self._text(table_row.get("table_name")),
            column_id=self._text(row.get("column_id")),
            column_name=self._text(row.get("column_name")),
            ordinal_position=self._int_or_none(row.get("ordinal_position")),
            data_type=self._text(row.get("data_type")),
            nullable=self._bool_or_none(row.get("nullable")),
            description=self._text(row.get("description")),
            raw=row.get("metadata") or {},
        )

    def _query_history_sql(
        self,
        *,
        start_time: datetime | str | None,
        end_time: datetime | str | None,
        catalog: str | None,
        schema: str | None,
        table: str | None,
        limit: int,
        offset: int,
    ) -> str:
        history_catalog = getattr(self._settings, "starburst_query_history_catalog", "galaxy_telemetry")
        history_schema = getattr(self._settings, "starburst_query_history_schema", "public")
        history_table = getattr(self._settings, "starburst_query_history_table", "query_history")
        source_name = (
            f"{self._quote_ident(history_catalog)}."
            f"{self._quote_ident(history_schema)}."
            f"{self._quote_ident(history_table)}"
        )

        filter_on_table_refs = bool(catalog or schema or table)
        from_clause = f"FROM (SELECT * FROM {source_name}) qh"
        if filter_on_table_refs:
            from_clause += '\n            CROSS JOIN UNNEST(qh.tables) AS t (catalog, "schema", "table")'

        where = ["qh.query IS NOT NULL"]
        if start_time:
            where.append(f"qh.create_time >= {self._sql_timestamp(start_time)}")
        if end_time:
            where.append(f"qh.create_time < {self._sql_timestamp(end_time)}")
        if catalog:
            literal = self._sql_literal(catalog)
            where.append(f"(qh.session_catalog = {literal} OR t.catalog = {literal})")
        if schema:
            literal = self._sql_literal(schema)
            where.append(f"(qh.session_schema = {literal} OR t.\"schema\" = {literal})")
        if table:
            literal = self._sql_literal(table)
            where.append(f"t.\"table\" = {literal}")

        sql = f"""
            SELECT
                qh.cluster_name,
                qh.email,
                qh.role_name,
                CAST(qh.create_time AS varchar) AS create_time,
                CAST(qh.execution_start_time AS varchar) AS execution_start_time,
                CAST(qh.end_time AS varchar) AS end_time,
                qh.session_catalog,
                qh.session_schema,
                json_format(CAST(qh.session_properties AS JSON)) AS session_properties_json,
                qh.remote_client_address,
                qh.user_agent,
                qh.query_type,
                qh.query_id,
                qh.query,
                qh.query_plan,
                qh.query_state,
                qh.update_type,
                json_format(CAST(qh.tables AS JSON)) AS tables_json,
                qh.internal_network_bytes,
                qh.internal_network_rows,
                qh.output_bytes,
                qh.output_rows,
                qh.peak_task_total_memory_bytes,
                qh.peak_task_user_memory_bytes,
                qh.peak_user_memory_bytes,
                qh.physical_input_bytes,
                qh.physical_input_rows,
                qh.read_bytes,
                qh.read_rows,
                qh.written_bytes,
                qh.written_rows,
                qh.original_query_id,
                qh.client_info,
                qh.source,
                qh.index_and_cache_usage_overall,
                qh.index_and_cache_usage_filtering,
                qh.index_and_cache_usage_projection,
                qh.planning_time_secs,
                qh.error_code_name,
                qh.error_code_category,
                qh.error_exception_message,
                qh.account_name,
                qh.date AS query_date,
                qh.hour AS query_hour
            {from_clause}
            WHERE {" AND ".join(where)}
            ORDER BY qh.create_time DESC, qh.query_id DESC
            OFFSET {int(offset)} ROWS
            LIMIT {int(limit)}
        """
        logger.debug(f"SQL for Query History Table: {sql}")
        return sql

    def _query_history_record(self, row: dict[str, Any]) -> BackendQueryHistoryRecord | None:
        query_id = self._text(row.get("query_id"))
        raw_sql = self._text(row.get("query"))
        if not query_id or not raw_sql:
            return None

        tables = self._parse_query_history_tables(row.get("tables_json"))
        raw = dict(row)
        metrics = {
            "execution_start_time": row.get("execution_start_time"),
            "remote_client_address": row.get("remote_client_address"),
            "user_agent": row.get("user_agent"),
            "update_type": row.get("update_type"),
            "read_bytes": row.get("read_bytes"),
            "read_rows": row.get("read_rows"),
            "written_bytes": row.get("written_bytes"),
            "written_rows": row.get("written_rows"),
            "output_bytes": row.get("output_bytes"),
            "output_rows": row.get("output_rows"),
            "physical_input_bytes": row.get("physical_input_bytes"),
            "physical_input_rows": row.get("physical_input_rows"),
            "internal_network_bytes": row.get("internal_network_bytes"),
            "internal_network_rows": row.get("internal_network_rows"),
            "peak_task_total_memory_bytes": row.get("peak_task_total_memory_bytes"),
            "peak_task_user_memory_bytes": row.get("peak_task_user_memory_bytes"),
            "peak_user_memory_bytes": row.get("peak_user_memory_bytes"),
            "original_query_id": row.get("original_query_id"),
            "client_info": row.get("client_info"),
            "index_and_cache_usage_overall": row.get("index_and_cache_usage_overall"),
            "index_and_cache_usage_filtering": row.get("index_and_cache_usage_filtering"),
            "index_and_cache_usage_projection": row.get("index_and_cache_usage_projection"),
            "planning_time_secs": row.get("planning_time_secs"),
            "error_code_category": row.get("error_code_category"),
            "error_code_name": row.get("error_code_name"),
            "error_exception_message": row.get("error_exception_message"),
            "account_name": row.get("account_name"),
            "query_date": row.get("query_date"),
            "query_hour": row.get("query_hour"),
            "session_properties": self._json_or_raw(row.get("session_properties_json")),
        }
        return BackendQueryHistoryRecord(
            engine=self.name,
            query_id=query_id,
            raw_sql=raw_sql,
            catalog_name=self._text(row.get("session_catalog")),
            schema_name=self._text(row.get("session_schema")),
            query_state=self._text(row.get("query_state")),
            query_type=self._text(row.get("query_type")),
            user_email=self._text(row.get("email")),
            role_name=self._text(row.get("role_name")),
            cluster_name=self._text(row.get("cluster_name")),
            source=self._text(row.get("source")),
            created_at=self._text(row.get("create_time")),
            ended_at=self._text(row.get("end_time")),
            tables=tables,
            metrics=metrics,
            raw=raw,
        )

    @staticmethod
    def _description(obj: dict[str, Any]) -> str | None:
        metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
        return (
            obj.get("description")
            or obj.get("comment")
            or obj.get("remarks")
            or metadata.get("description")
            or metadata.get("comment")
        )

    @staticmethod
    def _schema_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"name": col.get("name"), "type": col.get("type"), "typeSignature": col.get("typeSignature")}
            for col in payload.get("columns", [])
        ]

    # ---------------------------------------------------------------------
    # Small utilities
    # ---------------------------------------------------------------------

    def _connector(self) -> aiohttp.TCPConnector:
        return aiohttp.TCPConnector(ssl=self._ssl_context)

    @staticmethod
    def _build_ssl_context(verify_ssl: bool) -> ssl.SSLContext:
        context = ssl.create_default_context()
        if not verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context

    @staticmethod
    def _path_id(value: str) -> str:
        # Supports raw IDs and lookup forms like name=tpch.
        return quote(value, safe="")

    @staticmethod
    def _name_lookup_path_id(value: Any) -> str:
        plain = StarburstTrinoAdapter._plain_ref(value)
        return quote(f"name={plain}", safe="")

    @staticmethod
    def _metadata_path_id(record: dict[str, Any], id_key: str, *name_keys: str) -> str:
        record_id = str(record.get(id_key) or "").strip()
        if record_id:
            return StarburstTrinoAdapter._path_id(record_id)
        for key in name_keys:
            name = StarburstTrinoAdapter._plain_ref(record.get(key))
            if name:
                return StarburstTrinoAdapter._name_lookup_path_id(name)
        return StarburstTrinoAdapter._path_id("")

    @staticmethod
    def _plain_ref(value: Any) -> str:
        text = str(value or "").strip()
        if text.lower().startswith("name="):
            return text.split("=", 1)[1].strip()
        return text

    @staticmethod
    def _matches_metadata_ref(ref: str, *candidates: Any) -> bool:
        ref_key = StarburstTrinoAdapter._plain_ref(ref).lower()
        return any(
            StarburstTrinoAdapter._plain_ref(candidate).lower() == ref_key
            for candidate in candidates
        )

    @staticmethod
    def _result_row_to_dict(schema: list[dict[str, Any]], row: list[Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for idx, column in enumerate(schema):
            if idx >= len(row):
                break
            name = str(column.get("name") or f"col_{idx}")
            result[name] = row[idx]
        return result

    @staticmethod
    def _raise_for_trino_payload_error(payload: dict[str, Any], *, method: str, url: str) -> None:
        stats = payload.get("stats") or {}
        state = str(stats.get("state") or "").upper()
        error = payload.get("error")
        if not error and state != "FAILED":
            return

        query_id = payload.get("id")
        if isinstance(error, dict):
            message = error.get("message") or "Trino query failed"
            error_name = error.get("errorName")
            error_type = error.get("errorType")
            location = error.get("errorLocation") or {}
            line = location.get("lineNumber")
            column = location.get("columnNumber")
            location_text = f" line={line} column={column}" if line and column else ""
            logger.error(f"Failed on query execution: {error}")
            raise RuntimeError(
                "Trino query failed "
                f"query_id={query_id} state={state or 'UNKNOWN'} error_name={error_name} "
                f"error_type={error_type}{location_text} method={method} url={url}: {message}"
            )

        raise RuntimeError(
            f"Trino query failed query_id={query_id} state={state or 'UNKNOWN'} method={method} url={url}"
        )

    @classmethod
    def _parse_query_history_tables(cls, value: Any) -> list[str]:
        parsed = cls._json_or_raw(value)
        if not parsed:
            return []

        if not isinstance(parsed, list):
            parsed = [parsed]

        tables: list[str] = []
        seen: set[str] = set()
        for item in parsed:
            catalog = schema = table = ""
            if isinstance(item, dict):
                catalog = cls._text(item.get("catalog")) or ""
                schema = cls._text(item.get("schema")) or ""
                table = cls._text(item.get("table")) or ""
            elif isinstance(item, (list, tuple)):
                values = [cls._text(part) or "" for part in item]
                catalog = values[0] if len(values) > 0 else ""
                schema = values[1] if len(values) > 1 else ""
                table = values[2] if len(values) > 2 else ""
            else:
                table = cls._text(item) or ""

            parts = [part for part in (catalog, schema, table) if part]
            if not parts:
                continue
            qualified = ".".join(parts)
            key = qualified.lower()
            if key in seen:
                continue
            seen.add(key)
            tables.append(qualified)
        return tables

    @staticmethod
    def _json_or_raw(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    @staticmethod
    def _sql_literal(value: Any) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    @classmethod
    def _sql_timestamp(cls, value: datetime | str) -> str:
        if isinstance(value, datetime):
            dt = value
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return f"TIMESTAMP {cls._sql_literal(dt.isoformat(sep=' ', timespec='seconds'))}"

        text = str(value or "").strip()
        if not text:
            return "TIMESTAMP '1970-01-01 00:00:00'"
        text = text.replace("T", " ").replace("Z", "")
        if "+" in text:
            text = text.split("+", 1)[0].strip()
        if len(text) >= 6 and text[-6] in {"+", "-"} and text[-3] == ":":
            text = text[:-6].strip()
        return f"TIMESTAMP {cls._sql_literal(text)}"

    @staticmethod
    def _split_header_assignment(value: str) -> tuple[str, str]:
        if "=" not in value:
            return value.strip(), ""
        key, assigned = value.split("=", 1)
        return key.strip(), assigned.strip()

    @staticmethod
    def _parse_retry_after(value: str | None) -> float:
        if not value:
            return 1.0
        try:
            return max(0.0, float(value))
        except ValueError:
            return 1.0

    @staticmethod
    def _coerce_progress(value: Any) -> int:
        try:
            return max(0, min(100, int(float(value or 0))))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _bool_or_none(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "t", "1", "yes", "y"}:
            return True
        if text in {"false", "f", "0", "no", "n"}:
            return False
        return None

    @staticmethod
    def _quote_ident(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'
