"""Starburst Galaxy / Trino connector — production implementation.

Two explicitly separate auth paths
-----------------------------------
1. **Starburst Galaxy REST API** (metadata discovery)
   - Base URL: ``SourceConfig.host`` (e.g. ``https://acme.galaxy.starburst.io``)
   - Auth: OAuth 2.0 client-credentials, Bearer token.
   - Used for: listing catalogs, schemas, tables, columns.

2. **Trino REST protocol** (query execution / verification)
   - Base URL: ``SourceConfig.trino_host`` (e.g. ``https://acme.trino.galaxy.starburst.io``)
   - Auth: HTTP Basic (user/password).  Never send the Galaxy Bearer token here.
   - Used for: execute(), explain_plan(), health_check().

The connector is fully async (``aiohttp``).  Token caching avoids redundant
OAuth round-trips; tokens are refreshed 60 s before expiry.

Catalog hierarchy
-----------------
Starburst Galaxy exposes a 3-level hierarchy:

    catalog  (Galaxy catalog, e.g. "tpch" or "iceberg_prod")
      └─ schema
           └─ table
                └─ column

``get_metadata(catalog_name=..., db_schema=...)`` fetches one (catalog, schema)
slice.  ``list_catalogs()`` lists all Galaxy catalog names.  The full catalog
build loop in ``catalog/builder.py`` iterates over ``list_catalogs()`` and
calls ``get_metadata()`` for each (catalog, schema) pair.

Ported from the working ``app/adapters/starburst_trino_adapter.py``; this
version uses ``SourceConfig`` instead of ``Settings`` and returns the canonical
``ExecResult`` / ``PlanResult`` / ``Metadata`` DTOs.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import aiohttp

from ..observability.logging import get_logger
from .base import (
    ColumnMeta,
    ConnectorError,
    DataSourceConnector,
    ExecResult,
    Metadata,
    PlanResult,
    ReadOnlyViolation,
    SourceConfig,
    TableMeta,
)
from .factory import ConnectorFactory

_logger  = get_logger(__name__)


@dataclass(slots=True)
class _Token:
    access_token: str
    expires_at: float


class StarburstConnector(DataSourceConnector):
    """Starburst Galaxy + Trino connector."""

    def __init__(self, config: SourceConfig) -> None:
        super().__init__(config)
        c = config

        # Galaxy API base URL (account-level domain)
        self._api_base = (c.host or "").rstrip("/")
        if not self._api_base:
            raise ConnectorError(
                "StarburstConnector requires SourceConfig.host "
                "(e.g. https://acme.galaxy.starburst.io)"
            )

        # Trino cluster URL (cluster-level domain)
        self._trino_base = (c.trino_host or c.host or "").rstrip("/")

        # Query-history Trino URL (may point at a dedicated cluster)
        self._qh_trino_base = (c.qh_trino_host or self._trino_base).rstrip("/")

        self._timeout = aiohttp.ClientTimeout(total=c.timeout_ms / 1000)
        self._verify_ssl = bool(c.verify_ssl)
        self._ssl_context = _build_ssl_context(self._verify_ssl)
        if not self._verify_ssl:
            _logger.warning("starburst_ssl_verification_disabled")
        self._token: _Token | None = None

        # Trino session state (per-connection; safe to share across calls because
        # Starburst/Trino headers are idempotent for read-only sessions).
        self._session_props: dict[str, str] = {}
        self._prepared_stmts: dict[str, str] = {}
        self._tx_id: str | None = None

        role = (c.role or "").strip()
        self._role_header = f"system=ROLE{{{role}}}" if role else None

        qh_role = (c.qh_role or c.role or "").strip()
        self._qh_role_header = f"system=ROLE{{{qh_role}}}" if qh_role else None

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    def dialect(self) -> str:
        return "trino"

    # ── Core async operations ──────────────────────────────────────────────

    async def execute(
        self,
        sql: str,
        *,
        read_only: bool = True,
        timeout_s: int = 60,
        row_cap: int = 5000,
    ) -> ExecResult:
        if read_only:
            try:
                self.assert_read_only(sql)
            except ReadOnlyViolation as exc:
                return ExecResult(dialect=self.dialect, error=f"read_only_violation: {exc}")
        start = time.perf_counter()
        try:
            result = await self._trino_execute_full(sql, row_cap=row_cap)
            elapsed = (time.perf_counter() - start) * 1000.0
            result.elapsed_ms = elapsed
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000.0
            _logger.warning("starburst_execute_failed", sql_preview=sql[:80], error=str(exc))
            return ExecResult(dialect=self.dialect, elapsed_ms=elapsed, error=str(exc))

    async def explain_plan(self, sql: str) -> PlanResult:
        try:
            self.assert_read_only(sql)
        except ReadOnlyViolation as exc:
            return PlanResult(error=f"read_only_violation: {exc}")
        start = time.perf_counter()
        try:
            result = await self._trino_execute_full(
                f"EXPLAIN (TYPE DISTRIBUTED) {sql}", row_cap=500
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            if not result.ok:
                return PlanResult(error=result.error, elapsed_ms=elapsed)
            plan_text = "\n".join(str(r[0]) for r in result.rows if r)
            return PlanResult(plan_text=plan_text, elapsed_ms=elapsed)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000.0
            return PlanResult(error=str(exc), elapsed_ms=elapsed)

    async def get_metadata(
        self,
        *,
        catalog_name: str | None = None,
        db_schema: str | None = None,
    ) -> Metadata:
        """Discover tables + columns for one (catalog, schema) via the Galaxy API."""
        _logger.debug(f"Fetching metadata for catalog: {catalog_name} via Galaxy API...")
        cat = catalog_name or self.config.catalog_name
        if not cat:
            raise ConnectorError(
                "StarburstConnector.get_metadata() requires catalog_name "
                "(pass explicitly or set SourceConfig.catalog_name)"
            )
        schema = db_schema or self.config.db_schema

        catalog_label = _plain_ref(cat)
        catalog_key = _name_lookup_path_id(cat)
        tables: list[TableMeta] = []

        async def _for_schema(schema_meta: dict[str, Any]) -> None:
            schema_name = (
                _plain_ref(
                    _text(
                        schema_meta.get("schemaName")
                        or schema_meta.get("name")
                        or schema_meta.get("schemaId")
                    )
                )
                or ""
            )
            schema_key = _metadata_path_id(schema_meta, "schemaId", "schemaName", "name")
            raw_tables = await self._galaxy_get_paginated(
                f"/public/api/v1/catalog/{catalog_key}/schema/{schema_key}/table"
            )
            for rt in raw_tables:
                tname = _plain_ref(
                    _text(rt.get("tableName") or rt.get("name") or rt.get("tableId"))
                ) or ""
                table_key = _metadata_path_id(rt, "tableId", "tableName", "name")
                _logger.debug(
                    f"Table Key: {table_key}, Schema Key: {schema_key}, Table Name: {tname}"
                )
                cols: list[ColumnMeta] = []
                raw_cols = await self._galaxy_get_paginated(
                    f"/public/api/v1/catalog/{catalog_key}/schema/{schema_key}"
                    f"/table/{table_key}/column"
                )
                for ordinal, rc in enumerate(raw_cols, start=1):
                    cols.append(
                        ColumnMeta(
                            name=_plain_ref(
                                _text(
                                    rc.get("columnName")
                                    or rc.get("name")
                                    or rc.get("columnId")
                                )
                            )
                            or f"col_{ordinal}",
                            data_type=_text(rc.get("dataType") or rc.get("type")) or "UNKNOWN",
                            nullable=_bool_or_none(rc.get("nullable"))
                            if rc.get("nullable") is not None
                            else True,
                            comment=_text(_description(rc)),
                            ordinal=int(rc.get("ordinalPosition") or ordinal),
                        )
                    )
                tables.append(
                    TableMeta(
                        catalog_name=catalog_label,
                        db_schema=schema_name,
                        name=tname,
                        table_type=_text(rt.get("tableType")) or "BASE TABLE",
                        comment=_text(_description(rt)),
                        columns=cols,
                    )
                )

        if schema:
            schema_meta = await self._find_schema(catalog_key, schema)
            await _for_schema(schema_meta)
        else:
            async for sm in self._galaxy_iter_paginated(
                f"/public/api/v1/catalog/{catalog_key}/schema"
            ):
                await _for_schema(sm)

        _logger.info(
            "starburst_metadata_fetched",
            catalog=catalog_label,
            schema=schema,
            tables=len(tables),
        )
        return Metadata(
            source_type="starburst",
            catalog_name=catalog_label,
            db_schema=schema,
            tables=tables,
        )

    async def list_catalogs(self) -> list[str]:
        """List all Starburst Galaxy catalog names."""
        results: list[str] = []
        async for catalog in self._galaxy_iter_paginated("/public/api/v1/catalog"):
            name = _text(catalog.get("catalogName") or catalog.get("name"))
            if name:
                results.append(name)
        return results

    # ── Galaxy OAuth ──────────────────────────────────────────────────────

    async def _get_token(self) -> str:
        if self._token and time.time() < self._token.expires_at - 60:
            return self._token.access_token

        client_id = self.config.client_id
        client_secret = self.config.client_secret
        if not client_id or not client_secret:
            raise ConnectorError(
                "Starburst Galaxy API requires SourceConfig.client_id and "
                "SourceConfig.client_secret (OAuth client credentials)."
            )
        token_url = f"{self._api_base}/oauth/v2/token"
        async with (
            aiohttp.ClientSession(timeout=self._timeout, connector=self._connector()) as session,
            session.post(
                token_url,
                auth=aiohttp.BasicAuth(client_id, client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "client_credentials"},
            ) as resp,
        ):
            text = await resp.text()
            if resp.status >= 400:
                raise ConnectorError(
                    f"Galaxy token request failed status={resp.status} url={token_url} body={text}"
                )
            data = json.loads(text)

        token = data.get("access_token")
        if not token:
            raise ConnectorError("Galaxy token response missing access_token")
        self._token = _Token(
            access_token=token,
            expires_at=time.time() + int(data.get("expires_in", 600)),
        )
        return token

    async def _galaxy_headers(self) -> dict[str, str]:
        token = await self._get_token()
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    async def _galaxy_get_json(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self._api_base}{path}"
        async with (
            aiohttp.ClientSession(timeout=self._timeout, connector=self._connector()) as session,
            session.get(url, headers=await self._galaxy_headers(), params=params or {}) as resp,
        ):
            text = await resp.text()
            if resp.status >= 400:
                raise ConnectorError(
                    f"Galaxy API GET failed status={resp.status} url={url} body={text}"
                )
            return json.loads(text) if text else {}

    async def _galaxy_get_paginated(self, path: str, page_size: int = 100) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        async for item in self._galaxy_iter_paginated(path, page_size=page_size):
            results.append(item)
        return results

    async def _galaxy_iter_paginated(
        self, path: str, page_size: int = 100
    ) -> AsyncIterator[dict[str, Any]]:
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

    async def _find_schema(self, catalog_key: str, schema_ref: str) -> dict[str, Any]:
        async for schema in self._galaxy_iter_paginated(
            f"/public/api/v1/catalog/{catalog_key}/schema"
        ):
            if _matches_ref(
                schema_ref,
                schema.get("schemaId"),
                schema.get("schemaName"),
                schema.get("name"),
            ):
                return schema
        raise ConnectorError(f"Starburst schema not found: {schema_ref!r}")

    # ── Trino execution ────────────────────────────────────────────────────

    def _trino_auth(self) -> aiohttp.BasicAuth:
        user = self.config.user
        pwd = self.config.password
        if not user or not pwd:
            raise ConnectorError(
                "Trino execution requires SourceConfig.user and SourceConfig.password"
            )
        return aiohttp.BasicAuth(login=user, password=pwd)

    def _trino_headers(
        self, *, catalog: str | None = None, schema: str | None = None
    ) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "text/plain; charset=utf-8",
            "X-Trino-User": self.config.user or "",
            "X-Trino-Source": self.config.source,
        }
        effective_catalog = catalog or self.config.catalog_name
        effective_schema = schema or self.config.db_schema
        if effective_catalog:
            headers["X-Trino-Catalog"] = effective_catalog
        if effective_schema:
            headers["X-Trino-Schema"] = effective_schema
        if self._role_header:
            headers["X-Trino-Role"] = self._role_header
        if self._session_props:
            headers["X-Trino-Session"] = ",".join(
                f"{k}={v}" for k, v in sorted(self._session_props.items())
            )
        if self._tx_id:
            headers["X-Trino-Transaction-Id"] = self._tx_id
        return headers

    async def _trino_execute_full(self, sql: str, row_cap: int = 5000) -> ExecResult:
        """Submit a Trino statement and poll until completion, collecting all rows."""
        url = f"{self._trino_base}/v1/statement"
        payload = await self._trino_request("POST", url, data=sql.encode("utf-8"))

        columns: list[str] = []
        rows: list[list[Any]] = []
        schema: list[dict[str, Any]] = []

        def _collect(p: dict[str, Any]) -> None:
            nonlocal schema
            if p.get("columns") and not schema:
                schema = _schema_from_payload(p)
                columns.extend(col["name"] for col in schema)
            for row in p.get("data") or []:
                if len(rows) < row_cap + 1:
                    rows.append(list(row))

        _collect(payload)
        self._apply_response_headers(payload)

        while payload.get("nextUri"):
            payload = await self._trino_request("GET", payload["nextUri"])
            self._apply_response_headers(payload)
            _collect(payload)

        error = payload.get("error")
        if error:
            _raise_for_trino_error(payload, url=url)

        truncated = len(rows) > row_cap
        return ExecResult(
            columns=columns,
            rows=rows[:row_cap],
            row_count=len(rows[:row_cap]),
            truncated=truncated,
            dialect=self.dialect,
        )

    async def _trino_request(
        self,
        method: str,
        url: str,
        data: bytes | None = None,
        *,
        auth: aiohttp.BasicAuth | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        retry = 0
        while True:
            async with (
                aiohttp.ClientSession(
                    auth=auth or self._trino_auth(),
                    timeout=self._timeout,
                    connector=self._connector(),
                ) as session,
                session.request(
                    method, url, data=data, headers=headers or self._trino_headers()
                ) as resp,
            ):
                text = await resp.text()
                _logger.debug(
                    "trino_response",
                    method=method,
                    url=url,
                    status=resp.status,
                    bytes=len(text),
                )
                # Retry on transient errors
                if resp.status in {502, 503, 504} and retry < 3:
                    retry += 1
                    await asyncio.sleep(0.1 * retry)
                    continue
                if resp.status == 429 and retry < 3:
                    retry += 1
                    await asyncio.sleep(_parse_retry_after(resp.headers.get("Retry-After")))
                    continue
                if resp.status != 200:
                    raise ConnectorError(
                        f"Trino request failed status={resp.status} "
                        f"method={method} url={url} body={text[:200]}"
                    )
                try:
                    return json.loads(text) if text else {}
                except json.JSONDecodeError as exc:
                    raise ConnectorError(
                        f"Trino returned non-JSON status={resp.status} url={url}"
                    ) from exc

    def _apply_response_headers(self, payload: dict[str, Any]) -> None:
        # Not needed for the result-oriented interface; kept as a no-op for
        # compatibility if session state ever needs tracking.
        pass

    def _connector(self) -> aiohttp.TCPConnector:
        return aiohttp.TCPConnector(ssl=self._ssl_context)


# =============================================================================
# Module-level utility functions
# =============================================================================


def _path_id(value: str) -> str:
    """URL-encode a Galaxy catalog/schema/table ID for use in API paths."""
    return quote(str(value or ""), safe="")


def _name_lookup_path_id(value: Any) -> str:
    """URL-encode a Galaxy path lookup expression for a plain object name."""
    return quote(f"name={_plain_ref(value)}", safe="")


def _metadata_path_id(record: dict[str, Any], id_key: str, *name_keys: str) -> str:
    record_id = str(record.get(id_key) or "").strip()
    if record_id:
        return _path_id(record_id)
    for key in name_keys:
        name = _plain_ref(record.get(key))
        if name:
            return _name_lookup_path_id(name)
    return _path_id("")


def _plain_ref(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("name="):
        return text.split("=", 1)[1].strip()
    return text


def _build_ssl_context(verify_ssl: bool) -> ssl.SSLContext:
    context = ssl.create_default_context()
    if not verify_ssl:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def _matches_ref(ref: str, *candidates: Any) -> bool:
    key = _plain_ref(ref).lower()
    return any(_plain_ref(c).lower() == key for c in candidates)


def _description(obj: dict[str, Any]) -> str | None:
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    return (
        obj.get("description")
        or obj.get("comment")
        or obj.get("remarks")
        or (meta or {}).get("description")
        or (meta or {}).get("comment")
    )


def _schema_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": c.get("name"), "type": c.get("type")} for c in payload.get("columns", [])]


def _raise_for_trino_error(payload: dict[str, Any], url: str = "") -> None:
    error = payload.get("error")
    if not error:
        return
    if isinstance(error, dict):
        msg = error.get("message") or "Trino query failed"
        name = error.get("errorName", "")
        loc = error.get("errorLocation") or {}
        line = loc.get("lineNumber")
        col = loc.get("columnNumber")
        loc_txt = f" line={line} col={col}" if line and col else ""
        raise ConnectorError(f"Trino error {name}{loc_txt}: {msg} (url={url})")
    raise ConnectorError(f"Trino query failed (url={url})")


def _parse_retry_after(value: str | None) -> float:
    try:
        return max(0.0, float(value or 1.0))
    except ValueError:
        return 1.0


def _text(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).lower().strip()
    return True if s in {"true", "1", "yes"} else (False if s in {"false", "0", "no"} else None)


# Register with the factory
ConnectorFactory.register("starburst", StarburstConnector)
