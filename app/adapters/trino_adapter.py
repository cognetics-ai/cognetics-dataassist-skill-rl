from __future__ import annotations

import base64
import json
import ssl
import time
import uuid
from urllib.parse import quote

import aiohttp
from trino.auth import JWTAuthentication

from app.adapters.base import EngineAdapter, EngineHandle, EngineStatus, ExplainResult, ResultPage
from app.config import Settings


class StarburstTrinoAdapter(EngineAdapter):
    name = "starburst"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._use_jwt = settings.starburst_use_jwt

        self._base_url = (
                settings.starburst_url
                or f"https://{settings.starburst_host}:{settings.starburst_port}"
        ).rstrip("/")

        self._base_trino_url = (
                settings.starburst_trino_url
                or f"https://{settings.starburst_trino_host}:{settings.starburst_port}"
        ).rstrip("/")

        # Public API / OAuth token base URL. Prefer the account-domain URL.
        self._api_base_url = (
                getattr(settings, "starburst_api_url", None)
                or self._base_url
        ).rstrip("/")

        self._timeout = aiohttp.ClientTimeout(total=settings.starburst_timeout_ms / 1000)
        self._verify_ssl = bool(settings.starburst_verify_ssl)
        self._ssl_context = self._build_ssl_context(self._verify_ssl)
        self._headers: dict[str, str] = {}
        self._access_token = None
        self._access_token_expires_at = 0.0

        if settings.starburst_user:
            self._headers["X-Trino-User"] = settings.starburst_user
        if settings.starburst_catalog:
            self._headers["X-Trino-Catalog"] = settings.starburst_catalog
        if settings.starburst_schema:
            self._headers["X-Trino-Schema"] = settings.starburst_schema

        if self._use_jwt and settings.starburst_jwt_token:
            self._set_bearer(settings.starburst_jwt_token, expires_in=None)

        self._local_cancelled: set[str] = set()

    @property
    def _auth(self) -> aiohttp.BasicAuth | JWTAuthentication:
        self._ensure_bearer_token()
        if self._use_jwt:
            return JWTAuthentication(self._access_token)
        return aiohttp.BasicAuth(
            login=self._settings.starburst_user,
            password=self._settings.starburst_password,
        )

    async def explain(self, sql: str) -> ExplainResult:
        explain_sql = f"EXPLAIN (TYPE LOGICAL) {sql}"
        payload = await self._submit_statement(explain_sql)
        if payload.get("error"):
            return ExplainResult(ok=False, summary={"error": payload["error"], "engine": self.name})

        rows = payload.get("data") or []
        plan_text = "\n".join(str(col) for row in rows for col in row)
        return ExplainResult(ok=True,
                             summary={"engine": self.name, "plan": plan_text[:5000], "stats": payload.get("stats", {})})

    async def execute_async(self, sql: str) -> EngineHandle:
        payload = await self._submit_statement(sql)
        print(f"Payload: {payload}")
        handle_id = str(uuid.uuid4())
        handle_raw = {
            "query": sql,
            "nextUri": payload.get("nextUri"),
            "queryId": payload.get("id"),
            "lastPayload": payload,
            "rows": payload.get("data", []),
            "schema": [{"name": col.get("name"), "type": col.get("type")} for col in payload.get("columns", [])],
        }
        return EngineHandle(handle_id=handle_id, raw=handle_raw)

    async def get_status(self, handle: EngineHandle) -> EngineStatus:
        if handle.handle_id in self._local_cancelled:
            return EngineStatus(state="CANCELLED", done=True, progress_percentage=100)

        payload = handle.raw.get("lastPayload") or {}
        next_uri = handle.raw.get("nextUri")

        if next_uri:
            payload = await self._get_next(next_uri)
            handle.raw["lastPayload"] = payload
            handle.raw["nextUri"] = payload.get("nextUri")

            if payload.get("data"):
                handle.raw.setdefault("rows", []).extend(payload["data"])
            if payload.get("columns"):
                handle.raw["schema"] = [
                    {"name": col.get("name"), "type": col.get("type")} for col in payload.get("columns", [])
                ]

        stats = payload.get("stats", {})
        state = (stats.get("state") or payload.get("state") or "UNKNOWN").upper()
        error = payload.get("error")
        if error:
            state = "FAILED"

        done = state in {"FINISHED", "FAILED", "CANCELLED"} and not handle.raw.get("nextUri")
        progress = self._coerce_progress(stats.get("progressPercentage", 0))
        if state == "FINISHED":
            progress = 100

        return EngineStatus(state=state, done=done, progress_percentage=progress, stats=stats, error=error)

    async def fetch_results(self, handle: EngineHandle, page_token: str | None = None) -> ResultPage:
        schema = handle.raw.get("schema", [])
        rows = handle.raw.get("rows", [])
        return ResultPage(schema=schema, rows=rows, next_page_token=None)

    async def cancel(self, handle: EngineHandle) -> bool:
        self._local_cancelled.add(handle.handle_id)
        query_id = (handle.raw.get("lastPayload") or {}).get("id") or handle.raw.get("queryId")
        if not query_id:
            return True

        url = f"{self._base_url}/v1/query/{query_id}"
        connector = self._connector()
        async with aiohttp.ClientSession(auth=self._auth, timeout=self._timeout, connector=connector) as session:
            async with session.delete(url, headers=self._request_headers()) as resp:
                return resp.status in {200, 202, 204, 404}

    def _set_bearer(self, token: str, expires_in: int | None) -> None:
        token = token.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()

        self._headers["Authorization"] = f"Bearer {token}"
        self._access_token_expires_at = (
            float("inf") if expires_in is None else time.time() + expires_in
        )

    async def _ensure_bearer_token(self) -> None:
        if not self._use_jwt:
            return

        # Reuse token until near expiry.
        if self._headers.get("Authorization") and time.time() < self._access_token_expires_at - 60:
            return

        # If caller provided a ready access token, use it directly.
        if self._settings.starburst_jwt_token:
            token = self._settings.starburst_jwt_token.strip()
            if token.lower().startswith("bearer "):
                token = token[7:].strip()

            self._headers["Authorization"] = f"Bearer {token}"
            self._access_token_expires_at = float("inf")
            return

        client_id = self._settings.starburst_client_id
        client_secret = self._settings.starburst_client_secret

        if not client_id or not client_secret:
            raise ValueError(
                "STARBURST_USE_JWT=true requires either STARBURST_JWT_TOKEN "
                "or STARBURST_CLIENT_ID + STARBURST_CLIENT_SECRET"
            )

        # IMPORTANT:
        # This must be the Galaxy account domain, not the Trino cluster URL.
        # Example: https://myaccount.galaxy.starburst.io
        token_url = f"{self._api_base_url.rstrip('/')}/oauth/v2/token"
        print(f"Token URL: {token_url}")

        raw_basic = f"{client_id}:{client_secret}".encode("utf-8")
        print(f"Raw Basic: {raw_basic}")
        basic_token = base64.b64encode(raw_basic).decode("ascii")
        print(f"basic_token: {basic_token}")

        headers = {
            "Authorization": f"Basic {basic_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        body = "grant_type=client_credentials"

        connector = self._connector()

        async with aiohttp.ClientSession(timeout=self._timeout, connector=connector) as session:
            async with session.post(token_url, headers=headers, data=body) as resp:
                text = await resp.text()

                if resp.status >= 400:
                    raise RuntimeError(
                        "Starburst token request failed "
                        f"status={resp.status}, url={token_url}, body={text}"
                    )

                payload = json.loads(text)

        print(f"Payload: {payload} ")
        self._access_token = payload.get("access_token")
        if not self._access_token:
            raise RuntimeError(f"Starburst token response did not include access_token: {payload}")

        expires_in = int(payload.get("expires_in", 600))

        self._headers["Authorization"] = f"Bearer {self._access_token}"
        self._access_token_expires_at = time.time() + expires_in

    async def _api_get_json(self, path: str, params: dict | None = None) -> dict:
        await self._ensure_bearer_token()

        connector = self._connector()
        url = f"{self._api_base_url}{path}"

        async with aiohttp.ClientSession(timeout=self._timeout, connector=connector) as session:
            async with session.get(url, headers=self._request_headers(), params=params or {}) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"Starburst API GET failed: {resp.status} {text}")
                return json.loads(text)

    async def _api_get_paginated(self, path: str, page_size: int = 100) -> list[dict]:
        results: list[dict] = []
        page_token: str | None = None

        while True:
            params = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            payload = await self._api_get_json(path, params=params)
            results.extend(payload.get("result") or [])

            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        return results

    @staticmethod
    def _path_id(value: str) -> str:
        # Supports both raw IDs and lookup forms like name=my_catalog.
        # The Starburst spec says lookup forms must be URL encoded, including "=".
        return quote(value, safe="")

    async def extract_catalog_metadata(self, catalog_id: str) -> dict:
        """
        Returns:
        {
          "catalog": {...},
          "tables": [flattened table rows with catalog/schema/table metadata],
          "columns": [flattened column rows with catalog/schema/table/column metadata]
        }
        """
        catalog_key = self._path_id(catalog_id)

        catalog = await self._api_get_json(
            f"/public/api/v1/catalog/{catalog_key}/catalogMetadata"
        )

        schemas = await self._api_get_paginated(
            f"/public/api/v1/catalog/{catalog_key}/schema"
        )

        table_rows: list[dict] = []
        column_rows: list[dict] = []

        for schema in schemas:
            schema_id = schema["schemaId"]
            schema_key = self._path_id(schema_id)

            tables = await self._api_get_paginated(
                f"/public/api/v1/catalog/{catalog_key}/schema/{schema_key}/table"
            )

            for table in tables:
                table_id = table["tableId"]
                table_key = self._path_id(table_id)

                table_row = {
                    "catalogId": catalog.get("catalogId"),
                    "catalogName": catalog.get("catalogName"),
                    "catalogMetadata": catalog,
                    "schemaId": schema_id,
                    "schemaMetadata": schema,
                    "tableId": table_id,
                    "tableType": table.get("tableType"),
                    "tableMetadata": table,
                }
                table_rows.append(table_row)

                columns = await self._api_get_paginated(
                    f"/public/api/v1/catalog/{catalog_key}/schema/{schema_key}/table/{table_key}/column"
                )

                for column in columns:
                    column_rows.append({
                        **table_row,
                        "columnId": column.get("columnId"),
                        "dataType": column.get("dataType"),
                        "columnDefault": column.get("columnDefault"),
                        "nullable": column.get("nullable"),
                        "columnMetadata": column,
                    })

        return {
            "catalog": catalog,
            "tables": table_rows,
            "columns": column_rows,
        }

    async def _submit_statement(self, sql: str) -> dict:
        await self._ensure_bearer_token()
        print(f"Headers: {self._request_headers()}")

        url = f"{self._base_trino_url}/v1/statement"
        connector = self._connector()

        async with aiohttp.ClientSession(auth=self._auth, timeout=self._timeout, connector=connector) as session:
            async with session.post(url, data=sql.encode("utf-8"), headers=self._request_headers()) as resp:
                text = await resp.text()
                print(f"Response: {text}")
                resp.raise_for_status()
                return json.loads(text)

    async def _get_next(self, next_uri: str) -> dict:
        await self._ensure_bearer_token()

        connector = self._connector()

        async with aiohttp.ClientSession(auth=self._auth, timeout=self._timeout, connector=connector) as session:
            async with session.get(next_uri, headers=self._request_headers()) as resp:
                text = await resp.text()
                resp.raise_for_status()
                return json.loads(text)

    def _request_headers(self) -> dict[str, str]:
        return dict(self._headers)

    def _connector(self) -> aiohttp.TCPConnector:
        return aiohttp.TCPConnector(ssl=self._ssl_context)

    @staticmethod
    def _build_ssl_context(verify_ssl: bool) -> ssl.SSLContext:
        context = ssl.create_default_context()
        if not verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context
