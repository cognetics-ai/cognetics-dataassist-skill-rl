from __future__ import annotations

import ast
import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
from typing import Any
import zipfile

import aiohttp

from app.config import Settings
from app.core.sql_utils import extract_tables
from app.core.store import SQLStore


@dataclass
class CatalogSyncStats:
    step1_master_rows: int = 0
    step2_common_queries: int = 0
    step3_column_details: int = 0
    step2_failures: int = 0
    step3_failures: int = 0


class DiscoveryCatalogLoader:
    """Loads discovery catalog datasets via async POST APIs and persists to backend SQL tables."""

    def __init__(self, settings: Settings, store: SQLStore):
        self._settings = settings
        self._store = store

    async def sync_catalog(self, max_assets: int | None = None, concurrency: int = 8) -> dict[str, Any]:
        master_records = await self._fetch_master_records()
        if max_assets and max_assets > 0:
            master_records = master_records[:max_assets]

        stats = CatalogSyncStats()
        stats.step1_master_rows = await self._store.upsert_discovery_master_rows(master_records)

        unique_assets: dict[tuple[str, str], dict[str, Any]] = {}
        for row in master_records:
            table_name = str(row.get("table_name") or "").strip()
            target_db = str(row.get("target_db") or "").strip()
            if not table_name:
                continue
            unique_assets[(target_db, table_name)] = row

        assets = list(unique_assets.values())
        sem = asyncio.Semaphore(max(1, concurrency))

        step2_results = await asyncio.gather(
            *(self._load_common_queries_for_asset(asset, sem) for asset in assets),
            return_exceptions=True,
        )
        common_query_rows: list[dict[str, Any]] = []
        for result in step2_results:
            if isinstance(result, Exception):
                stats.step2_failures += 1
                continue
            common_query_rows.extend(result)
        stats.step2_common_queries = await self._store.upsert_common_query_rows(common_query_rows)

        step3_results = await asyncio.gather(
            *(self._load_column_details_for_asset(asset, sem) for asset in assets),
            return_exceptions=True,
        )
        column_rows: list[dict[str, Any]] = []
        for result in step3_results:
            if isinstance(result, Exception):
                stats.step3_failures += 1
                continue
            column_rows.extend(result)
        stats.step3_column_details = await self._store.upsert_column_detail_rows(column_rows)

        return {
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "master_row_count": stats.step1_master_rows,
            "common_query_row_count": stats.step2_common_queries,
            "column_detail_row_count": stats.step3_column_details,
            "step2_failures": stats.step2_failures,
            "step3_failures": stats.step3_failures,
            "asset_count_processed": len(assets),
        }

    async def _fetch_master_records(self) -> list[dict[str, Any]]:
        headers = self._headers(accept="application/zip")
        payload = {}
        timeout = aiohttp.ClientTimeout(total=self._settings.discovery_timeout_ms / 1000)
        connector = aiohttp.TCPConnector(verify_ssl=self._settings.discovery_verify_ssl)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.post(self._settings.discovery_step1_url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                content = await resp.read()
                content_type = resp.headers.get("Content-Type", "")
        records = self._parse_step1_payload(content, content_type)
        return [self._normalize_master_record(item) for item in records if isinstance(item, dict)]

    def _parse_step1_payload(self, content: bytes, content_type: str) -> list[dict[str, Any]]:
        if "zip" in content_type.lower():
            return self._parse_zip_records(content)
        try:
            return self._as_record_list(self._deserialize_blob(content, "payload.json"))
        except Exception:
            return self._parse_zip_records(content)

    def _parse_zip_records(self, zip_bytes: bytes) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                for filename in archive.namelist():
                    blob = archive.read(filename)
                    parsed = self._deserialize_blob(blob, filename)
                    records.extend(self._as_record_list(parsed))
        except zipfile.BadZipFile:
            parsed = self._deserialize_blob(zip_bytes, "step1_payload.txt")
            records.extend(self._as_record_list(parsed))
        return records

    def _deserialize_blob(self, blob: bytes, filename: str) -> Any:
        if filename.lower().endswith(".csv"):
            text = blob.decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            return [dict(row) for row in reader]

        text = blob.decode("utf-8", errors="replace").strip()
        if not text:
            return []

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(text)
        except Exception:
            pass

        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(candidate)
                except Exception:
                    continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return records

    @staticmethod
    def _as_record_list(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("resultList", "results", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        return []

    async def _load_common_queries_for_asset(self, asset: dict[str, Any], semaphore: asyncio.Semaphore) -> list[dict[str, Any]]:
        table_name = str(asset.get("table_name") or "").strip()
        if not table_name:
            return []
        target_db = str(asset.get("target_db") or "").strip()
        schema_table = f"{target_db}.{table_name}" if target_db else table_name

        async with semaphore:
            payload = {"schemaTable": schema_table}
            response = await self._post_json(self._settings.discovery_step2_url, payload)

        common_queries = response.get("commonQueries", []) if isinstance(response, dict) else []
        rows: list[dict[str, Any]] = []
        for item in common_queries:
            if not isinstance(item, dict):
                continue
            email = str(item.get("email") or "").strip() or None
            rows.append(
                {
                    "soeid": str(item.get("soeid") or self._derive_soeid_from_email(email) or "").strip() or None,
                    "email": email,
                    "name": item.get("mostFrequentUser") or item.get("name"),
                    "query": item.get("query"),
                    "tool": item.get("tool"),
                    "schema_table": schema_table,
                    "all_query_tables": extract_tables(str(item.get("query") or "")),
                    "success_percentage": item.get("successPercentage"),
                    "raw": item,
                }
            )
        return rows

    async def _load_column_details_for_asset(self, asset: dict[str, Any], semaphore: asyncio.Semaphore) -> list[dict[str, Any]]:
        table_name = str(asset.get("table_name") or "").strip()
        if not table_name:
            return []

        payload = {
            "product": asset.get("standardized_domain") or asset.get("domain"),
            "targetTableName": table_name,
            "sourceSystem": asset.get("source_system"),
            "region": asset.get("region"),
            "targetDB": asset.get("target_db"),
        }

        async with semaphore:
            response = await self._post_json(self._settings.discovery_step3_url, payload)

        result_list = response.get("resultList", []) if isinstance(response, dict) else []
        rows: list[dict[str, Any]] = []
        for item in result_list:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "domain": item.get("domain") or payload.get("product"),
                    "target_table_name": item.get("targetTableName") or table_name,
                    "target_column_name": item.get("targetColumnName"),
                    "target_column_name_desc": item.get("targetColumnNameDesc"),
                    "critical_data_element": item.get("criticalDataElement"),
                    "pii": item.get("pii"),
                    "primary_foreign_key": item.get("primary_foreign_key") or item.get("primary_Foreign_key"),
                    "target_db": payload.get("targetDB"),
                    "raw": item,
                }
            )
        return rows

    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self._settings.discovery_timeout_ms / 1000)
        connector = aiohttp.TCPConnector(verify_ssl=self._settings.discovery_verify_ssl)
        headers = self._headers(accept="application/json")
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)

    def _headers(self, accept: str) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
        }
        extras = self._settings.discovery_extra_headers_json.strip()
        if extras:
            try:
                parsed = json.loads(extras)
                if isinstance(parsed, dict):
                    headers.update({str(key): str(value) for key, value in parsed.items()})
            except json.JSONDecodeError:
                pass
        token = self._settings.discovery_api_token.strip()
        if token:
            header_name = self._settings.discovery_api_token_header.strip() or "Authorization"
            if header_name.lower() == "authorization" and not token.lower().startswith("bearer "):
                headers[header_name] = f"Bearer {token}"
            else:
                headers[header_name] = token
        return headers

    @staticmethod
    def _normalize_master_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "zone": record.get("zone"),
            "table_name": record.get("targetTableName") or record.get("tableName"),
            "table_description": record.get("targetTableDesc") or record.get("tableDescription"),
            "domain": record.get("domain"),
            "standardized_domain": record.get("standarizedDomain") or record.get("standardizedDomain"),
            "source_system": record.get("sourceSystem"),
            "region": record.get("region"),
            "country": record.get("country"),
            "target_db": record.get("targetDB"),
            "pii": record.get("pii"),
            "critical_data_element": record.get("criticalDataElement"),
            "asset_id": record.get("assetId") or record.get("assetid"),
            "raw": record,
        }

    @staticmethod
    def _derive_soeid_from_email(email: str | None) -> str | None:
        if not email:
            return None
        local = email.split("@", 1)[0].strip()
        return local or None
