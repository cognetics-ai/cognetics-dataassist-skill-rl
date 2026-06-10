from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import random
import re
import time
from typing import Any
from urllib.parse import quote_plus

import aiohttp

from app.adapters.registry import EngineRegistry
from app.config import Settings
from app.core.store import SQLStore
from app.core.sql_utils import extract_tables
from app.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class DataUsageNlpStats:
    source_rows: int = 0
    explain_passed: int = 0
    explain_failed: int = 0
    people_enriched: int = 0
    nlp_generated: int = 0
    inserted_rows: int = 0
    embeddings_generated: int = 0
    batches_processed: int = 0
    rows_failed: int = 0
    rate_limit_retries: int = 0
    embedding_retries: int = 0


@dataclass
class ProcessedRow:
    payload: dict[str, Any] | None = None
    explain_passed: bool = False
    explain_failed: bool = False
    people_enriched: bool = False
    nlp_generated: bool = False
    embeddings_generated: bool = False
    rate_limit_retries: int = 0
    embedding_retries: int = 0


class AsyncRateLimiter:
    """Simple in-process rate limiter enforcing global call spacing."""

    def __init__(self, max_per_second: float):
        self._interval = (1.0 / max_per_second) if max_per_second > 0 else 0.0
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self._interval <= 0:
            return

        async with self._lock:
            now = time.monotonic()
            if self._next_slot > now:
                await asyncio.sleep(self._next_slot - now)
                now = time.monotonic()
            self._next_slot = max(self._next_slot, now) + self._interval


class DataUsageNlpService:
    """Builds NLP-form query memory from DATA_USAGE_COMMON_QUERIES."""

    def __init__(
        self,
        settings: Settings,
        store: SQLStore,
        engines: EngineRegistry,
        embeddings: EmbeddingService,
    ):
        self._settings = settings
        self._store = store
        self._engines = engines
        self._embeddings = embeddings
        self._llm_rate_limiter = AsyncRateLimiter(max(0.0, settings.data_usage_nlp_llm_rps))
        self._people_cache: dict[str, dict[str, Any]] = {}
        self._column_context_cache: dict[str, list[dict[str, Any]]] = {}
        self._nlp_cache: dict[str, str] = {}
        self._prompt_examples = [
            (
                "SELECT cost_center, SUM(expense_amount) AS total_expense FROM finance.expenses WHERE fiscal_year = 2025 GROUP BY cost_center ORDER BY total_expense DESC LIMIT 200",
                "Shows total 2025 expense by cost center, sorted from highest to lowest, limited to 200 rows.",
            ),
            (
                "SELECT order_date, region, SUM(amount) AS revenue FROM sales.orders WHERE order_date >= date_add('day', -30, current_date) GROUP BY 1,2 ORDER BY 1 DESC LIMIT 1000",
                "Returns daily revenue by region for the last 30 days, most recent days first, capped at 1000 rows.",
            ),
        ]

    async def sync(
        self,
        limit: int | None = None,
        concurrency: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        sql = self._settings.data_usage_common_query_select_sql.strip().rstrip(";")
        return await self._sync_from_sql(
            sql=sql,
            limit=limit,
            concurrency=concurrency,
            batch_size=batch_size,
        )

    async def sync_by_ids(
        self,
        ids: list[int],
        concurrency: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        normalized_ids = sorted({int(item) for item in ids if int(item) > 0})
        if not normalized_ids:
            raise ValueError("At least one positive ID is required.")

        base_sql = self._settings.data_usage_common_query_select_sql.strip().rstrip(";")
        source_sql = self._ensure_id_selected(base_sql)
        placeholders = ", ".join("?" for _ in normalized_ids)
        filtered_sql = f"SELECT * FROM ({source_sql}) AS source_rows WHERE ID IN ({placeholders})"
        return await self._sync_from_sql(
            sql=filtered_sql,
            sql_params=tuple(normalized_ids),
            concurrency=concurrency,
            batch_size=batch_size,
        )

    async def sync_backend_query_history(
        self,
        *,
        engine: str = "starburst",
        limit: int | None = None,
        concurrency: int | None = None,
        batch_size: int | None = None,
        validate_with_explain: bool | None = None,
    ) -> dict[str, Any]:
        stats = DataUsageNlpStats()
        effective_concurrency = max(1, concurrency or self._settings.data_usage_nlp_concurrency)
        effective_batch_size = max(10, batch_size or self._settings.data_usage_nlp_batch_size)
        effective_validate = (
            self._settings.backend_query_nlp_validate_with_explain
            if validate_with_explain is None
            else bool(validate_with_explain)
        )
        remaining = int(limit) if limit and limit > 0 else None
        sem = asyncio.Semaphore(effective_concurrency)

        while True:
            page_size = effective_batch_size if remaining is None else max(0, min(effective_batch_size, remaining))
            if page_size <= 0:
                break

            source_batch = await self._store.list_backend_query_history_for_nlp(
                limit=page_size,
                engine=engine,
            )
            if not source_batch:
                break

            stats.batches_processed += 1
            stats.source_rows += len(source_batch)
            processed = await asyncio.gather(
                *(
                    self._process_row(
                        row,
                        sem,
                        validate_with_explain=effective_validate,
                    )
                    for row in source_batch
                ),
                return_exceptions=True,
            )

            payload_rows: list[dict[str, Any]] = []
            for item in processed:
                if isinstance(item, Exception):
                    stats.rows_failed += 1
                    logger.error(
                        "Backend query NLP row processing failed: %s",
                        item,
                        exc_info=(type(item), item, item.__traceback__),
                    )
                    continue
                if not isinstance(item, ProcessedRow):
                    continue
                stats.rate_limit_retries += item.rate_limit_retries
                stats.embedding_retries += item.embedding_retries
                if item.explain_passed:
                    stats.explain_passed += 1
                if item.explain_failed:
                    stats.explain_failed += 1
                if item.people_enriched:
                    stats.people_enriched += 1
                if item.nlp_generated:
                    stats.nlp_generated += 1
                if item.embeddings_generated:
                    stats.embeddings_generated += 1
                if item.payload:
                    payload_rows.append(item.payload)

            inserted = await self._store.upsert_backend_query_nlp_rows(payload_rows)
            stats.inserted_rows += inserted

            if remaining is not None:
                remaining -= len(source_batch)
                if remaining <= 0:
                    break
            if len(source_batch) < page_size or inserted == 0:
                break

        return {
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "source_rows": stats.source_rows,
            "explain_passed": stats.explain_passed,
            "explain_failed": stats.explain_failed,
            "people_enriched": stats.people_enriched,
            "nlp_generated": stats.nlp_generated,
            "inserted_rows": stats.inserted_rows,
            "embeddings_generated": stats.embeddings_generated,
            "batches_processed": stats.batches_processed,
            "rows_failed": stats.rows_failed,
            "rate_limit_retries": stats.rate_limit_retries,
            "embedding_retries": stats.embedding_retries,
        }

    async def _sync_from_sql(
        self,
        sql: str,
        limit: int | None = None,
        concurrency: int | None = None,
        batch_size: int | None = None,
        sql_params: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        stats = DataUsageNlpStats()
        effective_concurrency = max(1, concurrency or self._settings.data_usage_nlp_concurrency)
        effective_batch_size = max(10, batch_size or self._settings.data_usage_nlp_batch_size)
        remaining = int(limit) if limit and limit > 0 else None

        sem = asyncio.Semaphore(effective_concurrency)
        async for source_batch in self._iter_source_batches(
            sql,
            batch_size=effective_batch_size,
            limit=remaining,
            params=sql_params,
        ):
            stats.batches_processed += 1
            stats.source_rows += len(source_batch)
            logger.info(
                "NLP sync batch %s started (rows=%s, concurrency=%s)",
                stats.batches_processed,
                len(source_batch),
                effective_concurrency,
            )

            processed = await asyncio.gather(
                *(self._process_row(row, sem) for row in source_batch),
                return_exceptions=True,
            )

            payload_rows: list[dict[str, Any]] = []
            for item in processed:
                if isinstance(item, Exception):
                    stats.rows_failed += 1
                    logger.error(
                        "NLP sync row processing failed: %s",
                        item,
                        exc_info=(type(item), item, item.__traceback__),
                    )
                    continue
                if not isinstance(item, ProcessedRow):
                    continue
                stats.rate_limit_retries += item.rate_limit_retries
                stats.embedding_retries += item.embedding_retries
                if item.explain_passed:
                    stats.explain_passed += 1
                if item.explain_failed:
                    stats.explain_failed += 1
                if item.people_enriched:
                    stats.people_enriched += 1
                if item.nlp_generated:
                    stats.nlp_generated += 1
                if item.embeddings_generated:
                    stats.embeddings_generated += 1
                if item.payload:
                    payload_rows.append(item.payload)

            if payload_rows:
                stats.inserted_rows += await self._store.upsert_data_usage_nlp_rows(payload_rows)
            logger.info(
                "NLP sync batch %s completed (inserted=%s, failed_rows=%s)",
                stats.batches_processed,
                len(payload_rows),
                stats.rows_failed,
            )

        return {
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "source_rows": stats.source_rows,
            "explain_passed": stats.explain_passed,
            "explain_failed": stats.explain_failed,
            "people_enriched": stats.people_enriched,
            "nlp_generated": stats.nlp_generated,
            "inserted_rows": stats.inserted_rows,
            "embeddings_generated": stats.embeddings_generated,
            "batches_processed": stats.batches_processed,
            "rows_failed": stats.rows_failed,
            "rate_limit_retries": stats.rate_limit_retries,
            "embedding_retries": stats.embedding_retries,
        }

    @staticmethod
    def _ensure_id_selected(sql: str) -> str:
        match = re.match(r"^\s*select\s+(distinct\s+)?", sql, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError("DATA_USAGE_COMMON_QUERY_SELECT_SQL must start with SELECT.")

        projection = sql[match.end() :].lstrip()
        if re.match(r'^(?:"?id"?|[A-Za-z_][A-Za-z0-9_]*\."?id"?)\b', projection, flags=re.IGNORECASE):
            return sql

        distinct = "DISTINCT " if match.group(1) else ""
        return f"SELECT {distinct}ID, {sql[match.end():].lstrip()}"

    async def _iter_source_batches(
        self,
        base_sql: str,
        batch_size: int,
        limit: int | None,
        params: tuple[Any, ...] = (),
    ) -> Any:
        offset = 0
        remaining = limit
        while True:
            page_size = batch_size if remaining is None else max(0, min(batch_size, remaining))
            if page_size <= 0:
                break

            batch_sql = f"SELECT * FROM ({base_sql}) AS source_rows LIMIT {int(page_size)} OFFSET {int(offset)}"
            rows = await self._store.query_rows(batch_sql, params)
            if not rows:
                break

            yield rows

            loaded = len(rows)
            offset += loaded
            if remaining is not None:
                remaining -= loaded
                if remaining <= 0:
                    break
            if loaded < page_size:
                break

    async def _process_row(
        self,
        row: dict[str, Any],
        semaphore: asyncio.Semaphore,
        *,
        validate_with_explain: bool = True,
    ) -> ProcessedRow:
        normalized = {str(key).lower(): value for key, value in row.items()}

        query_text = str(normalized.get("query") or "").strip()
        if not query_text:
            return ProcessedRow()

        async with semaphore:
            explain_passed = False
            if validate_with_explain:
                engine_name = str(normalized.get("engine") or "starburst")
                explain = await self._engines.get(engine_name).explain(query_text)
                if not explain.ok:
                    return ProcessedRow(explain_failed=True)
                explain_passed = True

            email = str(normalized.get("email") or "").strip().lower()
            people, people_retries = await self._lookup_people_by_email(email) if email else ({}, 0)
            people_enriched = bool(people)

            schema_table = normalized.get("schema_table")
            all_query_tables = self._parse_table_list(normalized.get("all_query_tables"))
            if not all_query_tables:
                all_query_tables = extract_tables(query_text)
            if not schema_table and all_query_tables:
                schema_table = all_query_tables[0]
            schema_table_text = str(schema_table or "").strip()
            column_context = await self._load_column_context(schema_table_text)

            query_in_nlp, llm_retries = await self._sql_to_nlp(
                query_text,
                schema_table=schema_table_text or None,
                column_context=column_context,
            )
            if not query_in_nlp:
                return ProcessedRow(
                    explain_passed=explain_passed,
                    people_enriched=people_enriched,
                    rate_limit_retries=people_retries + llm_retries,
                )

            embeddings, embedding_retries = await self._embeddings.embed_document(query_in_nlp)
            embeddings_generated = bool(embeddings)

            segments = self._parse_managed_segments(people.get("managedsegmenthierarchy") or "")
            payload: dict[str, Any] = {
                "email": email,
                "soeid": people.get("soeid") or normalized.get("soeid"),
                "query": query_text,
                "query_in_nlp": query_in_nlp,
                "schema_table": schema_table_text or None,
                "all_query_tables": all_query_tables,
                "business_title": str(people.get("business_title") or "").strip() or None,
                "embeddings": embeddings or None,
            }
            if normalized.get("engine"):
                payload["engine"] = str(normalized.get("engine") or "").strip().lower()
            if normalized.get("query_id"):
                payload["query_id"] = str(normalized.get("query_id") or "").strip()
            if normalized.get("id") is not None:
                payload["raw_query_history_id"] = normalized.get("id")
            for index in range(1, 13):
                payload[f"managed_segment_l{index}"] = segments.get(index)
            return ProcessedRow(
                payload=payload,
                explain_passed=explain_passed,
                people_enriched=people_enriched,
                nlp_generated=True,
                embeddings_generated=embeddings_generated,
                rate_limit_retries=people_retries + llm_retries,
                embedding_retries=embedding_retries,
            )

    async def _sql_to_nlp(
        self,
        sql_text: str,
        schema_table: str | None = None,
        column_context: list[dict[str, Any]] | None = None,
    ) -> tuple[str, int]:
        cache_key = f"{schema_table or ''}::{sql_text}"
        cached = self._nlp_cache.get(cache_key)
        if cached:
            return cached, 0

        prompt = self._build_prompt(
            sql_text,
            schema_table=schema_table,
            column_context=column_context or [],
        )
        try:
            from google import genai  # type: ignore
        except Exception:
            return self._fallback_sql_summary(sql_text), 0

        def run_generation() -> str:
            client = genai.Client(vertexai=True, project=self._settings.vertex_project_id, location=self._settings.vertex_location)
            response = client.models.generate_content(model=self._settings.vertex_model, contents=prompt)
            text = getattr(response, "text", "") or ""
            return str(text).strip()

        retries = 0
        for attempt in range(max(0, self._settings.data_usage_nlp_max_retries) + 1):
            try:
                await self._llm_rate_limiter.wait()
                generated = await asyncio.to_thread(run_generation)
                result = generated or self._fallback_sql_summary(sql_text)
                self._cache_put(self._nlp_cache, cache_key, result)
                return result, retries
            except Exception as exc:
                if not self._is_rate_limited_exception(exc) or attempt >= self._settings.data_usage_nlp_max_retries:
                    result = self._fallback_sql_summary(sql_text)
                    self._cache_put(self._nlp_cache, cache_key, result)
                    return result, retries
                retries += 1
                await asyncio.sleep(self._backoff_seconds(attempt))

        result = self._fallback_sql_summary(sql_text)
        self._cache_put(self._nlp_cache, cache_key, result)
        return result, retries

    def _build_prompt(
        self,
        sql_text: str,
        schema_table: str | None,
        column_context: list[dict[str, Any]],
    ) -> str:
        examples = []
        for idx, (sample_sql, sample_nlp) in enumerate(self._prompt_examples, start=1):
            examples.append(
                f"### Example {idx}\n"
                f"```sql\n{sample_sql}\n```\n"
                f"Natural language: {sample_nlp}"
            )

        context_rows = self._column_context_markdown(column_context)
        schema_line = schema_table or "Unknown"

        return (
            "# Task\n"
            "Convert the SQL query into concise business-friendly natural language.\n\n"
            "# Instructions\n"
            "- Return only plain text (no markdown in output).\n"
            "- Keep response to 1-2 sentences.\n"
            "- Mention measures, filters, date window, joins, grouping, ordering, and limit when present.\n"
            "- Use column descriptions to expand abbreviations/business meaning when available.\n\n"
            "# Source Table\n"
            f"`{schema_line}`\n\n"
            "# Column Metadata Context\n"
            f"{context_rows}\n\n"
            "# SQL Query\n"
            f"```sql\n{sql_text}\n```\n\n"
            "# In-Context Examples\n"
            + "\n\n".join(examples)
            + "\n\n# Output\nNatural language summary:"
        )

    async def _load_column_context(self, schema_table: str) -> list[dict[str, Any]]:
        table_name = self._table_name(schema_table)
        if not table_name:
            return []

        cached = self._column_context_cache.get(table_name)
        if cached is not None:
            return cached

        grouped = await self._store.list_column_details_by_tables([table_name])
        direct = grouped.get(table_name) if grouped else None
        if direct:
            self._cache_put(self._column_context_cache, table_name, direct)
            return direct

        for key, value in grouped.items():
            if str(key).strip().upper() == table_name:
                self._cache_put(self._column_context_cache, table_name, value)
                return value

        backend_grouped = await self._store.list_backend_column_details_by_tables([table_name])
        backend_direct = backend_grouped.get(table_name) if backend_grouped else None
        if backend_direct:
            self._cache_put(self._column_context_cache, table_name, backend_direct)
            return backend_direct

        for key, value in backend_grouped.items():
            if str(key).strip().upper() == table_name:
                self._cache_put(self._column_context_cache, table_name, value)
                return value
        self._cache_put(self._column_context_cache, table_name, [])
        return []

    @staticmethod
    def _table_name(schema_table: str) -> str:
        if not schema_table:
            return ""
        cleaned = schema_table.strip().replace('"', "")
        if not cleaned:
            return ""
        if "." in cleaned:
            cleaned = cleaned.split(".")[-1]
        return cleaned.upper()

    def _column_context_markdown(self, column_context: list[dict[str, Any]]) -> str:
        if not column_context:
            return "_No column metadata available._"

        rows = ["| Column | Description | Primary/Foreign Key |", "|---|---|---|"]
        for item in column_context[:80]:
            name = self._md_cell(str(item.get("name") or ""))
            desc = self._md_cell(str(item.get("description") or ""))
            pkfk = self._md_cell(str(item.get("primary_foreign_key") or ""))
            rows.append(f"| {name} | {desc} | {pkfk} |")
        return "\n".join(rows)

    @staticmethod
    def _md_cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ").strip()

    @staticmethod
    def _fallback_sql_summary(sql_text: str) -> str:
        condensed = " ".join(sql_text.split())
        return f"SQL query over enterprise data: {condensed[:280]}{'...' if len(condensed) > 280 else ''}"

    async def _lookup_people_by_email(self, email: str) -> tuple[dict[str, Any], int]:
        if not email:
            return {}, 0

        cached = self._people_cache.get(email)
        if cached is not None:
            return cached, 0

        url = self._settings.people_search_url_template.format(email=quote_plus(email))
        timeout = aiohttp.ClientTimeout(total=self._settings.people_search_timeout_ms / 1000)
        headers = {"Accept": "application/json"}

        extras = self._settings.people_search_extra_headers_json.strip()
        if extras:
            try:
                parsed = json.loads(extras)
                if isinstance(parsed, dict):
                    headers.update({str(key): str(value) for key, value in parsed.items()})
            except json.JSONDecodeError:
                pass

        retries = 0
        for attempt in range(max(0, self._settings.data_usage_nlp_max_retries) + 1):
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, ssl=self._settings.people_search_verify_ssl) as resp:
                    if resp.status == 429:
                        if attempt >= self._settings.data_usage_nlp_max_retries:
                            return {}, retries
                        retries += 1
                        await asyncio.sleep(self._backoff_seconds(attempt))
                        continue

                    if resp.status >= 400:
                        return {}, retries
                    data = await resp.json(content_type=None)

            docs = (((data or {}).get("people") or {}).get("docs") or [])
            if not docs:
                self._cache_put(self._people_cache, email, {})
                return {}, retries
            first = docs[0] if isinstance(docs[0], dict) else {}
            soeid = self._first_item(first.get("soeid"))
            hierarchy = self._first_item(first.get("managedsegmenthierarchy"))
            business_title = self._first_item(first.get("ql_businesscardtitle"))
            payload = {
                "soeid": soeid,
                "managedsegmenthierarchy": hierarchy,
                "business_title": business_title,
            }
            self._cache_put(self._people_cache, email, payload)
            return payload, retries

        return {}, retries

    @staticmethod
    def _first_item(value: Any) -> str:
        if isinstance(value, list):
            return str(value[0]) if value else ""
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _parse_managed_segments(hierarchy: str) -> dict[int, str]:
        result: dict[int, str] = {}
        if not hierarchy:
            return result

        for token in hierarchy.split(";"):
            item = token.strip()
            if not item or item == "#":
                continue
            level_match = re.search(r"\[L(\d{1,2})\]", item, flags=re.IGNORECASE)
            if not level_match:
                continue
            level = int(level_match.group(1))
            if level < 1 or level > 12:
                continue

            if "#" in item:
                _, tail = item.split("#", 1)
            else:
                tail = item
            name = re.sub(r"\[L\d{1,2}\]", "", tail, flags=re.IGNORECASE).strip()
            name = re.sub(r"\s+", " ", name)
            if not name:
                continue
            result[level] = name
        return result

    @staticmethod
    def _parse_table_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                raw_items = parsed if isinstance(parsed, list) else [item.strip() for item in text.split(",")]
            except json.JSONDecodeError:
                raw_items = [item.strip() for item in text.split(",")]
        else:
            raw_items = [value]

        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            table = str(item or "").strip()
            if not table:
                continue
            key = table.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(table)
        return normalized

    def _backoff_seconds(self, attempt: int) -> float:
        base = max(1, self._settings.data_usage_nlp_backoff_initial_ms) / 1000
        cap = max(base, self._settings.data_usage_nlp_backoff_max_ms / 1000)
        delay = min(cap, base * (2**attempt))
        jitter = 0.8 + random.random() * 0.4
        return delay * jitter

    @staticmethod
    def _is_rate_limited_exception(exc: Exception) -> bool:
        text = str(exc).lower()
        return "429" in text or "rate limit" in text or "resource exhausted" in text or "quota" in text

    @staticmethod
    def _cache_put(cache: dict[str, Any], key: str, value: Any, max_entries: int = 5000) -> None:
        cache[key] = value
        if len(cache) > max_entries:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)
