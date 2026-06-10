from __future__ import annotations

import json
import re
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from app.config import Settings
from app.core.sql_utils import extract_tables
from app.core.store import SQLStore
from app.models import QueryHistoryEntry
from app.services.directory import DirectoryService
from app.services.embeddings import EmbeddingService


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(tok) > 2}


class DiscoveryService:
    def __init__(
        self,
        settings: Settings,
        store: SQLStore,
        embeddings: EmbeddingService,
        directory: DirectoryService | None = None,
    ):
        self._settings = settings
        self._store = store
        self._embeddings = embeddings
        self._directory = directory or DirectoryService(settings, store)
        self._directory_payload_cache: dict[str, dict[str, Any]] = {}

    async def role_context(
        self,
        soeid: str,
        limit: int = 10,
        segment_values: list[str] | None = None,
    ) -> tuple[str, list[QueryHistoryEntry], dict[str, Any]]:
        effective_limit = max(1, int(limit))
        candidate_limit = max(200, effective_limit * 20)
        profile = await self._resolve_profile_context(soeid, segment_values=segment_values)
        effective_role = profile["role"]
        business_title = profile["business_title"]
        segments = profile["segments"]

        nlp_rows: list[dict[str, Any]] = []
        if business_title:
            nlp_rows.extend(
                await self._store.list_nlp_queries_by_business_title(
                    business_title,
                    limit=candidate_limit,
                )
            )
        if segments:
            nlp_rows.extend(
                await self._store.list_nlp_queries_by_segments(
                    segments,
                    limit=candidate_limit,
                )
            )
        nlp_rows = self._dedupe_rows(nlp_rows)

        mapped = [self._nlp_row_to_query_history(row, role_id=effective_role) for row in nlp_rows]
        queries = [item for item in mapped if item is not None]
        queries = self._apply_token_budget(queries, self._settings.vertex_context_max_tokens)[
            :effective_limit
        ]
        if queries:
            metadata = await self._build_metadata(queries)
            metadata["segment_scope"] = segments
            metadata["business_title"] = business_title
            return effective_role, queries, metadata

        user = await self._store.get_user(soeid)
        if user:
            role_users = await self._store.list_users_by_role(user.role_id)
            user_ids = [entry.soeid for entry in role_users]
            common_rows = await self._store.list_common_queries_for_users(
                user_ids, limit=candidate_limit
            )
            mapped = [
                self._legacy_row_to_query_history(item, role_id=effective_role)
                for item in common_rows
            ]
            queries = [item for item in mapped if item is not None]
            queries = self._apply_token_budget(queries, self._settings.vertex_context_max_tokens)[
                :effective_limit
            ]
            if not queries:
                fallback = await self._store.list_queries_by_role(
                    user.role_id, limit=effective_limit
                )
                queries = self._apply_token_budget(
                    fallback, self._settings.vertex_context_max_tokens
                )[:effective_limit]
            metadata = await self._build_metadata(queries)
            metadata["segment_scope"] = segments
            metadata["business_title"] = business_title
            return effective_role, queries, metadata

        metadata = {
            "tables": [],
            "columns": {},
            "table_details": {},
            "segment_scope": segments,
            "business_title": business_title,
        }
        return effective_role, [], metadata

    async def similar_queries(
        self,
        soeid: str,
        prompt: str,
        limit: int = 5,
        segment_values: list[str] | None = None,
    ) -> list[QueryHistoryEntry]:
        effective_limit = max(1, int(limit))
        profile = await self._resolve_profile_context(soeid, segment_values=segment_values)
        role_id = profile["role"]
        business_title = profile["business_title"]
        segment_scope = profile["segments"]

        lexical_top_k = max(effective_limit, self._settings.discovery_similar_lexical_top_k)
        lexical_rows = await self._store.list_nlp_queries_by_full_text(prompt, limit=lexical_top_k)
        query_embedding, _ = await self._embeddings.embed_query(prompt)
        if not query_embedding and not lexical_rows:
            _, queries, _ = await self.role_context(
                soeid, limit=max(200, effective_limit * 10), segment_values=segment_scope
            )
            query_terms = _tokens(prompt)

            def lexical_score(item: QueryHistoryEntry) -> int:
                text = f"{item.sql2text} {item.sql_text} {' '.join(item.tables)}"
                return len(query_terms & _tokens(text))

            ranked = sorted(queries, key=lexical_score, reverse=True)
            filtered = [q for q in ranked if lexical_score(q) > 0]
            return self._apply_token_budget(filtered, self._settings.vertex_context_max_tokens)[
                :effective_limit
            ]

        semantic_rows: list[dict[str, Any]] = []
        if query_embedding:
            embedding_top_k = max(effective_limit, self._settings.discovery_similar_embedding_top_k)
            semantic_rows = await self._store.list_nlp_queries_by_embedding(
                query_embedding, limit=embedding_top_k
            )
        semantic_rows = self._fuse_rows_by_rrf(
            semantic_rows=semantic_rows, lexical_rows=lexical_rows
        )

        business_rows: list[dict[str, Any]] = []
        segment_rows: list[dict[str, Any]] = []
        if business_title:
            business_rows = await self._store.list_nlp_queries_by_business_title(
                business_title,
                limit=max(effective_limit, self._settings.discovery_similar_business_title_top_k),
            )
        if segment_scope:
            segment_rows = await self._store.list_nlp_queries_by_segments(
                segment_scope,
                limit=max(effective_limit, self._settings.discovery_similar_segment_top_k),
            )

        ranked_rows = self._rank_rows_by_context(
            semantic_rows=semantic_rows,
            business_rows=business_rows,
            segment_rows=segment_rows,
            business_title=business_title,
            segment_scope=segment_scope,
        )
        if not ranked_rows:
            return []

        mapped = [self._nlp_row_to_query_history(row, role_id=role_id) for row in ranked_rows]
        queries = [item for item in mapped if item is not None]
        queries = self._apply_token_budget(queries, self._settings.vertex_context_max_tokens)
        return queries[:effective_limit]

    async def metadata_for_tables(self, tables: list[str]) -> dict[str, Any]:
        normalized = self._normalize_table_names(tables)
        table_rows = (
            await self._store.list_discovery_master_by_tables(normalized) if normalized else []
        )
        columns = await self._store.list_column_details_by_tables(normalized) if normalized else {}
        backend_table_rows = (
            await self._store.list_backend_metadata_tables_by_tables(normalized)
            if normalized
            else []
        )
        backend_columns = (
            await self._store.list_backend_column_details_by_tables(normalized)
            if normalized
            else {}
        )
        table_details = {
            self._value(row, "TABLE_NAME"): {
                "zone": self._value(row, "ZONE"),
                "description": self._value(row, "TABLE_DESCRIPTION"),
                "domain": self._value(row, "DOMAIN") or self._value(row, "STANDARDIZED_DOMAIN"),
                "target_db": self._value(row, "TARGET_DB"),
                "source_system": self._value(row, "SOURCE_SYSTEM"),
                "region": self._value(row, "REGION"),
                "country": self._value(row, "COUNTRY"),
                "pii": self._value(row, "PII"),
                "criticalDataElement": self._value(row, "CRITICAL_DATA_ELEMENT"),
                "asset_id": self._value(row, "ASSET_ID"),
            }
            for row in table_rows
            if self._value(row, "TABLE_NAME")
        }
        for row in backend_table_rows:
            table_name = self._value(row, "TABLE_NAME")
            if not table_name or table_name in table_details:
                continue
            table_details[table_name] = {
                "zone": self._value(row, "ENGINE"),
                "description": self._value(row, "DESCRIPTION"),
                "domain": self._value(row, "CATALOG_NAME"),
                "target_db": self._value(row, "SCHEMA_NAME"),
                "source_system": self._value(row, "ENGINE"),
                "region": None,
                "country": None,
                "pii": None,
                "criticalDataElement": None,
                "asset_id": None,
                "table_type": self._value(row, "TABLE_TYPE"),
            }
        for table_name, backend_items in backend_columns.items():
            columns.setdefault(table_name, backend_items)
        return {
            "tables": normalized,
            "columns": columns,
            "table_details": table_details,
        }

    def _rank_rows_by_context(
        self,
        semantic_rows: list[dict[str, Any]],
        business_rows: list[dict[str, Any]],
        segment_rows: list[dict[str, Any]],
        business_title: str,
        segment_scope: list[str],
    ) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        seen: set[str] = set()
        segment_scope_lower = {value.lower() for value in segment_scope if value}
        business_title_lower = business_title.lower()

        semantic_scores: list[tuple[float, float, float, int, int, float, dict[str, Any]]] = []
        for row in semantic_rows:
            hybrid_score = float(self._value(row, "RRF_SCORE", 0.0) or 0.0)
            similarity = float(self._value(row, "COSINE_SIMILARITY", 0.0) or 0.0)
            lexical_score = float(self._value(row, "FTS_SCORE", 0.0) or 0.0)
            row_business = self._row_business_title(row)
            business_match = int(
                bool(business_title_lower and row_business == business_title_lower)
            )
            row_segments = self._row_segment_values(row)
            segment_match = (
                sum(1 for segment in segment_scope_lower if segment in row_segments)
                if segment_scope_lower
                else 0
            )
            updated_ts = self._updated_ts(row)
            semantic_scores.append(
                (
                    hybrid_score,
                    similarity,
                    lexical_score,
                    business_match,
                    segment_match,
                    updated_ts,
                    row,
                )
            )

        semantic_scores.sort(
            key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]), reverse=True
        )
        for _, _, _, _, _, _, row in semantic_scores:
            key = self._row_key(row)
            if key in seen:
                continue
            seen.add(key)
            ranked.append(row)

        business_ranked = sorted(
            business_rows,
            key=lambda row: (
                int(self._row_business_title(row) == business_title_lower),
                self._updated_ts(row),
            ),
            reverse=True,
        )
        for row in business_ranked:
            key = self._row_key(row)
            if key in seen:
                continue
            seen.add(key)
            ranked.append(row)

        segment_ranked = sorted(
            segment_rows,
            key=lambda row: (
                sum(
                    1 for segment in segment_scope_lower if segment in self._row_segment_values(row)
                ),
                self._updated_ts(row),
            ),
            reverse=True,
        )
        for row in segment_ranked:
            key = self._row_key(row)
            if key in seen:
                continue
            seen.add(key)
            ranked.append(row)
        return ranked

    @staticmethod
    def _row_segment_values(row: dict[str, Any]) -> set[str]:
        values: set[str] = set()
        for idx in range(1, 13):
            value = DiscoveryService._value(
                row, f"MANAGED_SEGMENT_L{idx}"
            ) or DiscoveryService._value(row, f"managed_segment_l{idx}")
            text = str(value or "").strip().lower()
            if text:
                values.add(text)
        return values

    async def _resolve_role(self, soeid: str) -> str:
        profile = await self._resolve_profile_context(soeid, segment_values=[])
        return profile["role"]

    async def _resolve_profile_context(
        self, soeid: str, segment_values: list[str] | None
    ) -> dict[str, Any]:
        business_title = ""
        directory_segments: list[str] = []
        try:
            payload = await self._directory_payload(soeid)
            info = payload.get("UserDirectoryInformation", {}) if isinstance(payload, dict) else {}
            work = info.get("UsersWorkInformation", {}) if isinstance(info, dict) else {}
            business_title = str(work.get("BusinessTitle") or "").strip()
            raw_last_two = (
                work.get("ManagedSegmentLastTwoLevels", []) if isinstance(work, dict) else []
            )
            if isinstance(raw_last_two, list):
                directory_segments = [
                    str(item).strip() for item in raw_last_two if str(item).strip()
                ]
        except Exception:
            business_title = ""
            directory_segments = []

        normalized_segments = self._normalize_segment_values(
            (segment_values or []) + directory_segments
        )
        role = business_title or await self._resolve_role_fallback(soeid)
        return {
            "role": role or self._settings.directory_default_role,
            "business_title": business_title,
            "segments": normalized_segments,
        }

    async def _resolve_role_fallback(self, soeid: str) -> str:
        user = await self._store.get_user(soeid)
        if user and user.role_id:
            return user.role_id
        return self._settings.directory_default_role

    async def _directory_payload(self, soeid: str) -> dict[str, Any]:
        key = str(soeid or "").strip()
        if not key:
            return {}
        cached = self._directory_payload_cache.get(key)
        if cached is not None:
            return cached
        payload = await self._directory.get_user_directory_information(key)
        self._directory_payload_cache[key] = payload
        if len(self._directory_payload_cache) > 2000:
            oldest_key = next(iter(self._directory_payload_cache))
            self._directory_payload_cache.pop(oldest_key, None)
        return payload

    async def _build_metadata(self, queries: list[QueryHistoryEntry]) -> dict[str, Any]:
        raw_tables = [table for query in queries for table in query.tables]
        return await self.metadata_for_tables(raw_tables)

    @staticmethod
    def _nlp_row_to_query_history(row: dict[str, Any], role_id: str) -> QueryHistoryEntry | None:
        sql_text = str(DiscoveryService._value(row, "QUERY") or "").strip()
        if not sql_text:
            return None
        schema_table = str(DiscoveryService._value(row, "SCHEMA_TABLE") or "").strip()
        table_list = DiscoveryService._extract_row_tables(row)
        tables = DiscoveryService._merge_tables(extract_tables(sql_text), table_list)
        if not tables and schema_table:
            tables = [schema_table]

        created_at = datetime.now(UTC)
        raw_created = DiscoveryService._value(row, "UPDATED_AT")
        if isinstance(raw_created, str):
            with suppress(ValueError):
                created_at = datetime.fromisoformat(raw_created)

        summary = str(DiscoveryService._value(row, "QUERY_IN_NLP") or "").strip()
        if not summary:
            preview = " ".join(sql_text.split())
            summary = preview[:150] + ("..." if len(preview) > 150 else "")

        return QueryHistoryEntry(
            query_id=str(DiscoveryService._value(row, "ID") or ""),
            soeid=str(DiscoveryService._value(row, "SOEID") or ""),
            role_id=role_id,
            engine="starburst",
            sql_text=sql_text,
            sql2text=summary,
            tables=tables,
            created_at=created_at,
            status="succeeded",
        )

    @staticmethod
    def _legacy_row_to_query_history(row: dict[str, Any], role_id: str) -> QueryHistoryEntry | None:
        sql_text = str(DiscoveryService._value(row, "QUERY") or "").strip()
        if not sql_text:
            return None
        schema_table = str(DiscoveryService._value(row, "SCHEMA_TABLE") or "").strip()
        table_list = DiscoveryService._extract_row_tables(row)
        tables = DiscoveryService._merge_tables(extract_tables(sql_text), table_list)
        if not tables and schema_table:
            tables = [schema_table]

        created_at = datetime.now(UTC)
        raw_created = DiscoveryService._value(row, "UPDATED_AT")
        if isinstance(raw_created, str):
            with suppress(ValueError):
                created_at = datetime.fromisoformat(raw_created)

        preview = " ".join(sql_text.split())
        sql2text = preview[:150] + ("..." if len(preview) > 150 else "")

        return QueryHistoryEntry(
            query_id=str(DiscoveryService._value(row, "ID") or ""),
            soeid=str(DiscoveryService._value(row, "SOEID") or ""),
            role_id=role_id,
            engine=str(DiscoveryService._value(row, "TOOL") or "starburst"),
            sql_text=sql_text,
            sql2text=sql2text,
            tables=tables,
            created_at=created_at,
            status="succeeded",
        )

    @staticmethod
    def _extract_row_tables(row: dict[str, Any]) -> list[str]:
        raw = DiscoveryService._value(row, "ALL_QUERY_TABLES")
        if raw is None:
            raw = DiscoveryService._value(row, "all_query_tables")
        if raw is None:
            raw = DiscoveryService._value(row, "TABLES_JSON")
        if raw is None:
            raw = DiscoveryService._value(row, "tables_json")
        if raw is None:
            return []
        if isinstance(raw, list):
            values = raw
        elif isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                values = (
                    parsed
                    if isinstance(parsed, list)
                    else [item.strip() for item in text.split(",")]
                )
            except json.JSONDecodeError:
                values = [item.strip() for item in text.split(",")]
        else:
            values = [raw]

        normalized: list[str] = []
        seen: set[str] = set()
        for item in values:
            if isinstance(item, dict):
                parts = [str(item.get(key) or "").strip() for key in ("catalog", "schema", "table")]
                value = ".".join(part for part in parts if part)
            elif isinstance(item, (list, tuple)):
                parts = [str(part or "").strip() for part in item[:3]]
                value = ".".join(part for part in parts if part)
            else:
                value = str(item or "").strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(value)
        return normalized

    @staticmethod
    def _merge_tables(primary: list[str], secondary: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for table in primary + secondary:
            value = str(table or "").strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
        return merged

    @staticmethod
    def _normalize_table_name(table: str) -> str:
        text = table.strip().replace('"', "")
        if not text:
            return ""
        if "." in text:
            text = text.split(".")[-1]
        return text.upper()

    def _normalize_table_names(self, tables: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for table in tables:
            item = self._normalize_table_name(str(table or ""))
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        normalized.sort()
        return normalized

    @staticmethod
    def _normalize_segment_values(values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    def _apply_token_budget(
        self, queries: list[QueryHistoryEntry], budget_tokens: int
    ) -> list[QueryHistoryEntry]:
        if budget_tokens <= 0:
            return queries
        selected: list[QueryHistoryEntry] = []
        used_tokens = 0
        for query in queries:
            estimate = self._estimate_query_tokens(query)
            if selected and used_tokens + estimate > budget_tokens:
                break
            selected.append(query)
            used_tokens += estimate
        return selected

    @staticmethod
    def _estimate_query_tokens(query: QueryHistoryEntry) -> int:
        text = f"{query.sql_text}\n{query.sql2text}\n{' '.join(query.tables)}"
        return max(1, (len(text) // 4) + 32)

    @staticmethod
    def _value(row: dict[str, Any], key: str, default: Any = None) -> Any:
        if key in row:
            return row[key]
        lower = key.lower()
        if lower in row:
            return row[lower]
        upper = key.upper()
        if upper in row:
            return row[upper]
        return default

    @staticmethod
    def _row_business_title(row: dict[str, Any]) -> str:
        value = DiscoveryService._value(row, "BUSINESS_TITLE") or DiscoveryService._value(
            row, "business_title"
        )
        return str(value or "").strip().lower()

    @staticmethod
    def _updated_ts(row: dict[str, Any]) -> float:
        raw = DiscoveryService._value(row, "UPDATED_AT")
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw).timestamp()
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _row_key(row: dict[str, Any]) -> str:
        query_id = str(DiscoveryService._value(row, "ID") or "").strip()
        if query_id:
            return f"id:{query_id}"
        query = str(DiscoveryService._value(row, "QUERY") or "").strip().lower()
        soeid = str(DiscoveryService._value(row, "SOEID") or "").strip().lower()
        return f"query:{soeid}:{query}"

    @classmethod
    def _dedupe_rows(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            key = cls._row_key(row)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _fuse_rows_by_rrf(
        self,
        semantic_rows: list[dict[str, Any]],
        lexical_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not semantic_rows and not lexical_rows:
            return []

        rrf_k = max(1, int(self._settings.discovery_similar_rrf_k))
        fused_by_key: dict[str, dict[str, Any]] = {}

        for rank, row in enumerate(semantic_rows, start=1):
            key = self._row_key(row)
            candidate = fused_by_key.setdefault(key, dict(row))
            candidate["RRF_SCORE"] = float(self._value(candidate, "RRF_SCORE", 0.0) or 0.0) + (
                1.0 / (rrf_k + rank)
            )

        for rank, row in enumerate(lexical_rows, start=1):
            key = self._row_key(row)
            candidate = fused_by_key.setdefault(key, dict(row))
            candidate["RRF_SCORE"] = float(self._value(candidate, "RRF_SCORE", 0.0) or 0.0) + (
                1.0 / (rrf_k + rank)
            )

            existing_cosine = float(self._value(candidate, "COSINE_SIMILARITY", 0.0) or 0.0)
            lexical_cosine = float(self._value(row, "COSINE_SIMILARITY", 0.0) or 0.0)
            if lexical_cosine > existing_cosine:
                candidate["COSINE_SIMILARITY"] = lexical_cosine

            existing_fts = float(self._value(candidate, "FTS_SCORE", 0.0) or 0.0)
            lexical_fts = float(self._value(row, "FTS_SCORE", 0.0) or 0.0)
            if lexical_fts > existing_fts:
                candidate["FTS_SCORE"] = lexical_fts

        fused = list(fused_by_key.values())
        fused.sort(
            key=lambda row: (
                float(self._value(row, "RRF_SCORE", 0.0) or 0.0),
                float(self._value(row, "COSINE_SIMILARITY", 0.0) or 0.0),
                float(self._value(row, "FTS_SCORE", 0.0) or 0.0),
                self._updated_ts(row),
            ),
            reverse=True,
        )
        return fused
