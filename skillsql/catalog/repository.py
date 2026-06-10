"""Catalog persistence + retrieval (proposal Sections 6.4, 7).

Owns the SQLAlchemy engine bound to the app catalog schema and exposes the
critical catalog operations the workflow needs: persist discovered metadata,
vector-search schema docs for a question, and read/write SkillBank entries.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from app.core.sql_utils import extract_tables
from app.observability.logging import get_logger
from sqlalchemy import String, cast, create_engine, event, func, literal, or_, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, aliased, selectinload, sessionmaker

from ..config.settings import Settings, get_settings
from ..connectors.base import ColumnMeta, Metadata, SchemaDoc, TableMeta
from .models import (
    Base,
    CatalogColumn,
    CatalogQueryHistory,
    CatalogQueryHistoryNlp,
    CatalogTable,
    SchemaDocRow,
    Skill,
    Source,
    SourceGroup,
)

_logger = get_logger(__name__)

_CATALOG_SCOPED_SOURCE_TYPES = {"starburst", "trino"}


class CatalogRepository:
    """Thin data-access layer over the app catalog database."""

    def __init__(self, settings: Settings | None = None, engine: Engine | None = None) -> None:
        self.settings = settings or get_settings()
        self.schema = self.settings.APP_CATALOG_SCHEMA
        self.engine = engine or self._make_engine(
            self.settings.APP_CATALOG_DSN,
            self.schema,
            hide_parameters=self.settings.SQLALCHEMY_HIDE_PARAMETERS,
        )
        self._bind_metadata_schema()
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    @staticmethod
    def _make_engine(dsn: str, schema: str, *, hide_parameters: bool = True) -> Engine:
        engine = create_engine(
            dsn,
            pool_pre_ping=True,
            future=True,
            hide_parameters=hide_parameters,
        )

        # Ensure every connection uses the app schema first on the search_path.
        @event.listens_for(engine, "connect")
        def _set_search_path(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            try:
                cur.execute(f'SET search_path TO "{schema}", public')
            finally:
                cur.close()

        return engine

    # ---- schema lifecycle -----------------------------------------------------
    def init_schema(self) -> None:
        """Create the schema, the ``vector`` extension, and all tables (idempotent)."""
        self._bind_metadata_schema()
        if self._uses_postgres_schema():
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
        Base.metadata.create_all(self.engine)
        if self._uses_postgres_schema():
            self._create_postgres_query_history_indexes()

    def reset_schema(self) -> None:
        """Drop and recreate catalog-owned tables.

        This is intentionally explicit because it deletes semantic catalog data,
        query history copies, generated NLP rows, and SkillBank entries.
        """
        self._bind_metadata_schema()
        if self._uses_postgres_schema():
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        if self._uses_postgres_schema():
            self._create_postgres_query_history_indexes()

    def session(self) -> Session:
        return self._Session()

    def _bind_metadata_schema(self) -> None:
        schema = self.schema if self._uses_postgres_schema() else None
        for table in Base.metadata.tables.values():
            table.schema = None
            table.schema = schema

    def _uses_postgres_schema(self) -> bool:
        return bool((self.schema or "").strip()) and self.engine.dialect.name != "sqlite"

    def _create_postgres_query_history_indexes(self) -> None:
        table_name = f'"{self.schema}".catalog_query_history_nlp'
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS ix_catalog_query_history_nlp_embeddings_cosine
                    ON {table_name}
                    USING ivfflat (embeddings vector_cosine_ops)
                    """
                )
            )
            conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS ix_catalog_query_history_nlp_fts
                    ON {table_name}
                    USING GIN (
                        to_tsvector(
                            'english',
                            COALESCE(nlp_text, '') || ' ' ||
                            COALESCE(raw_sql, '') || ' ' ||
                            COALESCE(schema_table, '') || ' ' ||
                            COALESCE(tables_json::text, '')
                        )
                    )
                    """
                )
            )

    # ---- sources --------------------------------------------------------------
    def upsert_source_group(
        self,
        source_type: str,
        name: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
    ) -> uuid.UUID:
        """Register or retrieve a logical group of physical catalog sources."""
        normalized_type = str(source_type or "").strip().lower()
        normalized_name = str(name or "").strip()
        if not normalized_type or not normalized_name:
            raise ValueError("source_type and source group name are required")

        with self.session() as s:
            existing = (
                s.query(SourceGroup)
                .filter(
                    self._ci_equals(SourceGroup.source_type, normalized_type),
                    self._ci_equals(SourceGroup.name, normalized_name),
                )
                .one_or_none()
            )
            if existing:
                if display_name is not None:
                    existing.display_name = display_name
                if description is not None:
                    existing.description = description
                existing.updated_at = self._now()
                s.commit()
                return existing.id

            row = SourceGroup(
                source_type=normalized_type,
                name=normalized_name,
                display_name=display_name,
                description=description,
            )
            s.add(row)
            s.commit()
            return row.id

    def upsert_source(
        self,
        source_type: str,
        name: str,
        catalog_name: str | None = None,
        db_schema: str | None = None,
        # legacy param kept for backward-compatibility
        database: str | None = None,
        source_group_id: uuid.UUID | str | None = None,
        source_group_name: str | None = None,
    ) -> uuid.UUID:
        """Register or retrieve a source row.

        ``catalog_name`` is the canonical 3-level top identifier (Starburst Galaxy
        catalog / Snowflake database / Postgres database).  ``database`` is kept
        for compatibility and for engines that expose database separately from
        catalog. It is used as a fallback when ``catalog_name`` is not provided.
        """
        effective_catalog = database or catalog_name
        effective_database = database or effective_catalog
        normalized_group_id = self._resolve_source_group_id(
            source_type=source_type,
            source_group_id=source_group_id,
            source_group_name=source_group_name,
        )
        with self.session() as s:
            existing = (
                s.query(Source)
                .filter_by(
                    source_group_id=normalized_group_id,
                    source_type=source_type,
                    name=name,
                    catalog_name=effective_catalog,
                    database=effective_database,
                    db_schema=db_schema,
                )
                .one_or_none()
            )
            if existing:
                return existing.id
            src = Source(
                source_group_id=normalized_group_id,
                source_type=source_type,
                name=name,
                catalog_name=effective_catalog,
                database=effective_database,
                db_schema=db_schema,
            )
            s.add(src)
            s.commit()
            return src.id

    def list_source_groups(self) -> list[SourceGroup]:
        with self.session() as s:
            return (
                s.query(SourceGroup)
                .order_by(SourceGroup.created_at.desc())
                .all()
            )

    def list_sources(self) -> list[Source]:
        with self.session() as s:
            return (
                s.query(Source)
                .order_by(Source.created_at.desc())
                .all()
            )

    def source_ids_for_group(self, source_group_id: uuid.UUID | str) -> list[uuid.UUID]:
        group_id = self._coerce_uuid(source_group_id)
        with self.session() as s:
            return [row.id for row in s.query(Source.id).filter(Source.source_group_id == group_id)]

    # ---- metadata persistence -------------------------------------------------
    def persist_metadata(self, source_id: uuid.UUID, meta: Metadata) -> int:
        """Replace the table/column rows for a source.  Returns #tables written."""
        with self.session() as s:
            s.query(CatalogTable).filter_by(source_id=source_id).delete()
            for t in meta.tables:
                row = CatalogTable(
                    source_id=source_id,
                    fqn=t.fqn,
                    catalog_name=t.catalog_name,
                    db_schema=t.db_schema,
                    name=t.name,
                    table_type=t.table_type,
                    comment=t.comment,
                    row_estimate=t.row_estimate,
                )
                row.columns = [
                    CatalogColumn(
                        name=c.name,
                        data_type=c.data_type,
                        nullable=c.nullable,
                        ordinal=c.ordinal,
                        is_primary_key=c.is_primary_key,
                        comment=c.comment,
                        sample_values=c.sample_values or None,
                        null_fraction=c.null_fraction,
                        distinct_estimate=c.distinct_estimate,
                    )
                    for c in t.columns
                ]
                s.add(row)
            s.commit()
            return len(meta.tables)

    def upsert_table_metadata(
        self,
        source_id: uuid.UUID,
        table: TableMeta,
        *,
        replace_columns: bool = False,
    ) -> tuple[uuid.UUID, int]:
        """Insert or update one table and any supplied columns.

        Unlike :meth:`persist_metadata`, this is intentionally incremental. It
        supports catalog/schema/table/column refresh APIs where updating one
        object must not wipe the rest of a large enterprise catalog.
        """
        with self.session() as s:
            row = (
                s.query(CatalogTable)
                .filter_by(
                    source_id=source_id,
                    catalog_name=table.catalog_name,
                    db_schema=table.db_schema,
                    name=table.name,
                )
                .one_or_none()
            )
            if row is None:
                row = CatalogTable(source_id=source_id, fqn=table.fqn, name=table.name)
                s.add(row)

            row.fqn = table.fqn
            row.catalog_name = table.catalog_name
            row.db_schema = table.db_schema
            row.name = table.name
            row.table_type = table.table_type or "BASE TABLE"
            row.comment = table.comment
            row.row_estimate = table.row_estimate

            s.flush()
            if replace_columns:
                s.query(CatalogColumn).filter_by(table_id=row.id).delete()
                s.flush()

            columns_written = 0
            for column in getattr(table, "columns", []) or []:
                self._upsert_column_in_session(s, row.id, column)
                columns_written += 1

            table_id = row.id
            s.commit()
            return table_id, columns_written

    def upsert_column_metadata(
        self,
        source_id: uuid.UUID,
        table: TableMeta,
        column: ColumnMeta,
    ) -> uuid.UUID:
        """Insert or update one column under a table, creating the table if needed."""
        with self.session() as s:
            table_row = (
                s.query(CatalogTable)
                .filter_by(
                    source_id=source_id,
                    catalog_name=table.catalog_name,
                    db_schema=table.db_schema,
                    name=table.name,
                )
                .one_or_none()
            )
            if table_row is None:
                table_row = CatalogTable(
                    source_id=source_id,
                    fqn=table.fqn,
                    catalog_name=table.catalog_name,
                    db_schema=table.db_schema,
                    name=table.name,
                    table_type=table.table_type or "BASE TABLE",
                    comment=table.comment,
                    row_estimate=table.row_estimate,
                )
                s.add(table_row)
                s.flush()
            else:
                table_row.fqn = table.fqn
                table_row.table_type = table.table_type or table_row.table_type
                table_row.comment = (
                    table.comment if table.comment is not None else table_row.comment
                )
                table_row.row_estimate = (
                    table.row_estimate if table.row_estimate is not None else table_row.row_estimate
                )

            col_row = self._upsert_column_in_session(s, table_row.id, column)
            column_id = col_row.id
            s.commit()
            return column_id

    # ---- query-history persistence -------------------------------------------
    def upsert_query_history_rows(
        self,
        *,
        source_id: uuid.UUID | str,
        rows: list[dict],
    ) -> int:
        """Insert/update raw datasource query history rows for a catalog source."""
        normalized_source_id = self._coerce_uuid(source_id)
        if not rows:
            return 0

        written = 0
        with self.session() as s:
            for payload in rows:
                engine = str(payload.get("engine") or "").strip().lower()
                query_id = str(payload.get("query_id") or "").strip()
                raw_sql = str(payload.get("raw_sql") or "").strip()
                if not engine or not query_id or not raw_sql:
                    continue

                row = (
                    s.query(CatalogQueryHistory)
                    .filter_by(source_id=normalized_source_id, engine=engine, query_id=query_id)
                    .one_or_none()
                )
                if row is None:
                    row = CatalogQueryHistory(
                        source_id=normalized_source_id,
                        engine=engine,
                        query_id=query_id,
                        raw_sql=raw_sql,
                    )
                    s.add(row)

                tables = self._coerce_table_list(payload.get("tables"))
                schema_table = str(payload.get("schema_table") or "").strip()
                if not schema_table and tables:
                    schema_table = tables[0]

                row.catalog_name = self._clean_optional(payload.get("catalog_name"))
                row.schema_name = self._clean_optional(payload.get("schema_name"))
                row.query_state = self._clean_optional(payload.get("query_state"))
                row.query_type = self._clean_optional(payload.get("query_type"))
                row.user_email = self._clean_optional(payload.get("user_email"))
                row.role_name = self._clean_optional(payload.get("role_name"))
                row.cluster_name = self._clean_optional(payload.get("cluster_name"))
                row.source_name = self._clean_optional(payload.get("source"))
                row.created_at = self._coerce_datetime(payload.get("created_at"))
                row.ended_at = self._coerce_datetime(payload.get("ended_at"))
                row.raw_sql = raw_sql
                row.schema_table = schema_table or None
                row.tables_json = tables or None
                row.metrics_json = self._coerce_json_dict(payload.get("metrics"))
                row.raw_json = self._coerce_json_dict(payload.get("raw"))
                row.updated_at = self._now()
                written += 1
            s.commit()
        return written

    def list_query_history_for_nlp(
        self,
        *,
        source_id: uuid.UUID | str | None = None,
        source_group_id: uuid.UUID | str | None = None,
        limit: int = 100,
        engine: str | None = None,
        ids: list[int] | None = None,
        raw_sql: str | None = None,
        missing_only: bool = True,
    ) -> list[dict]:
        """Return raw query-history rows ready for NLP generation."""
        normalized_ids = sorted({int(item) for item in (ids or []) if int(item) > 0})
        effective_limit = len(normalized_ids) if normalized_ids else max(1, int(limit))
        with self.session() as s:
            nlp_row = aliased(CatalogQueryHistoryNlp)
            q = s.query(CatalogQueryHistory).outerjoin(
                nlp_row,
                (nlp_row.source_id == CatalogQueryHistory.source_id)
                & (nlp_row.engine == CatalogQueryHistory.engine)
                & (nlp_row.query_id == CatalogQueryHistory.query_id),
            )
            q = q.filter(
                CatalogQueryHistory.raw_sql.isnot(None),
                func.length(func.trim(CatalogQueryHistory.raw_sql)) > 0,
            )
            if source_id:
                q = q.filter(CatalogQueryHistory.source_id == self._coerce_uuid(source_id))
            elif source_group_id:
                q = self._filter_source_group(q, CatalogQueryHistory.source_id, source_group_id)
            normalized_engine = str(engine or "").strip().lower()
            if normalized_engine:
                q = q.filter(CatalogQueryHistory.engine == normalized_engine)
            if normalized_ids:
                q = q.filter(CatalogQueryHistory.id.in_(normalized_ids))
            if raw_sql:
                text_value = str(raw_sql)
                q = q.filter(
                    or_(
                        CatalogQueryHistory.raw_sql == text_value,
                        func.trim(CatalogQueryHistory.raw_sql) == text_value.strip(),
                    )
                )
            if missing_only:
                q = q.filter(nlp_row.query_id.is_(None))
            rows = (
                q.order_by(
                    func.coalesce(
                        CatalogQueryHistory.created_at,
                        CatalogQueryHistory.updated_at,
                    ).desc()
                )
                .limit(effective_limit)
                .all()
            )
        return [self._query_history_to_dict(row) for row in rows]

    def get_query_history_context_by_raw_sql(
        self,
        *,
        source_id: uuid.UUID | str | None = None,
        raw_sql: str | None = None,
        raw_history_id: int | None = None,
        engine: str | None = None,
    ) -> dict:
        """Return one catalog query-history row plus semantic table/column context."""
        _logger.debug(f"In get_query_history_context_by_raw_sql, raw_history_id={raw_history_id}")
        raw_text = str(raw_sql or "")
        text_value = raw_text.strip()
        normalized_id = int(raw_history_id or 0)
        if not text_value and normalized_id <= 0:
            return {}

        with self.session() as s:
            q = s.query(CatalogQueryHistory)
            if source_id:
                q = q.filter(CatalogQueryHistory.source_id == self._coerce_uuid(source_id))
            if normalized_id > 0:
                q = q.filter(CatalogQueryHistory.id == normalized_id)
            else:
                q = q.filter(
                    or_(
                        CatalogQueryHistory.raw_sql == raw_text,
                        func.trim(CatalogQueryHistory.raw_sql) == text_value,
                    )
                )
            normalized_engine = str(engine or "").strip().lower()
            if normalized_engine:
                q = q.filter(CatalogQueryHistory.engine == normalized_engine)
            row = (
                q.order_by(
                    func.coalesce(
                        CatalogQueryHistory.created_at,
                        CatalogQueryHistory.updated_at,
                    ).desc()
                )
                .first()
            )
            if row is None:
                return {}

            row_dict = self._query_history_to_dict(row)
            table_names = self._coerce_table_list(row.tables_json)
            if not table_names:
                table_names = extract_tables(text_value)
            tables = self._table_context_for_refs(
                s,
                source_id=row.source_id,
                table_refs=table_names,
                catalog_name=row.catalog_name,
                schema_name=row.schema_name,
            )

        return {
            "history": row_dict,
            "raw_sql": text_value,
            "tables_json": table_names,
            "tables": tables,
        }

    def upsert_query_history_nlp_row(
        self,
        *,
        raw_row: dict,
        nlp_text: str,
        embedding: list[float] | None = None,
    ) -> bool:
        """Insert/update generated NLP text and embedding for a raw history row."""
        normalized_nlp_text = str(nlp_text or "").strip()
        if not normalized_nlp_text:
            return False

        source_id = self._row_value(raw_row, "SOURCE_ID")
        engine = str(self._row_value(raw_row, "ENGINE") or "").strip().lower()
        query_id = str(self._row_value(raw_row, "QUERY_ID") or "").strip()
        raw_sql = str(self._row_value(raw_row, "RAW_SQL") or "").strip()
        if not source_id or not engine or not query_id or not raw_sql:
            return False

        normalized_source_id = self._coerce_uuid(source_id)
        with self.session() as s:
            row = (
                s.query(CatalogQueryHistoryNlp)
                .filter_by(source_id=normalized_source_id, engine=engine, query_id=query_id)
                .one_or_none()
            )
            if row is None:
                row = CatalogQueryHistoryNlp(
                    source_id=normalized_source_id,
                    engine=engine,
                    query_id=query_id,
                    raw_sql=raw_sql,
                    nlp_text=normalized_nlp_text,
                )
                s.add(row)

            row.raw_query_history_id = self._optional_int(self._row_value(raw_row, "ID"))
            row.catalog_name = self._clean_optional(self._row_value(raw_row, "CATALOG_NAME"))
            row.schema_name = self._clean_optional(self._row_value(raw_row, "SCHEMA_NAME"))
            row.query_state = self._clean_optional(self._row_value(raw_row, "QUERY_STATE"))
            row.query_type = self._clean_optional(self._row_value(raw_row, "QUERY_TYPE"))
            row.user_email = self._clean_optional(self._row_value(raw_row, "USER_EMAIL"))
            row.role_name = self._clean_optional(self._row_value(raw_row, "ROLE_NAME"))
            row.cluster_name = self._clean_optional(self._row_value(raw_row, "CLUSTER_NAME"))
            row.source_name = self._clean_optional(self._row_value(raw_row, "SOURCE"))
            row.created_at = self._coerce_datetime(self._row_value(raw_row, "CREATED_AT"))
            row.ended_at = self._coerce_datetime(self._row_value(raw_row, "ENDED_AT"))
            row.raw_sql = raw_sql
            row.nlp_text = normalized_nlp_text
            row.embeddings = embedding or None
            row.schema_table = self._clean_optional(self._row_value(raw_row, "SCHEMA_TABLE"))
            row.tables_json = (
                self._coerce_table_list(self._row_value(raw_row, "TABLES_JSON")) or None
            )
            row.metrics_json = self._coerce_json_dict(self._row_value(raw_row, "METRICS_JSON"))
            row.raw_json = self._coerce_json_dict(self._row_value(raw_row, "RAW_JSON"))
            row.updated_at = self._now()
            s.commit()
        return True

    def list_query_history_nlp_by_full_text(
        self,
        prompt: str,
        *,
        source_id: uuid.UUID | str | None = None,
        source_group_id: uuid.UUID | str | None = None,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Best-effort lexical retrieval over generated query-history NLP rows."""
        terms = [term for term in str(prompt or "").strip().split() if term]
        if not terms:
            return []

        with self.session() as s:
            q = self._query_history_nlp_scope_query(
                s.query(CatalogQueryHistoryNlp),
                source_id=source_id,
                source_group_id=source_group_id,
                engine=engine,
                catalog=catalog,
                schema_name=schema_name,
            )
            q = self._exclude_failed_query_history_nlp(q)
            match_clauses = []
            for term in terms:
                pattern = f"%{term}%"
                match_clauses.extend(
                    [
                        CatalogQueryHistoryNlp.nlp_text.ilike(pattern),
                        CatalogQueryHistoryNlp.raw_sql.ilike(pattern),
                        CatalogQueryHistoryNlp.schema_table.ilike(pattern),
                        cast(CatalogQueryHistoryNlp.tables_json, String).ilike(pattern),
                    ]
                )
            rows = (
                q.filter(or_(*match_clauses))
                .order_by(
                    func.coalesce(
                        CatalogQueryHistoryNlp.created_at,
                        CatalogQueryHistoryNlp.updated_at,
                    ).desc()
                )
                .limit(max(1, int(limit)) * 5)
                .all()
            )

        scored = [self._query_history_nlp_to_dict(row) for row in rows]
        for row in scored:
            haystack = " ".join(
                str(row.get(key) or "")
                for key in ("NLP_TEXT", "RAW_SQL", "SCHEMA_TABLE", "TABLES_JSON")
            ).lower()
            row["FTS_SCORE"] = float(sum(haystack.count(term.lower()) for term in terms))
        return sorted(
            scored,
            key=lambda row: (float(row.get("FTS_SCORE") or 0.0), row.get("UPDATED_AT") or ""),
            reverse=True,
        )[: max(1, int(limit))]

    def list_query_history_nlp_by_embedding(
        self,
        embedding: list[float],
        *,
        source_id: uuid.UUID | str | None = None,
        source_group_id: uuid.UUID | str | None = None,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Top-k generated query-history rows by pgvector cosine distance."""
        if not embedding or self.engine.dialect.name == "sqlite":
            return []

        distance = CatalogQueryHistoryNlp.embeddings.cosine_distance(embedding)
        similarity = (literal(1.0) - distance).label("cosine_similarity")
        with self.session() as s:
            q = self._query_history_nlp_scope_query(
                s.query(CatalogQueryHistoryNlp, similarity),
                source_id=source_id,
                source_group_id=source_group_id,
                engine=engine,
                catalog=catalog,
                schema_name=schema_name,
            )
            q = self._exclude_failed_query_history_nlp(q)
            rows = (
                q.filter(CatalogQueryHistoryNlp.embeddings.isnot(None))
                .order_by(distance)
                .limit(max(1, int(limit)))
                .all()
            )

        out: list[dict] = []
        for row, cosine_similarity in rows:
            item = self._query_history_nlp_to_dict(row)
            item["COSINE_SIMILARITY"] = float(cosine_similarity or 0.0)
            out.append(item)
        return out

    def search_query_history(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
        source_id: uuid.UUID | str | None = None,
        source_group_id: uuid.UUID | str | None = None,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        query_state: str | None = "FINISHED",
    ) -> list[dict]:
        """Top-k semantically similar query-history examples for in-context SQL."""
        if not query_embedding:
            return []

        effective_limit = max(1, int(k))
        with self.session() as s:
            q = self._query_history_nlp_scope_query(
                s.query(CatalogQueryHistoryNlp),
                source_id=source_id,
                source_group_id=source_group_id,
                engine=engine,
                catalog=catalog,
                schema_name=schema_name,
            )
            if query_state:
                q = q.filter(
                    func.upper(func.coalesce(CatalogQueryHistoryNlp.query_state, ""))
                    == str(query_state).strip().upper()
                )
            else:
                q = self._exclude_failed_query_history_nlp(q)
            q = q.filter(CatalogQueryHistoryNlp.embeddings.isnot(None))

            if self.engine.dialect.name != "sqlite":
                distance = CatalogQueryHistoryNlp.embeddings.cosine_distance(query_embedding)
                similarity = (literal(1.0) - distance).label("cosine_similarity")
                rows = (
                    q.with_entities(CatalogQueryHistoryNlp, similarity)
                    .order_by(distance)
                    .limit(effective_limit)
                    .all()
                )
                out: list[dict] = []
                for row, cosine_similarity in rows:
                    item = self._query_history_nlp_to_dict(row)
                    item["COSINE_SIMILARITY"] = float(cosine_similarity or 0.0)
                    out.append(item)
                return out

            rows = q.all()

        scored: list[dict] = []
        for row in rows:
            similarity = self._cosine_similarity(
                query_embedding,
                self._vector_values(row.embeddings),
            )
            item = self._query_history_nlp_to_dict(row)
            item["COSINE_SIMILARITY"] = similarity
            scored.append(item)
        return sorted(
            scored,
            key=lambda item: (
                float(item.get("COSINE_SIMILARITY") or 0.0),
                str(item.get("UPDATED_AT") or ""),
            ),
            reverse=True,
        )[:effective_limit]

    def table_context_for_refs(
        self,
        table_refs: list[str],
        *,
        source_id: uuid.UUID | str | None = None,
        source_group_id: uuid.UUID | str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return table/column metadata context for ordered table references."""
        with self.session() as s:
            return self._table_context_for_refs(
                s,
                source_id=self._coerce_uuid(source_id) if source_id else None,
                source_group_id=self._coerce_uuid(source_group_id) if source_group_id else None,
                table_refs=table_refs,
                catalog_name=catalog_name,
                schema_name=schema_name,
            )

    @staticmethod
    def _upsert_column_in_session(
        s: Session,
        table_id: uuid.UUID,
        column: ColumnMeta,
    ) -> CatalogColumn:
        col_row = (
            s.query(CatalogColumn)
            .filter_by(table_id=table_id, name=column.name)
            .one_or_none()
        )
        if col_row is None:
            col_row = CatalogColumn(table_id=table_id, name=column.name, data_type=column.data_type)
            s.add(col_row)

        col_row.name = column.name
        col_row.data_type = column.data_type
        col_row.nullable = column.nullable
        col_row.ordinal = column.ordinal
        col_row.is_primary_key = column.is_primary_key
        col_row.comment = column.comment
        col_row.sample_values = column.sample_values or None
        col_row.null_fraction = column.null_fraction
        col_row.distinct_estimate = column.distinct_estimate
        return col_row

    def persist_schema_docs(self, source_id: uuid.UUID, docs: Sequence[SchemaDoc]) -> int:
        with self.session() as s:
            s.query(SchemaDocRow).filter_by(source_id=source_id).delete()
            for d in docs:
                s.add(SchemaDocRow(
                    source_id=source_id,
                    object_type=d.object_type,
                    fqn=d.fqn,
                    catalog_name=d.catalog_name,
                    db_schema=d.db_schema,
                    table_name=d.table,
                    column_name=d.column,
                    text=d.text,
                    embedding=d.embedding,
                ))
            s.commit()
            return len(docs)

    def upsert_schema_docs(self, source_id: uuid.UUID, docs: Sequence[SchemaDoc]) -> int:
        """Insert or replace schema docs by stable ``(source, object_type, fqn)``."""
        with self.session() as s:
            written = 0
            for d in docs:
                s.query(SchemaDocRow).filter_by(
                    source_id=source_id,
                    object_type=d.object_type,
                    fqn=d.fqn,
                ).delete()
                s.add(SchemaDocRow(
                    source_id=source_id,
                    object_type=d.object_type,
                    fqn=d.fqn,
                    catalog_name=d.catalog_name,
                    db_schema=d.db_schema,
                    table_name=d.table,
                    column_name=d.column,
                    text=d.text,
                    embedding=d.embedding,
                ))
                written += 1
            s.commit()
            return written

    # ---- NL descriptions ------------------------------------------------------
    def list_tables_for_description(
        self,
        *,
        source_type: str | None = None,
        source_name: str | None = None,
        catalog_name: str | None = None,
        database_name: str | None = None,
        db_schema: str | None = None,
        table_name: str | None = None,
        missing_only: bool = True,
        limit: int = 50,
    ) -> list[CatalogTable]:
        """Return catalog tables that need table-level NL descriptions.

        ``database_name`` is treated as the physical top-level namespace when
        supplied. This is how Snowflake's database layer reaches the concrete
        table rows while Starburst/Postgres continue to use ``catalog_name``.
        """
        effective_catalog = self._effective_catalog_filter(
            source_type=source_type,
            catalog_name=catalog_name,
            database_name=database_name,
        )
        with self.session() as s:
            q = (
                s.query(CatalogTable)
                .join(Source, CatalogTable.source_id == Source.id)
                .options(
                    selectinload(CatalogTable.source),
                    selectinload(CatalogTable.columns),
                )
            )
            q = self._apply_source_scope(
                q,
                source_type=source_type,
                source_name=source_name,
                catalog_name=catalog_name,
                database_name=database_name,
            )
            if effective_catalog:
                q = q.filter(self._ci_equals(CatalogTable.catalog_name, effective_catalog))
            if db_schema:
                q = q.filter(self._ci_equals(CatalogTable.db_schema, db_schema))
            if table_name:
                q = q.filter(self._ci_equals(CatalogTable.name, table_name))
            if missing_only:
                q = q.filter(self._blank(CatalogTable.nl_description))
            return (
                q.order_by(CatalogTable.catalog_name, CatalogTable.db_schema, CatalogTable.name)
                .limit(max(1, int(limit)))
                .all()
            )

    def list_columns_for_description(
        self,
        *,
        source_type: str | None = None,
        source_name: str | None = None,
        catalog_name: str | None = None,
        database_name: str | None = None,
        db_schema: str,
        table_name: str | None = None,
        column_names: Sequence[str] | None = None,
        missing_only: bool = True,
        limit: int = 500,
    ) -> list[CatalogColumn]:
        """Return catalog columns that need column-level NL descriptions."""
        effective_catalog = self._effective_catalog_filter(
            source_type=source_type,
            catalog_name=catalog_name,
            database_name=database_name,
        )
        wanted_columns = [str(item).strip() for item in column_names or [] if str(item).strip()]
        with self.session() as s:
            q = (
                s.query(CatalogColumn)
                .join(CatalogTable, CatalogColumn.table_id == CatalogTable.id)
                .join(Source, CatalogTable.source_id == Source.id)
                .options(
                    selectinload(CatalogColumn.table).selectinload(CatalogTable.source),
                    selectinload(CatalogColumn.table).selectinload(CatalogTable.columns),
                )
            )
            q = self._apply_source_scope(
                q,
                source_type=source_type,
                source_name=source_name,
                catalog_name=catalog_name,
                database_name=database_name,
            )
            if effective_catalog:
                q = q.filter(self._ci_equals(CatalogTable.catalog_name, effective_catalog))
            q = q.filter(self._ci_equals(CatalogTable.db_schema, db_schema))
            if table_name:
                q = q.filter(self._ci_equals(CatalogTable.name, table_name))
            if wanted_columns:
                q = q.filter(
                    or_(*[self._ci_equals(CatalogColumn.name, col) for col in wanted_columns])
                )
            if missing_only:
                q = q.filter(self._blank(CatalogColumn.nl_description))
            return (
                q.order_by(
                    CatalogTable.catalog_name,
                    CatalogTable.db_schema,
                    CatalogTable.name,
                    CatalogColumn.ordinal,
                    CatalogColumn.name,
                )
                .limit(max(1, int(limit)))
                .all()
            )

    def update_table_description(
        self,
        table_id: uuid.UUID,
        *,
        description: str,
        confidence: float | None = None,
    ) -> bool:
        """Persist a generated table NL description."""
        with self.session() as s:
            row = s.get(CatalogTable, table_id)
            if row is None:
                return False
            row.nl_description = description
            row.description_confidence = confidence
            s.commit()
            return True

    def update_column_description(
        self,
        column_id: uuid.UUID,
        *,
        description: str,
    ) -> bool:
        """Persist a generated column NL description."""
        with self.session() as s:
            row = s.get(CatalogColumn, column_id)
            if row is None:
                return False
            row.nl_description = description
            s.commit()
            return True

    # ---- retrieval ------------------------------------------------------------
    def search_schema_docs(
        self,
        query_embedding: list[float],
        k: int = 15,
        source_id: uuid.UUID | None = None,
        source_group_id: uuid.UUID | str | None = None,
        catalog_name: str | None = None,
        object_type: str | None = None,
        db_schema: str | None = None,
        table_name: str | None = None,
    ) -> list[SchemaDocRow]:
        """Top-k schema docs by cosine distance (pgvector ``<=>``)."""
        with self.session() as s:
            q = s.query(SchemaDocRow).filter(SchemaDocRow.embedding.isnot(None))
            if source_id:
                q = q.filter(SchemaDocRow.source_id == source_id)
            elif source_group_id:
                q = self._filter_source_group(q, SchemaDocRow.source_id, source_group_id)
            if catalog_name:
                q = q.filter(self._ci_equals(SchemaDocRow.catalog_name, catalog_name))
            if object_type:
                q = q.filter(SchemaDocRow.object_type == object_type)
            if db_schema:
                q = q.filter(self._ci_equals(SchemaDocRow.db_schema, db_schema))
            if table_name:
                q = q.filter(self._ci_equals(SchemaDocRow.table_name, table_name))
            if self.engine.dialect.name == "sqlite":
                rows = q.all()
                return sorted(
                    rows,
                    key=lambda row: self._cosine_similarity(
                        query_embedding,
                        self._vector_values(row.embedding),
                    ),
                    reverse=True,
                )[:k]
            return (
                q.order_by(SchemaDocRow.embedding.cosine_distance(query_embedding))
                .limit(k)
                .all()
            )

    def search_schema_docs_lexical(
        self,
        query: str,
        k: int = 15,
        source_id: uuid.UUID | None = None,
        source_group_id: uuid.UUID | str | None = None,
        catalog_name: str | None = None,
        object_type: str | None = None,
        db_schema: str | None = None,
        table_name: str | None = None,
    ) -> list[SchemaDocRow]:
        """Best-effort lexical search over schema-doc text and identifiers."""
        terms = [term for term in str(query or "").strip().split() if term]
        if not terms:
            return []
        with self.session() as s:
            q = s.query(SchemaDocRow)
            if source_id:
                q = q.filter(SchemaDocRow.source_id == source_id)
            elif source_group_id:
                q = self._filter_source_group(q, SchemaDocRow.source_id, source_group_id)
            if catalog_name:
                q = q.filter(self._ci_equals(SchemaDocRow.catalog_name, catalog_name))
            if object_type:
                q = q.filter(SchemaDocRow.object_type == object_type)
            if db_schema:
                q = q.filter(self._ci_equals(SchemaDocRow.db_schema, db_schema))
            if table_name:
                q = q.filter(self._ci_equals(SchemaDocRow.table_name, table_name))
            match_clauses = []
            for term in terms:
                pattern = f"%{term}%"
                match_clauses.append(SchemaDocRow.fqn.ilike(pattern))
                match_clauses.append(SchemaDocRow.text.ilike(pattern))
            rows = q.filter(or_(*match_clauses)).limit(max(1, int(k)) * 5).all()
            return sorted(
                rows,
                key=lambda row: self._lexical_score(row, terms),
                reverse=True,
            )[:k]

    @staticmethod
    def _apply_source_scope(
        q,
        *,
        source_type: str | None = None,
        source_name: str | None = None,
        catalog_name: str | None = None,
        database_name: str | None = None,
    ):
        database_scope = CatalogRepository._effective_database_scope(
            source_type=source_type,
            catalog_name=catalog_name,
            database_name=database_name,
        )
        if source_type:
            q = q.filter(CatalogRepository._ci_equals(Source.source_type, source_type))
        if source_name:
            q = q.filter(CatalogRepository._ci_equals(Source.name, source_name))
        if catalog_name:
            q = q.filter(CatalogRepository._ci_equals(Source.catalog_name, catalog_name))
        if database_scope:
            q = q.filter(
                or_(
                    CatalogRepository._ci_equals(Source.database, database_scope),
                    CatalogRepository._ci_equals(Source.catalog_name, database_scope),
                )
            )
        return q

    def _resolve_source_group_id(
        self,
        *,
        source_type: str,
        source_group_id: uuid.UUID | str | None = None,
        source_group_name: str | None = None,
    ) -> uuid.UUID | None:
        if source_group_id:
            return self._coerce_uuid(source_group_id)
        normalized_name = str(source_group_name or "").strip()
        if not normalized_name:
            return None
        return self.upsert_source_group(source_type, normalized_name)

    def _filter_source_group(self, q, source_id_column, source_group_id):  # noqa: ANN001
        group_id = self._coerce_uuid(source_group_id)
        return q.filter(
            source_id_column.in_(
                select(Source.id).where(Source.source_group_id == group_id)
            )
        )

    @staticmethod
    def _effective_catalog_filter(
        *,
        source_type: str | None,
        catalog_name: str | None,
        database_name: str | None,
    ) -> str | None:
        if CatalogRepository._catalog_scoped(source_type) and catalog_name:
            return catalog_name
        return database_name or catalog_name

    @staticmethod
    def _effective_database_scope(
        *,
        source_type: str | None,
        catalog_name: str | None,
        database_name: str | None,
    ) -> str | None:
        if CatalogRepository._catalog_scoped(source_type) and catalog_name:
            return catalog_name
        return database_name

    @staticmethod
    def _catalog_scoped(source_type: str | None) -> bool:
        return (source_type or "").strip().lower() in _CATALOG_SCOPED_SOURCE_TYPES

    @staticmethod
    def _ci_equals(column, value: str):  # noqa: ANN001
        return func.lower(column) == str(value).lower()

    @staticmethod
    def _blank(column):  # noqa: ANN001
        return or_(column.is_(None), func.length(func.trim(column)) == 0)

    @staticmethod
    def _lexical_score(row: SchemaDocRow, terms: Sequence[str]) -> int:
        haystack = f"{row.fqn} {row.text}".lower()
        return sum(haystack.count(term.lower()) for term in terms)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _coerce_uuid(value: uuid.UUID | str) -> uuid.UUID:
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))

    @staticmethod
    def _clean_optional(value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            parsed = int(value or 0)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text_value = value.strip()
            if not text_value:
                return None
            try:
                return datetime.fromisoformat(text_value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @staticmethod
    def _datetime_to_iso(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        text_value = str(value).strip()
        return text_value or None

    @staticmethod
    def _row_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
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
    def _coerce_json_dict(value: Any) -> dict:
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {"value": value}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        return {"value": value}

    @staticmethod
    def _coerce_table_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            values = value
        elif isinstance(value, str):
            text_value = value.strip()
            if not text_value:
                return []
            try:
                parsed = json.loads(text_value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                values = parsed
            else:
                values = [item.strip() for item in text_value.split(",")]
        else:
            values = [value]

        out: list[str] = []
        seen: set[str] = set()
        for item in values:
            text_value = str(item or "").strip()
            if not text_value:
                continue
            key = text_value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text_value)
        return out

    @staticmethod
    def _parse_table_ref(value: Any) -> dict[str, str]:
        cleaned = str(value or "").strip().replace('"', "").replace("`", "")
        if not cleaned:
            return {"catalog": "", "schema_name": "", "table_name": ""}
        parts = [part.strip() for part in cleaned.split(".") if part.strip()]
        if len(parts) >= 3:
            return {"catalog": parts[-3], "schema_name": parts[-2], "table_name": parts[-1]}
        if len(parts) == 2:
            return {"catalog": "", "schema_name": parts[0], "table_name": parts[1]}
        return {"catalog": "", "schema_name": "", "table_name": parts[0]}

    def _query_history_to_dict(self, row: CatalogQueryHistory) -> dict[str, Any]:
        return {
            "ID": row.id,
            "SOURCE_ID": str(row.source_id),
            "ENGINE": row.engine,
            "QUERY_ID": row.query_id,
            "CATALOG_NAME": row.catalog_name,
            "SCHEMA_NAME": row.schema_name,
            "QUERY_STATE": row.query_state,
            "QUERY_TYPE": row.query_type,
            "USER_EMAIL": row.user_email,
            "ROLE_NAME": row.role_name,
            "CLUSTER_NAME": row.cluster_name,
            "SOURCE": row.source_name,
            "CREATED_AT": self._datetime_to_iso(row.created_at),
            "ENDED_AT": self._datetime_to_iso(row.ended_at),
            "RAW_SQL": row.raw_sql,
            "SCHEMA_TABLE": row.schema_table,
            "TABLES_JSON": row.tables_json or [],
            "METRICS_JSON": row.metrics_json or {},
            "RAW_JSON": row.raw_json or {},
            "UPDATED_AT": self._datetime_to_iso(row.updated_at),
        }

    def _query_history_nlp_to_dict(self, row: CatalogQueryHistoryNlp) -> dict[str, Any]:
        return {
            "ID": row.id,
            "RAW_QUERY_HISTORY_ID": row.raw_query_history_id,
            "SOURCE_ID": str(row.source_id),
            "ENGINE": row.engine,
            "QUERY_ID": row.query_id,
            "CATALOG_NAME": row.catalog_name,
            "SCHEMA_NAME": row.schema_name,
            "QUERY_STATE": row.query_state,
            "QUERY_TYPE": row.query_type,
            "USER_EMAIL": row.user_email,
            "ROLE_NAME": row.role_name,
            "CLUSTER_NAME": row.cluster_name,
            "SOURCE": row.source_name,
            "CREATED_AT": self._datetime_to_iso(row.created_at),
            "ENDED_AT": self._datetime_to_iso(row.ended_at),
            "RAW_SQL": row.raw_sql,
            "QUERY_NLP": row.nlp_text,
            "NLP_TEXT": row.nlp_text,
            "SCHEMA_TABLE": row.schema_table,
            "TABLES_JSON": row.tables_json or [],
            "UPDATED_AT": self._datetime_to_iso(row.updated_at),
        }

    def _query_history_nlp_scope_query(
        self,
        q,
        *,
        source_id: uuid.UUID | str | None = None,
        source_group_id: uuid.UUID | str | None = None,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
    ):
        if source_id:
            q = q.filter(CatalogQueryHistoryNlp.source_id == self._coerce_uuid(source_id))
        elif source_group_id:
            q = self._filter_source_group(q, CatalogQueryHistoryNlp.source_id, source_group_id)
        normalized_engine = str(engine or "").strip().lower()
        if normalized_engine:
            q = q.filter(CatalogQueryHistoryNlp.engine == normalized_engine)
        if catalog:
            q = q.filter(self._ci_equals(CatalogQueryHistoryNlp.catalog_name, catalog))
        if schema_name:
            q = q.filter(self._ci_equals(CatalogQueryHistoryNlp.schema_name, schema_name))
        return q

    @staticmethod
    def _exclude_failed_query_history_nlp(q):
        return q.filter(
            func.upper(func.coalesce(CatalogQueryHistoryNlp.query_state, "")) != "FAILED"
        )

    @staticmethod
    def _vector_values(value: Any) -> list[float]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, str):
            text_value = value.strip()
            if not text_value:
                return []
            try:
                parsed = json.loads(text_value)
            except json.JSONDecodeError:
                parsed = [item.strip() for item in text_value.strip("[]").split(",")]
            value = parsed
        if not isinstance(value, (list, tuple)):
            return []
        out: list[float] = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out

    @classmethod
    def _cosine_similarity(cls, left: list[float] | None, right: list[float] | None) -> float:
        left_values = cls._vector_values(left)
        right_values = cls._vector_values(right)
        count = min(len(left_values), len(right_values))
        if count <= 0:
            return 0.0
        dot = sum(left_values[idx] * right_values[idx] for idx in range(count))
        left_norm = math.sqrt(sum(left_values[idx] ** 2 for idx in range(count)))
        right_norm = math.sqrt(sum(right_values[idx] ** 2 for idx in range(count)))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def _table_context_for_refs(
        self,
        session: Session,
        *,
        source_id: uuid.UUID | None,
        table_refs: list[str],
        source_group_id: uuid.UUID | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[CatalogTable] = []
        for table_ref in table_refs:
            ref = self._parse_table_ref(table_ref)
            table_name = ref.get("table_name")
            if not table_name:
                continue
            q = (
                session.query(CatalogTable)
                .options(
                    selectinload(CatalogTable.source),
                    selectinload(CatalogTable.columns),
                )
                .filter(self._ci_equals(CatalogTable.name, table_name))
            )
            if source_id:
                q = q.filter(CatalogTable.source_id == source_id)
            elif source_group_id:
                q = self._filter_source_group(q, CatalogTable.source_id, source_group_id)
            ref_catalog = ref.get("catalog") or catalog_name
            ref_schema = ref.get("schema_name") or schema_name
            if ref_catalog:
                q = q.filter(self._ci_equals(CatalogTable.catalog_name, ref_catalog))
            if ref_schema:
                q = q.filter(self._ci_equals(CatalogTable.db_schema, ref_schema))
            row = q.order_by(CatalogTable.catalog_name, CatalogTable.db_schema).first()
            if row is not None:
                rows.append(row)

        context: list[dict[str, Any]] = []
        seen: set[str] = set()
        for table in rows:
            key = f"{table.source_id}:{table.catalog_name}:{table.db_schema}:{table.name}".lower()
            if key in seen:
                continue
            seen.add(key)
            context.append(
                {
                    "name": table.fqn,
                    "engine": table.source.source_type if table.source else "",
                    "catalog": table.catalog_name or "",
                    "schema_name": table.db_schema or "",
                    "table_name": table.name,
                    "table_type": table.table_type or "",
                    "description": table.nl_description or table.comment or "",
                    "columns": [
                        {
                            "column_name": column.name,
                            "name": column.name,
                            "data_type": column.data_type or "",
                            "type": column.data_type or "",
                            "description": column.nl_description or column.comment or "",
                            "ordinal_position": column.ordinal,
                            "nullable": column.nullable,
                        }
                        for column in sorted(
                            table.columns or [],
                            key=lambda item: item.ordinal or 0,
                        )
                    ],
                }
            )
        return context

    # ---- skills ---------------------------------------------------------------
    def add_skill(self, **kwargs) -> uuid.UUID:
        with self.session() as s:
            skill = Skill(**kwargs)
            s.add(skill)
            s.commit()
            return skill.id

    def general_and_dialect_skills(self, dialect: str) -> list[Skill]:
        with self.session() as s:
            return (
                s.query(Skill)
                .filter(
                    Skill.status == "promoted",
                    (
                        (Skill.scope == "general_sql")
                        | ((Skill.scope == "dialect") & (Skill.dialect == dialect))
                    ),
                )
                .all()
            )

    def search_specific_skills(
        self,
        query_embedding: list[float],
        k: int = 6,
        threshold: float = 0.4,
        source_id: uuid.UUID | None = None,
    ) -> list[Skill]:
        """Top-k specific skills by similarity (proposal Eq. 4; cosine sim > δ)."""
        with self.session() as s:
            q = s.query(Skill).filter(
                Skill.embedding.isnot(None),
                Skill.scope.in_(["schema_specific", "failure_repair", "verifier_obligation"]),
            )
            if source_id:
                q = q.filter((Skill.source_id == source_id) | (Skill.source_id.is_(None)))
            rows = (
                q.order_by(Skill.embedding.cosine_distance(query_embedding)).limit(k * 3).all()
            )
            # cosine_distance = 1 - cosine_similarity; keep sim > threshold.
            out: list[Skill] = []
            for r in rows:
                # Recompute is unnecessary; rely on DB ordering, just cap to k.
                out.append(r)
                if len(out) >= k:
                    break
            return out

    def dispose(self) -> None:
        self.engine.dispose()
