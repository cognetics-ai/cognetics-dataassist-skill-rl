"""SQLAlchemy ORM for the unified semantic catalog.

Everything lives in a dedicated Postgres schema (``APP_CATALOG_SCHEMA``), kept
separate from the ADK session/event store (``ADK_SESSION_DSN``). Vector columns
use ``pgvector``; the dimension is configurable (``EMBEDDING_DIM``) and must
match the embedding model in use.

Hierarchy support
-----------------
The data model accommodates three-level datasource hierarchies:

    Starburst Galaxy:  catalog_name (Galaxy catalog) / db_schema / table_name
    Snowflake:         catalog_name (database)        / db_schema / table_name
    Postgres / Oracle: catalog_name (database or None)/ db_schema / table_name

``catalog_tables.catalog_name`` stores the top-level namespace so that FQNs
are always resolvable without heuristics.

Tables
------
    source_groups  -- logical datasource groups (uuid, type, name)
    sources        -- physical/catalog datasource scopes (uuid, type, name)
    catalog_tables -- one row per discovered table (catalog/schema/name/type)
    catalog_columns-- one row per discovered column (data type, comment, stats)
    schema_docs    -- retrievable descriptions + pgvector embeddings
    catalog_query_history     -- raw datasource query history scoped to source_id
    catalog_query_history_nlp -- generated NL query text + embeddings
    skills         -- SqlSkillBank entries (5 scopes) + pgvector embeddings
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ..config.settings import get_settings


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class Base(DeclarativeBase):
    """Declarative base; the concrete schema is bound at engine-creation time."""


class SourceGroup(Base):
    __tablename__ = "source_groups"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(128))
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    sources: Mapped[list[Source]] = relationship(
        back_populates="source_group", cascade="all, delete-orphan"
    )
    __table_args__ = (
        UniqueConstraint("source_type", "name"),
    )


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    source_group_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("source_groups.id", ondelete="CASCADE"), nullable=True, index=True
    )
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(128))
    # catalog_name: Starburst Galaxy catalog / Snowflake database
    catalog_name: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    # database kept for backward-compatibility; prefer catalog_name for new code
    database: Mapped[str | None] = mapped_column(String(256), nullable=True)
    db_schema: Mapped[str | None] = mapped_column(String(256), nullable=True)
    origin_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    source_group: Mapped[SourceGroup | None] = relationship(back_populates="sources")
    tables: Mapped[list[CatalogTable]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )
    query_history_rows: Mapped[list[CatalogQueryHistory]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )
    query_history_nlp_rows: Mapped[list[CatalogQueryHistoryNlp]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )
    __table_args__ = (
        UniqueConstraint("source_group_id", "source_type", "name", "catalog_name", "db_schema"),
    )


class CatalogTable(Base):
    __tablename__ = "catalog_tables"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )
    fqn: Mapped[str] = mapped_column(String(512), index=True)
    # 3-level location: catalog_name.db_schema.name
    catalog_name: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    db_schema: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(256), index=True)
    table_type: Mapped[str] = mapped_column(String(64), default="BASE TABLE")
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    nl_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    description_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    source: Mapped[Source] = relationship(back_populates="tables")
    columns: Mapped[list[CatalogColumn]] = relationship(
        back_populates="table", cascade="all, delete-orphan"
    )
    __table_args__ = (
        UniqueConstraint("source_id", "catalog_name", "db_schema", "name"),
    )


class CatalogColumn(Base):
    __tablename__ = "catalog_columns"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    table_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalog_tables.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(256))
    data_type: Mapped[str] = mapped_column(String(128))
    nullable: Mapped[bool] = mapped_column(Boolean, default=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    is_primary_key: Mapped[bool] = mapped_column(Boolean, default=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    sample_values: Mapped[list | None] = mapped_column(JSON, nullable=True)
    null_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    distinct_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    nl_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    table: Mapped[CatalogTable] = relationship(back_populates="columns")
    __table_args__ = (
        UniqueConstraint("table_id", "name"),
    )


def _dim() -> int:
    return get_settings().EMBEDDING_DIM


class SchemaDocRow(Base):
    __tablename__ = "schema_docs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )
    object_type: Mapped[str] = mapped_column(String(16))  # "table" | "column"
    fqn: Mapped[str] = mapped_column(String(512), index=True)
    # 3-level location for catalog-aware retrieval filtering
    catalog_name: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    db_schema: Mapped[str | None] = mapped_column(String(256), nullable=True)
    table_name: Mapped[str] = mapped_column(String(256))
    column_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_dim()), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    __table_args__ = (
        UniqueConstraint("source_id", "object_type", "fqn"),
    )


class CatalogQueryHistory(Base):
    """Raw datasource query-history row loaded under a catalog source."""

    __tablename__ = "catalog_query_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )
    engine: Mapped[str] = mapped_column(String(32), index=True)
    query_id: Mapped[str] = mapped_column(String(256))
    catalog_name: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    schema_name: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    query_state: Mapped[str | None] = mapped_column(String(64), nullable=True)
    query_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_email: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    role_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    cluster_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source: Mapped[Source] = relationship(back_populates="query_history_rows")
    source_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    raw_sql: Mapped[str] = mapped_column(Text)
    schema_table: Mapped[str | None] = mapped_column(String(768), nullable=True)
    tables_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    nlp_row: Mapped[CatalogQueryHistoryNlp | None] = relationship(
        back_populates="raw_history",
        cascade="all, delete-orphan",
        uselist=False,
    )
    __table_args__ = (
        UniqueConstraint("source_id", "engine", "query_id"),
        Index(
            "ix_catalog_query_history_source_scope",
            "source_id",
            "engine",
            "catalog_name",
            "schema_name",
        ),
    )


class CatalogQueryHistoryNlp(Base):
    """Natural-language query-history text and embedding for retrieval."""

    __tablename__ = "catalog_query_history_nlp"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    raw_query_history_id: Mapped[int | None] = mapped_column(
        ForeignKey("catalog_query_history.id", ondelete="CASCADE"), nullable=True, index=True
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )
    engine: Mapped[str] = mapped_column(String(32), index=True)
    query_id: Mapped[str] = mapped_column(String(256))
    catalog_name: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    schema_name: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    query_state: Mapped[str | None] = mapped_column(String(64), nullable=True)
    query_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_email: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    role_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    cluster_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source: Mapped[Source] = relationship(back_populates="query_history_nlp_rows")
    source_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    raw_sql: Mapped[str] = mapped_column(Text)
    nlp_text: Mapped[str] = mapped_column(Text)
    embeddings: Mapped[list[float] | None] = mapped_column(Vector(_dim()), nullable=True)
    schema_table: Mapped[str | None] = mapped_column(String(768), nullable=True)
    tables_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    raw_history: Mapped[CatalogQueryHistory | None] = relationship(back_populates="nlp_row")
    __table_args__ = (
        UniqueConstraint("source_id", "engine", "query_id"),
        Index(
            "ix_catalog_query_history_nlp_source_scope",
            "source_id",
            "engine",
            "catalog_name",
            "schema_name",
        ),
    )


class Skill(Base):
    """A SqlSkillBank entry (proposal Section 7)."""

    __tablename__ = "skills"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    scope: Mapped[str] = mapped_column(String(32), index=True)  # general_sql|dialect|...
    skill_type: Mapped[str] = mapped_column(String(32), default="strategy")
    title: Mapped[str] = mapped_column(String(256))
    principle: Mapped[str] = mapped_column(Text)
    when_to_apply: Mapped[str | None] = mapped_column(Text, nullable=True)
    positive_example: Mapped[str | None] = mapped_column(Text, nullable=True)
    negative_example: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sources.id", ondelete="SET NULL"), nullable=True
    )
    dialect: Mapped[str | None] = mapped_column(String(32), nullable=True)
    provenance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="candidate")  # candidate|promoted
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_dim()), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
