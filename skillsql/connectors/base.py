"""Unified async datasource connector abstraction.

Architecture
------------
``DataSourceConnector`` is the single interface every downstream component
depends on — the catalog builder, verifier, and benchmark runner.  No vendor
SDK is ever imported outside a connector subclass.

All primary methods are **async**:
- Starburst is natively async (Galaxy REST API + Trino REST protocol).
- Snowflake and Postgres are synchronous DB-API drivers wrapped in
  ``asyncio.to_thread()`` via :class:`SyncConnectorMixin`.

Hierarchy
---------
Different datasources express different levels of nesting:

    Starburst   :  catalog  → schema → table       (3 levels, Galaxy API)
    Snowflake   :  database → schema → table       (3 levels; database ≡ catalog)
    Postgres    :  database → schema → table       (3 levels; usually 1 DB)
    Oracle      :  database → schema (owner) → table

``TableMeta.catalog_name`` captures the top level for all backends.  The FQN
property ``catalog.schema.table`` is pre-computed and used by all SQL builders.

Read-only enforcement
---------------------
``assert_read_only()`` is synchronous (parse-only, no I/O) and is called
inside ``execute()`` before any network call.
"""

from __future__ import annotations

import abc
import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any

import sqlglot
import sqlglot.expressions as exp
from pydantic import BaseModel, Field, field_validator

from ..observability.logging import get_logger

log = get_logger(__name__)

# Allowed root statement types in read-only mode.
_READ_ONLY_TOP_LEVEL = (
    exp.Select, exp.Union, exp.Intersect, exp.Except,
    exp.With, exp.Describe, exp.Show,
)
_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge,
    exp.Create, exp.Drop, exp.Alter,
)


# =============================================================================
# SourceConfig
# =============================================================================

class SourceConfig(BaseModel):
    """Vendor-agnostic connection configuration.

    Named attributes cover the three first-class backends (Starburst, Snowflake,
    Postgres).  The ``extra`` dict absorbs any overflow without schema churn.
    Secrets must come from a secret manager in production.
    """

    source_type: str
    name: str = "default"

    # Generic
    host: str | None = None           # Galaxy API base URL for Starburst
    trino_host: str | None = None     # Starburst Trino cluster URL
    port: int | None = None
    user: str | None = None
    password: str | None = None
    role: str | None = None

    # Snowflake
    account: str | None = None
    warehouse: str | None = None
    authenticator: str | None = None
    private_key_path: str | None = None

    # Catalog scope
    database: str | None = None       # Snowflake database / Postgres DB
    catalog_name: str | None = None   # Starburst Galaxy catalog
    db_schema: str | None = None

    # Starburst Galaxy OAuth (separate from Trino Basic Auth)
    client_id: str | None = None
    client_secret: str | None = None

    # Execution safety / housekeeping
    query_tag: str = "skillsql_rl"
    verify_ssl: bool = True
    timeout_ms: int = 300_000
    source: str = "skillsql"          # X-Trino-Source / Snowflake query tag

    # Query history (Starburst only)
    qh_catalog: str = "galaxy_telemetry"
    qh_schema: str = "public"
    qh_table: str = "query_history"
    qh_trino_host: str | None = None  # separate cluster for query history queries
    qh_user: str | None = None
    qh_password: str | None = None
    qh_role: str | None = None
    qh_source: str = "skillsql-qh"

    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower().strip()


# =============================================================================
# DTOs
# =============================================================================

class ColumnMeta(BaseModel):
    name: str
    data_type: str
    nullable: bool = True
    comment: str | None = None
    ordinal: int = 0
    is_primary_key: bool = False
    sample_values: list[str] = Field(default_factory=list)
    null_fraction: float | None = None
    distinct_estimate: int | None = None


class TableMeta(BaseModel):
    """A discovered table with its full 3-level location.

    catalog_name  : Starburst Galaxy catalog name / Snowflake database
    db_schema     : schema name
    name          : bare table name
    """

    catalog_name: str | None = None
    db_schema: str | None = None
    name: str
    table_type: str = "BASE TABLE"
    comment: str | None = None
    row_estimate: int | None = None
    columns: list[ColumnMeta] = Field(default_factory=list)

    @property
    def fqn(self) -> str:
        """Fully-qualified name: catalog.schema.table (omits None levels)."""
        parts = [p for p in (self.catalog_name, self.db_schema, self.name) if p]
        return ".".join(parts)

    @property
    def schema_fqn(self) -> str:
        parts = [p for p in (self.catalog_name, self.db_schema) if p]
        return ".".join(parts) if parts else ""


class Metadata(BaseModel):
    """Schema metadata returned by get_metadata()."""

    source_type: str
    catalog_name: str | None = None
    db_schema: str | None = None
    tables: list[TableMeta] = Field(default_factory=list)
    collected_at: float = Field(default_factory=time.time)


class SchemaDoc(BaseModel):
    """Retrievable description of a schema object (table or column)."""

    object_type: str             # "table" | "column"
    fqn: str                     # full 3-level path
    catalog_name: str | None = None
    db_schema: str | None = None
    table: str
    column: str | None = None
    text: str
    embedding: list[float] | None = None


class ExecResult(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0
    dialect: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class PlanResult(BaseModel):
    plan_text: str = ""
    elapsed_ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# =============================================================================
# Errors
# =============================================================================

class ConnectorError(RuntimeError):
    """Base error for connector failures."""


class ReadOnlyViolation(ConnectorError):
    """Raised for non-read-only SQL under read-only enforcement."""


class CatalogNotFound(ConnectorError):
    """Raised when a requested catalog or database does not exist."""


# =============================================================================
# Abstract base
# =============================================================================

class DataSourceConnector(abc.ABC):
    """Async-first connector interface.

    Concrete implementations must provide:  dialect, execute, explain_plan,
    get_metadata.  All other methods have safe default implementations.
    """

    def __init__(self, config: SourceConfig) -> None:
        self.config = config

    @property
    @abc.abstractmethod
    def dialect(self) -> str:
        """sqlglot dialect: 'snowflake', 'trino', 'postgres', 'oracle'."""

    @property
    def source_type(self) -> str:
        return self.config.source_type

    # ── Core async operations ──────────────────────────────────────────────

    @abc.abstractmethod
    async def execute(
        self,
        sql: str,
        *,
        read_only: bool = True,
        timeout_s: int = 60,
        row_cap: int = 5000,
    ) -> ExecResult:
        """Execute sql safely and return a complete, bounded result.

        Never raises for SQL-level errors; those appear in ExecResult.error.
        Network / auth errors propagate as ConnectorError.
        """

    @abc.abstractmethod
    async def explain_plan(self, sql: str) -> PlanResult:
        """Return the query plan (EXPLAIN / compile-only, cheap)."""

    @abc.abstractmethod
    async def get_metadata(
        self,
        *,
        catalog_name: str | None = None,
        db_schema: str | None = None,
    ) -> Metadata:
        """Discover tables and columns for a (catalog, schema) scope.

        When catalog_name / db_schema are None, fall back to the connector's
        configured defaults (SourceConfig.catalog_name / db_schema).
        """

    # ── Optional operations ────────────────────────────────────────────────

    async def list_catalogs(self) -> list[str]:
        """List available catalog / database names.

        Starburst: lists all Galaxy catalog names via the API.
        Snowflake/Postgres: returns [config.database] or [].
        """
        if self.config.database:
            return [self.config.database]
        if self.config.catalog_name:
            return [self.config.catalog_name]
        return []

    # ── Shared, vendor-independent behavior ───────────────────────────────

    def assert_read_only(self, sql: str) -> None:
        """Synchronous parse-level guard — runs before any network I/O.

        Raises ReadOnlyViolation on DDL/DML, multi-statement scripts, or
        administrative commands.
        """
        try:
            stmts = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except Exception as exc:  # noqa: BLE001
            raise ReadOnlyViolation(f"unparseable SQL: {exc}") from exc
        if len(stmts) != 1:
            raise ReadOnlyViolation("multiple statements not permitted")
        root = stmts[0]
        if isinstance(root, exp.Command):
            raise ReadOnlyViolation(f"command not permitted: {root.sql()[:60]}")
        if not isinstance(root, _READ_ONLY_TOP_LEVEL):
            raise ReadOnlyViolation(f"{type(root).__name__} not permitted in read-only mode")
        if any(root.find(f) for f in _FORBIDDEN_NODES):
            raise ReadOnlyViolation("DDL/DML modification not permitted")

    async def health_check(self) -> bool:
        """Lightweight connectivity probe."""
        result = await self.execute("SELECT 1", read_only=True, timeout_s=10, row_cap=1)
        return result.ok

    def to_schema_docs(self, meta: Metadata) -> list[SchemaDoc]:
        """Build retrievable text documents — one per table, one per column.

        Embeddings are injected separately by the catalog builder so the
        embedding model stays pluggable.
        """
        docs: list[SchemaDoc] = []
        for t in meta.tables:
            col_summary = ", ".join(f"{c.name} {c.data_type}" for c in t.columns[:40])
            table_text = (
                f"Table {t.fqn} ({t.table_type}). "
                f"{t.comment + '. ' if t.comment else ''}"
                f"Columns: {col_summary}."
            )
            docs.append(SchemaDoc(
                object_type="table",
                fqn=t.fqn,
                catalog_name=t.catalog_name,
                db_schema=t.db_schema,
                table=t.name,
                text=table_text,
            ))
            for c in t.columns:
                samples = (
                    f" Examples: {', '.join(c.sample_values[:5])}."
                    if c.sample_values else ""
                )
                col_text = (
                    f"Column {t.fqn}.{c.name} of type {c.data_type}"
                    f"{' (nullable)' if c.nullable else ' (not null)'}"
                    f"{'. ' + c.comment if c.comment else ''}.{samples}"
                )
                docs.append(SchemaDoc(
                    object_type="column",
                    fqn=f"{t.fqn}.{c.name}",
                    catalog_name=t.catalog_name,
                    db_schema=t.db_schema,
                    table=t.name,
                    column=c.name,
                    text=col_text,
                ))
        return docs

    async def embed_schema(
        self,
        meta: Metadata,
        embedder: Callable[[Sequence[str]], list[list[float]]],
    ) -> list[SchemaDoc]:
        docs = self.to_schema_docs(meta)
        vectors = embedder([d.text for d in docs])
        for d, v in zip(docs, vectors, strict=True):
            d.embedding = v
        return docs

    def close(self) -> None:
        """Release resources.  Override for connection-pool cleanup."""


# =============================================================================
# SyncConnectorMixin
# =============================================================================

class SyncConnectorMixin(DataSourceConnector, abc.ABC):
    """Mixin for synchronous DB-API connectors.

    Subclasses implement ``_sync_execute``, ``_sync_explain_plan``, and
    ``_sync_get_metadata``.  This mixin wraps each in ``asyncio.to_thread()``
    so the full async interface is satisfied without blocking the event loop.

    Subclasses should also override ``list_catalogs()`` if they can enumerate
    databases without a DB-API query.
    """

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
        return await asyncio.to_thread(
            self._sync_execute, sql, read_only=read_only,
            timeout_s=timeout_s, row_cap=row_cap,
        )

    async def explain_plan(self, sql: str) -> PlanResult:
        try:
            self.assert_read_only(sql)
        except ReadOnlyViolation as exc:
            return PlanResult(error=f"read_only_violation: {exc}")
        return await asyncio.to_thread(self._sync_explain_plan, sql)

    async def get_metadata(
        self,
        *,
        catalog_name: str | None = None,
        db_schema: str | None = None,
    ) -> Metadata:
        return await asyncio.to_thread(
            self._sync_get_metadata,
            catalog_name=catalog_name,
            db_schema=db_schema,
        )

    # ── Sync implementation hooks (subclasses must implement) ─────────────

    @abc.abstractmethod
    def _sync_execute(
        self, sql: str, *, read_only: bool, timeout_s: int, row_cap: int
    ) -> ExecResult: ...

    @abc.abstractmethod
    def _sync_explain_plan(self, sql: str) -> PlanResult: ...

    @abc.abstractmethod
    def _sync_get_metadata(
        self, *, catalog_name: str | None, db_schema: str | None
    ) -> Metadata: ...
