"""Postgres connector — production implementation.

Uses ``psycopg`` (v3), imported lazily.  Suitable for:
  - Local development and testing (no cloud account needed)
  - Hosting the SkillSQL-RL app catalog itself (APP_CATALOG_DSN)
  - Spider-2.0 schemas mirrored into Postgres for benchmarking
  - Any self-hosted or cloud Postgres / Redshift / AlloyDB instance

Async wrapping:
    Synchronous psycopg calls run in ``asyncio.to_thread()`` via
    SyncConnectorMixin.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from ..observability.logging import get_logger
from .base import (
    ColumnMeta,
    ConnectorError,
    ExecResult,
    Metadata,
    PlanResult,
    SyncConnectorMixin,
    TableMeta,
)
from .factory import ConnectorFactory

log = get_logger(__name__)


def _libpq_dsn(dsn: str) -> str:
    """Strip a SQLAlchemy async driver prefix for use with psycopg."""
    return (
        dsn.replace("postgresql+psycopg://", "postgresql://")
           .replace("postgresql+asyncpg://", "postgresql://")
    )


class PostgresConnector(SyncConnectorMixin):

    @property
    def dialect(self) -> str:
        return "postgres"

    def _dsn(self) -> str:
        raw = self.config.extra.get("dsn") or self.config.host
        if not raw:
            raise ConnectorError(
                "PostgresConnector requires a DSN in SourceConfig.extra['dsn'] "
                "or SourceConfig.host (e.g. postgresql://user:pw@host/db)"
            )
        return _libpq_dsn(str(raw))

    @contextmanager
    def _cursor(self, *, read_only: bool, timeout_s: int):
        try:
            import psycopg
        except ImportError as exc:
            raise ConnectorError(
                "psycopg is not installed; pip install 'psycopg[binary]'"
            ) from exc
        conn = psycopg.connect(self._dsn(), autocommit=True)
        try:
            cur = conn.cursor()
            try:
                cur.execute(f"SET statement_timeout = {int(timeout_s) * 1000}")
                if read_only:
                    cur.execute("SET default_transaction_read_only = on")
                yield cur
            finally:
                cur.close()
        finally:
            conn.close()

    # ── SyncConnectorMixin hooks ──────────────────────────────────────────

    def _sync_execute(
        self, sql: str, *, read_only: bool, timeout_s: int, row_cap: int
    ) -> ExecResult:
        start = time.perf_counter()
        try:
            with self._cursor(read_only=read_only, timeout_s=timeout_s) as cur:
                cur.execute(sql)
                columns = [d.name for d in (cur.description or [])]
                rows = cur.fetchmany(row_cap + 1)
                truncated = len(rows) > row_cap
                rows = [list(r) for r in rows[:row_cap]]
            elapsed = (time.perf_counter() - start) * 1000.0
            return ExecResult(
                columns=columns, rows=rows, row_count=len(rows),
                truncated=truncated, elapsed_ms=elapsed, dialect=self.dialect,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000.0
            log.warning("postgres_execute_failed", error=str(exc), sql_preview=sql[:80])
            return ExecResult(dialect=self.dialect, elapsed_ms=elapsed, error=str(exc))

    def _sync_explain_plan(self, sql: str) -> PlanResult:
        start = time.perf_counter()
        try:
            with self._cursor(read_only=True, timeout_s=30) as cur:
                cur.execute(f"EXPLAIN {sql}")
                plan = "\n".join(str(r[0]) for r in cur.fetchall())
            return PlanResult(
                plan_text=plan, elapsed_ms=(time.perf_counter() - start) * 1000.0
            )
        except Exception as exc:  # noqa: BLE001
            return PlanResult(
                error=str(exc), elapsed_ms=(time.perf_counter() - start) * 1000.0
            )

    def _sync_get_metadata(
        self, *, catalog_name: str | None, db_schema: str | None
    ) -> Metadata:
        schema = db_schema or self.config.db_schema or "public"
        db_name = catalog_name or self.config.database or self.config.catalog_name
        tables: dict[str, TableMeta] = {}

        with self._cursor(read_only=True, timeout_s=120) as cur:
            # Table-level discovery (with optional comment from pg_description)
            cur.execute("""
                SELECT t.table_name,
                       t.table_type,
                       obj_description(
                           (quote_ident(t.table_schema)||'.'||quote_ident(t.table_name))::regclass,
                           'pg_class'
                       ) AS comment
                FROM information_schema.tables t
                WHERE t.table_schema = %s
                ORDER BY t.table_name
            """, (schema,))
            for name, ttype, comment in cur.fetchall():
                tables[name] = TableMeta(
                    catalog_name=db_name,
                    db_schema=schema,
                    name=name,
                    table_type=ttype or "BASE TABLE",
                    comment=comment,
                )

            # Column-level discovery
            cur.execute("""
                SELECT c.table_name,
                       c.column_name,
                       c.data_type,
                       c.is_nullable,
                       c.ordinal_position,
                       col_description(
                           (quote_ident(c.table_schema)||'.'||quote_ident(c.table_name))::regclass,
                           c.ordinal_position
                       ) AS comment
                FROM information_schema.columns c
                WHERE c.table_schema = %s
                ORDER BY c.table_name, c.ordinal_position
            """, (schema,))
            for tname, cname, dtype, nullable, ordinal, ccomment in cur.fetchall():
                if tname in tables:
                    tables[tname].columns.append(ColumnMeta(
                        name=cname,
                        data_type=dtype,
                        nullable=(str(nullable).upper() == "YES"),
                        comment=ccomment,
                        ordinal=int(ordinal) if ordinal is not None else 0,
                    ))

        log.info(
            "postgres_metadata_fetched",
            schema=schema, tables=len(tables),
        )
        return Metadata(
            source_type="postgres",
            catalog_name=db_name,
            db_schema=schema,
            tables=list(tables.values()),
        )

    async def list_catalogs(self) -> list[str]:
        if self.config.database or self.config.catalog_name:
            return [self.config.database or self.config.catalog_name]  # type: ignore[list-item]
        return []


ConnectorFactory.register("postgres", PostgresConnector)
