"""Snowflake connector — production implementation.

Uses ``snowflake-connector-python`` (imported lazily).

Hardening:
    * A read-only role is expected (SNOWFLAKE_ROLE); the parse-level guard in
      SyncConnectorMixin is defense-in-depth.
    * STATEMENT_TIMEOUT_IN_SECONDS bounds every query.
    * QUERY_TAG stamps every statement for cost attribution.
    * Key-pair auth is preferred over passwords in production.

Async wrapping:
    The synchronous DB-API calls run in ``asyncio.to_thread()`` via
    SyncConnectorMixin so the event loop is never blocked.
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
    SourceConfig,
    SyncConnectorMixin,
    TableMeta,
)
from .factory import ConnectorFactory

log = get_logger(__name__)


class SnowflakeConnector(SyncConnectorMixin):

    @property
    def dialect(self) -> str:
        return "snowflake"

    # ── Connection ────────────────────────────────────────────────────────

    def _connect_kwargs(self) -> dict[str, Any]:
        c = self.config
        kwargs: dict[str, Any] = {
            "account":    c.account,
            "user":       c.user,
            "role":       c.role,
            "warehouse":  c.warehouse,
            "database":   c.database or c.catalog_name,
            "schema":     c.db_schema,
            "authenticator": c.authenticator or "snowflake",
            "session_parameters": {
                "QUERY_TAG": c.query_tag or "skillsql_rl",
                "STATEMENT_TIMEOUT_IN_SECONDS": 60,
            },
            "client_session_keep_alive": False,
        }
        if c.private_key_path and len(c.private_key_path) > 0:
            kwargs["private_key"] = _load_private_key(c.private_key_path)
        elif c.password:
            kwargs["password"] = c.password
        return {k: v for k, v in kwargs.items() if v is not None}

    @contextmanager
    def _cursor(self, *, timeout_s: int):
        try:
            import snowflake.connector as sf
        except ImportError as exc:
            raise ConnectorError(
                "snowflake-connector-python not installed; "
                "pip install snowflake-connector-python"
            ) from exc
        conn = sf.connect(**self._connect_kwargs())
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {int(timeout_s)}"
                )
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
            with self._cursor(timeout_s=timeout_s) as cur:
                cur.execute(sql)
                columns = [d[0] for d in (cur.description or [])]
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
            log.warning("snowflake_execute_failed", error=str(exc), sql_preview=sql[:80])
            return ExecResult(dialect=self.dialect, elapsed_ms=elapsed, error=str(exc))

    def _sync_explain_plan(self, sql: str) -> PlanResult:
        start = time.perf_counter()
        try:
            with self._cursor(timeout_s=30) as cur:
                cur.execute(f"EXPLAIN USING TEXT {sql}")
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
        database = catalog_name or self.config.database or self.config.catalog_name
        schema = db_schema or self.config.db_schema
        if not database or not schema:
            raise ConnectorError(
                "SnowflakeConnector.get_metadata() requires catalog_name (database) "
                "and db_schema."
            )
        info = f"{database}.INFORMATION_SCHEMA"
        tables: dict[str, TableMeta] = {}

        with self._cursor(timeout_s=120) as cur:
            cur.execute(
                f"SELECT table_name, table_type, comment, row_count "
                f"FROM {info}.TABLES WHERE table_schema = %s ORDER BY table_name",
                (schema,),
            )
            for name, ttype, comment, row_count in cur.fetchall():
                tables[name] = TableMeta(
                    catalog_name=database,
                    db_schema=schema,
                    name=name,
                    table_type=ttype or "BASE TABLE",
                    comment=comment,
                    row_estimate=int(row_count) if row_count is not None else None,
                )
            cur.execute(
                f"SELECT table_name, column_name, data_type, is_nullable, "
                f"comment, ordinal_position "
                f"FROM {info}.COLUMNS WHERE table_schema = %s "
                f"ORDER BY table_name, ordinal_position",
                (schema,),
            )
            for tname, cname, dtype, nullable, ccomment, ordinal in cur.fetchall():
                if tname in tables:
                    tables[tname].columns.append(ColumnMeta(
                        name=cname,
                        data_type=dtype,
                        nullable=(str(nullable).upper() == "YES"),
                        comment=ccomment,
                        ordinal=int(ordinal) if ordinal is not None else 0,
                    ))

        log.info(
            "snowflake_metadata_fetched",
            database=database, schema=schema, tables=len(tables),
        )
        return Metadata(
            source_type="snowflake",
            catalog_name=database,
            db_schema=schema,
            tables=list(tables.values()),
        )

    async def list_catalogs(self) -> list[str]:
        if self.config.database or self.config.catalog_name:
            return [self.config.database or self.config.catalog_name]  # type: ignore[list-item]
        return []


def _load_private_key(path: str) -> bytes:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    with open(path, "rb") as fh:
        key = serialization.load_pem_private_key(
            fh.read(), password=None, backend=default_backend()
        )
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


ConnectorFactory.register("snowflake", SnowflakeConnector)
