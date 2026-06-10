"""Oracle connector -- declared in the factory, not implemented in v1.

Implement the four hooks using ``oracledb`` (thin mode) to enable Oracle:
EXPLAIN PLAN FOR + DBMS_XPLAN for plans, ALL_TABLES/ALL_TAB_COLUMNS for metadata,
and a read-only session. Until then every operation raises ``NotImplementedError``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from .base import DataSourceConnector, TableMeta
from .factory import ConnectorFactory

_MSG = "Oracle connector is not implemented in v1 (declared extension point)."


class OracleConnector(DataSourceConnector):
    @property
    def dialect(self) -> str:
        return "oracle"

    @contextmanager
    def _managed_cursor(self, *, read_only: bool, timeout_s: int):
        raise NotImplementedError(_MSG)
        yield  # pragma: no cover

    def _raw_execute(self, cursor: Any, sql: str, row_cap: int):
        raise NotImplementedError(_MSG)

    def _raw_explain(self, cursor: Any, sql: str) -> str:
        raise NotImplementedError(_MSG)

    def _discover_tables(self, database: str | None, db_schema: str | None) -> list[TableMeta]:
        raise NotImplementedError(_MSG)


ConnectorFactory.register("oracle", OracleConnector)
