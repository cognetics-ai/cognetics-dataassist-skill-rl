"""Test doubles: a connector that returns canned results without a live DB.

Updated for the async connector interface (DataSourceConnector.execute is now
async). Tests that call compute_reward() use asyncio.run() internally via the
reward module's sync wrapper, so the fake's async execute works transparently.
"""

from __future__ import annotations

from typing import Any

from skillsql.connectors.base import (
    DataSourceConnector,
    ExecResult,
    Metadata,
    PlanResult,
    SourceConfig,
)


class FakeConnector(DataSourceConnector):
    """A connector whose ``execute`` consults a mapping (sql -> ExecResult).

    The static gates in ``compute_reward`` run for real; only the *execution*
    step is faked, which lets us exercise every branch of the reward cascade.
    """

    def __init__(
        self,
        dialect: str = "snowflake",
        results: dict[str, ExecResult] | None = None,
        default: ExecResult | None = None,
    ) -> None:
        super().__init__(SourceConfig(source_type="fake"))
        self._dialect = dialect
        self._results = results or {}
        self._default = default or ExecResult(columns=["c"], rows=[[1]], row_count=1)

    @property
    def dialect(self) -> str:
        return self._dialect

    async def execute(
        self, sql: str, *, read_only: bool = True, timeout_s: int = 60, row_cap: int = 5000
    ) -> ExecResult:
        if read_only:
            try:
                self.assert_read_only(sql)
            except Exception as exc:  # noqa: BLE001
                return ExecResult(dialect=self._dialect, error=f"read_only_violation: {exc}")
        return self._results.get(sql.strip(), self._default)

    async def explain_plan(self, sql: str) -> PlanResult:
        return PlanResult(plan_text="FAKE PLAN")

    async def get_metadata(
        self, *, catalog_name: str | None = None, db_schema: str | None = None
    ) -> Metadata:
        return Metadata(source_type="fake", catalog_name=catalog_name, db_schema=db_schema)


def ok(cols: list[str], rows: list) -> ExecResult:
    return ExecResult(columns=cols, rows=[list(r) for r in rows], row_count=len(rows))
