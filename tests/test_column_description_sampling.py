from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.adapters.base import EngineHandle, ResultPage
from app.agents.column_description_agent.tools import build_tools


class FakeSampleAdapter:
    def __init__(self) -> None:
        self.sqls: list[str] = []

    async def execute_async(self, sql: str) -> EngineHandle:
        self.sqls.append(sql)
        return EngineHandle(handle_id=str(len(self.sqls)), raw={"query": sql})

    async def fetch_results(
        self,
        handle: EngineHandle,
        page_token: str | None = None,
    ) -> ResultPage:
        sql = str(handle.raw["query"])
        if '"AMOUNT"' in sql:
            return ResultPage(
                schema=[{"name": "AMOUNT", "type": "NUMBER"}],
                rows=[[10], [20], [10]],
            )
        return ResultPage(
            schema=[{"name": "STATUS", "type": "VARCHAR"}],
            rows=[["OPEN"], ["CLOSED"], ["OPEN"]],
        )


@pytest.mark.asyncio
async def test_column_description_tool_samples_distinct_values_per_column():
    adapter = FakeSampleAdapter()
    deps = SimpleNamespace(engines=SimpleNamespace(get=lambda engine: adapter))
    sample_column_values = build_tools(deps)[0]

    payload = await sample_column_values(
        catalog="ANALYTICS_DB",
        schema_name="PUBLIC",
        table_name="ORDERS",
        column_names=["AMOUNT", "STATUS"],
        sample_size=5,
        engine="snowflake",
    )

    assert len(adapter.sqls) == 2
    assert all("SELECT DISTINCT" in sql for sql in adapter.sqls)
    assert payload["column_samples"] == [
        {
            "column_name": "AMOUNT",
            "sample_values": [10, 20],
            "row_count": 2,
            "sql": adapter.sqls[0],
            "schema": [{"name": "AMOUNT", "type": "NUMBER"}],
        },
        {
            "column_name": "STATUS",
            "sample_values": ["OPEN", "CLOSED"],
            "row_count": 2,
            "sql": adapter.sqls[1],
            "schema": [{"name": "STATUS", "type": "VARCHAR"}],
        },
    ]
    assert payload["rows"] == [
        {"AMOUNT": 10, "STATUS": "OPEN"},
        {"AMOUNT": 20, "STATUS": "CLOSED"},
    ]
