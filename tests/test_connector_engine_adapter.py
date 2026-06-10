from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.adapters.connector_engine_adapter import ConnectorEngineAdapter
from skillsql.connectors.base import ColumnMeta, Metadata, TableMeta

from tests._fakes import FakeConnector


class MetadataConnector(FakeConnector):
    def __init__(self):
        super().__init__(dialect="postgres")
        self.metadata_calls = []

    async def get_metadata(self, *, catalog_name: str | None = None, db_schema: str | None = None):
        self.metadata_calls.append((catalog_name, db_schema))
        return Metadata(
            source_type="fake",
            catalog_name=catalog_name,
            db_schema=db_schema,
            tables=[
                TableMeta(
                    catalog_name=catalog_name,
                    db_schema=db_schema,
                    name="orders",
                    columns=[
                        ColumnMeta(name="order_id", data_type="integer", nullable=False, ordinal=1),
                    ],
                )
            ],
        )


@pytest.mark.asyncio
async def test_connector_engine_adapter_streams_table_metadata():
    connector = MetadataConnector()
    adapter = ConnectorEngineAdapter(
        "postgres",
        connector,
        SimpleNamespace(max_runtime_seconds=60, default_limit=100),
    )

    rows = [
        record
        async for record in adapter.iter_table_metadata(
            "warehouse",
            "public",
            "orders",
            database_name="analytics_db",
            include_columns=True,
        )
    ]

    assert connector.metadata_calls == [("analytics_db", "public")]
    assert [row.entity_type for row in rows] == ["catalog", "schema", "table", "column"]
    assert rows[2].table_name == "orders"
    assert rows[2].catalog_name == "analytics_db"
    assert rows[2].database_name == "analytics_db"
    assert rows[3].column_name == "order_id"
