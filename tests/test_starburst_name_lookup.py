from __future__ import annotations

import pytest
from app.adapters.starburst_trino_adapter import StarburstTrinoAdapter
from skillsql.connectors.base import SourceConfig
from skillsql.connectors.starburst_connector import StarburstConnector


def test_starburst_adapter_separates_name_lookup_from_persisted_name():
    assert StarburstTrinoAdapter._name_lookup_path_id("sample") == "name%3Dsample"
    assert StarburstTrinoAdapter._name_lookup_path_id("name=sample") == "name%3Dsample"
    assert StarburstTrinoAdapter._plain_ref("name=sample") == "sample"

    catalog_record = StarburstTrinoAdapter._catalog_record(
        StarburstTrinoAdapter.__new__(StarburstTrinoAdapter),
        {"catalogName": "name=sample"},
    )

    assert catalog_record.catalog_name == "sample"


class FakeStarburstConnector(StarburstConnector):
    def __init__(self) -> None:
        super().__init__(
            SourceConfig(
                source_type="starburst",
                host="https://example.galaxy.starburst.io",
                trino_host="https://example.trino.galaxy.starburst.io",
            )
        )
        self.paths: list[str] = []

    async def _galaxy_get_paginated(self, path: str, params=None):
        self.paths.append(path)
        if path == "/public/api/v1/catalog/name%3Dsample/schema":
            return [{"schemaName": "PUBLIC"}]
        if path == "/public/api/v1/catalog/name%3Dsample/schema/name%3DPUBLIC/table":
            return [{"tableName": "ORDERS"}]
        if (
            path
            == "/public/api/v1/catalog/name%3Dsample/schema/name%3DPUBLIC"
            "/table/name%3DORDERS/column"
        ):
            return [{"columnName": "STATUS", "dataType": "VARCHAR"}]
        return []

    async def _galaxy_iter_paginated(self, path: str, params=None):
        for item in await self._galaxy_get_paginated(path, params=params):
            yield item


@pytest.mark.asyncio
async def test_starburst_connector_uses_name_lookup_but_persists_plain_catalog_name():
    connector = FakeStarburstConnector()

    metadata = await connector.get_metadata(catalog_name="name=sample", db_schema="PUBLIC")

    assert metadata.catalog_name == "sample"
    assert metadata.tables[0].catalog_name == "sample"
    assert metadata.tables[0].db_schema == "PUBLIC"
    assert metadata.tables[0].name == "ORDERS"
    assert metadata.tables[0].columns[0].name == "STATUS"
    assert connector.paths == [
        "/public/api/v1/catalog/name%3Dsample/schema",
        "/public/api/v1/catalog/name%3Dsample/schema/name%3DPUBLIC/table",
        "/public/api/v1/catalog/name%3Dsample/schema/name%3DPUBLIC/table/name%3DORDERS/column",
    ]
