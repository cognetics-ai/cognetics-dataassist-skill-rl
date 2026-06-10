from __future__ import annotations

from types import SimpleNamespace

import app.services.catalog as catalog_svc
import pytest


class FakeAdapter:
    name = "snowflake"

    def __init__(self) -> None:
        self.schema_calls: list[tuple[str, str, str | None, bool]] = []

    def iter_schema_metadata(
        self,
        catalog: str,
        schema: str,
        *,
        database_name: str | None = None,
        include_columns: bool = True,
    ):
        self.schema_calls.append((catalog, schema, database_name, include_columns))

        async def records():
            if False:
                yield None

        return records()


class FakeEngines:
    def __init__(self, adapter: FakeAdapter) -> None:
        self.adapter = adapter

    def get(self, engine_name: str) -> FakeAdapter:
        assert engine_name == "snowflake"
        return self.adapter


@pytest.mark.asyncio
async def test_build_catalog_schema_scope_syncs_metadata_then_descriptions(monkeypatch):
    adapter = FakeAdapter()
    table = SimpleNamespace(
        catalog_name="ANALYTICS_DB",
        db_schema="PUBLIC",
        name="ORDERS",
        columns=[
            SimpleNamespace(name="ORDER_ID"),
            SimpleNamespace(name="STATUS"),
        ],
    )
    repo = SimpleNamespace(
        list_tables_for_description=lambda **kwargs: [table],
    )
    monkeypatch.setattr(
        catalog_svc,
        "get_resources",
        lambda: SimpleNamespace(
            settings=SimpleNamespace(DATASOURCE_TYPE="snowflake"),
            repo=repo,
        ),
    )

    metadata_calls: list[dict[str, object]] = []

    async def fake_sync_metadata_records(records, **kwargs):
        metadata_calls.append(kwargs)
        return {
            "source_id": "source-1",
            "scope": {
                "catalog": kwargs["catalog_name"],
                "database": kwargs["database_name"],
                "schema": kwargs["db_schema"],
                "table": None,
                "column": None,
            },
            "table_rows": 1,
            "column_rows": 2,
            "docs": 3,
        }

    table_description_calls: list[dict[str, object]] = []

    async def fake_sync_table_descriptions(*args, **kwargs):
        table_description_calls.append(kwargs)
        return {
            "candidate_count": 1,
            "updated_count": 1,
            "skipped_count": 0,
            "embeddings_generated": 1,
            "embedding_retries": 0,
        }

    column_description_calls: list[dict[str, object]] = []

    async def fake_sync_column_descriptions(*args, **kwargs):
        column_description_calls.append(kwargs)
        return {
            "candidate_count": 2,
            "updated_count": 2,
            "skipped_count": 0,
            "embeddings_generated": 2,
            "embedding_retries": 0,
        }

    monkeypatch.setattr(catalog_svc, "_sync_metadata_records", fake_sync_metadata_records)
    monkeypatch.setattr(catalog_svc, "sync_table_descriptions", fake_sync_table_descriptions)
    monkeypatch.setattr(catalog_svc, "sync_column_descriptions", fake_sync_column_descriptions)

    result = await catalog_svc.build_catalog(
        FakeEngines(adapter),
        source_type="snowflake",
        source_name="analytics",
        database_name="ANALYTICS_DB",
        db_schema="PUBLIC",
        describe=True,
        description_runtime=object(),
        sample_size=7,
    )

    assert adapter.schema_calls == [("ANALYTICS_DB", "PUBLIC", "ANALYTICS_DB", True)]
    assert metadata_calls[0]["scope_type"] == "schema"
    assert metadata_calls[0]["describe"] is False
    assert table_description_calls[0]["sample_size"] == 7
    assert table_description_calls[0]["missing_only"] is True
    assert column_description_calls[0]["table_name"] == "ORDERS"
    assert column_description_calls[0]["limit"] == 2
    assert result["tables"] == 1
    assert result["columns"] == 2
    assert result["description_sync"]["tables"]["updated_count"] == 1
    assert result["description_sync"]["columns"]["updated_count"] == 2
