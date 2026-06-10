from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import app.services.catalog as catalog_svc
import pytest
from skillsql.catalog.sync import sync_metadata_stream


class FakeCatalogRepo:
    def __init__(self) -> None:
        self.settings = SimpleNamespace()
        self.source_id = uuid.uuid4()
        self.sources = []
        self.tables = []
        self.columns = []
        self.docs = []

    def upsert_source(
        self,
        source_type,
        name,
        catalog_name=None,
        db_schema=None,
        database=None,
        source_group_id=None,
        source_group_name=None,
    ):
        self.sources.append(
            (
                source_type,
                name,
                catalog_name,
                db_schema,
                database,
                source_group_id,
                source_group_name,
            )
        )
        return self.source_id

    def upsert_table_metadata(self, source_id, table, *, replace_columns=False):
        self.tables.append(table)
        return uuid.uuid4(), len(table.columns)

    def upsert_column_metadata(self, source_id, table, column):
        self.columns.append((table, column))
        return uuid.uuid4()

    def upsert_schema_docs(self, source_id, docs):
        self.docs.extend(docs)
        return len(docs)


async def _records():
    yield SimpleNamespace(
        entity_type="catalog",
        engine="snowflake",
        catalog_name="SPIDER_DB",
    )
    yield SimpleNamespace(
        entity_type="schema",
        engine="snowflake",
        catalog_name="SPIDER_DB",
        schema_name="PUBLIC",
    )
    yield SimpleNamespace(
        entity_type="table",
        engine="snowflake",
        catalog_name="SPIDER_DB",
        schema_name="PUBLIC",
        table_name="ORDERS",
        object_type="BASE TABLE",
        description="Orders table",
        raw={"row_count": 10},
    )
    yield SimpleNamespace(
        entity_type="column",
        engine="snowflake",
        catalog_name="SPIDER_DB",
        schema_name="PUBLIC",
        table_name="ORDERS",
        column_name="ORDER_ID",
        ordinal_position=1,
        data_type="NUMBER",
        nullable=False,
        description="Primary order identifier",
        raw={},
    )


@pytest.mark.asyncio
async def test_sync_metadata_stream_upserts_table_column_and_docs():
    repo = FakeCatalogRepo()

    result = await sync_metadata_stream(
        _records(),
        repo,
        source_type="snowflake",
        source_name="spider2-snow",
        scope_type="table",
        catalog_name="SPIDER_DB",
        database_name="SPIDER_DB",
        db_schema="PUBLIC",
        table_name="ORDERS",
        embedder=lambda texts: [[0.0] for _ in texts],
    )

    assert result.source_id == repo.source_id
    assert result.as_dict()["scope"]["database"] == "SPIDER_DB"
    assert result.catalog_rows == 1
    assert result.schema_rows == 1
    assert result.table_rows == 1
    assert result.column_rows == 1
    assert result.docs == 2
    assert repo.tables[0].fqn == "SPIDER_DB.PUBLIC.ORDERS"
    assert repo.columns[0][1].name == "ORDER_ID"
    assert {doc.object_type for doc in repo.docs} == {"table", "column"}


@pytest.mark.asyncio
async def test_sync_metadata_stream_awaits_async_describer():
    repo = FakeCatalogRepo()

    async def describe(text: str) -> str:
        await asyncio.sleep(0)
        assert text
        return "Async catalog description."

    result = await sync_metadata_stream(
        _records(),
        repo,
        source_type="snowflake",
        source_name="spider2-snow",
        scope_type="table",
        catalog_name="SPIDER_DB",
        database_name="SPIDER_DB",
        db_schema="PUBLIC",
        table_name="ORDERS",
        embedder=lambda texts: [[0.0] for _ in texts],
        describer=describe,
    )

    assert result.warnings == []
    assert repo.docs
    assert all("Async catalog description." in doc.text for doc in repo.docs)


@pytest.mark.asyncio
async def test_app_metadata_sync_does_not_attach_generic_describer(monkeypatch):
    captured: dict[str, object] = {}
    source_id = uuid.uuid4()

    class FakeSyncResult:
        def __init__(self) -> None:
            self.source_id = source_id

        def as_dict(self) -> dict[str, object]:
            return {"source_id": str(source_id), "warnings": []}

    async def fake_sync_metadata_stream(records, repo, **kwargs):
        captured.update(kwargs)
        return FakeSyncResult()

    monkeypatch.setattr("skillsql.catalog.sync.sync_metadata_stream", fake_sync_metadata_stream)
    monkeypatch.setattr(
        catalog_svc,
        "get_resources",
        lambda: SimpleNamespace(repo=object(), embedder=lambda texts: [[0.0] for _ in texts]),
    )

    result = await catalog_svc._sync_metadata_records(
        _records(),
        source_type="snowflake",
        source_name="spider2-snow",
        scope_type="schema",
        catalog_name="SPIDER_DB",
        database_name="SPIDER_DB",
        db_schema="PUBLIC",
        table_name=None,
        column_name=None,
        describe=True,
        doc_batch_size=128,
    )

    assert captured["describer"] is None
    assert "describe ignored during metadata sync" in result["warnings"][0]
