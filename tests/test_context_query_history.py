from __future__ import annotations

import uuid
from types import SimpleNamespace

import app.services.catalog as catalog_svc
import pytest
from app.adapters.base import BackendQueryHistoryRecord
from skillsql.catalog.models import CatalogQueryHistoryNlp
from skillsql.catalog.repository import CatalogRepository
from skillsql.config.settings import get_settings
from sqlalchemy import create_engine


class TrackingRepo:
    def __init__(self, repo: CatalogRepository) -> None:
        self._repo = repo
        self.source_id: uuid.UUID | None = None
        self.sources: list[dict[str, object]] = []

    def __getattr__(self, name: str):
        return getattr(self._repo, name)

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
            {
                "source_type": source_type,
                "name": name,
                "catalog_name": catalog_name,
                "db_schema": db_schema,
                "database": database,
                "source_group_id": source_group_id,
                "source_group_name": source_group_name,
            }
        )
        self.source_id = self._repo.upsert_source(
            source_type,
            name,
            catalog_name=catalog_name,
            db_schema=db_schema,
            database=database,
        )
        return self.source_id


class FakeResources:
    def __init__(self, repo: TrackingRepo) -> None:
        self.repo = repo


class FakeAdapter:
    name = "starburst"

    async def iter_query_history(self, **kwargs):
        yield BackendQueryHistoryRecord(
            engine="starburst",
            query_id="q-1",
            raw_sql="select * from sample.burstbank.account",
            catalog_name="sample",
            schema_name="burstbank",
            query_state="FINISHED",
            user_email="analyst@example.com",
            tables=["sample.burstbank.account"],
            metrics={"cpu_ms": 12},
            raw={"query_id": "q-1"},
        )


class FakeFailedAdapter:
    name = "starburst"

    async def iter_query_history(self, **kwargs):
        yield BackendQueryHistoryRecord(
            engine="starburst",
            query_id="q-failed",
            raw_sql="select missing_column from sample.burstbank.account",
            catalog_name="sample",
            schema_name="burstbank",
            query_state="FAILED",
            user_email="analyst@example.com",
            tables=["sample.burstbank.account"],
            metrics={
                "error_code_name": "COLUMN_NOT_FOUND",
                "error_code_category": "USER_ERROR",
                "error_exception_message": "Column 'missing_column' cannot be resolved",
            },
            raw={"query_id": "q-failed"},
        )


class FakeEngines:
    def __init__(self, adapter=None) -> None:
        self.adapter = adapter or FakeAdapter()

    def get(self, engine: str):
        assert engine == "starburst"
        return self.adapter


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def describe_query_history_nlp(self, **kwargs):
        self.calls.append(kwargs)
        return {"query_nlp": "Show account records in burstbank.", "caveats": []}


class FakeEmbeddings:
    async def embed_document(self, text: str):
        assert text == "Show account records in burstbank."
        return [0.1] * get_settings().EMBEDDING_DIM, 0


class FlexibleFakeEmbeddings:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed_document(self, text: str):
        self.texts.append(text)
        return [0.1] * get_settings().EMBEDDING_DIM, 0


def _repo() -> TrackingRepo:
    engine = create_engine("sqlite:///:memory:", future=True)
    repo = CatalogRepository(
        settings=SimpleNamespace(
            APP_CATALOG_SCHEMA="",
            SQLALCHEMY_HIDE_PARAMETERS=True,
            APP_CATALOG_DSN="sqlite:///:memory:",
        ),
        engine=engine,
    )
    repo.init_schema()
    return TrackingRepo(repo)


@pytest.mark.asyncio
async def test_catalog_query_history_sync_writes_source_provenance(monkeypatch):
    resources = FakeResources(_repo())
    monkeypatch.setattr(catalog_svc, "get_resources", lambda: resources)

    result = await catalog_svc.sync_query_history(
        FakeEngines(),
        None,
        engine="starburst",
        source_name="starburst",
        catalog="sample",
        database_name="starburst",
        schema_name="burstbank",
        limit=10,
    )

    source_id = str(resources.repo.source_id)
    assert result["source_id"] == source_id
    assert result["query_history_rows"] == 1
    assert resources.repo.sources[0] == {
        "source_type": "starburst",
        "name": "starburst",
        "catalog_name": "sample",
        "db_schema": "burstbank",
        "database": "sample",
        "source_group_id": None,
        "source_group_name": None,
    }

    rows = resources.repo.list_query_history_for_nlp(
        source_id=source_id,
        engine="starburst",
        missing_only=False,
    )
    assert len(rows) == 1
    assert rows[0]["SOURCE_ID"] == source_id
    assert rows[0]["QUERY_ID"] == "q-1"


@pytest.mark.asyncio
async def test_catalog_query_history_nlp_sync_writes_nlp_text_and_embeddings(monkeypatch):
    resources = FakeResources(_repo())
    monkeypatch.setattr(catalog_svc, "get_resources", lambda: resources)
    raw = await catalog_svc.sync_query_history(
        FakeEngines(),
        None,
        engine="starburst",
        source_name="starburst",
        catalog="sample",
        database_name="starburst",
        schema_name="burstbank",
        limit=10,
    )
    runtime = FakeRuntime()

    result = await catalog_svc.sync_query_history_nlp(
        None,
        runtime,
        FakeEmbeddings(),
        engine="starburst",
        source_id=raw["source_id"],
        limit=10,
    )

    assert result["source_rows"] == 1
    assert result["inserted_rows"] == 1
    assert result["embeddings_generated"] == 1
    assert result["items"][0]["source_id"] == raw["source_id"]
    assert result["items"][0]["nlp_text"] == "Show account records in burstbank."
    assert runtime.calls[0]["source_id"] == raw["source_id"]

    rows = resources.repo.list_query_history_nlp_by_full_text(
        "account records",
        source_id=raw["source_id"],
        engine="starburst",
    )
    assert len(rows) == 1
    assert rows[0]["NLP_TEXT"] == "Show account records in burstbank."


@pytest.mark.asyncio
async def test_failed_query_history_nlp_includes_failure_details_and_filters_search(
    monkeypatch,
):
    resources = FakeResources(_repo())
    monkeypatch.setattr(catalog_svc, "get_resources", lambda: resources)
    raw = await catalog_svc.sync_query_history(
        FakeEngines(FakeFailedAdapter()),
        None,
        engine="starburst",
        source_name="starburst",
        catalog="sample",
        database_name="starburst",
        schema_name="burstbank",
        limit=10,
    )
    runtime = FakeRuntime()
    embeddings = FlexibleFakeEmbeddings()

    result = await catalog_svc.sync_query_history_nlp(
        None,
        runtime,
        embeddings,
        engine="starburst",
        source_id=raw["source_id"],
        limit=10,
    )

    nlp_text = result["items"][0]["nlp_text"]
    assert "Show account records in burstbank." in nlp_text
    assert "Failure cause:" in nlp_text
    assert "COLUMN_NOT_FOUND" in nlp_text
    assert "USER_ERROR" in nlp_text
    assert "Column 'missing_column' cannot be resolved" in nlp_text
    assert "Suggested fix:" in nlp_text
    assert embeddings.texts == [nlp_text]

    with resources.repo.session() as session:
        persisted = session.query(CatalogQueryHistoryNlp).one()
        assert persisted.nlp_text == nlp_text
        assert persisted.query_state == "FAILED"

    rows = resources.repo.list_query_history_nlp_by_full_text(
        "missing_column account",
        source_id=raw["source_id"],
        engine="starburst",
    )
    assert rows == []
