from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.api.routes import catalog as catalog_routes
from app.schemas import CatalogColumnDescriptionSyncRequest
from pydantic import ValidationError


@pytest.mark.asyncio
async def test_catalog_build_request_forwards_database_name(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_build_catalog(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(catalog_routes.svc, "build_catalog", fake_build_catalog)
    ctx = SimpleNamespace(engines=object(), adk_runtime=object())

    request = catalog_routes.CatalogBuildRequest(
        source_type="snowflake",
        source_name="analytics",
        catalog="ANALYTICS_DB",
        database_name="ANALYTICS_DB",
        db_schema="PUBLIC",
        profile=False,
    )

    result = await catalog_routes.catalog_build(request, ctx)

    assert result == {"status": "ok"}
    assert "database_name" in catalog_routes.CatalogBuildRequest.model_fields
    assert "catalog" in catalog_routes.CatalogBuildRequest.model_fields
    assert captured["args"] == (ctx.engines,)
    assert captured["catalog"] == "ANALYTICS_DB"
    assert captured["database_name"] == "ANALYTICS_DB"
    assert captured["description_runtime"] is ctx.adk_runtime


def test_column_description_request_allows_schema_wide_sync_without_table_name():
    request = CatalogColumnDescriptionSyncRequest(
        catalog="sample",
        schema_name="burstbank",
    )

    assert request.table_name is None


def test_column_description_request_requires_table_for_column_filter():
    with pytest.raises(ValidationError, match="table_name is required"):
        CatalogColumnDescriptionSyncRequest(
            catalog="sample",
            schema_name="burstbank",
            column_names=["acctkey"],
        )
