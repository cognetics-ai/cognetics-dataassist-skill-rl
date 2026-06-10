from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import create_engine

from skillsql.catalog.repository import CatalogRepository
from skillsql.catalog.source_resolver import resolve_source_for_database


def test_resolve_source_for_database_matches_snowflake_database_catalog_and_name():
    repo = CatalogRepository(
        settings=SimpleNamespace(
            APP_CATALOG_SCHEMA="",
            SQLALCHEMY_HIDE_PARAMETERS=True,
            APP_CATALOG_DSN="sqlite:///:memory:",
        ),
        engine=create_engine("sqlite:///:memory:", future=True),
    )
    repo.init_schema()
    snowflake_source = repo.upsert_source(
        "snowflake",
        "CENSUS_BENCH",
        catalog_name="CENSUS_DB",
        database="CENSUS_DB",
        db_schema="PUBLIC",
    )
    starburst_source = repo.upsert_source(
        "starburst",
        "starburst",
        catalog_name="sample",
        database="sample",
        db_schema="burstbank",
    )

    assert (
        resolve_source_for_database(repo, "census_db", source_type="snowflake")
        == snowflake_source
    )
    assert (
        resolve_source_for_database(repo, "census_bench", source_type="snowflake")
        == snowflake_source
    )
    assert resolve_source_for_database(repo, "sample") == starburst_source
    assert resolve_source_for_database(repo, "sample", source_type="snowflake") is None
