from __future__ import annotations

from types import SimpleNamespace

from skillsql.catalog.repository import CatalogRepository
from skillsql.config.settings import get_settings
from skillsql.connectors.base import ColumnMeta, SchemaDoc, TableMeta
from sqlalchemy import create_engine


def _repo() -> CatalogRepository:
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
    return repo


def _embedding() -> list[float]:
    return [1.0, *([0.0] * (get_settings().EMBEDDING_DIM - 1))]


def test_source_group_links_multiple_physical_sources():
    repo = _repo()
    group_id = repo.upsert_source_group("snowflake", "spider2-snow")
    census_id = repo.upsert_source(
        "snowflake",
        "spider2-snow",
        catalog_name="CENSUS_DB",
        database="CENSUS_DB",
        db_schema="PUBLIC",
        source_group_id=group_id,
    )
    finance_id = repo.upsert_source(
        "snowflake",
        "spider2-snow",
        catalog_name="FINANCE_DB",
        database="FINANCE_DB",
        db_schema="PUBLIC",
        source_group_id=group_id,
    )

    assert set(repo.source_ids_for_group(group_id)) == {census_id, finance_id}

    sources = repo.list_sources()
    assert {row.source_group_id for row in sources} == {group_id}
    groups = repo.list_source_groups()
    assert len(groups) == 1
    assert groups[0].name == "spider2-snow"
    assert len(repo.source_ids_for_group(groups[0].id)) == 2


def test_search_schema_docs_can_scope_to_source_group():
    repo = _repo()
    group_id = repo.upsert_source_group("snowflake", "spider2-snow")
    in_group_source = repo.upsert_source(
        "snowflake",
        "spider2-snow",
        catalog_name="CENSUS_DB",
        database="CENSUS_DB",
        db_schema="PUBLIC",
        source_group_id=group_id,
    )
    other_source = repo.upsert_source(
        "snowflake",
        "other",
        catalog_name="OTHER_DB",
        database="OTHER_DB",
        db_schema="PUBLIC",
    )
    repo.upsert_schema_docs(
        in_group_source,
        [
            SchemaDoc(
                object_type="table",
                fqn="CENSUS_DB.PUBLIC.POPULATION",
                catalog_name="CENSUS_DB",
                db_schema="PUBLIC",
                table="POPULATION",
                column=None,
                text="population census residents",
                embedding=_embedding(),
            )
        ],
    )
    repo.upsert_schema_docs(
        other_source,
        [
            SchemaDoc(
                object_type="table",
                fqn="OTHER_DB.PUBLIC.POPULATION",
                catalog_name="OTHER_DB",
                db_schema="PUBLIC",
                table="POPULATION",
                column=None,
                text="population outside group",
                embedding=_embedding(),
            )
        ],
    )

    rows = repo.search_schema_docs(_embedding(), k=10, source_group_id=group_id)

    assert [row.source_id for row in rows] == [in_group_source]
    assert rows[0].catalog_name == "CENSUS_DB"


def test_query_history_and_table_context_can_scope_to_source_group():
    repo = _repo()
    group_id = repo.upsert_source_group("snowflake", "spider2-snow")
    source_id = repo.upsert_source(
        "snowflake",
        "spider2-snow",
        catalog_name="CENSUS_DB",
        database="CENSUS_DB",
        db_schema="PUBLIC",
        source_group_id=group_id,
    )
    other_source = repo.upsert_source(
        "snowflake",
        "other",
        catalog_name="OTHER_DB",
        database="OTHER_DB",
        db_schema="PUBLIC",
    )
    table = TableMeta(
        name="POPULATION",
        catalog_name="CENSUS_DB",
        db_schema="PUBLIC",
        columns=[
            ColumnMeta(name="STATE", data_type="TEXT", ordinal=1),
            ColumnMeta(name="POP", data_type="INT", ordinal=2),
        ],
    )
    repo.upsert_table_metadata(source_id, table, replace_columns=True)
    repo.upsert_query_history_rows(
        source_id=source_id,
        rows=[
            {
                "engine": "snowflake",
                "query_id": "q1",
                "catalog_name": "CENSUS_DB",
                "schema_name": "PUBLIC",
                "query_state": "FINISHED",
                "raw_sql": "SELECT STATE, POP FROM CENSUS_DB.PUBLIC.POPULATION",
                "tables": ["CENSUS_DB.PUBLIC.POPULATION"],
            }
        ],
    )
    repo.upsert_query_history_rows(
        source_id=other_source,
        rows=[
            {
                "engine": "snowflake",
                "query_id": "q2",
                "catalog_name": "OTHER_DB",
                "schema_name": "PUBLIC",
                "query_state": "FINISHED",
                "raw_sql": "SELECT * FROM OTHER_DB.PUBLIC.POPULATION",
                "tables": ["OTHER_DB.PUBLIC.POPULATION"],
            }
        ],
    )
    raw_rows = repo.list_query_history_for_nlp(source_group_id=group_id, engine="snowflake")
    assert [row["QUERY_ID"] for row in raw_rows] == ["q1"]
    repo.upsert_query_history_nlp_row(
        raw_row=raw_rows[0],
        nlp_text="Find population by state",
        embedding=_embedding(),
    )

    history = repo.search_query_history(
        _embedding(),
        source_group_id=group_id,
        engine="snowflake",
        query_state="FINISHED",
    )
    table_context = repo.table_context_for_refs(
        ["CENSUS_DB.PUBLIC.POPULATION"],
        source_group_id=group_id,
    )

    assert [row["QUERY_ID"] for row in history] == ["q1"]
    assert table_context[0]["name"] == "CENSUS_DB.PUBLIC.POPULATION"
    assert [col["name"] for col in table_context[0]["columns"]] == ["STATE", "POP"]
