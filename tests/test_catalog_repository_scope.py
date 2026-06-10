from __future__ import annotations

from types import SimpleNamespace

from skillsql.catalog.models import (
    Base,
    CatalogColumn,
    CatalogQueryHistory,
    CatalogQueryHistoryNlp,
    CatalogTable,
    SchemaDocRow,
    Skill,
    Source,
    SourceGroup,
)
from skillsql.catalog.repository import CatalogRepository
from sqlalchemy import create_engine, create_mock_engine, inspect, select
from sqlalchemy.schema import CreateTable


def _repo_with_catalog_rows() -> CatalogRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    for table in (Source.__table__, CatalogTable.__table__, CatalogColumn.__table__):
        table.schema = None
        table.create(engine)

    repo = CatalogRepository(
        settings=SimpleNamespace(APP_CATALOG_SCHEMA="", SQLALCHEMY_HIDE_PARAMETERS=True),
        engine=engine,
    )
    with repo.session() as session:
        source = Source(
            source_type="starburst",
            name="starburst",
            catalog_name="sample",
            database="sample",
            db_schema="burstbank",
        )
        session.add(source)
        session.flush()
        table = CatalogTable(
            source_id=source.id,
            fqn="sample.burstbank.account",
            catalog_name="sample",
            db_schema="burstbank",
            name="account",
            table_type="BASE TABLE",
        )
        session.add(table)
        session.flush()
        customer_table = CatalogTable(
            source_id=source.id,
            fqn="sample.burstbank.customer",
            catalog_name="sample",
            db_schema="burstbank",
            name="customer",
            table_type="BASE TABLE",
        )
        session.add(customer_table)
        session.flush()
        session.add(
            CatalogColumn(
                table_id=table.id,
                name="account_id",
                data_type="VARCHAR",
                nullable=False,
                ordinal=1,
            )
        )
        session.add(
            CatalogColumn(
                table_id=customer_table.id,
                name="customer_id",
                data_type="VARCHAR",
                nullable=False,
                ordinal=1,
            )
        )
        session.commit()
    return repo


def test_starburst_description_scope_uses_catalog_not_database_label():
    repo = _repo_with_catalog_rows()

    tables = repo.list_tables_for_description(
        source_type="starburst",
        source_name="starburst",
        catalog_name="sample",
        database_name="starburst",
        db_schema="burstbank",
        table_name="account",
        missing_only=False,
    )

    assert [table.fqn for table in tables] == ["sample.burstbank.account"]


def test_starburst_column_description_scope_uses_catalog_not_database_label():
    repo = _repo_with_catalog_rows()

    columns = repo.list_columns_for_description(
        source_type="starburst",
        source_name="starburst",
        catalog_name="sample",
        database_name="starburst",
        db_schema="burstbank",
        table_name="account",
        missing_only=False,
    )

    assert [column.name for column in columns] == ["account_id"]


def test_column_description_scope_without_table_lists_schema_columns():
    repo = _repo_with_catalog_rows()

    columns = repo.list_columns_for_description(
        source_type="starburst",
        source_name="starburst",
        catalog_name="sample",
        database_name="starburst",
        db_schema="burstbank",
        missing_only=False,
    )

    assert [(column.table.name, column.name) for column in columns] == [
        ("account", "account_id"),
        ("customer", "customer_id"),
    ]


def test_repository_binds_configured_schema_for_all_postgres_catalog_tables():
    engine = create_mock_engine("postgresql+psycopg://", lambda *args, **kwargs: None)
    models = [
        (SourceGroup, "source_groups"),
        (Source, "sources"),
        (CatalogTable, "catalog_tables"),
        (CatalogColumn, "catalog_columns"),
        (SchemaDocRow, "schema_docs"),
        (CatalogQueryHistory, "catalog_query_history"),
        (CatalogQueryHistoryNlp, "catalog_query_history_nlp"),
        (Skill, "skills"),
    ]
    try:
        CatalogRepository(
            settings=SimpleNamespace(
                APP_CATALOG_SCHEMA="skillsql_catalog",
                SQLALCHEMY_HIDE_PARAMETERS=True,
            ),
            engine=engine,
        )

        for model, table_name in models:
            compiled = str(
                select(model).where(model.id.is_not(None)).compile(dialect=engine.dialect)
            )
            assert f"FROM skillsql_catalog.{table_name}" in compiled
    finally:
        for table in Base.metadata.tables.values():
            table.schema = None


def test_repository_schema_binding_qualifies_postgres_foreign_keys():
    engine = create_mock_engine("postgresql+psycopg://", lambda *args, **kwargs: None)
    try:
        CatalogRepository(
            settings=SimpleNamespace(
                APP_CATALOG_SCHEMA="skillsql_catalog",
                SQLALCHEMY_HIDE_PARAMETERS=True,
            ),
            engine=engine,
        )

        table_ddl = str(CreateTable(CatalogTable.__table__).compile(dialect=engine.dialect))
        source_ddl = str(CreateTable(Source.__table__).compile(dialect=engine.dialect))
        column_ddl = str(CreateTable(CatalogColumn.__table__).compile(dialect=engine.dialect))
        history_ddl = str(
            CreateTable(CatalogQueryHistory.__table__).compile(dialect=engine.dialect)
        )
        history_nlp_ddl = str(
            CreateTable(CatalogQueryHistoryNlp.__table__).compile(dialect=engine.dialect)
        )

        assert "REFERENCES skillsql_catalog.source_groups" in source_ddl
        assert "CREATE TABLE skillsql_catalog.catalog_tables" in table_ddl
        assert "REFERENCES skillsql_catalog.sources" in table_ddl
        assert "CREATE TABLE skillsql_catalog.catalog_columns" in column_ddl
        assert "REFERENCES skillsql_catalog.catalog_tables" in column_ddl
        assert "CREATE TABLE skillsql_catalog.catalog_query_history" in history_ddl
        assert "REFERENCES skillsql_catalog.sources" in history_ddl
        assert "CREATE TABLE skillsql_catalog.catalog_query_history_nlp" in history_nlp_ddl
        assert "REFERENCES skillsql_catalog.catalog_query_history" in history_nlp_ddl
        assert "REFERENCES skillsql_catalog.sources" in history_nlp_ddl
    finally:
        for table in Base.metadata.tables.values():
            table.schema = None


def test_init_schema_creates_catalog_query_history_tables():
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

    table_names = set(inspect(engine).get_table_names())
    assert "catalog_query_history" in table_names
    assert "catalog_query_history_nlp" in table_names
    assert "source_groups" in table_names
