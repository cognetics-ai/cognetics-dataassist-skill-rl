from __future__ import annotations

from types import SimpleNamespace

import app.services.catalog as catalog_svc
from skillsql.catalog.models import CatalogColumn, CatalogTable, SchemaDocRow
from skillsql.catalog.repository import CatalogRepository
from skillsql.config.settings import get_settings
from sqlalchemy import create_engine


def _vector(first: float = 1.0) -> list[float]:
    values = [0.0] * get_settings().EMBEDDING_DIM
    values[0] = first
    return values


def test_generate_context_groups_schema_docs_and_finished_query_examples(monkeypatch):
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
    source_id = repo.upsert_source(
        "starburst",
        "starburst",
        catalog_name="sample",
        database="sample",
        db_schema="burstbank",
    )

    with repo.session() as session:
        account = CatalogTable(
            source_id=source_id,
            fqn="sample.burstbank.account",
            catalog_name="sample",
            db_schema="burstbank",
            name="account",
            table_type="BASE TABLE",
            nl_description="Account balances and lending product status by account.",
        )
        customer = CatalogTable(
            source_id=source_id,
            fqn="sample.burstbank.customer",
            catalog_name="sample",
            db_schema="burstbank",
            name="customer",
            table_type="BASE TABLE",
            nl_description="Customer profile and ownership information.",
        )
        session.add_all([account, customer])
        session.flush()
        session.add_all(
            [
                CatalogColumn(
                    table_id=account.id,
                    name="acctkey",
                    data_type="BIGINT",
                    nullable=False,
                    ordinal=1,
                    nl_description="Unique account identifier.",
                ),
                CatalogColumn(
                    table_id=account.id,
                    name="cc_balance",
                    data_type="DECIMAL",
                    nullable=True,
                    ordinal=2,
                    nl_description="Credit card balance amount.",
                ),
                CatalogColumn(
                    table_id=customer.id,
                    name="custkey",
                    data_type="BIGINT",
                    nullable=False,
                    ordinal=1,
                    nl_description="Unique customer identifier.",
                ),
            ]
        )
        session.add_all(
            [
                SchemaDocRow(
                    source_id=source_id,
                    object_type="column",
                    fqn="sample.burstbank.account.cc_balance",
                    catalog_name="sample",
                    db_schema="burstbank",
                    table_name="account",
                    column_name="cc_balance",
                    text="Column cc_balance is the current credit card balance.",
                    embedding=_vector(1.0),
                ),
                SchemaDocRow(
                    source_id=source_id,
                    object_type="table",
                    fqn="sample.burstbank.account",
                    catalog_name="sample",
                    db_schema="burstbank",
                    table_name="account",
                    text="Table account stores banking account balances and loan states.",
                    embedding=_vector(0.9),
                ),
            ]
        )
        session.commit()

    rows_written = repo.upsert_query_history_rows(
        source_id=source_id,
        rows=[
            {
                "engine": "starburst",
                "query_id": "finished-1",
                "raw_sql": "select acctkey, cc_balance from sample.burstbank.account",
                "catalog_name": "sample",
                "schema_name": "burstbank",
                "query_state": "FINISHED",
                "tables": ["sample.burstbank.account", "sample.burstbank.customer"],
            },
            {
                "engine": "starburst",
                "query_id": "failed-1",
                "raw_sql": "select missing_column from sample.burstbank.account",
                "catalog_name": "sample",
                "schema_name": "burstbank",
                "query_state": "FAILED",
                "tables": ["sample.burstbank.account"],
            },
        ],
    )
    assert rows_written == 2
    raw_rows = repo.list_query_history_for_nlp(
        source_id=source_id,
        engine="starburst",
        missing_only=False,
        limit=10,
    )
    for row in raw_rows:
        repo.upsert_query_history_nlp_row(
            raw_row=row,
            nlp_text=(
                "Show account credit card balances."
                if row["QUERY_ID"] == "finished-1"
                else "Failed query asking for a missing account column."
            ),
            embedding=_vector(1.0),
        )

    monkeypatch.setattr(
        catalog_svc,
        "get_resources",
        lambda: SimpleNamespace(repo=repo, embedder=lambda texts: [_vector(1.0) for _ in texts]),
    )

    result = catalog_svc.generate_context(
        "show account balances",
        source_id=str(source_id),
        engine="starburst",
        catalog="sample",
        schema_name="burstbank",
        schema_k=5,
        query_k=5,
    )

    context = result["context"]
    assert "## In-Context SQL Examples" in context
    assert "**SQL:**" in context
    assert "select acctkey, cc_balance from sample.burstbank.account" in context
    assert "**Natural Language Query:** Show account credit card balances." in context
    assert "missing_column" not in context
    assert context.count("### `sample.burstbank.account`") == 1
    assert "### `sample.burstbank.customer`" in context
    assert "**Table Description:** Table account stores banking account balances and loan states." in context
    assert "| cc_balance | DECIMAL | Column cc_balance is the current credit card balance. |" in context
    assert "| custkey | BIGINT | Unique customer identifier. |" in context
