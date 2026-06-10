import ssl
from types import SimpleNamespace

import pytest
from skillsql.connectors.base import ReadOnlyViolation
from skillsql.connectors.factory import ConnectorFactory, _ensure_registered, get_connector
from skillsql.connectors.starburst_connector import StarburstConnector

from _fakes import FakeConnector


def test_assert_read_only_raises_on_dml():
    conn = FakeConnector()
    with pytest.raises(ReadOnlyViolation):
        conn.assert_read_only("UPDATE orders SET x = 1")


def test_assert_read_only_allows_select():
    FakeConnector().assert_read_only("SELECT 1")  # no raise


@pytest.mark.asyncio
async def test_execute_read_only_returns_error_not_raises():
    conn = FakeConnector()
    res = await conn.execute("DROP TABLE orders")
    assert res.ok is False and "read_only_violation" in res.error


def test_factory_registers_all_backends():
    _ensure_registered()
    avail = ConnectorFactory.available()
    for name in ("snowflake", "postgres", "starburst", "oracle"):
        assert name in avail


def test_starburst_factory_reads_uppercase_ssl_setting():
    settings = SimpleNamespace(
        DATASOURCE_TYPE="starburst",
        STARBURST_API_URL="https://example.galaxy.starburst.io",
        STARBURST_TRINO_URL="https://example.trino.galaxy.starburst.io",
        STARBURST_USER="user@example.com",
        STARBURST_PASSWORD="password",
        STARBURST_CLIENT_ID="client",
        STARBURST_CLIENT_SECRET="secret",
        STARBURST_VERIFY_SSL=False,
    )

    connector = get_connector(settings=settings)

    assert isinstance(connector, StarburstConnector)
    assert connector.config.verify_ssl is False
    assert connector._ssl_context.verify_mode == ssl.CERT_NONE
