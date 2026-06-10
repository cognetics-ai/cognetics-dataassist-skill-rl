"""Abstract factory for datasource connectors.

Downstream code calls :func:`get_connector` (or ``ConnectorFactory().create(...)``)
and receives a :class:`~skillsql.connectors.base.DataSourceConnector` without
knowing the concrete vendor class. New backends register themselves via
:meth:`ConnectorFactory.register`.
"""

from __future__ import annotations

from typing import ClassVar

from ..config.settings import Settings, get_settings
from .base import DataSourceConnector, SourceConfig


class ConnectorFactory:
    """Registry-backed abstract factory."""

    _registry: ClassVar[dict[str, type[DataSourceConnector]]] = {}

    @classmethod
    def register(cls, source_type: str, connector_cls: type[DataSourceConnector]) -> None:
        cls._registry[source_type.lower()] = connector_cls

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)

    def create(self, source_type: str, config: SourceConfig) -> DataSourceConnector:
        key = source_type.lower()
        try:
            connector_cls = self._registry[key]
        except KeyError as e:
            raise ValueError(
                f"unknown datasource '{source_type}'. registered: {self.available()}"
            ) from e
        return connector_cls(config)


def source_config_from_settings(
    settings: Settings | None = None, source_type: str | None = None
) -> SourceConfig:
    """Build a :class:`SourceConfig` for the configured (or given) datasource.

    Reads all connection parameters from :class:`Settings` (populated from
    environment variables / ``.env``).  Secrets come from the environment;
    never hard-code them here.
    """
    s = settings or get_settings()
    stype = (source_type or s.DATASOURCE_TYPE).lower()

    if stype == "snowflake":
        return SourceConfig(
            source_type="snowflake",
            account=s.SNOWFLAKE_ACCOUNT,
            user=s.SNOWFLAKE_USER,
            password=s.SNOWFLAKE_PASSWORD.get_secret_value() if s.SNOWFLAKE_PASSWORD else None,
            role=s.SNOWFLAKE_ROLE,
            warehouse=s.SNOWFLAKE_WAREHOUSE,
            # Snowflake uses 'database' at the catalog level
            database=s.SNOWFLAKE_DATABASE,
            catalog_name=s.SNOWFLAKE_DATABASE,
            db_schema=s.SNOWFLAKE_SCHEMA,
            authenticator=s.SNOWFLAKE_AUTHENTICATOR,
            private_key_path=s.SNOWFLAKE_PRIVATE_KEY_PATH,
            query_tag=s.SNOWFLAKE_QUERY_TAG,
        )

    if stype == "starburst":
        # Starburst has two explicit hosts: Galaxy API + Trino cluster.
        def _get(k: str, default=None):
            if hasattr(s, k):
                return getattr(s, k)
            upper_key = k.upper()
            if hasattr(s, upper_key):
                return getattr(s, upper_key)
            return default

        api_url = (
            _get("starburst_api_url")
            or _get("starburst_url")
            or (f"https://{_get('starburst_host', '')}:{_get('starburst_port', 443)}")
        )
        trino_url = _get("starburst_trino_url") or (
            f"https://{_get('starburst_trino_host', '')}:{_get('starburst_port', 443)}"
        )
        qh_trino_url = (
            _get("starburst_query_history_trino_url")
            or (
                f"https://{_get('starburst_query_history_trino_host', '')}"
                if _get("starburst_query_history_trino_host")
                else ""
            )
            or trino_url
        )
        pwd = _get("starburst_password")
        if hasattr(pwd, "get_secret_value"):
            pwd = pwd.get_secret_value()
        qh_pwd = _get("starburst_query_history_password") or pwd
        if hasattr(qh_pwd, "get_secret_value"):
            qh_pwd = qh_pwd.get_secret_value()
        cs = _get("starburst_client_secret")
        if hasattr(cs, "get_secret_value"):
            cs = cs.get_secret_value()

        return SourceConfig(
            source_type="starburst",
            host=api_url,
            trino_host=trino_url,
            user=_get("starburst_user"),
            password=pwd,
            role=_get("starburst_role"),
            catalog_name=_get("starburst_catalog"),
            db_schema=_get("starburst_schema"),
            client_id=_get("starburst_client_id"),
            client_secret=cs,
            verify_ssl=_get("starburst_verify_ssl", True),
            timeout_ms=_get("starburst_timeout_ms", 300_000),
            source=_get("starburst_source", "skillsql"),
            query_tag=_get("starburst_query_tag", "skillsql_rl") or "skillsql_rl",
            # Query-history cluster (may differ from primary Trino cluster)
            qh_trino_host=qh_trino_url or None,
            qh_user=_get("starburst_query_history_user") or _get("starburst_user"),
            qh_password=qh_pwd,
            qh_role=_get("starburst_query_history_role") or _get("starburst_role"),
            qh_catalog=_get("starburst_query_history_catalog", "galaxy_telemetry"),
            qh_schema=_get("starburst_query_history_schema", "public"),
            qh_table=_get("starburst_query_history_table", "query_history"),
            qh_source=_get("starburst_query_history_source", "skillsql-qh"),
        )

    if stype == "postgres":
        return SourceConfig(
            source_type="postgres",
            catalog_name=getattr(s, "SNOWFLAKE_DATABASE", None),
            extra={"dsn": s.APP_CATALOG_DSN},
        )

    if stype == "oracle":
        return SourceConfig(
            source_type="oracle",
            host=getattr(s, "ORACLE_HOST", None),
            port=getattr(s, "ORACLE_PORT", 1521),
            user=getattr(s, "ORACLE_USER", None),
            password=getattr(s, "ORACLE_PASSWORD", None),
            db_schema=getattr(s, "ORACLE_SCHEMA", None),
        )

    # Unknown type — return minimal config so the factory can raise a clear error.
    return SourceConfig(source_type=stype)


def get_connector(
    source_type: str | None = None, settings: Settings | None = None
) -> DataSourceConnector:
    """Convenience: build the connector for the configured (or given) datasource."""
    _ensure_registered()
    s = settings or get_settings()
    stype = source_type or s.DATASOURCE_TYPE
    cfg = source_config_from_settings(s, stype)
    return ConnectorFactory().create(stype, cfg)


def _ensure_registered() -> None:
    """Import concrete connectors so they self-register (idempotent)."""
    from . import (  # noqa: F401
        oracle_connector,
        postgres_connector,
        snowflake_connector,
        starburst_connector,
    )
