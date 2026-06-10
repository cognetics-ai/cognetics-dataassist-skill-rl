"""Process-level resources.

A small, lazily-initialized container for the expensive shared objects (the
datasource connector, the catalog repository/engine, and the embedder). Workflow
nodes and the API/service layer pull from here instead of constructing their own,
which keeps connection pools and the SQLAlchemy engine singular per process.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .catalog.embeddings import Embedder, get_embedder
from .catalog.repository import CatalogRepository
from .config.settings import Settings, get_settings
from .connectors.base import DataSourceConnector
from .connectors.factory import get_connector


@dataclass
class Resources:
    settings: Settings
    repo: CatalogRepository
    embedder: Embedder
    _connector: DataSourceConnector | None = field(default=None, repr=False)

    @property
    def connector(self) -> DataSourceConnector:
        if self._connector is None:
            self._connector = get_connector(settings=self.settings)
        return self._connector


_RESOURCES: Resources | None = None


def get_resources(settings: Settings | None = None) -> Resources:
    global _RESOURCES
    if _RESOURCES is None:
        s = settings or get_settings()
        _RESOURCES = Resources(settings=s, repo=CatalogRepository(s), embedder=get_embedder(s))
    return _RESOURCES


def reset_resources() -> None:
    """Dispose and clear the cached resources (tests / config reloads)."""
    global _RESOURCES
    if _RESOURCES is not None:
        try:
            _RESOURCES.repo.dispose()
            _RESOURCES.connector.close()
        except Exception:  # noqa: BLE001
            pass
    _RESOURCES = None


# The "active" catalog source the workflow should retrieve against. The benchmark
# sets this per task (different Spider-2.0 databases); inference uses the default.
_ACTIVE_SOURCE = None


def set_active_source(source_id) -> None:
    global _ACTIVE_SOURCE
    _ACTIVE_SOURCE = source_id


def get_active_source():
    """Return the active source id (env override, else the explicitly set one,
    else the most recently registered source, else None)."""
    import os
    import uuid as _uuid

    env_sid = os.environ.get("SKILLSQL_ACTIVE_SOURCE_ID")
    if env_sid:
        try:
            return _uuid.UUID(env_sid)
        except ValueError:
            return None
    if _ACTIVE_SOURCE is not None:
        return _ACTIVE_SOURCE
    try:
        from .catalog.models import Source

        res = get_resources()
        with res.repo.session() as s:
            row = s.query(Source).order_by(Source.created_at.desc()).first()
            return row.id if row else None
    except Exception:  # noqa: BLE001
        return None
