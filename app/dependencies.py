from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.adapters.registry import EngineRegistry
from app.adk.runtime import AdkDataAssistRuntime
from app.config import settings
from app.core.events import EventBus
from app.core.policy import PolicyChecker
from app.core.store import SQLStore
from app.observability.logging import get_logger
from app.services.auth import AuthService
from app.services.backend_metadata_sync import BackendMetadataSyncService
from app.services.data_usage_nlp import DataUsageNlpService
from app.services.directory import DirectoryService
from app.services.discovery import DiscoveryService
from app.services.discovery_catalog_loader import DiscoveryCatalogLoader
from app.services.embeddings import EmbeddingService
from app.services.query_to_nlp import QueryToNlpService

_logger = get_logger(__name__)


@dataclass
class AppContext:
    store: SQLStore
    engines: EngineRegistry
    auth_service: AuthService
    directory_service: DirectoryService
    discovery_service: DiscoveryService
    discovery_catalog_loader: DiscoveryCatalogLoader
    backend_metadata_sync_service: BackendMetadataSyncService
    data_usage_nlp_service: DataUsageNlpService
    query_to_nlp_service: QueryToNlpService
    embeddings: EmbeddingService
    adk_runtime: AdkDataAssistRuntime
    event_bus: EventBus


def _try_build_skillsql_resources():
    """Attempt to initialize SkillSQL-RL resources (catalog repo + embedder + connector).

    Returns None when the SkillSQL stack is unavailable (missing deps, no catalog
    DSN, or Postgres not reachable). Failures are logged as warnings so the app
    boots cleanly in dev / SQLite mode without a full catalog setup.
    """
    try:
        from skillsql.resources import get_resources
        res = get_resources()
        _logger.info("skillsql_catalog_connected", dialect=res.connector.dialect)
        return res
    except Exception as e:  # noqa: BLE001
        _logger.warning(
            "skillsql_catalog_unavailable",
            error=str(e),
            note="Skill retrieval and formal verification will be skipped.",
        )
        return None


async def build_context() -> AppContext:
    store = SQLStore(
        backend=settings.db_backend,
        sqlite_path=settings.sqlite_path,
        postgres_dsn=settings.postgres_dsn,
        postgres_schema=settings.postgres_schema,
        embedding_dimension=settings.active_embedding_dimension,
    )
    await store.seed()

    event_bus = EventBus()
    engines = EngineRegistry(settings)
    embeddings = EmbeddingService(settings)
    policy = PolicyChecker(settings)
    directory = DirectoryService(settings, store)
    discovery = DiscoveryService(settings, store, embeddings, directory=directory)
    catalog_loader = DiscoveryCatalogLoader(settings, store)
    backend_metadata_sync_service = BackendMetadataSyncService(store, engines)
    data_usage_nlp_service = DataUsageNlpService(settings, store, engines, embeddings)

    # SkillSQL-RL resources (optional; gracefully None when catalog is unavailable)
    skillsql_res = _try_build_skillsql_resources()

    runtime = AdkDataAssistRuntime(
        settings=settings,
        store=store,
        directory=directory,
        policy=policy,
        engines=engines,
        embeddings=embeddings,
        event_bus=event_bus,
        skillsql_resources=skillsql_res,
    )
    query_to_nlp_service = QueryToNlpService(runtime)

    return AppContext(
        store=store,
        engines=engines,
        auth_service=AuthService(directory),
        directory_service=directory,
        discovery_service=discovery,
        discovery_catalog_loader=catalog_loader,
        backend_metadata_sync_service=backend_metadata_sync_service,
        data_usage_nlp_service=data_usage_nlp_service,
        query_to_nlp_service=query_to_nlp_service,
        embeddings=embeddings,
        adk_runtime=runtime,
        event_bus=event_bus,
    )


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx
