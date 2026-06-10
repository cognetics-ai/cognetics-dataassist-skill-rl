"""Shared runtime dependencies for all ADK agent tools.

``AgentDependencies`` is constructed once per request (via ``app/dependencies.py``)
and injected into every agent's tool functions. It carries both the existing
app-layer services (directory, policy, engines, embeddings) and the SkillSQL-RL
resources (catalog repo, embedder, connector) so tools can access both without
duplicating service construction.

``skillsql_resources`` is optional — the app starts cleanly without a catalog
(dev / SQLite mode). Tools check ``deps.has_catalog`` before calling it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.adapters.registry import EngineRegistry
from app.config import Settings
from app.core.policy import PolicyChecker
from app.core.store import SQLStore
from app.services.directory import DirectoryService

if TYPE_CHECKING:
    from skillsql.resources import Resources

    from app.services.embeddings import EmbeddingService


@dataclass(frozen=True)
class AgentDependencies:
    """Container for runtime services needed by ADK tools.

    Attributes:
        settings:           Global application settings.
        directory:          Directory service for user identity / work profile.
        policy_checker:     Guardrail policy checker for SQL risk and compliance.
        engines:            Registry for pluggable execution engines (Starburst etc.).
        store:              SQL-backed repository for metadata and query history.
        embeddings:         Embedding service for legacy query-history vectors.
        skillsql_resources: SkillSQL-RL resources (catalog repo, embedder, connector).
                            ``None`` when running without a catalog (dev mode).
    """

    settings: Settings
    directory: DirectoryService
    policy_checker: PolicyChecker
    engines: EngineRegistry
    store: SQLStore
    embeddings: EmbeddingService
    skillsql_resources: Resources | None = field(default=None, compare=False)

    @property
    def has_catalog(self) -> bool:
        """True when the SkillSQL-RL catalog is available for skill/schema retrieval."""
        return self.skillsql_resources is not None
