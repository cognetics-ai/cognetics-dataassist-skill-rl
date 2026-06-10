"""App-specific data catalog + SkillBank (persisted to Postgres/pgvector)."""

from .builder import CatalogBuildResult, build_catalog, get_schema_context
from .embeddings import Embedder, OllamaEmbedder, get_embedder
from .models import (
    Base,
    CatalogColumn,
    CatalogTable,
    SchemaDocRow,
    Skill,
    Source,
)
from .repository import CatalogRepository
from .sync import CatalogMetadataSyncResult, sync_metadata_stream

__all__ = [
    "CatalogRepository", "build_catalog", "get_schema_context", "CatalogBuildResult",
    "get_embedder", "Embedder", "OllamaEmbedder",
    "Base", "Source", "CatalogTable", "CatalogColumn", "SchemaDocRow", "Skill",
    "CatalogMetadataSyncResult", "sync_metadata_stream",
]
