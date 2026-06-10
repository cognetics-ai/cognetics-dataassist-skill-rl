"""Catalog builder (Algorithm 1, line "BuildCatalog").

Pipeline
--------
1. ``list_catalogs()``           enumerate all top-level catalogs / databases
2. ``get_metadata(catalog, schema)`` discover tables + columns per (catalog, schema)
3. Column profiling              bounded SELECT DISTINCT to attach sample values
4. LLM description (optional)   enrich text with NL descriptions before embedding
5. Embedding                     vectorize text with the configured embedder
6. Persistence                   upsert into Postgres catalog (sources, tables,
                                  columns, schema_docs with pgvector embeddings)

Starburst multi-catalog support
--------------------------------
For Starburst, ``list_catalogs()`` returns all Galaxy catalog names.  The
builder iterates over every (catalog, schema) pair.  For single-catalog sources
(Snowflake, Postgres), the loop runs once.

All I/O is async.  Column profiling is the only blocking step; it uses the
connector's ``execute()`` which is also async.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from inspect import isawaitable

from ..connectors.base import DataSourceConnector, Metadata
from ..observability.logging import get_logger
from .embeddings import Embedder, get_embedder
from .repository import CatalogRepository

_logger = get_logger(__name__)

# A describer converts a table/column text blob into a short NL description.
# Typically backed by the table_description or column_description ADK agent.
Describer = Callable[[str], str | Awaitable[str]]


@dataclass
class CatalogBuildResult:
    source_type: str
    source_name: str
    source_ids: list[uuid.UUID] = field(default_factory=list)
    catalogs_processed: int = 0
    tables: int = 0
    docs: int = 0
    profiled_columns: int = 0


async def _profile_columns(
    connector: DataSourceConnector,
    meta: Metadata,
    max_samples: int = 5,
) -> int:
    """Attach sample values per column via bounded async queries.

    Failures on individual columns are logged and skipped — one bad column
    never aborts cataloging.
    """
    profiled = 0
    for t in meta.tables:
        for c in t.columns:
            sql = (
                f"SELECT DISTINCT {c.name} AS v "
                f"FROM {t.fqn} "
                f"WHERE {c.name} IS NOT NULL "
                f"LIMIT {max_samples}"
            )
            res = await connector.execute(sql, read_only=True, timeout_s=15, row_cap=max_samples)
            if res.ok and res.rows:
                c.sample_values = [str(r[0]) for r in res.rows]
                profiled += 1
            elif not res.ok:
                _logger.debug(
                    "profile_column_skip",
                    column=f"{t.fqn}.{c.name}", error=res.error,
                )
    return profiled


async def _build_one_catalog(
    connector: DataSourceConnector,
    repo: CatalogRepository,
    embedder: Embedder,
    *,
    catalog_name: str | None,
    db_schema: str | None,
    source_name: str,
    profile: bool,
    describer: Describer | None,
    source_group_id: uuid.UUID | str | None = None,
    source_group_name: str | None = None,
) -> tuple[uuid.UUID, int, int, int]:
    """Discover, enrich, embed, and persist one (catalog, schema) slice.

    Returns (source_id, n_tables, n_docs, n_profiled).
    """
    _logger.debug(
        "Building one catalog: "
        f"catalog_name={catalog_name}, profile {profile}, describer {describer}"
    )
    meta = await connector.get_metadata(catalog_name=catalog_name, db_schema=db_schema)
    _logger.info(
        "catalog_slice_discovered",
        catalog=catalog_name, schema=db_schema, tables=len(meta.tables),
    )

    profiled = await _profile_columns(connector, meta) if profile else 0

    docs = connector.to_schema_docs(meta)
    if describer is not None:
        for d in docs:
            try:
                enriched = describer(d.text)
                if isawaitable(enriched):
                    enriched = await enriched
                d.text = f"{d.text} {enriched}".strip()
            except Exception as exc:  # noqa: BLE001
                _logger.warning("describe_skip", fqn=d.fqn, error=str(exc))

    if docs:
        vectors = embedder([d.text for d in docs])
        for d, v in zip(docs, vectors, strict=True):
            d.embedding = v

    source_id = repo.upsert_source(
        connector.config.source_type,
        source_name,
        catalog_name=catalog_name or meta.catalog_name,
        db_schema=db_schema or meta.db_schema,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
    )
    n_tables = repo.persist_metadata(source_id, meta)
    n_docs = repo.persist_schema_docs(source_id, docs)
    _logger.info(
        "catalog_slice_persisted",
        source_id=str(source_id), catalog=catalog_name,
        tables=n_tables, docs=n_docs,
    )
    return source_id, n_tables, n_docs, profiled


async def build_catalog(
    connector: DataSourceConnector,
    repo: CatalogRepository,
    *,
    embedder: Embedder | None = None,
    describer: Describer | None = None,
    profile: bool = True,
    source_name: str = "default",
    source_group_id: uuid.UUID | str | None = None,
    source_group_name: str | None = None,
    catalog_names: list[str] | None = None,
    db_schema: str | None = None,
) -> CatalogBuildResult:
    """Discover, profile, embed, and persist all catalogs for a datasource.

    Args:
        connector:     The datasource connector (Starburst, Snowflake, Postgres …).
        repo:          Catalog repository (Postgres + pgvector).
        embedder:      Text embedder; defaults to the one configured in settings.
        describer:     Optional LLM describer for richer NL text before embedding.
        profile:       Whether to run column-sampling queries (slow but improves
                       retrieval quality).
        source_name:   Human-readable label for this datasource registration.
        catalog_names: Override the list of catalogs to process.  ``None`` means
                       call ``connector.list_catalogs()`` to enumerate them all.
                       Pass ``[None]`` to force a single call without a catalog filter.
        db_schema:     Restrict discovery to one schema (optional).

    Returns:
        :class:`CatalogBuildResult` aggregating counts across all catalogs.
    """
    embedder = embedder or get_embedder(repo.settings)
    result = CatalogBuildResult(
        source_type=connector.config.source_type,
        source_name=source_name,
    )

    _logger.info(
        "catalog_build_start....",
        source_type=connector.config.source_type,
        source_name=source_name,
        profile=profile,
    )

    # Determine which catalogs to iterate over.
    if catalog_names is None:
        try:
            catalog_names = await connector.list_catalogs()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("list_catalogs_failed", error=str(exc), note="falling back to config")
            catalog_names = [connector.config.catalog_name or connector.config.database]

    # Fall back to a single None-keyed pass for connectors that don't support
    # list_catalogs() (e.g. Postgres with only a DSN and no explicit database name).
    if not catalog_names:
        catalog_names = [None]  # type: ignore[list-item]

    for cat in catalog_names:
        try:
            sid, n_t, n_d, n_p = await _build_one_catalog(
                connector, repo, embedder,
                catalog_name=cat,
                db_schema=db_schema,
                source_name=source_name,
                source_group_id=source_group_id,
                source_group_name=source_group_name,
                profile=profile,
                describer=describer,
            )
            result.source_ids.append(sid)
            result.catalogs_processed += 1
            result.tables += n_t
            result.docs += n_d
            result.profiled_columns += n_p
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "catalog_slice_failed",
                catalog=cat, schema=db_schema, error=str(exc),
                exc_info=True,
            )

    _logger.info(
        "catalog_build_done",
        source_type=connector.config.source_type,
        catalogs=result.catalogs_processed,
        tables=result.tables,
        docs=result.docs,
    )
    return result


def build_catalog_sync(
    connector: DataSourceConnector,
    repo: CatalogRepository,
    **kwargs,
) -> CatalogBuildResult:
    """Synchronous entry point (wraps the async build in a new event loop)."""
    return asyncio.run(build_catalog(connector, repo, **kwargs))


def get_schema_context(
    repo: CatalogRepository,
    question: str,
    *,
    embedder: Embedder | None = None,
    k: int = 15,
    source_id: uuid.UUID | None = None,
    catalog_name: str | None = None,
) -> str:
    """Retrieve top-k schema docs for a question (synchronous).

    Used at inference time; embedding is typically fast (local Ollama).

    Args:
        repo:         Catalog repository.
        question:     Natural-language question for similarity search.
        embedder:     Embedder; defaults to settings embedder.
        k:            Number of schema docs to return.
        source_id:    Restrict to one source (optional).
        catalog_name: Further restrict to one catalog within a source (optional).

    Returns:
        Formatted prompt block: ``- <table/column description>\n…``
    """
    embedder = embedder or get_embedder(repo.settings)
    q_vec = embedder([question])[0]
    docs = repo.search_schema_docs(
        q_vec, k=k, source_id=source_id, catalog_name=catalog_name
    )
    return "\n".join(f"- {d.text}" for d in docs)
