"""Catalog and SkillSQL-RL service layer.

Provides the business-logic functions that HTTP routes and CLI commands call.
All operations are thin adaptors over the pure framework modules in ``skillsql/``:
no SQL dialect knowledge, no protocol code, no embedding math lives here.

Planes served
-------------
Cataloging plane (read/write catalog):
    init_db, build_catalog, get_schema_context, search_columns

Inference plane (verification, scoring, SQL generation, end-to-end workflow):
    verify_sql, score_sql, generate_sql, run_text2sql

These functions are intentionally callable both from FastAPI route handlers and
from scripts / CLI commands without any HTTP context.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from skillsql.resources import get_active_source, get_resources, set_active_source
from skillsql.verification.static_gates import run_static_lattice

from app.adapters.base import BackendMetadataRecord, BackendQueryHistoryRecord
from app.adapters.registry import EngineRegistry
from app.core.sql_utils import extract_tables
from app.observability.logging import get_logger

_logger = get_logger(__name__)  # use 'log' consistently — matches every log.info/log.debug below

_METADATA_DESCRIBE_IGNORED_WARNING = (
    "describe ignored during metadata sync; use "
    "/catalog/metadata/table-description/sync or "
    "/catalog/metadata/column-description/sync for natural-language descriptions."
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_source(source_id: str | uuid.UUID | None) -> uuid.UUID | None:
    """Return the active source UUID, accepting string or UUID input."""
    if source_id is None:
        return get_active_source()
    return uuid.UUID(source_id) if isinstance(source_id, str) else source_id


def _known_tables(source_id: uuid.UUID | None) -> set[str] | None:
    """Return lowercased table identifiers for the static bind gate."""
    res = get_resources()
    try:
        from skillsql.catalog.models import CatalogTable

        with res.repo.session() as s:
            q = s.query(CatalogTable)
            if source_id:
                q = q.filter(CatalogTable.source_id == source_id)
            rows = q.all()
        known: set[str] = set()
        for r in rows:
            known.add(r.name.lower())
            known.add(r.fqn.lower())
        return known or None
    except Exception:  # noqa: BLE001
        return None


def get_resources_for_api():
    """Convenience re-export for routes that need the Resources object directly."""
    return get_resources()


# ── Cataloging plane ───────────────────────────────────────────────────────────

def init_db(*, reset: bool = False) -> dict[str, Any]:
    """Create the app catalog Postgres schema, pgvector extension, and all tables.

    Idempotent by default. With ``reset=True``, drops and recreates all
    catalog-owned tables.
    """
    repo = get_resources().repo
    if reset:
        repo.reset_schema()
        _logger.warning("catalog_schema_reset", schema=repo.schema)
        return {"status": "ok", "schema": repo.schema, "reset": True}
    repo.init_schema()
    _logger.info("catalog_schema_initialized", schema=repo.schema)
    return {"status": "ok", "schema": repo.schema, "reset": False}


async def build_catalog(
    engines: EngineRegistry | None = None,
    *,
    source_type: str | None = None,
    source_name: str = "default",
    source_group_id: str | None = None,
    source_group_name: str | None = None,
    catalog: str | None = None,
    catalog_names: list[str] | None = None,
    database_name: str | None = None,
    db_schema: str | None = None,
    profile: bool = True,
    describe: bool = False,
    description_runtime: Any | None = None,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Discover + embed + persist a datasource catalog into Postgres.

    Args:
        engines:       Optional app engine registry. When supplied with a concrete
                       catalog/database and schema, the build uses engine metadata
                       streaming plus ADK description agents.
        source_type:   Override the default DATASOURCE_TYPE setting.
        source_name:   Human label for this catalog registration.
        catalog:       Single Starburst catalog or logical top-level namespace.
        catalog_names: Starburst: specific Galaxy catalogs to index. ``None`` indexes all.
        database_name: Physical database name for database-scoped sources. Used as the
                       catalog/database to index when catalog_names is not supplied.
        db_schema:     Restrict to one schema (optional).
        profile:       Run bounded column-sampling queries to collect sample values.
        describe:      Generate and persist NL table/column descriptions (slower).
        description_runtime: Runtime exposing table/column description agents.
        sample_size:   Sample rows/values per table or column for description context.

    Returns:
        Dict with source_ids, catalogs_processed, tables, docs counts.
    """
    from skillsql.catalog.builder import build_catalog as _build
    from skillsql.connectors.factory import get_connector

    res = get_resources()
    source_type = source_type or res.settings.DATASOURCE_TYPE
    _logger.debug(f"Incoming list of catalogs: {catalog_names}")
    requested_catalog = _single_requested_catalog(
        catalog=catalog,
        catalog_names=catalog_names,
        database_name=database_name,
    )
    _logger.debug(f"_single_requested_catalog:: Catalog requested: {requested_catalog}")
    if engines is not None and db_schema and requested_catalog:
        return await _build_catalog_schema_scope(
            engines,
            description_runtime,
            engine=source_type,
            source_name=source_name,
            source_group_id=source_group_id,
            source_group_name=source_group_name,
            catalog=requested_catalog,
            database_name=database_name,
            db_schema=db_schema,
            describe=describe,
            sample_size=sample_size,
        )
    if engines is not None and describe:
        raise ValueError(
            "Catalog build with describe=true requires catalog or database_name plus db_schema."
        )

    connector = get_connector(source_type=source_type, settings=res.settings)
    if describe:
        _logger.info(
            "catalog_build_describe_ignored_without_schema_runtime",
            note=(
                "Generic catalog_describer was removed. Use schema-scoped /catalog/build "
                "or the table/column description sync endpoints for NL descriptions."
            ),
        )
    requested_catalog_names = catalog_names
    if requested_catalog_names is None and requested_catalog:
        requested_catalog_names = [requested_catalog]

    result = await _build(
        connector,
        res.repo,
        embedder=res.embedder,
        describer=None,
        profile=profile,
        source_name=source_name,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
        catalog_names=requested_catalog_names,
        db_schema=db_schema,
    )
    # Set the first source as active so subsequent calls default to it.
    if result.source_ids:
        set_active_source(result.source_ids[0])

    return {
        "source_ids": [str(sid) for sid in result.source_ids],
        "catalogs_processed": result.catalogs_processed,
        "tables": result.tables,
        "docs": result.docs,
        "profiled_columns": result.profiled_columns,
        "source_type": result.source_type,
        "source_name": result.source_name,
    }


def _single_requested_catalog(
    *,
    catalog: str | None,
    catalog_names: list[str] | None,
    database_name: str | None,
) -> str | None:
    catalog_value = str(catalog or "").strip()
    if catalog_value:
        return catalog_value
    names = [str(item).strip() for item in catalog_names or [] if str(item).strip()]
    if len(names) == 1:
        return names[0]
    database_value = str(database_name or "").strip()
    return database_value or None


async def _build_catalog_schema_scope(
    engines: EngineRegistry,
    description_runtime: Any | None,
    *,
    engine: str,
    source_name: str,
    source_group_id: str | None,
    source_group_name: str | None,
    catalog: str,
    database_name: str | None,
    db_schema: str,
    describe: bool,
    sample_size: int,
) -> dict[str, Any]:
    """Build one catalog/database + schema and enrich all tables/columns."""
    _logger.debug(f"_build_catalog_schema_scope:: Incoming catalog: {catalog}")
    adapter = engines.get(engine)
    metadata_result = await _sync_metadata_records(
        adapter.iter_schema_metadata(
            catalog,
            db_schema,
            database_name=database_name,
            include_columns=True,
        ),
        source_type=adapter.name,
        source_name=source_name,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
        scope_type="schema",
        catalog_name=catalog,
        database_name=database_name,
        db_schema=db_schema,
        table_name=None,
        column_name=None,
        describe=False,
        doc_batch_size=128,
    )

    description_sync: dict[str, Any] = {
        "enabled": bool(describe),
        "tables": _empty_description_counts(),
        "columns": _empty_description_counts(),
    }
    if describe:
        _logger.info("Describing Catalog.....")
        if description_runtime is None:
            raise ValueError(
                "Catalog build with describe=true requires an ADK description runtime."
            )
        description_sync = await _describe_built_schema_scope(
            engines,
            description_runtime,
            engine=adapter.name,
            source_name=source_name,
            catalog=catalog,
            database_name=database_name,
            db_schema=db_schema,
            table_count=int(metadata_result.get("table_rows") or 0),
            sample_size=sample_size,
        )

    return {
        "source_ids": [metadata_result["source_id"]],
        "catalogs_processed": 1,
        "tables": metadata_result["table_rows"],
        "columns": metadata_result["column_rows"],
        "docs": metadata_result["docs"],
        "profiled_columns": 0,
        "source_type": adapter.name,
        "source_name": source_name,
        "scope": metadata_result["scope"],
        "metadata_sync": metadata_result,
        "description_sync": description_sync,
    }


async def _describe_built_schema_scope(
    engines: EngineRegistry,
    description_runtime: Any,
    *,
    engine: str,
    source_name: str,
    catalog: str,
    database_name: str | None,
    db_schema: str,
    table_count: int,
    sample_size: int,
) -> dict[str, Any]:
    """Run table and column agents for every table in a built schema scope."""
    _logger.debug(f"_describe_built_schema_scope:: Incoming catalog: {catalog}, table count: "
                  f"{table_count}, sample_size: {sample_size}, source_name: {source_name}")
    res = get_resources()
    table_limit = max(1, table_count)
    tables = res.repo.list_tables_for_description(
        source_type=engine,
        source_name=source_name,
        catalog_name=catalog,
        database_name=database_name,
        db_schema=db_schema,
        missing_only=False,
        limit=table_limit,
    )

    _logger.debug(f"Tables in describe built schema scope: {tables}")
    
    table_descriptions = await sync_table_descriptions(
        engines,
        description_runtime,
        engine=engine,
        source_name=source_name,
        catalog=catalog,
        database_name=database_name,
        schema_name=db_schema,
        missing_only=True,
        limit=table_limit,
        sample_size=sample_size,
    )

    column_results: list[dict[str, Any]] = []
    for table in tables:
        column_results.append(
            await sync_column_descriptions(
                engines,
                description_runtime,
                engine=engine,
                source_name=source_name,
                catalog=table.catalog_name or catalog,
                database_name=database_name,
                schema_name=table.db_schema or db_schema,
                table_name=table.name,
                missing_only=True,
                limit=max(1, len(table.columns or [])),
                sample_size=sample_size,
            )
        )

    return {
        "enabled": True,
        "tables": _description_counts(table_descriptions),
        "columns": _combine_description_counts(column_results),
    }


def _empty_description_counts() -> dict[str, int]:
    return {
        "candidate_count": 0,
        "updated_count": 0,
        "skipped_count": 0,
        "embeddings_generated": 0,
        "embedding_retries": 0,
    }


def _description_counts(result: dict[str, Any]) -> dict[str, int]:
    return {
        "candidate_count": int(result.get("candidate_count") or 0),
        "updated_count": int(result.get("updated_count") or 0),
        "skipped_count": int(result.get("skipped_count") or 0),
        "embeddings_generated": int(result.get("embeddings_generated") or 0),
        "embedding_retries": int(result.get("embedding_retries") or 0),
    }


def _combine_description_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    combined = _empty_description_counts()
    for result in results:
        counts = _description_counts(result)
        for key in combined:
            combined[key] += counts[key]
    return combined


def build_catalog_sync(**kwargs) -> dict[str, Any]:
    """Synchronous wrapper for scripts and CLI commands."""
    return asyncio.run(build_catalog(**kwargs))


def available_datasources(engines: EngineRegistry) -> dict[str, Any]:
    """Return engine names available through the shared adapter factory."""
    return {"datasources": engines.available()}


async def sync_catalog_metadata(
    engines: EngineRegistry,
    *,
    engine: str,
    catalog: str,
    database_name: str | None = None,
    source_name: str | None = None,
    source_group_id: str | None = None,
    source_group_name: str | None = None,
    include_columns: bool = True,
    describe: bool = False,
    doc_batch_size: int = 128,
) -> dict[str, Any]:
    """Stream a catalog scope into the SkillSQL semantic catalog."""
    adapter = engines.get(engine)
    records = adapter.iter_catalog_metadata(
        catalog,
        database_name=database_name,
        include_columns=include_columns,
    )
    return await _sync_metadata_records(
        records,
        source_type=adapter.name,
        source_name=source_name or adapter.name,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
        scope_type="catalog",
        catalog_name=catalog,
        database_name=database_name,
        db_schema=None,
        table_name=None,
        column_name=None,
        describe=describe,
        doc_batch_size=doc_batch_size,
    )


async def sync_schema_metadata(
    engines: EngineRegistry,
    *,
    engine: str,
    catalog: str,
    database_name: str | None = None,
    schema_name: str,
    source_name: str | None = None,
    source_group_id: str | None = None,
    source_group_name: str | None = None,
    include_columns: bool = True,
    describe: bool = False,
    doc_batch_size: int = 128,
) -> dict[str, Any]:
    """Stream a schema scope into the SkillSQL semantic catalog."""
    _logger.debug(
        "sync_schema_metadata_start",
        engine=engine,
        catalog=catalog,
        database_name=database_name,
        schema_name=schema_name,
    )
    adapter = engines.get(engine)
    records = adapter.iter_schema_metadata(
        catalog,
        schema_name,
        database_name=database_name,
        include_columns=include_columns,
    )
    return await _sync_metadata_records(
        records,
        source_type=adapter.name,
        source_name=source_name or adapter.name,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
        scope_type="schema",
        catalog_name=catalog,
        database_name=database_name,
        db_schema=schema_name,
        table_name=None,
        column_name=None,
        describe=describe,
        doc_batch_size=doc_batch_size,
    )


async def sync_table_metadata(
    engines: EngineRegistry,
    *,
    engine: str,
    catalog: str,
    database_name: str | None = None,
    schema_name: str,
    table_name: str,
    source_name: str | None = None,
    source_group_id: str | None = None,
    source_group_name: str | None = None,
    include_columns: bool = True,
    describe: bool = False,
    doc_batch_size: int = 128,
) -> dict[str, Any]:
    """Stream a table scope into the SkillSQL semantic catalog."""
    adapter = engines.get(engine)
    _logger.debug("Iterating over metadata...")
    records = adapter.iter_table_metadata(
        catalog,
        schema_name,
        table_name,
        database_name=database_name,
        include_columns=include_columns,
    )
    return await _sync_metadata_records(
        records,
        source_type=adapter.name,
        source_name=source_name or adapter.name,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
        scope_type="table",
        catalog_name=catalog,
        database_name=database_name,
        db_schema=schema_name,
        table_name=table_name,
        column_name=None,
        describe=describe,
        doc_batch_size=doc_batch_size,
    )


async def sync_column_metadata(
    engines: EngineRegistry,
    *,
    engine: str,
    catalog: str,
    database_name: str | None = None,
    schema_name: str,
    table_name: str,
    column_name: str,
    source_name: str | None = None,
    source_group_id: str | None = None,
    source_group_name: str | None = None,
    describe: bool = False,
    doc_batch_size: int = 128,
) -> dict[str, Any]:
    """Stream one column under a table into the SkillSQL semantic catalog."""
    adapter = engines.get(engine)
    records = _filter_column_records(
        adapter.iter_table_metadata(
            catalog,
            schema_name,
            table_name,
            database_name=database_name,
            include_columns=True,
        ),
        column_name=column_name,
    )
    return await _sync_metadata_records(
        records,
        source_type=adapter.name,
        source_name=source_name or adapter.name,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
        scope_type="column",
        catalog_name=catalog,
        database_name=database_name,
        db_schema=schema_name,
        table_name=table_name,
        column_name=column_name,
        describe=describe,
        doc_batch_size=doc_batch_size,
    )


async def sync_table_descriptions(
    engines: EngineRegistry,
    description_runtime: Any,
    *,
    engine: str,
    catalog: str,
    database_name: str | None = None,
    schema_name: str,
    table_name: str | None = None,
    source_name: str | None = None,
    missing_only: bool = True,
    limit: int = 50,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Generate and persist table descriptions in the SkillSQL catalog backend."""
    adapter = engines.get(engine)
    res = get_resources()
    candidates = res.repo.list_tables_for_description(
        source_type=adapter.name,
        source_name=source_name,
        catalog_name=catalog,
        database_name=database_name,
        db_schema=schema_name,
        table_name=table_name,
        missing_only=missing_only,
        limit=limit,
    )
    _logger.debug(f"Candidate tables for description: {candidates}")

    items: list[dict[str, Any]] = []
    updated_count = 0
    embeddings_generated = 0
    embedding_retries = 0
    for table in candidates:
        physical_catalog = table.catalog_name or database_name or catalog
        response_base = _table_description_response_base(
            adapter.name,
            catalog,
            database_name or table.catalog_name,
            table,
        )
        try:
            output = await description_runtime.describe_catalog_table(
                engine=adapter.name,
                catalog=physical_catalog,
                schema_name=table.db_schema or schema_name,
                table_name=table.name,
                sample_size=sample_size,
            )
            description = str(output.get("description") or "").strip()
            caveats = [str(item) for item in output.get("caveats") or [] if str(item).strip()]
            confidence = _float(output.get("confidence"))
            updated = False
            embedding_generated = False
            if description:
                _logger.info(f"Updating description for table {table.name}....desc: {description}")
                updated = res.repo.update_table_description(
                    table.id,
                    description=description,
                    confidence=confidence,
                )
                doc = _table_schema_doc(table, description)
                _logger.debug(f"_table_schema_doc:: Table doc: {doc}")
                vectors = res.embedder([doc.text])
                if vectors:
                    doc.embedding = vectors[0]
                    embedding_generated = True
                    embeddings_generated += 1
                res.repo.upsert_schema_docs(table.source_id, [doc])
            if updated:
                updated_count += 1
            items.append(
                {
                    **response_base,
                    "description": description,
                    "confidence": confidence,
                    "observed_entities": list(output.get("observed_entities") or []),
                    "likely_grain": str(output.get("likely_grain") or ""),
                    "embedding_generated": embedding_generated,
                    "embedding_retries": 0,
                    "updated": updated,
                    "caveats": caveats,
                }
            )
        except Exception as exc:  # noqa: BLE001
            _logger.exception(
                "catalog_table_description_sync_failed",
                catalog=catalog,
                database=database_name,
                schema=table.db_schema,
                table=table.name,
            )
            items.append(
                {
                    **response_base,
                    "description": "",
                    "confidence": 0.0,
                    "observed_entities": [],
                    "likely_grain": "",
                    "embedding_generated": False,
                    "embedding_retries": 0,
                    "updated": False,
                    "caveats": [str(exc)],
                }
            )

    return {
        "synced_at": datetime.now(UTC).isoformat(),
        "candidate_count": len(candidates),
        "updated_count": updated_count,
        "embeddings_generated": embeddings_generated,
        "embedding_retries": embedding_retries,
        "skipped_count": len(candidates) - updated_count,
        "items": items,
    }


async def sync_column_descriptions(
    engines: EngineRegistry,
    description_runtime: Any,
    *,
    engine: str,
    catalog: str,
    database_name: str | None = None,
    schema_name: str,
    table_name: str | None = None,
    column_name: str | None = None,
    column_names: list[str] | None = None,
    source_name: str | None = None,
    missing_only: bool = True,
    limit: int = 500,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Generate and persist column descriptions in the SkillSQL catalog backend."""
    _logger.info(
        f"Sync column descriptions for {engine} engine, "
        f"schema_name: {schema_name}, table_name: {table_name or '*'}, column_name: {column_name}"
    )
    adapter = engines.get(engine)
    res = get_resources()
    requested_columns = _normalize_column_names(column_name, column_names)
    candidates = res.repo.list_columns_for_description(
        source_type=adapter.name,
        source_name=source_name,
        catalog_name=catalog,
        database_name=database_name,
        db_schema=schema_name,
        table_name=table_name,
        column_names=requested_columns or None,
        missing_only=missing_only,
        limit=limit,
    )
    _logger.debug(f"Candidates for col desc: {candidates}")

    grouped: dict[uuid.UUID, list[Any]] = {}
    for column in candidates:
        grouped.setdefault(column.table_id, []).append(column)

    items: list[dict[str, Any]] = []
    updated_count = 0
    embeddings_generated = 0
    embedding_retries = 0
    for columns in grouped.values():
        table = columns[0].table
        physical_catalog = table.catalog_name or database_name or catalog
        metadata = [_column_metadata_item(column) for column in columns]
        try:
            output = await description_runtime.describe_catalog_columns(
                engine=adapter.name,
                catalog=physical_catalog,
                schema_name=table.db_schema or schema_name,
                table_name=table.name,
                column_name=columns[0].name if len(columns) == 1 else None,
                column_metadata=metadata,
                sample_size=sample_size,
            )
            output_caveats = [
                str(item) for item in output.get("caveats") or [] if str(item).strip()
            ]
            output_by_name = {
                _name_key(item.get("column_name")): item
                for item in output.get("columns") or []
                if _name_key(item.get("column_name"))
            }
            for column in columns:
                response_base = _column_description_response_base(
                    adapter.name,
                    catalog,
                    database_name or table.catalog_name,
                    column,
                )
                output_item = output_by_name.get(_name_key(column.name))
                if not output_item:
                    items.append(
                        {
                            **response_base,
                            "description": "",
                            "confidence": 0.0,
                            "semantic_type": "",
                            "sample_values": [],
                            "embedding_generated": False,
                            "embedding_retries": 0,
                            "updated": False,
                            "caveats": output_caveats
                            + ["Column description agent did not return this column."],
                        }
                    )
                    continue

                description = str(output_item.get("description") or "").strip()
                caveats = [
                    str(item)
                    for item in output_item.get("caveats") or []
                    if str(item).strip()
                ]
                updated = False
                embedding_generated = False
                if description:
                    updated = res.repo.update_column_description(column.id, description=description)
                    doc = _column_schema_doc(column, description)
                    vectors = res.embedder([doc.text])
                    if vectors:
                        doc.embedding = vectors[0]
                        embedding_generated = True
                        embeddings_generated += 1
                    res.repo.upsert_schema_docs(table.source_id, [doc])
                if updated:
                    updated_count += 1
                items.append(
                    {
                        **response_base,
                        "description": description,
                        "confidence": _float(output_item.get("confidence")),
                        "semantic_type": str(output_item.get("semantic_type") or ""),
                        "sample_values": [
                            str(value) for value in output_item.get("sample_values") or []
                        ],
                        "embedding_generated": embedding_generated,
                        "embedding_retries": 0,
                        "updated": updated,
                        "caveats": output_caveats + caveats,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            _logger.exception(
                "catalog_column_description_sync_failed",
                catalog=catalog,
                database=database_name,
                schema=table.db_schema,
                table=table.name,
            )
            for column in columns:
                items.append(
                    {
                        **_column_description_response_base(
                            adapter.name,
                            catalog,
                            database_name or table.catalog_name,
                            column,
                        ),
                        "description": "",
                        "confidence": 0.0,
                        "semantic_type": "",
                        "sample_values": [],
                        "embedding_generated": False,
                        "embedding_retries": 0,
                        "updated": False,
                        "caveats": [str(exc)],
                    }
                )

    return {
        "synced_at": datetime.now(UTC).isoformat(),
        "candidate_count": len(candidates),
        "updated_count": updated_count,
        "embeddings_generated": embeddings_generated,
        "embedding_retries": embedding_retries,
        "skipped_count": len(candidates) - updated_count,
        "items": items,
    }


async def sync_query_history(
    engines: EngineRegistry,
    store: Any,
    *,
    engine: str,
    source_name: str | None = None,
    source_group_id: str | None = None,
    source_group_name: str | None = None,
    start_time: datetime | str | None = None,
    end_time: datetime | str | None = None,
    catalog: str | None = None,
    database_name: str | None = None,
    schema_name: str | None = None,
    table_name: str | None = None,
    limit: int | None = None,
    page_size: int = 1000,
    batch_size: int = 500,
) -> dict[str, Any]:
    """Load datasource query history into catalog-owned history tables."""
    adapter = engines.get(engine)
    repo = get_resources().repo
    source_id = _ensure_catalog_source_id(
        source_type=adapter.name,
        source_name=source_name,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
        catalog=catalog,
        database_name=database_name,
        schema_name=schema_name,
    )
    rows_synced = await _sync_query_history_records(
        repo,
        adapter.iter_query_history(
            start_time=start_time,
            end_time=end_time,
            catalog=catalog,
            schema=schema_name,
            table=table_name,
            limit=limit,
            page_size=page_size,
        ),
        source_id=str(source_id),
        batch_size=batch_size,
    )
    return {
        "synced_at": datetime.now(UTC).isoformat(),
        "engine": adapter.name,
        "source_id": str(source_id),
        "scope": {
            "source_name": source_name or adapter.name,
            "source_group_id": source_group_id,
            "source_group_name": source_group_name,
            "start_time": start_time,
            "end_time": end_time,
            "catalog": catalog,
            "database": database_name,
            "schema": schema_name,
            "table": table_name,
            "limit": limit,
        },
        "query_history_rows": rows_synced,
    }


async def sync_query_history_nlp(
    store: Any,
    description_runtime: Any,
    embeddings: Any,
    *,
    engine: str,
    source_id: str | None = None,
    source_name: str | None = None,
    source_group_id: str | None = None,
    source_group_name: str | None = None,
    catalog: str | None = None,
    database_name: str | None = None,
    schema_name: str | None = None,
    ids: list[int] | None = None,
    raw_sql: str | None = None,
    limit: int = 100,
    missing_only: bool = True,
) -> dict[str, Any]:
    """Generate NLP text and embeddings for catalog-owned raw query history."""
    repo = get_resources().repo
    normalized_source_id = source_id
    if not normalized_source_id and not source_group_id:
        normalized_source_id = str(
            _ensure_catalog_source_id(
                source_type=engine,
                source_name=source_name,
                source_group_id=source_group_id,
                source_group_name=source_group_name,
                catalog=catalog,
                database_name=database_name,
                schema_name=schema_name,
            )
        )
    normalized_ids = sorted({int(item) for item in (ids or []) if int(item) > 0})
    if ids is not None and not normalized_ids:
        return {
            "synced_at": datetime.now(UTC).isoformat(),
            "source_rows": 0,
            "inserted_rows": 0,
            "embeddings_generated": 0,
            "embedding_retries": 0,
            "rows_failed": 0,
            "items": [],
        }

    source_rows = repo.list_query_history_for_nlp(
        source_id=normalized_source_id,
        source_group_id=source_group_id if not normalized_source_id else None,
        engine=engine,
        ids=normalized_ids or None,
        raw_sql=raw_sql,
        limit=len(normalized_ids) if normalized_ids else (1 if raw_sql else limit),
        missing_only=missing_only,
    )

    items: list[dict[str, Any]] = []
    inserted_count = 0
    embeddings_generated = 0
    embedding_retries = 0
    for row in source_rows:
        row_engine = str(_row_value(row, "ENGINE") or engine).strip()
        row_query_id = str(_row_value(row, "QUERY_ID") or "").strip()
        row_id = int(_row_value(row, "ID") or 0)
        row_sql = str(_row_value(row, "RAW_SQL") or "")
        row_source_id = str(_row_value(row, "SOURCE_ID") or normalized_source_id or "").strip()
        if not row_source_id or not row_engine or not row_query_id or not row_sql.strip():
            items.append(
                _query_nlp_history_item(
                    row,
                    engine,
                    "",
                    False,
                    [
                        "Catalog query history row is missing source id, "
                        "engine, query id, or raw SQL."
                    ],
                )
            )
            continue

        try:
            output = await description_runtime.describe_query_history_nlp(
                engine=row_engine,
                raw_history_id=row_id,
                raw_sql=row_sql,
                source_id=row_source_id,
            )
            nlp_text = str(output.get("query_nlp") or output.get("nlp_text") or "").strip()
            nlp_text = _query_history_nlp_with_failure_context(row, nlp_text)
            inserted = False
            embedding: list[float] = []
            embedding_retry_count = 0
            caveats = [str(item) for item in output.get("caveats") or [] if str(item).strip()]
            if nlp_text:
                embedding, embedding_retry_count = await embeddings.embed_document(nlp_text)
                embedding_retries += embedding_retry_count
                if embedding:
                    embeddings_generated += 1
                else:
                    caveats.append("Embedding generation returned no vector.")
                inserted = repo.upsert_query_history_nlp_row(
                    raw_row=row,
                    nlp_text=nlp_text,
                    embedding=embedding or None,
                )
            if inserted:
                inserted_count += 1
            items.append(
                _query_nlp_history_item(
                    row,
                    engine,
                    nlp_text,
                    inserted,
                    caveats,
                    embedding_generated=bool(embedding),
                    embedding_retries=embedding_retry_count,
                )
            )
        except Exception as exc:  # noqa: BLE001
            _logger.exception("Catalog query NLP history sync failed for query_id=%s", row_query_id)
            items.append(_query_nlp_history_item(row, engine, "", False, [str(exc)]))

    return {
        "synced_at": datetime.now(UTC).isoformat(),
        "source_rows": len(source_rows),
        "inserted_rows": inserted_count,
        "embeddings_generated": embeddings_generated,
        "embedding_retries": embedding_retries,
        "rows_failed": len(source_rows) - inserted_count,
        "items": items,
    }


def get_schema_context(
    question: str,
    *,
    source_id: str | None = None,
    source_group_id: str | None = None,
    k: int = 15,
    query_k: int = 5,
    engine: str | None = None,
    catalog: str | None = None,
    database_name: str | None = None,
    schema_name: str | None = None,
) -> dict[str, Any]:
    """Build grouped schema context plus FINISHED query-history examples."""
    return generate_context(
        question,
        source_id=source_id,
        source_group_id=source_group_id,
        engine=engine,
        catalog=catalog,
        database_name=database_name,
        schema_name=schema_name,
        schema_k=k,
        query_k=query_k,
    )


def get_query_history_context(
    question: str,
    *,
    source_id: str | None = None,
    source_group_id: str | None = None,
    engine: str | None = None,
    catalog: str | None = None,
    database_name: str | None = None,
    schema_name: str | None = None,
    k: int = 5,
    query_embedding: list[float] | None = None,
) -> dict[str, Any]:
    """Retrieve FINISHED query-history examples for in-context SQL samples."""
    normalized_question = str(question or "").strip()
    if not normalized_question:
        raise ValueError("question is required")

    res = get_resources()
    sid = _resolve_source(source_id)
    group_id = _resolve_source(source_group_id)
    q_vec = query_embedding or res.embedder([normalized_question])[0]
    effective_catalog = database_name or catalog
    rows = res.repo.search_query_history(
        q_vec,
        k=k,
        source_id=sid,
        source_group_id=group_id if not sid else None,
        engine=engine,
        catalog=effective_catalog,
        schema_name=schema_name,
        query_state="FINISHED",
    )
    examples = [_query_history_context_example(row) for row in rows]
    table_refs = _dedupe_backend_table_refs(
        [table for row in rows for table in _extract_row_tables(row)]
    )
    return {
        "question": normalized_question,
        "source_id": str(sid) if sid else None,
        "source_group_id": str(group_id) if group_id else None,
        "examples": examples,
        "table_refs": table_refs,
        "query_examples_retrieved": len(examples),
    }


def generate_context(
    question: str,
    *,
    source_id: str | None = None,
    source_group_id: str | None = None,
    engine: str | None = None,
    catalog: str | None = None,
    database_name: str | None = None,
    schema_name: str | None = None,
    schema_k: int = 15,
    query_k: int = 5,
) -> dict[str, Any]:
    """Generate markdown context from schema docs and FINISHED query-history NLP.

    The table/column section is grouped by table. Tables are ordered by first
    evidence: schema-doc rank first, then tables referenced by similar queries.
    """
    normalized_question = str(question or "").strip()
    if not normalized_question:
        raise ValueError("question is required")

    res = get_resources()
    sid = _resolve_source(source_id)
    group_id = _resolve_source(source_group_id)
    effective_catalog = database_name or catalog
    q_vec = res.embedder([normalized_question])[0]

    schema_docs = res.repo.search_schema_docs(
        q_vec,
        k=schema_k,
        source_id=sid,
        source_group_id=group_id if not sid else None,
        catalog_name=effective_catalog,
        db_schema=schema_name,
    )
    query_history = get_query_history_context(
        normalized_question,
        source_id=str(sid) if sid else None,
        source_group_id=str(group_id) if group_id and not sid else None,
        engine=engine,
        catalog=catalog,
        database_name=database_name,
        schema_name=schema_name,
        k=query_k,
        query_embedding=q_vec,
    )
    examples = list(query_history.get("examples") or [])

    schema_doc_table_refs = _schema_doc_table_refs(schema_docs)
    query_table_refs = list(query_history.get("table_refs") or [])
    table_refs = _dedupe_backend_table_refs([*schema_doc_table_refs, *query_table_refs])
    table_context = res.repo.table_context_for_refs(
        table_refs,
        source_id=sid,
        source_group_id=group_id if not sid else None,
        catalog_name=effective_catalog,
        schema_name=schema_name,
    )
    table_context = _merge_schema_docs_into_table_context(table_context, schema_docs)
    table_context = _append_schema_doc_only_tables(table_context, schema_docs)

    context_text = _format_generated_context_markdown(examples, table_context)
    return {
        "question": normalized_question,
        "source_id": str(sid) if sid else None,
        "source_group_id": str(group_id) if group_id else None,
        "engine": engine or "",
        "catalog": catalog or "",
        "database_name": database_name or "",
        "schema_name": schema_name or "",
        "context": context_text,
        "schema_context": context_text,
        "docs_retrieved": len(schema_docs),
        "query_examples_retrieved": len(examples),
        "tables": table_context,
        "examples": examples,
        "table_refs": table_refs,
    }


def search_columns(
    query: str,
    *,
    source_id: str | None = None,
    source_group_id: str | None = None,
    k: int = 15,
) -> dict[str, Any]:
    """Vector search over column descriptions for a query string.

    Returns the top-k matching column docs ranked by cosine similarity.
    """
    res = get_resources()
    sid = _resolve_source(source_id)
    group_id = _resolve_source(source_group_id)
    q_vec = res.embedder([query])[0]
    rows = res.repo.search_schema_docs(
        q_vec,
        k=k * 2,
        source_id=sid,
        source_group_id=group_id if not sid else None,
    )
    cols = [
        {
            "fqn": r.fqn,
            "catalog": r.catalog_name,
            "schema": r.db_schema,
            "table": r.table_name,
            "column": r.column_name,
            "text": r.text,
        }
        for r in rows
        if r.object_type == "column"
    ][:k]
    return {
        "query": query,
        "source_id": str(sid) if sid else None,
        "source_group_id": str(group_id) if group_id else None,
        "results": cols,
    }


def search_tables(
    engines: EngineRegistry,
    *,
    query: str,
    engine: str | None = "starburst",
    catalog: str | None = None,
    database_name: str | None = None,
    schema_name: str | None = None,
    top_k: int = 10,
    semantic_top_k: int | None = None,
    lexical_top_k: int | None = None,
) -> dict[str, Any]:
    """Hybrid lexical + semantic table search over the SkillSQL catalog."""
    adapter_name = engines.get(engine).name if engine else ""
    res = get_resources()
    q_vec = res.embedder([query])[0]
    effective_catalog = database_name or catalog
    semantic_limit = max(top_k, int(semantic_top_k or top_k))
    lexical_limit = max(top_k, int(lexical_top_k or top_k))
    semantic_rows = res.repo.search_schema_docs(
        q_vec,
        k=semantic_limit,
        catalog_name=effective_catalog,
        object_type="table",
        db_schema=schema_name,
    )
    lexical_rows = res.repo.search_schema_docs_lexical(
        query,
        k=lexical_limit,
        catalog_name=effective_catalog,
        object_type="table",
        db_schema=schema_name,
    )
    fused = _fuse_schema_docs_by_rrf(
        semantic_rows=semantic_rows,
        lexical_rows=lexical_rows,
        query_embedding=q_vec,
    )
    tables = [
        _table_search_item(
            row,
            adapter_name,
            requested_catalog=catalog,
            requested_database=database_name,
            rank_data=rank_data,
        )
        for row, rank_data in fused[:top_k]
    ]
    return {
        "query": query,
        "engine": adapter_name,
        "catalog": catalog or "",
        "database_name": database_name or "",
        "schema_name": schema_name or "",
        "top_k": top_k,
        "semantic_candidate_count": len(semantic_rows),
        "lexical_candidate_count": len(lexical_rows),
        "embedding_generated": bool(q_vec),
        "embedding_retries": 0,
        "tables": tables,
    }


def search_columns_hybrid(
    engines: EngineRegistry,
    *,
    query: str,
    engine: str | None = "starburst",
    catalog: str | None = None,
    database_name: str | None = None,
    schema_name: str | None = None,
    table_name: str | None = None,
    top_k: int = 10,
    semantic_top_k: int | None = None,
    lexical_top_k: int | None = None,
    matched_columns_limit: int = 10,
) -> dict[str, Any]:
    """Hybrid lexical + semantic column search over the SkillSQL catalog."""
    adapter_name = engines.get(engine).name if engine else ""
    res = get_resources()
    q_vec = res.embedder([query])[0]
    effective_catalog = database_name or catalog
    semantic_limit = max(top_k, int(semantic_top_k or top_k))
    lexical_limit = max(top_k, int(lexical_top_k or top_k))
    semantic_rows = res.repo.search_schema_docs(
        q_vec,
        k=semantic_limit,
        catalog_name=effective_catalog,
        object_type="column",
        db_schema=schema_name,
        table_name=table_name,
    )
    lexical_rows = res.repo.search_schema_docs_lexical(
        query,
        k=lexical_limit,
        catalog_name=effective_catalog,
        object_type="column",
        db_schema=schema_name,
        table_name=table_name,
    )
    fused = _fuse_schema_docs_by_rrf(
        semantic_rows=semantic_rows,
        lexical_rows=lexical_rows,
        query_embedding=q_vec,
    )
    tables_by_key: dict[tuple[str | None, str | None, str], dict[str, Any]] = {}
    for row, rank_data in fused:
        key = (row.catalog_name, row.db_schema, row.table_name)
        table = tables_by_key.setdefault(
            key,
            _column_search_table_item(
                row,
                adapter_name,
                requested_catalog=catalog,
                requested_database=database_name,
            ),
        )
        table["rrf_score"] += rank_data["rrf_score"]
        table["cosine_similarity"] = max(
            table["cosine_similarity"],
            rank_data["cosine_similarity"],
        )
        table["fts_score"] = max(table["fts_score"], rank_data["fts_score"])
        table["semantic_rank"] = _min_rank(table.get("semantic_rank"), rank_data["semantic_rank"])
        table["lexical_rank"] = _min_rank(table.get("lexical_rank"), rank_data["lexical_rank"])
        if len(table["matched_columns"]) < matched_columns_limit:
            table["matched_columns"].append(_column_search_column_item(row, rank_data))
            table["matched_column_count"] = len(table["matched_columns"])

    tables = sorted(
        tables_by_key.values(),
        key=lambda item: (
            item["rrf_score"],
            item["cosine_similarity"],
            item["fts_score"],
        ),
        reverse=True,
    )[:top_k]
    return {
        "query": query,
        "engine": adapter_name,
        "catalog": catalog or "",
        "database_name": database_name or "",
        "schema_name": schema_name or "",
        "table_name": table_name or "",
        "top_k": top_k,
        "semantic_candidate_count": len(semantic_rows),
        "lexical_candidate_count": len(lexical_rows),
        "embedding_generated": bool(q_vec),
        "embedding_retries": 0,
        "tables": tables,
    }


async def search_query_history(
    settings: Any,
    store: Any,
    embeddings: Any,
    *,
    query: str,
    source_id: str | None = None,
    engine: str | None = "starburst",
    catalog: str | None = None,
    schema_name: str | None = None,
    top_k: int = 10,
    semantic_top_k: int | None = None,
    lexical_top_k: int | None = None,
) -> dict[str, Any]:
    """Hybrid lexical + semantic search over backend query NLP history."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("query is required")

    effective_top_k = max(1, int(top_k))
    semantic_limit = max(
        effective_top_k,
        int(semantic_top_k or settings.discovery_similar_embedding_top_k),
    )
    lexical_limit = max(
        effective_top_k,
        int(lexical_top_k or settings.discovery_similar_lexical_top_k),
    )

    repo = None
    try:
        repo = get_resources().repo
    except Exception as exc:  # noqa: BLE001
        _logger.warning("catalog_query_history_repo_unavailable", error=str(exc))

    lexical_rows = (
        repo.list_query_history_nlp_by_full_text(
            normalized_query,
            source_id=source_id,
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
            limit=lexical_limit,
        )
        if repo is not None
        else []
    )
    if not lexical_rows:
        lexical_rows = await store.list_backend_query_nlp_history_by_full_text(
            normalized_query,
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
            limit=lexical_limit,
        )
    lexical_rows = _filter_successful_query_history_rows(lexical_rows)
    query_embedding, embedding_retries = await embeddings.embed_query(normalized_query)
    semantic_rows: list[dict[str, Any]] = []
    if query_embedding:
        semantic_rows = (
            repo.list_query_history_nlp_by_embedding(
                query_embedding,
                source_id=source_id,
                engine=engine,
                catalog=catalog,
                schema_name=schema_name,
                limit=semantic_limit,
            )
            if repo is not None
            else []
        )
        if not semantic_rows:
            semantic_rows = await store.list_backend_query_nlp_history_by_embedding(
                query_embedding,
                engine=engine,
                catalog=catalog,
                schema_name=schema_name,
                limit=semantic_limit,
            )
        semantic_rows = _filter_successful_query_history_rows(semantic_rows)

    fused_rows = _fuse_query_history_rows_by_rrf(
        semantic_rows=semantic_rows,
        lexical_rows=lexical_rows,
        rrf_k=max(1, int(settings.discovery_similar_rrf_k)),
    )
    queries = [_backend_query_history_item(row) for row in fused_rows[:effective_top_k]]
    tables = _dedupe_backend_table_refs(
        [table for item in queries for table in item.get("tables", [])]
    )
    return {
        "query": normalized_query,
        "engine": engine or "",
        "catalog": catalog or "",
        "schema_name": schema_name or "",
        "top_k": effective_top_k,
        "semantic_candidate_count": len(semantic_rows),
        "lexical_candidate_count": len(lexical_rows),
        "embedding_generated": bool(query_embedding),
        "embedding_retries": embedding_retries,
        "queries": queries,
        "tables": tables,
    }


async def build_backend_context(
    settings: Any,
    store: Any,
    embeddings: Any,
    engines: EngineRegistry,
    query: str,
    *,
    engine: str | None = "starburst",
    catalog: str | None = None,
    database_name: str | None = None,
    schema_name: str | None = None,
    table_top_k: int = 10,
    column_top_k: int = 10,
    query_top_k: int = 5,
) -> dict[str, Any]:
    """Build production Text2SQL backend context from the catalog plane."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("query is required")

    _logger.info(
        "catalog_build_backend_context_start",
        engine=engine,
        catalog=catalog,
        database=database_name,
        schema=schema_name,
    )
    warnings: list[str] = []
    try:
        table_search = search_tables(
            engines,
            query=normalized_query,
            engine=engine,
            catalog=catalog,
            database_name=database_name,
            schema_name=schema_name,
            top_k=table_top_k,
            semantic_top_k=settings.discovery_similar_embedding_top_k,
            lexical_top_k=settings.discovery_similar_lexical_top_k,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"catalog table search unavailable: {exc}")
        table_search = _empty_table_search(
            normalized_query,
            engine=engine,
            catalog=catalog,
            database_name=database_name,
            schema_name=schema_name,
            top_k=table_top_k,
        )
    try:
        column_search = search_columns_hybrid(
            engines,
            query=normalized_query,
            engine=engine,
            catalog=catalog,
            database_name=database_name,
            schema_name=schema_name,
            top_k=column_top_k,
            semantic_top_k=settings.discovery_similar_embedding_top_k,
            lexical_top_k=settings.discovery_similar_lexical_top_k,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"catalog column search unavailable: {exc}")
        column_search = _empty_column_search(
            normalized_query,
            engine=engine,
            catalog=catalog,
            database_name=database_name,
            schema_name=schema_name,
            top_k=column_top_k,
        )
    query_history_search = await search_query_history(
        settings,
        store,
        embeddings,
        query=normalized_query,
        engine=engine,
        catalog=database_name or catalog,
        schema_name=schema_name,
        top_k=query_top_k,
    )

    table_refs = _dedupe_backend_table_refs(
        [
            _backend_table_ref_from_search_item(item)
            for item in table_search.get("tables", [])
        ]
        + [
            _backend_table_ref_from_search_item(item)
            for item in column_search.get("tables", [])
        ]
        + [
            table
            for item in query_history_search.get("queries", [])
            for table in item.get("tables", [])
        ]
    )
    table_context = _list_catalog_table_context(
        table_refs,
        catalog_name=database_name or catalog,
        schema_name=schema_name,
    )
    if len(table_context) < len(table_refs):
        table_context = _merge_table_context(
            table_context,
            await store.list_backend_table_context(table_refs, engine=engine),
        )

    examples = list(query_history_search.get("queries", []))
    context_text = _format_backend_context_text(table_context, examples)
    context_pack = {
        "TABLES": table_context,
        "EXAMPLE QUERIES": examples,
        "CONTEXT TEXT": context_text,
    }
    context = {
        "query": normalized_query,
        "engine": engine or "",
        "catalog": catalog or "",
        "database_name": database_name or "",
        "schema_name": schema_name or "",
        "tables": [
            str(item.get("name") or "").strip()
            for item in table_context
            if str(item.get("name") or "").strip()
        ],
        "table_context": table_context,
        "examples": [
            {
                "query_id": str(item.get("query_id") or ""),
                "sql": str(item.get("raw_sql") or ""),
                "sql2text": str(item.get("query_nlp") or ""),
                "tables": list(item.get("tables") or []),
            }
            for item in examples
        ],
        "queries": [
            {
                "query_id": str(item.get("query_id") or ""),
                "sql": str(item.get("raw_sql") or ""),
                "sql2text": str(item.get("query_nlp") or ""),
                "tables": list(item.get("tables") or []),
            }
            for item in examples
        ],
        "similar_queries": examples,
        "metadata": {
            "tables": table_context,
            "examples": examples,
            "warnings": warnings,
        },
        "metadata_summary": {
            "table_count": len(table_context),
            "example_query_count": len(examples),
            "context_text": context_text,
            "warnings": warnings,
        },
        "context_pack": context_pack,
        "backend_search": {
            "tables": table_search,
            "columns": column_search,
            "query_history": query_history_search,
        },
    }
    _logger.debug("catalog_build_backend_context_done", table_count=len(table_context))
    return context


def list_sources() -> dict[str, Any]:
    """List all registered catalog sources."""
    res = get_resources()
    rows = res.repo.list_sources()
    groups = res.repo.list_source_groups()
    group_by_id = {r.id: r for r in groups}
    source_count_by_group: dict[uuid.UUID, int] = {}
    for row in rows:
        if row.source_group_id:
            source_count_by_group[row.source_group_id] = (
                source_count_by_group.get(row.source_group_id, 0) + 1
            )
    return {
        "source_groups": [
            {
                "id": str(r.id),
                "source_type": r.source_type,
                "name": r.name,
                "display_name": r.display_name,
                "description": r.description,
                "source_count": source_count_by_group.get(r.id, 0),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in groups
        ],
        "sources": [
            {
                "id": str(r.id),
                "source_group_id": str(r.source_group_id) if r.source_group_id else None,
                "source_group_name": (
                    group_by_id[r.source_group_id].name
                    if r.source_group_id and r.source_group_id in group_by_id
                    else None
                ),
                "source_type": r.source_type,
                "name": r.name,
                "catalog_name": r.catalog_name,
                "database": r.database,
                "db_schema": r.db_schema,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }


def seed_skillbank() -> dict[str, Any]:
    """Insert curated general_sql and Snowflake dialect seed skills (idempotent)."""
    from skillsql.skillbank.seeds import load_seeds

    n = load_seeds(get_resources().repo)
    _logger.info("skillbank_seeds_loaded", inserted=n)
    return {"inserted": n}


def sync_live_feedback_skills(
    query_runs: list[Any],
    *,
    promote: bool = False,
) -> dict[str, Any]:
    """Distill production QueryRun records into SkillBank candidate skills."""
    from skillsql.rl.live_feedback import distill_live_query_runs

    res = get_resources()
    result = distill_live_query_runs(
        query_runs,
        res.repo,
        embedder=res.embedder,
        promote=promote,
    )
    _logger.info(
        "live_feedback_skills_synced",
        runs_seen=result.runs_seen,
        trajectories_used=result.trajectories_used,
        skills_inserted=result.skills_inserted,
        promote=promote,
    )
    return result.to_dict()


def list_skills(*, scope: str | None = None, dialect: str | None = None) -> dict[str, Any]:
    """List all promoted SqlSkillBank skills."""
    from skillsql.catalog.models import Skill

    res = get_resources()
    with res.repo.session() as s:
        q = s.query(Skill).filter(Skill.status == "promoted")
        if scope:
            q = q.filter(Skill.scope == scope)
        if dialect:
            q = q.filter(Skill.dialect == dialect)
        rows = q.all()
    return {
        "count": len(rows),
        "skills": [
            {
                "id": str(r.id),
                "scope": r.scope,
                "title": r.title,
                "dialect": r.dialect,
                "when_to_apply": r.when_to_apply,
            }
            for r in rows
        ],
    }


# ── Inference plane — verification & scoring ──────────────────────────────────

def verify_sql(
    sql: str, *, source_id: str | None = None, execute: bool = True
) -> dict[str, Any]:
    """Run the static-lattice gates (and optionally execute) a SQL candidate.

    Args:
        sql:       The candidate SQL string.
        source_id: Restrict the bind gate to this source's known tables.
        execute:   Execute the SQL when all static gates pass.

    Returns:
        Gate results plus optional execution summary.
    """
    res = get_resources()
    sid = _resolve_source(source_id)
    report = run_static_lattice(sql, res.connector.dialect, known_tables=_known_tables(sid))
    out: dict[str, Any] = {
        "safe": report.safe,
        "parses": report.parses,
        "binds": report.binds,
        "scope_ok": report.scope_ok,
        "join_ok": report.join_ok,
        "passed_all": report.passed_all,
        "messages": report.messages,
        "first_failure": report.first_failure,
        "dialect": res.connector.dialect,
    }
    if execute and report.passed_all:
        s = res.settings
        er = asyncio.run(
            res.connector.execute(
                sql, read_only=True,
                timeout_s=s.SQL_STATEMENT_TIMEOUT_S,
                row_cap=s.SQL_ROW_CAP,
            )
        )
        out["execution"] = {
            "ok": er.ok,
            "row_count": er.row_count,
            "truncated": er.truncated,
            "elapsed_ms": er.elapsed_ms,
            "error": er.error,
            "columns": er.columns,
            "sample_rows": er.rows[:5],
        }
    return out


async def score_sql(
    question: str,
    sql: str,
    *,
    gold_sql: str | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Compute the composite verifier reward R(τ) for a candidate.

    When ``gold_sql`` is supplied it is executed to provide the equivalence
    reference; without it the score is capped at the exec_nogold ceiling.
    """
    from skillsql.verification.reward import RewardConfig, compute_reward

    res = get_resources()
    sid = _resolve_source(source_id)
    s = res.settings
    gold = None
    if gold_sql:
        gold = await res.connector.execute(
            gold_sql, read_only=True,
            timeout_s=s.SQL_STATEMENT_TIMEOUT_S, row_cap=s.SQL_ROW_CAP,
        )
    rb = await compute_reward(
        question=question,
        sql=sql,
        connector=res.connector,
        gold=gold,
        known_tables=_known_tables(sid),
        timeout_s=s.SQL_STATEMENT_TIMEOUT_S,
        row_cap=s.SQL_ROW_CAP,
        config=RewardConfig(),
    )
    return {
        "total": rb.total,
        "stage": rb.stage,
        "equivalent": rb.equivalent,
        "obligation_score": rb.obligation_score,
        "efficiency": rb.efficiency,
        "self_consistency": rb.self_consistency,
        "components": rb.components,
        "gate_messages": rb.gate_report.messages if rb.gate_report else [],
    }


# ── Inference plane — SQL generation and end-to-end workflow ──────────────────

async def generate_sql(
    question: str,
    *,
    source_id: str | None = None,
    source_group_id: str | None = None,
) -> dict[str, Any]:
    """Generate a single SQL candidate from the Arctic agent.

    Uses the SkillSQL-RL training-path workflow (retrieve → generate × 1).
    No critic/refiner loop; for the full production workflow use run_text2sql.
    """
    from skillsql.agents.sql_generator.agent import build_prompt, get_agent
    from skillsql.workflow._adk import clean_sql

    from app.adk.skillsql_runner import run_agent_once

    res = get_resources()
    sid = _resolve_source(source_id)
    group_id = _resolve_source(source_group_id)
    schema_ctx = _schema_context_for_generation(question, sid, group_id)
    skills = _skills_block(res.connector.dialect)
    prompt = build_prompt(question, res.connector.dialect, schema_ctx, skills)
    raw = await run_agent_once(get_agent(), prompt)
    return {
        "question": question,
        "sql": clean_sql(raw),
        "source_id": str(sid) if sid else None,
        "source_group_id": str(group_id) if group_id else None,
    }


async def run_text2sql(
    question: str, *, source_id: str | None = None
) -> dict[str, Any]:
    """End-to-end Text-to-SQL via the production inference workflow.

    Routes through the ADK SequentialAgent workflow:
    Directory → ContextBuilder[+SkillBank] → Generator → Critic/Refiner loop
    → Validator[+Reward] → Optimizer → SkillDistillation.
    """
    from app.adk.skillsql_runner import run_text2sql as _run

    if source_id:
        set_active_source(uuid.UUID(source_id))
    return await _run(question)


async def health_check() -> dict[str, Any]:
    """Check connector + catalog liveness."""
    res = get_resources()
    try:
        ok = await res.connector.health_check()
        dialect = res.connector.dialect
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "connector_ok": False, "error": str(exc)}
    return {"status": "ok", "dialect": dialect, "connector_ok": ok}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _schema_context_for_generation(
    question: str,
    source_id: uuid.UUID | None,
    source_group_id: uuid.UUID | None = None,
) -> str:
    try:
        context = generate_context(
            question,
            source_id=str(source_id) if source_id else None,
            source_group_id=str(source_group_id) if source_group_id and not source_id else None,
            schema_k=15,
            query_k=5,
        )
        return str(context.get("context") or "")
    except Exception:  # noqa: BLE001
        return ""


async def _sync_metadata_records(
    records: AsyncIterator[BackendMetadataRecord],
    *,
    source_type: str,
    source_name: str,
    scope_type: str,
    source_group_id: str | None = None,
    source_group_name: str | None = None,
    catalog_name: str | None,
    database_name: str | None,
    db_schema: str | None,
    table_name: str | None,
    column_name: str | None,
    describe: bool,
    doc_batch_size: int,
) -> dict[str, Any]:
    from skillsql.catalog.sync import sync_metadata_stream

    res = get_resources()
    if describe:
        _logger.info("metadata_sync_describe_ignored", reason="generic_catalog_describer_removed")

    _logger.debug(f"Syncing metadata records for catalog: {catalog_name}, table: {table_name}, "
                  f"column: {column_name} ")
    result = await sync_metadata_stream(
        records,
        res.repo,
        source_type=source_type,
        source_name=source_name,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
        scope_type=scope_type,
        catalog_name=catalog_name,
        database_name=database_name,
        db_schema=db_schema,
        table_name=table_name,
        column_name=column_name,
        embedder=res.embedder,
        describer=None,
        doc_batch_size=doc_batch_size,
    )
    set_active_source(result.source_id)
    payload = result.as_dict()
    if describe:
        payload.setdefault("warnings", []).append(_METADATA_DESCRIBE_IGNORED_WARNING)
    return payload


async def _sync_query_history_records(
    repo: Any,
    records: AsyncIterator[BackendQueryHistoryRecord],
    *,
    source_id: uuid.UUID,
    batch_size: int,
) -> int:
    rows_synced = 0
    batch: list[dict[str, Any]] = []

    async for record in records:
        batch.append(asdict(record))
        if len(batch) >= max(1, int(batch_size)):
            rows_synced += repo.upsert_query_history_rows(
                source_id=source_id,
                rows=batch,
            )
            batch.clear()

    if batch:
        rows_synced += repo.upsert_query_history_rows(
            source_id=source_id,
            rows=batch,
        )

    return rows_synced


def _ensure_catalog_source_id(
    *,
    source_type: str,
    source_name: str | None,
    source_group_id: str | None = None,
    source_group_name: str | None = None,
    catalog: str | None,
    database_name: str | None,
    schema_name: str | None,
) -> uuid.UUID:
    res = get_resources()
    normalized_source_type = str(source_type or "").strip().lower()
    normalized_source_name = str(source_name or "").strip() or normalized_source_type or "default"
    catalog_value = str(catalog or "").strip() or None
    database_value = str(database_name or "").strip() or None
    if _catalog_scoped_source(normalized_source_type) or database_value is None:
        database_value = catalog_value
    return res.repo.upsert_source(
        normalized_source_type,
        normalized_source_name,
        catalog_name=catalog_value or database_value,
        database=database_value,
        db_schema=str(schema_name or "").strip() or None,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
    )


def _catalog_scoped_source(source_type: str | None) -> bool:
    return str(source_type or "").strip().lower() in {"starburst", "trino", "starburst_galaxy"}


async def _filter_column_records(
    records: AsyncIterator[BackendMetadataRecord],
    *,
    column_name: str,
) -> AsyncIterator[BackendMetadataRecord]:
    wanted = column_name.lower()
    found = False
    async for record in records:
        entity_type = (record.entity_type or "").lower()
        if entity_type != "column":
            yield record
            continue
        if (record.column_name or "").lower() == wanted:
            found = True
            yield record
    if not found:
        raise ValueError(f"Column not found in metadata stream: {column_name}")


def _normalize_column_names(
    column_name: str | None,
    column_names: list[str] | None,
) -> list[str]:
    names: list[str] = []
    for value in [column_name, *(column_names or [])]:
        text = str(value or "").strip()
        if text and _name_key(text) not in {_name_key(existing) for existing in names}:
            names.append(text)
    return names


def _table_description_response_base(
    engine: str,
    catalog: str,
    database_name: str | None,
    table: Any,
) -> dict[str, Any]:
    return {
        "engine": engine,
        "catalog": catalog,
        "database_name": database_name or "",
        "schema_name": table.db_schema or "",
        "table_name": table.name,
    }


def _column_description_response_base(
    engine: str,
    catalog: str,
    database_name: str | None,
    column: Any,
) -> dict[str, Any]:
    table = column.table
    return {
        "engine": engine,
        "catalog": catalog,
        "database_name": database_name or "",
        "schema_name": table.db_schema or "",
        "table_name": table.name,
        "column_name": column.name,
        "data_type": column.data_type or "",
    }


def _column_metadata_item(column: Any) -> dict[str, Any]:
    return {
        "column_name": column.name,
        "data_type": column.data_type or "",
        "nullable": column.nullable,
        "ordinal_position": column.ordinal,
        "sample_values": column.sample_values or [],
        "current_description": column.nl_description or column.comment or "",
    }


def _table_schema_doc(table: Any, description: str):
    from skillsql.connectors.base import SchemaDoc

    col_summary = ", ".join(
        f"{column.name} {column.data_type}" for column in (table.columns or [])[:40]
    )
    columns_text = f" Columns: {col_summary}." if col_summary else ""
    text = (
        f"Table {table.fqn} ({table.table_type}). "
        f"{description}. "
        f"{columns_text}"
    ).strip()
    return SchemaDoc(
        object_type="table",
        fqn=table.fqn,
        catalog_name=table.catalog_name,
        db_schema=table.db_schema,
        table=table.name,
        text=text,
    )


def _column_schema_doc(column: Any, description: str):
    from skillsql.connectors.base import SchemaDoc

    table = column.table
    nullable = "nullable" if column.nullable else "not null"
    text = (
        f"Column {table.fqn}.{column.name} of type {column.data_type or 'UNKNOWN'} "
        f"({nullable}). {description}."
    ).strip()
    return SchemaDoc(
        object_type="column",
        fqn=f"{table.fqn}.{column.name}",
        catalog_name=table.catalog_name,
        db_schema=table.db_schema,
        table=table.name,
        column=column.name,
        text=text,
    )


def _name_key(value: Any) -> str:
    return str(value or "").strip().strip('"').strip("`").lower()


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fuse_schema_docs_by_rrf(
    *,
    semantic_rows: list[Any],
    lexical_rows: list[Any],
    query_embedding: list[float],
    rrf_k: int = 60,
) -> list[tuple[Any, dict[str, Any]]]:
    candidates: dict[str, tuple[Any, dict[str, Any]]] = {}

    for rank, row in enumerate(semantic_rows, start=1):
        key = str(row.fqn)
        _, data = candidates.setdefault(key, (row, _empty_rank_data()))
        data["semantic_rank"] = _min_rank(data["semantic_rank"], rank)
        data["rrf_score"] += 1.0 / (rrf_k + rank)
        data["cosine_similarity"] = max(
            data["cosine_similarity"],
            _cosine_similarity(query_embedding, row.embedding),
        )

    for rank, row in enumerate(lexical_rows, start=1):
        key = str(row.fqn)
        _, data = candidates.setdefault(key, (row, _empty_rank_data()))
        data["lexical_rank"] = _min_rank(data["lexical_rank"], rank)
        data["rrf_score"] += 1.0 / (rrf_k + rank)
        data["fts_score"] = max(data["fts_score"], 1.0 / rank)

    return sorted(
        candidates.values(),
        key=lambda item: (
            item[1]["rrf_score"],
            item[1]["cosine_similarity"],
            item[1]["fts_score"],
        ),
        reverse=True,
    )


def _empty_rank_data() -> dict[str, Any]:
    return {
        "rrf_score": 0.0,
        "cosine_similarity": 0.0,
        "fts_score": 0.0,
        "semantic_rank": None,
        "lexical_rank": None,
    }


def _table_search_item(
    row: Any,
    engine: str,
    *,
    requested_catalog: str | None,
    requested_database: str | None,
    rank_data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "engine": engine,
        "catalog": requested_catalog or row.catalog_name or "",
        "database_name": requested_database or "",
        "schema_name": row.db_schema or "",
        "table_name": row.table_name,
        "table_type": "",
        "description": row.text or "",
        "catalog_id": "",
        "schema_id": "",
        "table_id": "",
        "rrf_score": rank_data["rrf_score"],
        "cosine_similarity": rank_data["cosine_similarity"],
        "fts_score": rank_data["fts_score"],
        "semantic_rank": rank_data["semantic_rank"],
        "lexical_rank": rank_data["lexical_rank"],
        "updated_at": getattr(row, "created_at", None),
    }


def _column_search_table_item(
    row: Any,
    engine: str,
    *,
    requested_catalog: str | None,
    requested_database: str | None,
) -> dict[str, Any]:
    return {
        "engine": engine,
        "catalog": requested_catalog or row.catalog_name or "",
        "database_name": requested_database or "",
        "schema_name": row.db_schema or "",
        "table_name": row.table_name,
        "catalog_id": "",
        "schema_id": "",
        "table_id": "",
        "rrf_score": 0.0,
        "cosine_similarity": 0.0,
        "fts_score": 0.0,
        "semantic_rank": None,
        "lexical_rank": None,
        "matched_column_count": 0,
        "matched_columns": [],
        "updated_at": getattr(row, "created_at", None),
    }


def _column_search_column_item(row: Any, rank_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "column_name": row.column_name or "",
        "column_id": "",
        "data_type": "",
        "description": row.text or "",
        "ordinal_position": None,
        "nullable": None,
        "rrf_score": rank_data["rrf_score"],
        "cosine_similarity": rank_data["cosine_similarity"],
        "fts_score": rank_data["fts_score"],
        "semantic_rank": rank_data["semantic_rank"],
        "lexical_rank": rank_data["lexical_rank"],
    }


def _min_rank(existing: int | None, candidate: int | None) -> int | None:
    if candidate is None:
        return existing
    if existing is None:
        return candidate
    return min(existing, candidate)


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    count = min(len(left), len(right))
    if count == 0:
        return 0.0
    dot = sum(float(left[idx]) * float(right[idx]) for idx in range(count))
    left_norm = math.sqrt(sum(float(left[idx]) ** 2 for idx in range(count)))
    right_norm = math.sqrt(sum(float(right[idx]) ** 2 for idx in range(count)))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _fuse_query_history_rows_by_rrf(
    *,
    semantic_rows: list[dict[str, Any]],
    lexical_rows: list[dict[str, Any]],
    rrf_k: int,
) -> list[dict[str, Any]]:
    if not semantic_rows and not lexical_rows:
        return []

    fused_by_key: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(semantic_rows, start=1):
        key = _backend_query_history_row_key(row)
        candidate = fused_by_key.setdefault(key, dict(row))
        candidate["SEMANTIC_RANK"] = _min_rank(candidate.get("SEMANTIC_RANK"), rank)
        candidate["RRF_SCORE"] = float(_row_value(candidate, "RRF_SCORE", 0.0) or 0.0) + (
            1.0 / (rrf_k + rank)
        )
        candidate["COSINE_SIMILARITY"] = max(
            float(_row_value(candidate, "COSINE_SIMILARITY", 0.0) or 0.0),
            float(_row_value(row, "COSINE_SIMILARITY", 0.0) or 0.0),
        )

    for rank, row in enumerate(lexical_rows, start=1):
        key = _backend_query_history_row_key(row)
        candidate = fused_by_key.setdefault(key, dict(row))
        candidate["LEXICAL_RANK"] = _min_rank(candidate.get("LEXICAL_RANK"), rank)
        candidate["RRF_SCORE"] = float(_row_value(candidate, "RRF_SCORE", 0.0) or 0.0) + (
            1.0 / (rrf_k + rank)
        )
        candidate["FTS_SCORE"] = max(
            float(_row_value(candidate, "FTS_SCORE", 0.0) or 0.0),
            float(_row_value(row, "FTS_SCORE", 0.0) or 0.0),
        )

    return sorted(
        fused_by_key.values(),
        key=lambda row: (
            float(_row_value(row, "RRF_SCORE", 0.0) or 0.0),
            float(_row_value(row, "COSINE_SIMILARITY", 0.0) or 0.0),
            float(_row_value(row, "FTS_SCORE", 0.0) or 0.0),
            _updated_ts(row),
        ),
        reverse=True,
    )


def _query_history_nlp_with_failure_context(row: dict[str, Any], nlp_text: str) -> str:
    base = str(nlp_text or "").strip()
    if not _query_history_failed(row):
        return base

    details = _query_failure_details(row)
    code = details.get("error_code_name") or "unknown"
    category = details.get("error_code_category") or "unknown"
    message = (
        details.get("error_exception_message")
        or "The engine did not return a detailed exception message."
    )
    fix = _query_failure_fix(code, category, message)
    failure_text = (
        f"Failure cause: query failed with error_code_name={code} "
        f"and error_code_category={category}. "
        f"Detailed explanation: {message} "
        f"Suggested fix: {fix}"
    )
    if not base:
        return failure_text
    if "failure cause:" in base.lower() and "suggested fix:" in base.lower():
        return base
    return f"{base}\n\n{failure_text}"


def _query_failure_details(row: dict[str, Any]) -> dict[str, str]:
    metrics = _row_mapping(row, "METRICS_JSON")
    raw = _row_mapping(row, "RAW_JSON")
    return {
        "error_code_name": _first_failure_value(row, metrics, raw, key="error_code_name"),
        "error_code_category": _first_failure_value(row, metrics, raw, key="error_code_category"),
        "error_exception_message": _first_failure_value(
            row,
            metrics,
            raw,
            key="error_exception_message",
        ),
    }


def _first_failure_value(*rows: dict[str, Any], key: str) -> str:
    for row in rows:
        value = _row_value(row, key)
        text_value = str(value or "").strip()
        if text_value:
            return text_value
    return ""


def _query_failure_fix(error_code_name: str, error_code_category: str, message: str) -> str:
    haystack = f"{error_code_name} {error_code_category} {message}".lower()
    if any(
        term in haystack
        for term in ("column", "cannot be resolved", "not_found", "not found")
    ):
        return (
            "Verify table and column names against the catalog metadata, confirm the "
            "catalog/schema scope, refresh metadata if needed, and quote case-sensitive "
            "identifiers."
        )
    if any(term in haystack for term in ("table", "schema", "catalog", "does not exist")):
        return (
            "Confirm the referenced catalog, schema, and table exist for this source, then refresh "
            "catalog metadata or correct the fully qualified table name."
        )
    if any(term in haystack for term in ("syntax", "parse", "mismatched input")):
        return (
            "Correct the SQL syntax near the reported token and rerun validation before "
            "execution."
        )
    if any(term in haystack for term in ("access", "permission", "denied", "unauthorized")):
        return (
            "Use a role with the required privileges or request grants for the referenced "
            "objects."
        )
    if any(term in haystack for term in ("timeout", "resource", "memory", "exceeded", "limit")):
        return (
            "Reduce the scanned data with filters or limits, simplify joins/aggregations, "
            "or run on a larger execution resource."
        )
    return (
        "Review the engine error, correct the SQL or datasource scope, verify permissions "
        "and object names, then rerun the query."
    )


def _filter_successful_query_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not _query_history_failed(row)]


def _query_history_failed(row: dict[str, Any]) -> bool:
    return str(_row_value(row, "QUERY_STATE") or "").strip().upper() == "FAILED"


def _row_mapping(row: dict[str, Any], key: str) -> dict[str, Any]:
    raw = _row_value(row, key)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _query_history_context_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": str(_row_value(row, "QUERY_ID") or ""),
        "sql": str(_row_value(row, "RAW_SQL") or ""),
        "natural_language_query": str(
            _row_value(row, "NLP_TEXT") or _row_value(row, "QUERY_NLP") or ""
        ),
        "tables": _extract_row_tables(row),
        "cosine_similarity": float(_row_value(row, "COSINE_SIMILARITY", 0.0) or 0.0),
    }


def _schema_doc_table_refs(schema_docs: list[Any]) -> list[str]:
    refs: list[str] = []
    for doc in schema_docs:
        table_name = str(getattr(doc, "table_name", "") or "").strip()
        if not table_name:
            continue
        catalog_name = str(getattr(doc, "catalog_name", "") or "").strip()
        schema_name = str(getattr(doc, "db_schema", "") or "").strip()
        fqn = str(getattr(doc, "fqn", "") or "").strip()
        if catalog_name or schema_name:
            refs.append(".".join(part for part in [catalog_name, schema_name, table_name] if part))
        elif fqn:
            refs.append(fqn)
        else:
            refs.append(table_name)
    return _dedupe_backend_table_refs(refs)


def _merge_schema_docs_into_table_context(
    table_context: list[dict[str, Any]],
    schema_docs: list[Any],
) -> list[dict[str, Any]]:
    table_docs: dict[str, str] = {}
    column_docs: dict[str, str] = {}
    for doc in schema_docs:
        key = _schema_doc_table_key(doc)
        object_type = str(getattr(doc, "object_type", "") or "").strip().lower()
        text = str(getattr(doc, "text", "") or "").strip()
        if not key or not text:
            continue
        if object_type == "table":
            table_docs.setdefault(key, text)
        elif object_type == "column":
            column_name = str(getattr(doc, "column_name", "") or "").strip().lower()
            if column_name:
                column_docs.setdefault(f"{key}.{column_name}", text)

    for table in table_context:
        key = _table_dict_key(table)
        if key in table_docs:
            table["description"] = table_docs[key]
        for column in table.get("columns") or []:
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("column_name") or column.get("name") or "").strip().lower()
            doc_text = column_docs.get(f"{key}.{column_name}")
            if doc_text:
                column["description"] = doc_text
    return table_context


def _append_schema_doc_only_tables(
    table_context: list[dict[str, Any]],
    schema_docs: list[Any],
) -> list[dict[str, Any]]:
    existing = {_table_dict_key(table) for table in table_context}
    docs_by_table: dict[str, dict[str, Any]] = {}
    for doc in schema_docs:
        key = _schema_doc_table_key(doc)
        if not key or key in existing:
            continue
        table = docs_by_table.setdefault(
            key,
            {
                "name": _schema_doc_table_fqn(doc),
                "engine": "",
                "catalog": str(getattr(doc, "catalog_name", "") or ""),
                "schema_name": str(getattr(doc, "db_schema", "") or ""),
                "table_name": str(getattr(doc, "table_name", "") or ""),
                "table_type": "",
                "description": "",
                "columns": [],
            },
        )
        object_type = str(getattr(doc, "object_type", "") or "").strip().lower()
        text = str(getattr(doc, "text", "") or "").strip()
        if object_type == "table" and text:
            table["description"] = text
        elif object_type == "column":
            column_name = str(getattr(doc, "column_name", "") or "").strip()
            if column_name:
                table["columns"].append(
                    {
                        "column_name": column_name,
                        "name": column_name,
                        "data_type": "",
                        "type": "",
                        "description": text,
                        "ordinal_position": len(table["columns"]) + 1,
                        "nullable": None,
                    }
                )
    table_context.extend(docs_by_table.values())
    return table_context


def _format_generated_context_markdown(
    examples: list[dict[str, Any]],
    table_context: list[dict[str, Any]],
) -> str:
    sections: list[str] = []
    if examples:
        blocks: list[str] = []
        for idx, example in enumerate(examples, start=1):
            sql = str(example.get("sql") or "").strip()
            nlq = str(
                example.get("natural_language_query") or example.get("sql2text") or ""
            ).strip()
            tables = [
                str(item).strip()
                for item in example.get("tables") or []
                if str(item).strip()
            ]
            table_line = ", ".join(f"`{item}`" for item in tables) if tables else "None listed"
            blocks.append(
                "\n".join(
                    [
                        f"### Example {idx}",
                        "**SQL:**",
                        f"```sql\n{sql}\n```",
                        f"**Natural Language Query:** {nlq}",
                        f"**Referenced Tables:** {table_line}",
                    ]
                )
            )
        sections.append("## In-Context SQL Examples\n\n" + "\n\n".join(blocks))

    table_blocks: list[str] = []
    for table in table_context:
        name = str(table.get("name") or table.get("table_name") or "").strip()
        if not name:
            continue
        description = (
            str(table.get("description") or "").strip()
            or "No table description available."
        )
        rows = [
            "| Column | Type | Natural Language Description |",
            "| --- | --- | --- |",
        ]
        columns = table.get("columns") if isinstance(table.get("columns"), list) else []
        for column in columns:
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("column_name") or column.get("name") or "").strip()
            if not column_name:
                continue
            data_type = str(column.get("data_type") or column.get("type") or "").strip()
            column_desc = str(column.get("description") or "").strip()
            rows.append(
                "| "
                + " | ".join(
                    [
                        _markdown_table_cell(column_name),
                        _markdown_table_cell(data_type or "unknown"),
                        _markdown_table_cell(column_desc or "No column description available."),
                    ]
                )
                + " |"
            )
        if len(rows) == 2:
            rows.append("| _No column metadata found._ |  |  |")
        table_blocks.append(
            "\n".join(
                [
                    f"### `{name}`",
                    f"**Table Description:** {description}",
                    "",
                    *rows,
                ]
            )
        )
    if table_blocks:
        sections.append("## Relevant Tables and Columns\n\n" + "\n\n".join(table_blocks))
    return "\n\n".join(sections).strip()


def _schema_doc_table_key(doc: Any) -> str:
    return _normalized_table_key(
        getattr(doc, "catalog_name", None),
        getattr(doc, "db_schema", None),
        getattr(doc, "table_name", None),
    )


def _schema_doc_table_fqn(doc: Any) -> str:
    parts = [
        str(getattr(doc, "catalog_name", "") or "").strip(),
        str(getattr(doc, "db_schema", "") or "").strip(),
        str(getattr(doc, "table_name", "") or "").strip(),
    ]
    value = ".".join(part for part in parts if part)
    return value or str(getattr(doc, "fqn", "") or "").strip()


def _table_dict_key(table: dict[str, Any]) -> str:
    return _normalized_table_key(
        table.get("catalog"),
        table.get("schema_name"),
        table.get("table_name") or table.get("name"),
    )


def _normalized_table_key(catalog: Any, schema_name: Any, table_name: Any) -> str:
    return "|".join(
        str(item or "").strip().lower()
        for item in (catalog, schema_name, table_name)
    )


def _markdown_table_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _empty_table_search(
    query: str,
    *,
    engine: str | None,
    catalog: str | None,
    database_name: str | None,
    schema_name: str | None,
    top_k: int,
) -> dict[str, Any]:
    return {
        "query": query,
        "engine": engine or "",
        "catalog": catalog or "",
        "database_name": database_name or "",
        "schema_name": schema_name or "",
        "top_k": top_k,
        "semantic_candidate_count": 0,
        "lexical_candidate_count": 0,
        "embedding_generated": False,
        "embedding_retries": 0,
        "tables": [],
    }


def _empty_column_search(
    query: str,
    *,
    engine: str | None,
    catalog: str | None,
    database_name: str | None,
    schema_name: str | None,
    top_k: int,
) -> dict[str, Any]:
    return {
        "query": query,
        "engine": engine or "",
        "catalog": catalog or "",
        "database_name": database_name or "",
        "schema_name": schema_name or "",
        "table_name": "",
        "top_k": top_k,
        "semantic_candidate_count": 0,
        "lexical_candidate_count": 0,
        "embedding_generated": False,
        "embedding_retries": 0,
        "tables": [],
    }


def _backend_query_history_item(row: dict[str, Any]) -> dict[str, Any]:
    tables = _merge_tables(
        _extract_row_tables(row),
        extract_tables(str(_row_value(row, "RAW_SQL") or "")),
    )
    schema_table = str(_row_value(row, "SCHEMA_TABLE") or "").strip()
    if schema_table:
        tables = _merge_tables(tables, [schema_table])
    return {
        "id": _row_value(row, "ID"),
        "source_id": str(_row_value(row, "SOURCE_ID") or "") or None,
        "engine": str(_row_value(row, "ENGINE") or ""),
        "query_id": str(_row_value(row, "QUERY_ID") or ""),
        "catalog": str(_row_value(row, "CATALOG_NAME") or ""),
        "schema_name": str(_row_value(row, "SCHEMA_NAME") or ""),
        "raw_sql": str(_row_value(row, "RAW_SQL") or ""),
        "query_nlp": str(_row_value(row, "QUERY_NLP") or _row_value(row, "NLP_TEXT") or ""),
        "nlp_text": str(_row_value(row, "NLP_TEXT") or _row_value(row, "QUERY_NLP") or ""),
        "tables": tables,
        "rrf_score": float(_row_value(row, "RRF_SCORE", 0.0) or 0.0),
        "cosine_similarity": float(_row_value(row, "COSINE_SIMILARITY", 0.0) or 0.0),
        "fts_score": float(_row_value(row, "FTS_SCORE", 0.0) or 0.0),
        "semantic_rank": _optional_int(_row_value(row, "SEMANTIC_RANK")),
        "lexical_rank": _optional_int(_row_value(row, "LEXICAL_RANK")),
        "created_at": _row_value(row, "CREATED_AT"),
        "updated_at": _row_value(row, "UPDATED_AT"),
    }


def _query_nlp_history_item(
    row: dict[str, Any],
    fallback_engine: str,
    nlp_text: str,
    inserted: bool,
    caveats: list[str],
    *,
    embedding_generated: bool = False,
    embedding_retries: int = 0,
) -> dict[str, Any]:
    return {
        "raw_query_history_id": int(_row_value(row, "ID") or 0),
        "source_id": str(_row_value(row, "SOURCE_ID") or "") or None,
        "engine": str(_row_value(row, "ENGINE") or fallback_engine or ""),
        "query_id": str(_row_value(row, "QUERY_ID") or ""),
        "raw_sql": str(_row_value(row, "RAW_SQL") or ""),
        "query_nlp": nlp_text,
        "nlp_text": nlp_text,
        "embedding_generated": embedding_generated,
        "embedding_retries": embedding_retries,
        "inserted": inserted,
        "caveats": caveats,
    }


def _list_catalog_table_context(
    table_refs: list[str],
    *,
    catalog_name: str | None = None,
    schema_name: str | None = None,
) -> list[dict[str, Any]]:
    from skillsql.catalog.models import CatalogTable
    from sqlalchemy.orm import selectinload

    parsed_refs = [_parse_backend_table_ref(ref) for ref in table_refs]
    if not parsed_refs:
        return []

    res = get_resources()
    rows = []
    with res.repo.session() as session:
        for ref in parsed_refs:
            table_name = ref.get("table_name")
            if not table_name:
                continue
            q = session.query(CatalogTable).options(
                selectinload(CatalogTable.source),
                selectinload(CatalogTable.columns),
            )
            ref_catalog = ref.get("catalog") or catalog_name
            ref_schema = ref.get("schema_name") or schema_name
            if ref_catalog:
                q = q.filter(CatalogTable.catalog_name.ilike(ref_catalog))
            if ref_schema:
                q = q.filter(CatalogTable.db_schema.ilike(ref_schema))
            q = q.filter(CatalogTable.name.ilike(table_name))
            row = q.order_by(CatalogTable.catalog_name, CatalogTable.db_schema).first()
            if row is not None:
                rows.append(row)

    context: list[dict[str, Any]] = []
    seen: set[str] = set()
    for table in rows:
        key = _table_context_key(
            table.source.source_type if table.source else "",
            table.catalog_name,
            table.db_schema,
            table.name,
        )
        if key in seen:
            continue
        seen.add(key)
        context.append(
            {
                "name": table.fqn,
                "engine": table.source.source_type if table.source else "",
                "catalog": table.catalog_name or "",
                "schema_name": table.db_schema or "",
                "table_name": table.name,
                "table_type": table.table_type or "",
                "description": table.nl_description or table.comment or "",
                "columns": [
                    {
                        "column_name": column.name,
                        "name": column.name,
                        "data_type": column.data_type or "",
                        "type": column.data_type or "",
                        "description": column.nl_description or column.comment or "",
                        "ordinal_position": column.ordinal,
                        "nullable": column.nullable,
                    }
                    for column in sorted(table.columns or [], key=lambda item: item.ordinal or 0)
                ],
            }
        )
    return context


def _merge_table_context(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*primary, *fallback]:
        key = _table_context_key(
            item.get("engine"),
            item.get("catalog"),
            item.get("schema_name"),
            item.get("table_name") or item.get("name"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _format_backend_context_text(
    tables: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    blocks: list[str] = []
    for table in tables:
        table_name = str(table.get("name") or table.get("table_name") or "").strip()
        description = str(table.get("description") or "").strip()
        if not table_name:
            continue
        if description:
            table_lines = [f"Table: {table_name}: {description}", "Columns:"]
        else:
            table_lines = [f"Table: {table_name}", "Columns:"]
        columns = table.get("columns") if isinstance(table.get("columns"), list) else []
        for column in columns:
            column_name = str(column.get("column_name") or column.get("name") or "").strip()
            if not column_name:
                continue
            data_type = str(column.get("data_type") or column.get("type") or "").strip()
            column_description = str(column.get("description") or "").strip()
            type_text = f" ({data_type})" if data_type else ""
            desc_text = f": {column_description}" if column_description else ""
            table_lines.append(f"- {column_name}{type_text}{desc_text}")
        blocks.append("\n".join(table_lines))

    for example in examples:
        query_nlp = str(example.get("query_nlp") or example.get("sql2text") or "").strip()
        raw_sql = str(example.get("raw_sql") or example.get("sql") or "").strip()
        if not query_nlp and not raw_sql:
            continue
        blocks.append(
            "\n".join(
                [
                    f"Example Query in Natural language: {query_nlp}",
                    f"Associated SQL: {raw_sql}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _backend_table_ref_from_search_item(item: dict[str, Any]) -> str:
    catalog = str(item.get("catalog") or item.get("CATALOG_NAME") or "").strip()
    schema_name = str(item.get("schema_name") or item.get("SCHEMA_NAME") or "").strip()
    table_name = str(
        item.get("table_name") or item.get("TABLE_NAME") or item.get("name") or ""
    ).strip()
    return ".".join(part for part in (catalog, schema_name, table_name) if part)


def _dedupe_backend_table_refs(tables: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for table in tables:
        value = str(table or "").strip().replace('"', "").replace("`", "")
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _parse_backend_table_ref(table: str) -> dict[str, str]:
    parts = [part.strip() for part in str(table or "").split(".") if part.strip()]
    if len(parts) >= 3:
        return {"catalog": parts[-3], "schema_name": parts[-2], "table_name": parts[-1]}
    if len(parts) == 2:
        return {"schema_name": parts[0], "table_name": parts[1]}
    if len(parts) == 1:
        return {"table_name": parts[0]}
    return {}


def _extract_row_tables(row: dict[str, Any]) -> list[str]:
    raw = _row_value(row, "ALL_QUERY_TABLES")
    if raw is None:
        raw = _row_value(row, "all_query_tables")
    if raw is None:
        raw = _row_value(row, "TABLES_JSON")
    if raw is None:
        raw = _row_value(row, "tables_json")
    if raw is None:
        return []
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                values = parsed
            else:
                values = [item.strip() for item in text.split(",")]
        except json.JSONDecodeError:
            values = [item.strip() for item in text.split(",")]
    else:
        values = [raw]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, dict):
            parts = [str(item.get(key) or "").strip() for key in ("catalog", "schema", "table")]
            value = ".".join(part for part in parts if part)
        elif isinstance(item, (list, tuple)):
            parts = [str(part or "").strip() for part in item[:3]]
            value = ".".join(part for part in parts if part)
        else:
            value = str(item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized


def _merge_tables(primary: list[str], secondary: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for table in primary + secondary:
        value = str(table or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged


def _backend_query_history_row_key(row: dict[str, Any]) -> str:
    query_id = str(_row_value(row, "QUERY_ID") or "").strip().lower()
    engine = str(_row_value(row, "ENGINE") or "").strip().lower()
    if query_id:
        return f"{engine}:{query_id}"
    row_id = str(_row_value(row, "ID") or "").strip().lower()
    if row_id:
        return f"id:{row_id}"
    raw_sql = str(_row_value(row, "RAW_SQL") or "").strip().lower()
    return f"sql:{raw_sql}" if raw_sql else f"row:{id(row)}"


def _table_context_key(
    engine: Any,
    catalog: Any,
    schema_name: Any,
    table_name: Any,
) -> str:
    return ".".join(
        str(part or "").strip().lower()
        for part in (engine, catalog, schema_name, table_name)
        if str(part or "").strip()
    )


def _row_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row[key]
    lower = key.lower()
    if lower in row:
        return row[lower]
    upper = key.upper()
    if upper in row:
        return row[upper]
    return default


def _updated_ts(row: dict[str, Any]) -> float:
    raw = _row_value(row, "UPDATED_AT")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _skills_block(dialect: str) -> str:
    res = get_resources()
    try:
        skills = res.repo.general_and_dialect_skills(dialect)
        return "\n".join(f"- {sk.title}: {sk.principle}" for sk in skills)
    except Exception:  # noqa: BLE001
        return ""
