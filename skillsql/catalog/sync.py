"""Streaming semantic-catalog sync from engine-adapter metadata records.

The older ``builder.py`` path still supports connector-first full catalog builds.
This module is the production API path: it consumes the async metadata stream
exposed by ``app.adapters.EngineAdapter`` without importing app types directly.
That keeps large catalog refreshes incremental and makes table/column-specific
API sync safe.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any

from ..connectors.base import ColumnMeta, SchemaDoc, TableMeta
from ..observability.logging import get_logger
from .embeddings import Embedder, get_embedder
from .repository import CatalogRepository

log = get_logger(__name__)

Describer = Callable[[str], str | Awaitable[str]]


@dataclass(slots=True)
class CatalogMetadataSyncResult:
    source_type: str
    source_name: str
    source_id: uuid.UUID
    scope_type: str
    source_group_id: uuid.UUID | None = None
    source_group_name: str | None = None
    catalog_name: str | None = None
    database_name: str | None = None
    db_schema: str | None = None
    table_name: str | None = None
    column_name: str | None = None
    catalog_rows: int = 0
    schema_rows: int = 0
    table_rows: int = 0
    column_rows: int = 0
    docs: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_name": self.source_name,
            "source_id": str(self.source_id),
            "source_group_id": str(self.source_group_id) if self.source_group_id else None,
            "source_group_name": self.source_group_name,
            "scope_type": self.scope_type,
            "scope": {
                "catalog": self.catalog_name,
                "database": self.database_name,
                "schema": self.db_schema,
                "table": self.table_name,
                "column": self.column_name,
            },
            "catalog_rows": self.catalog_rows,
            "schema_rows": self.schema_rows,
            "table_rows": self.table_rows,
            "column_rows": self.column_rows,
            "docs": self.docs,
            "warnings": self.warnings,
        }


async def sync_metadata_stream(
    records: AsyncIterator[Any],
    repo: CatalogRepository,
    *,
    source_type: str,
    source_name: str,
    scope_type: str,
    source_group_id: uuid.UUID | str | None = None,
    source_group_name: str | None = None,
    catalog_name: str | None = None,
    database_name: str | None = None,
    db_schema: str | None = None,
    table_name: str | None = None,
    column_name: str | None = None,
    embedder: Embedder | None = None,
    describer: Describer | None = None,
    doc_batch_size: int = 128,
) -> CatalogMetadataSyncResult:
    """Persist a stream of catalog/schema/table/column records.

    ``records`` is intentionally duck-typed. The app layer passes
    ``BackendMetadataRecord`` instances, while tests can pass simple objects
    with matching attributes.
    """
    source_id = repo.upsert_source(
        source_type,
        source_name,
        catalog_name=catalog_name,
        database=database_name,
        db_schema=db_schema,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
    )
    resolved_group_id = repo._coerce_uuid(source_group_id) if source_group_id else None
    if resolved_group_id is None and source_group_name:
        resolved_group_id = repo.upsert_source_group(source_type, source_group_name)
    result = CatalogMetadataSyncResult(
        source_type=source_type,
        source_name=source_name,
        source_group_id=resolved_group_id,
        source_group_name=source_group_name,
        source_id=source_id,
        scope_type=scope_type,
        catalog_name=catalog_name,
        database_name=database_name,
        db_schema=db_schema,
        table_name=table_name,
        column_name=column_name,
    )
    embedder = embedder or get_embedder(repo.settings)

    table_cache: dict[tuple[str | None, str | None, str], TableMeta] = {}
    current_table_key: tuple[str | None, str | None, str] | None = None
    doc_buffer: list[SchemaDoc] = []

    async def flush_docs() -> None:
        if not doc_buffer:
            return
        docs = list(doc_buffer)
        doc_buffer.clear()
        if describer is not None:
            for doc in docs:
                try:
                    enriched = describer(doc.text)
                    if isawaitable(enriched):
                        enriched = await enriched
                    doc.text = f"{doc.text} {enriched}".strip()
                except Exception as exc:  # noqa: BLE001
                    result.warnings.append(f"describe skipped for {doc.fqn}: {exc}")
        if docs:
            vectors = embedder([doc.text for doc in docs])
            for doc, vector in zip(docs, vectors, strict=True):
                doc.embedding = vector
        result.docs += repo.upsert_schema_docs(source_id, docs)

    async def flush_current_table_doc(next_key: tuple[str | None, str | None, str] | None) -> None:
        nonlocal current_table_key
        if current_table_key is None or current_table_key == next_key:
            return
        table = table_cache.get(current_table_key)
        if table is not None:
            doc_buffer.append(_table_doc(table))
            if len(doc_buffer) >= doc_batch_size:
                await flush_docs()
        current_table_key = next_key

    async for record in records:
        entity_type = _text(_attr(record, "entity_type")).lower()
        if entity_type == "catalog":
            result.catalog_rows += 1
            continue
        if entity_type == "schema":
            result.schema_rows += 1
            continue
        if entity_type == "table":
            table = _table_from_record(record)
            key = _table_key(table)
            await flush_current_table_doc(key)
            repo.upsert_table_metadata(source_id, table)
            table_cache[key] = table
            current_table_key = key
            result.table_rows += 1
            continue
        if entity_type == "column":
            table, column = _column_from_record(record, table_cache)
            key = _table_key(table)
            if current_table_key is None:
                current_table_key = key
            table_cache[key] = table
            _merge_column(table, column)
            repo.upsert_column_metadata(source_id, table, column)
            doc_buffer.append(_column_doc(table, column))
            result.column_rows += 1
            if len(doc_buffer) >= doc_batch_size:
                await flush_docs()
            continue
        result.warnings.append(f"unsupported metadata entity_type={entity_type!r}")

    await flush_current_table_doc(None)
    await flush_docs()
    log.info(
        "semantic_catalog_sync_done",
        source_id=str(source_id),
        source_type=source_type,
        scope_type=scope_type,
        catalog=catalog_name,
        database=database_name,
        schema=db_schema,
        table=table_name,
        column=column_name,
        tables=result.table_rows,
        columns=result.column_rows,
        docs=result.docs,
    )
    return result


def _table_from_record(record: Any) -> TableMeta:
    catalog = _text(_attr(record, "database_name")) or _text(_attr(record, "catalog_name"))
    schema = _text(_attr(record, "schema_name"))
    name = _required_text(record, "table_name")
    raw = _raw(record)
    return TableMeta(
        catalog_name=catalog,
        db_schema=schema,
        name=name,
        table_type=_text(_attr(record, "object_type")) or "BASE TABLE",
        comment=_text(_attr(record, "description")),
        row_estimate=_int_or_none(
            raw.get("row_count")
            or raw.get("rowCount")
            or raw.get("row_estimate")
            or raw.get("estimatedRows")
        ),
    )


def _column_from_record(
    record: Any,
    table_cache: dict[tuple[str | None, str | None, str], TableMeta],
) -> tuple[TableMeta, ColumnMeta]:
    catalog = _text(_attr(record, "database_name")) or _text(_attr(record, "catalog_name"))
    schema = _text(_attr(record, "schema_name"))
    table_name = _required_text(record, "table_name")
    key = (catalog, schema, table_name)
    table = table_cache.get(key)
    if table is None:
        table = TableMeta(
            catalog_name=catalog,
            db_schema=schema,
            name=table_name,
            table_type="BASE TABLE",
        )

    column = ColumnMeta(
        name=_required_text(record, "column_name"),
        data_type=_text(_attr(record, "data_type")) or "UNKNOWN",
        nullable=_bool_or_default(_attr(record, "nullable"), True),
        comment=_text(_attr(record, "description")),
        ordinal=_int_or_none(_attr(record, "ordinal_position")) or 0,
    )
    return table, column


def _table_doc(table: TableMeta) -> SchemaDoc:
    col_summary = ", ".join(f"{c.name} {c.data_type}" for c in table.columns[:40])
    columns_text = f" Columns: {col_summary}." if col_summary else ""
    text = (
        f"Table {table.fqn} ({table.table_type}). "
        f"{table.comment + '. ' if table.comment else ''}"
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


def _column_doc(table: TableMeta, column: ColumnMeta) -> SchemaDoc:
    text = (
        f"Column {table.fqn}.{column.name} of type {column.data_type}"
        f"{' (nullable)' if column.nullable else ' (not null)'}"
        f"{'. ' + column.comment if column.comment else ''}."
    )
    return SchemaDoc(
        object_type="column",
        fqn=f"{table.fqn}.{column.name}",
        catalog_name=table.catalog_name,
        db_schema=table.db_schema,
        table=table.name,
        column=column.name,
        text=text,
    )


def _merge_column(table: TableMeta, column: ColumnMeta) -> None:
    for idx, existing in enumerate(table.columns):
        if existing.name.lower() == column.name.lower():
            table.columns[idx] = column
            return
    table.columns.append(column)


def _table_key(table: TableMeta) -> tuple[str | None, str | None, str]:
    return (table.catalog_name, table.db_schema, table.name)


def _attr(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _raw(record: Any) -> dict[str, Any]:
    raw = _attr(record, "raw")
    return raw if isinstance(raw, dict) else {}


def _text(value: Any) -> str:
    return str(value).strip() if value is not None and str(value).strip() else ""


def _required_text(record: Any, field: str) -> str:
    value = _text(_attr(record, field))
    if not value:
        raise ValueError(f"metadata record missing required field {field!r}: {record!r}")
    return value


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _bool_or_default(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return default
