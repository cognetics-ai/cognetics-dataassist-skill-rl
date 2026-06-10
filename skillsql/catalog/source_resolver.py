"""Helpers for mapping benchmark/training tasks to catalog sources."""

from __future__ import annotations

import uuid

from sqlalchemy import func, or_

from .models import Source
from .repository import CatalogRepository


def resolve_source_for_database(
    repo: CatalogRepository,
    db_id: str | None,
    *,
    source_type: str | None = None,
) -> uuid.UUID | None:
    """Resolve a task database id to a catalog source.

    Spider-2.0-Snow uses Snowflake database ids. In the catalog, that same value
    may appear as ``sources.database`` or ``sources.catalog_name`` depending on
    which build path created the source. Starburst catalog names also land in
    ``catalog_name``. We therefore match all stable source labels exactly,
    case-insensitively.
    """
    target = str(db_id or "").strip().lower()
    if not target:
        return None

    with repo.session() as session:
        query = session.query(Source)
        if source_type:
            query = query.filter(func.lower(Source.source_type) == source_type.strip().lower())
        row = (
            query.filter(
                or_(
                    func.lower(Source.database) == target,
                    func.lower(Source.catalog_name) == target,
                    func.lower(Source.name) == target,
                )
            )
            .order_by(Source.updated_at.desc(), Source.created_at.desc())
            .first()
        )
        return row.id if row else None
