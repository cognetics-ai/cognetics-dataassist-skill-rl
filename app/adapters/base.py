from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExplainResult:
    ok: bool
    summary: dict[str, Any]


@dataclass
class EngineHandle:
    handle_id: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineStatus:
    state: str
    done: bool
    progress_percentage: int = 0
    stats: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None


@dataclass
class ResultPage:
    schema: list[dict[str, Any]]
    rows: list[list[Any]]
    next_page_token: str | None = None


@dataclass(slots=True)
class BackendMetadataRecord:
    """Normalized metadata row streamed from a backend engine."""

    entity_type: str
    engine: str
    catalog_id: str | None = None
    catalog_name: str | None = None
    database_name: str | None = None
    schema_id: str | None = None
    schema_name: str | None = None
    table_id: str | None = None
    table_name: str | None = None
    column_id: str | None = None
    column_name: str | None = None
    ordinal_position: int | None = None
    data_type: str | None = None
    nullable: bool | None = None
    object_type: str | None = None
    description: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackendQueryHistoryRecord:
    """Normalized raw query-history row streamed from a backend engine."""

    engine: str
    query_id: str
    raw_sql: str
    catalog_name: str | None = None
    schema_name: str | None = None
    query_state: str | None = None
    query_type: str | None = None
    user_email: str | None = None
    role_name: str | None = None
    cluster_name: str | None = None
    source: str | None = None
    created_at: datetime | str | None = None
    ended_at: datetime | str | None = None
    tables: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class EngineAdapter(ABC):
    name: str

    @abstractmethod
    async def explain(self, sql: str) -> ExplainResult:
        raise NotImplementedError

    @abstractmethod
    async def execute_async(self, sql: str) -> EngineHandle:
        raise NotImplementedError

    @abstractmethod
    async def get_status(self, handle: EngineHandle) -> EngineStatus:
        raise NotImplementedError

    @abstractmethod
    async def fetch_results(
        self,
        handle: EngineHandle,
        page_token: str | None = None,
    ) -> ResultPage:
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, handle: EngineHandle) -> bool:
        raise NotImplementedError

    async def iter_catalog_metadata(
        self,
        catalog: str,
        *,
        database_name: str | None = None,
        include_columns: bool = True,
    ) -> AsyncIterator[BackendMetadataRecord]:
        raise NotImplementedError(f"{self.name} does not expose catalog metadata sync")
        yield

    async def iter_schema_metadata(
        self,
        catalog: str,
        schema: str,
        *,
        database_name: str | None = None,
        include_columns: bool = True,
    ) -> AsyncIterator[BackendMetadataRecord]:
        raise NotImplementedError(f"{self.name} does not expose schema metadata sync")
        yield

    async def iter_table_metadata(
        self,
        catalog: str,
        schema: str,
        table: str,
        *,
        database_name: str | None = None,
        include_columns: bool = True,
    ) -> AsyncIterator[BackendMetadataRecord]:
        raise NotImplementedError(f"{self.name} does not expose table metadata sync")
        yield

    async def iter_query_history(
        self,
        *,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        table: str | None = None,
        limit: int | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[BackendQueryHistoryRecord]:
        raise NotImplementedError(f"{self.name} does not expose query-history sync")
        yield
