from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.adk.runtime import AdkDataAssistRuntime


class QueryToNlpService:
    """Service wrapper for the query-to-NLP ADK workflow."""

    def __init__(self, runtime: AdkDataAssistRuntime):
        self._runtime = runtime

    async def sync_backend_query_nlp_history(
        self,
        *,
        engine: str = "starburst",
        ids: list[int] | None = None,
        raw_sql: str | None = None,
        limit: int = 100,
        missing_only: bool = True,
    ) -> dict[str, Any]:
        return await self._runtime.sync_query_nlp_history(
            engine=engine,
            ids=ids,
            raw_sql=raw_sql,
            limit=limit,
            missing_only=missing_only,
        )
