from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ValidationError

from app.adapters.base import EngineHandle
from app.adapters.registry import EngineRegistry
from app.adk.model_provider import build_adk_model, build_query_generator_adk_model, uses_vertex_provider
from app.adk.postgres_memory_service import PostgresMemoryService
from app.adk.retry import AdkRetryConfig, adk_retry
from app.agents.column_description_agent.agent import build_agent as build_column_description_agent
from app.agents.common.output_schemas import (
    ColumnDescriptionOutput,
    ContextBuilderOutput,
    CriticPackageOutput,
    DraftPackageOutput,
    OptimizationPackageOutput,
    QueryToNlpOutput,
    RefinementPackageOutput,
    RouteDecisionOutput,
    TableDescriptionOutput,
    ValidationPackageOutput,
)
from app.agents.common.runtime_context import AgentDependencies
from app.agents.common.sql_quality import evaluate_sql_quality, optimize_sql_with_guardrails
from app.agents.input_router.agent import build_agent as build_input_router_agent
from app.agents.query_to_nlp_agent.agent import build_agent as build_query_to_nlp_agent
from app.agents.table_description_agent.agent import build_agent as build_table_description_agent
from app.agents.text2sql_workflow.agent import build_agent as build_text2sql_workflow_agent
from app.config import Settings
from app.core.events import EventBus
from app.core.policy import PolicyChecker
from app.core.sql_utils import statement_kind
from app.core.store import SQLStore
from app.models import QueryRun, RunEvent
from app.observability.logging import get_logger
from app.services.directory import DirectoryService
from app.services.embeddings import EmbeddingService

_logger = get_logger(__name__)


class AdkDataAssistRuntime:
    """Runtime orchestrator using Google ADK workflows + deterministic execution path.

    Agent responsibilities:
        - InputRouterAgent chooses SQL-direct vs NL workflow route.
        - Text2SQL workflow (Sequential + LoopAgent) generates/refines/validates SQL.

    Deterministic responsibilities:
        - Query execution, polling, cancellation, and result collection against engine adapter.
    """

    ROUTER_APP = "data_assist_router"
    TEXT2SQL_APP = "data_assist_text2sql"
    TABLE_DESCRIPTION_APP = "data_assist_table_description"
    COLUMN_DESCRIPTION_APP = "data_assist_column_description"
    QUERY_TO_NLP_APP = "data_assist_query_to_nlp"

    def __init__(
        self,
        settings: Settings,
        store: SQLStore,
        directory: DirectoryService,
        policy: PolicyChecker,
        engines: EngineRegistry,
        embeddings: EmbeddingService,
        event_bus: EventBus,
        skillsql_resources=None,  # skillsql.resources.Resources | None
    ):
        _logger.info(f"Initializing AdkDataAssistRuntime....")
        self._settings = settings
        self._store = store
        self._policy = policy
        self._engines = engines
        self._embeddings = embeddings
        self._event_bus = event_bus
        self._skillsql_resources = skillsql_resources

        _logger.info(f"Inserting ADK dependencies (catalog={'yes' if skillsql_resources else 'no'})")
        self._deps = AgentDependencies(
            settings=settings,
            directory=directory,
            policy_checker=policy,
            engines=engines,
            store=store,
            embeddings=embeddings,
            skillsql_resources=skillsql_resources,
        )

        self._configure_model_env()
        self._Runner, self._DatabaseSessionService, self._Content, self._Part = self._load_adk_components()
        model = build_adk_model(self._settings)
        query_generator_model = build_query_generator_adk_model(self._settings)

        state_schema = self._validate_pg_identifier(self._settings.adk_state_schema)
        memory_schema = self._validate_pg_identifier(self._settings.adk_memory_schema)
        self._ensure_postgres_schema(self._settings.adk_postgres_dsn, state_schema)
        state_db_url = self._build_adk_state_db_url(self._settings.adk_postgres_dsn, state_schema)
        state_connect_args = self._build_adk_state_connect_args(state_schema)

        _logger.info(f"Initializing database session service with state={state_db_url}")
        self._session_service = self._DatabaseSessionService(
            db_url=state_db_url,
            connect_args=state_connect_args,
        )
        _logger.info(f"Initializing database memory service")
        self._memory_service = PostgresMemoryService(
            dsn=self._settings.adk_postgres_dsn,
            schema=memory_schema,
        )
        self._router_runner = self._Runner(
            app_name=self.ROUTER_APP,
            agent=build_input_router_agent(model),
            session_service=self._session_service,
            memory_service=self._memory_service,
        )
        self._text2sql_runner = self._Runner(
            app_name=self.TEXT2SQL_APP,
            agent=build_text2sql_workflow_agent(
                model=model,
                query_generator_model=query_generator_model,
                deps=self._deps,
                max_refinement_iterations=self._settings.critic_refiner_max_iterations,
            ),
            session_service=self._session_service,
            memory_service=self._memory_service,
        )
        self._table_description_runner = self._Runner(
            app_name=self.TABLE_DESCRIPTION_APP,
            agent=build_table_description_agent(model, self._deps),
            session_service=self._session_service,
            memory_service=self._memory_service,
        )
        self._column_description_runner = self._Runner(
            app_name=self.COLUMN_DESCRIPTION_APP,
            agent=build_column_description_agent(model, self._deps),
            session_service=self._session_service,
            memory_service=self._memory_service,
        )
        self._query_to_nlp_runner = self._Runner(
            app_name=self.QUERY_TO_NLP_APP,
            agent=build_query_to_nlp_agent(model, self._deps),
            session_service=self._session_service,
            memory_service=self._memory_service,
        )

        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_requested: set[str] = set()
        self._handles: dict[str, tuple[str, EngineHandle]] = {}

    async def draft(self, soeid: str, prompt: str, engine: str) -> dict[str, Any]:
        """Generate SQL draft from natural language prompt using ADK workflow."""

        if not prompt.strip():
            raise ValueError("Prompt is required for draft generation")

        workflow = await self._run_text2sql_workflow(
            soeid=soeid,
            prompt=prompt,
            engine=engine,
            run_id=None,
        )

        return {
            "draft_sql": workflow["final_sql"],
            "explanation": workflow["explanation"],
            "warnings": workflow["warnings"],
            "context_refs": workflow["context_refs"],
            "confidence": workflow["confidence"],
            "assumptions": workflow["assumptions"],
        }

    async def validate(self, soeid: str, sql: str, engine: str) -> dict[str, Any]:
        """Deterministically validate SQL using policy + explain checks."""

        role = await self._deps.directory.get_user_role(soeid)
        report = await evaluate_sql_quality(
            deps=self._deps,
            sql=sql,
            engine=engine,
            role_id=role,
            prompt="",
        )

        return {
            "is_valid": bool(report["approved"]),
            "policy_findings": report["policy_findings"],
            "explain_summary": report["explain_summary"],
            "risk_score": float(report["risk_score"]),
            "fixes": report["recommendations"],
        }

    async def run_query(
        self,
        soeid: str,
        sql: str | None,
        engine: str,
        prompt: str | None = None,
        input_mode: str = "auto",
        run_id: str | None = None,
        source_id: str | None = None,
    ) -> str:
        """Start asynchronous query run with routing + deterministic execution."""

        submitted = (sql or prompt or "").strip()
        if not submitted:
            raise ValueError("Either sql or prompt must be provided")

        if run_id:
            run = await self._prepare_successful_run_for_rerun(
                run_id=run_id,
                soeid=soeid,
                sql=sql,
                prompt=prompt,
                engine=engine,
                source_id=source_id,
            )
            sql = run.submitted_sql
            prompt = run.submitted_prompt
            engine = run.engine
            input_mode = "sql"
            await self._event_bus.clear_history(run.run_id)
        else:
            natural_language_query = self._natural_language_query_from_inputs(
                sql=sql,
                prompt=prompt,
                input_mode=input_mode,
            )
            run = await self._store.create_run(
                soeid=soeid,
                engine=engine,
                submitted_text=submitted,
                input_mode=input_mode,
                submitted_sql=(sql or "").strip() or None,
                submitted_prompt=(prompt or "").strip() or None,
                natural_language_query=natural_language_query,
                source_id=source_id,
            )
        await self._emit(run.run_id, "RUN_CREATED", {"status": run.status, "engine": run.engine})

        task = asyncio.create_task(
            self._run_query_task(
                run_id=run.run_id,
                soeid=soeid,
                sql=sql,
                prompt=prompt,
                engine=engine,
                input_mode=input_mode,
                source_id=run.source_id or source_id,
            ),
            name=f"run-{run.run_id}",
        )
        self._tasks[run.run_id] = task
        return run.run_id

    async def _prepare_successful_run_for_rerun(
        self,
        *,
        run_id: str,
        soeid: str,
        sql: str | None,
        prompt: str | None,
        engine: str,
        source_id: str | None = None,
    ) -> QueryRun:
        existing = await self._store.get_run(run_id)
        if not existing:
            raise ValueError("Run not found")
        if existing.soeid != soeid:
            raise ValueError("Run does not belong to the current user")
        if existing.status != "succeeded":
            raise ValueError("Only successful historical runs can be rerun in place")

        rerun_sql = (sql or existing.final_sql or existing.submitted_sql or "").strip()
        if not rerun_sql:
            raise ValueError("Historical run does not have SQL to rerun")

        rerun_prompt = (prompt or existing.natural_language_query or existing.submitted_prompt or "").strip() or None
        natural_language_query = (
            existing.natural_language_query
            or self._natural_language_query_from_inputs(sql=rerun_sql, prompt=rerun_prompt, input_mode="sql")
        )
        now = datetime.now(timezone.utc)
        updated = await self._store.update_run(
            run_id,
            engine=existing.engine,
            submitted_text=natural_language_query or rerun_prompt or existing.submitted_text or rerun_sql,
            input_mode="sql",
            route_mode=None,
            submitted_sql=rerun_sql,
            submitted_prompt=rerun_prompt,
            natural_language_query=natural_language_query,
            source_id=existing.source_id or source_id,
            reward_json=None,
            final_sql=rerun_sql,
            status="queued",
            started_at=now,
            ended_at=None,
            error_message=None,
            stats={},
            schema=[],
            rows=[],
        )
        if not updated:
            raise ValueError("Run not found")
        return updated

    async def cancel_run(self, run_id: str) -> bool:
        """Request cancellation for an active run and propagate to engine adapter."""

        run = await self._store.get_run(run_id)
        if not run:
            return False

        self._cancel_requested.add(run_id)
        handle_data = self._handles.get(run_id)
        if handle_data:
            engine_name, handle = handle_data
            await self._engines.get(engine_name).cancel(handle)

        await self._emit(run_id, "RUN_CANCEL_REQUESTED", {"message": "Cancellation requested by user"})
        return True

    async def describe_catalog_table(
        self,
        *,
        engine: str,
        catalog: str,
        schema_name: str,
        table_name: str,
        sample_size: int = 5,
    ) -> dict[str, Any]:
        """Generate a table description without persisting it."""
        _logger.debug(f"Describing Table Name: {table_name}")
        return await self._run_table_description_agent(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
            table_name=table_name,
            sample_size=sample_size,
        )

    async def describe_catalog_columns(
        self,
        *,
        engine: str,
        catalog: str,
        schema_name: str,
        table_name: str,
        column_name: str | None = None,
        column_metadata: list[dict[str, Any]],
        sample_size: int = 5,
    ) -> dict[str, Any]:
        """Generate column descriptions without persisting them."""
        return await self._run_column_description_agent(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
            table_name=table_name,
            column_name=column_name,
            column_metadata=column_metadata,
            sample_size=sample_size,
        )

    async def describe_query_history_nlp(
        self,
        *,
        engine: str,
        raw_history_id: int,
        raw_sql: str,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate NLP text for a catalog query-history row without persisting it."""
        return await self._run_query_to_nlp_agent(
            engine=engine,
            raw_history_id=raw_history_id,
            raw_sql=raw_sql,
            source_id=source_id,
        )

    async def sync_table_descriptions(
        self,
        *,
        engine: str = "starburst",
        catalog: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        missing_only: bool = True,
        limit: int = 50,
        sample_size: int = 5,
    ) -> dict[str, Any]:
        _logger.debug(f"In sync_table_descriptions for table name: {table_name}, catalog={catalog}, schema={schema_name}, engine={engine} ")
        candidates = await self._store.list_backend_metadata_tables_for_description(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
            table_name=table_name,
            missing_only=missing_only,
            limit=limit,
        )

        items: list[dict[str, Any]] = []
        updated_count = 0
        embeddings_generated = 0
        embedding_retries = 0
        for row in candidates:
            row_engine = str(self._row_value(row, "ENGINE") or engine).strip()
            row_catalog = str(self._row_value(row, "CATALOG_NAME") or "").strip()
            row_schema = str(self._row_value(row, "SCHEMA_NAME") or "").strip()
            row_table = str(self._row_value(row, "TABLE_NAME") or "").strip()
            _logger.debug(f"Table: {row_table}, Catalog: {row_catalog}, Schema: {row_schema}")
            if not row_engine or not row_catalog or not row_schema or not row_table:
                items.append(
                    {
                        "engine": row_engine or engine,
                        "catalog": row_catalog,
                        "schema_name": row_schema,
                        "table_name": row_table,
                        "description": "",
                        "confidence": 0.0,
                        "observed_entities": [],
                        "likely_grain": "",
                        "updated": False,
                        "caveats": ["Metadata row is missing engine, catalog, schema, or table name."],
                    }
                )
                continue

            try:
                _logger.info(f"Generating table description for table {row_table}")
                output = await self._run_table_description_agent(
                    engine=row_engine,
                    catalog=row_catalog,
                    schema_name=row_schema,
                    table_name=row_table,
                    sample_size=sample_size,
                )
                description = str(output.get("description") or "").strip()
                caveats = list(output.get("caveats") or [])
                embedding: list[float] = []
                embedding_retry_count = 0
                updated = False
                if description:
                    embedding, embedding_retry_count = await self._embeddings.embed_document(description)
                    embedding_retries += embedding_retry_count
                    if embedding:
                        embeddings_generated += 1
                    else:
                        _logger.warning("Embedding generation failed and returned no vector")
                        caveats.append("Embedding generation returned no vector.")
                    updated = await self._store.update_backend_metadata_table_description(
                        engine=row_engine,
                        catalog_id=str(self._row_value(row, "CATALOG_ID") or ""),
                        schema_id=str(self._row_value(row, "SCHEMA_ID") or ""),
                        table_id=str(self._row_value(row, "TABLE_ID") or ""),
                        description=description,
                        embedding=embedding or None,
                    )
                if updated:
                    updated_count += 1
                items.append(
                    {
                        "engine": row_engine,
                        "catalog": row_catalog,
                        "schema_name": row_schema,
                        "table_name": row_table,
                        "description": description,
                        "confidence": float(output.get("confidence") or 0.0),
                        "observed_entities": list(output.get("observed_entities") or []),
                        "likely_grain": str(output.get("likely_grain") or ""),
                        "embedding_generated": bool(embedding),
                        "embedding_retries": embedding_retry_count,
                        "updated": updated,
                        "caveats": caveats,
                    }
                )
            except Exception as exc:
                _logger.exception("Table description sync failed for %s.%s.%s", row_catalog, row_schema, row_table)
                items.append(
                    {
                        "engine": row_engine,
                        "catalog": row_catalog,
                        "schema_name": row_schema,
                        "table_name": row_table,
                        "description": "",
                        "confidence": 0.0,
                        "observed_entities": [],
                        "likely_grain": "",
                        "updated": False,
                        "caveats": [str(exc)],
                    }
                )

        return {
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(candidates),
            "updated_count": updated_count,
            "embeddings_generated": embeddings_generated,
            "embedding_retries": embedding_retries,
            "skipped_count": len(candidates) - updated_count,
            "items": items,
        }

    async def sync_column_descriptions(
        self,
        *,
        engine: str = "starburst",
        catalog: str,
        schema_name: str,
        table_name: str,
        column_name: str | None = None,
        missing_only: bool = True,
        limit: int = 500,
        sample_size: int = 5,
    ) -> dict[str, Any]:
        _logger.debug(f"In ADK runtime sync_column_descriptions with engine {engine}, catalog {catalog}, schema_name {schema_name}, table_name {table_name}")
        candidates = await self._store.list_backend_metadata_columns_for_description(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
            table_name=table_name,
            column_name=column_name,
            missing_only=missing_only,
            limit=limit,
        )

        groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for row in candidates:
            row_engine = str(self._row_value(row, "ENGINE") or engine).strip()
            row_catalog = str(self._row_value(row, "CATALOG_NAME") or "").strip()
            row_schema = str(self._row_value(row, "SCHEMA_NAME") or "").strip()
            row_table = str(self._row_value(row, "TABLE_NAME") or "").strip()
            groups.setdefault((row_engine, row_catalog, row_schema, row_table), []).append(row)

        items: list[dict[str, Any]] = []
        updated_count = 0
        embeddings_generated = 0
        embedding_retries = 0
        for (row_engine, row_catalog, row_schema, row_table), rows in groups.items():
            if not row_engine or not row_catalog or not row_schema or not row_table:
                for row in rows:
                    items.append(
                        self._column_description_item(
                            row,
                            engine,
                            "",
                            False,
                            ["Metadata row is missing engine, catalog, schema, or table name."],
                        )
                    )
                continue

            column_metadata = [
                {
                    "column_name": str(self._row_value(row, "COLUMN_NAME") or "").strip(),
                    "data_type": str(self._row_value(row, "DATA_TYPE") or ""),
                    "nullable": self._row_value(row, "NULLABLE"),
                    "ordinal_position": self._row_value(row, "ORDINAL_POSITION"),
                    "current_description": self._row_value(row, "DESCRIPTION"),
                }
                for row in rows
                if str(self._row_value(row, "COLUMN_NAME") or "").strip()
            ]
            if not column_metadata:
                for row in rows:
                    items.append(
                        self._column_description_item(
                            row,
                            engine,
                            "",
                            False,
                            ["Metadata row is missing column name."],
                        )
                    )
                continue

            try:
                output = await self._run_column_description_agent(
                    engine=row_engine,
                    catalog=row_catalog,
                    schema_name=row_schema,
                    table_name=row_table,
                    column_name=column_name,
                    column_metadata=column_metadata,
                    sample_size=sample_size,
                )
                output_caveats = [str(item) for item in output.get("caveats") or [] if str(item).strip()]
                output_by_name = {
                    self._metadata_name_key(item.get("column_name")): item
                    for item in output.get("columns") or []
                    if self._metadata_name_key(item.get("column_name"))
                }

                for row in rows:
                    row_column = str(self._row_value(row, "COLUMN_NAME") or "").strip()
                    output_item = output_by_name.get(self._metadata_name_key(row_column))
                    if not output_item:
                        caveats = output_caveats + ["Column description agent did not return this column."]
                        items.append(self._column_description_item(row, engine, "", False, caveats))
                        continue

                    description = str(output_item.get("description") or "").strip()
                    item_caveats = list(output_item.get("caveats") or [])
                    embedding: list[float] = []
                    embedding_retry_count = 0
                    updated = False
                    if description:
                        embedding, embedding_retry_count = await self._embeddings.embed_document(description)
                        embedding_retries += embedding_retry_count
                        if embedding:
                            embeddings_generated += 1
                        else:
                            item_caveats.append("Embedding generation returned no vector.")
                        updated = await self._store.update_backend_metadata_column_description(
                            engine=row_engine,
                            catalog_id=str(self._row_value(row, "CATALOG_ID") or ""),
                            schema_id=str(self._row_value(row, "SCHEMA_ID") or ""),
                            table_id=str(self._row_value(row, "TABLE_ID") or ""),
                            column_id=str(self._row_value(row, "COLUMN_ID") or ""),
                            description=description,
                            embedding=embedding or None,
                        )
                    if updated:
                        updated_count += 1
                    items.append(
                        self._column_description_item(
                            row,
                            engine,
                            description,
                            updated,
                            item_caveats,
                            output_item,
                            embedding_generated=bool(embedding),
                            embedding_retries=embedding_retry_count,
                        )
                    )
            except Exception as exc:
                _logger.exception("Column description sync failed for %s.%s.%s", row_catalog, row_schema, row_table)
                for row in rows:
                    items.append(self._column_description_item(row, engine, "", False, [str(exc)]))

        return {
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(candidates),
            "updated_count": updated_count,
            "embeddings_generated": embeddings_generated,
            "embedding_retries": embedding_retries,
            "skipped_count": len(candidates) - updated_count,
            "items": items,
        }

    async def sync_query_nlp_history(
        self,
        *,
        engine: str = "starburst",
        ids: list[int] | None = None,
        raw_sql: str | None = None,
        limit: int = 100,
        missing_only: bool = True,
    ) -> dict[str, Any]:
        normalized_ids = sorted({int(item) for item in (ids or []) if int(item) > 0})
        if ids is not None and not normalized_ids:
            return {
                "synced_at": datetime.now(timezone.utc).isoformat(),
                "source_rows": 0,
                "inserted_rows": 0,
                "embeddings_generated": 0,
                "embedding_retries": 0,
                "rows_failed": 0,
                "items": [],
            }
        source_rows = await self._store.list_backend_query_history_raw_for_nlp_history(
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
            row_engine = str(self._row_value(row, "ENGINE") or engine).strip()
            row_query_id = str(self._row_value(row, "QUERY_ID") or "").strip()
            row_id = int(self._row_value(row, "ID") or 0)
            row_sql = str(self._row_value(row, "RAW_SQL") or "")
            if not row_engine or not row_query_id or not row_sql.strip():
                items.append(
                    self._query_nlp_history_item(
                        row,
                        engine,
                        "",
                        False,
                        ["Raw query history row is missing engine, query id, or raw SQL."],
                    )
                )
                continue

            try:
                output = await self._run_query_to_nlp_agent(
                    engine=row_engine,
                    raw_history_id=row_id,
                    raw_sql=row_sql,
                )
                query_nlp = str(output.get("query_nlp") or "").strip()
                _logger.debug(f"Response from Agent: query_nlp={query_nlp}")
                inserted = False
                embedding: list[float] = []
                embedding_retry_count = 0
                caveats = list(output.get("caveats") or [])
                _logger.info(f"Inserting NLP history for row_query_id={row_query_id}")
                if query_nlp:
                    embedding, embedding_retry_count = await self._embeddings.embed_document(query_nlp)
                    embedding_retries += embedding_retry_count
                    if embedding:
                        embeddings_generated += 1
                    else:
                        caveats.append("Embedding generation returned no vector.")
                    inserted = await self._store.upsert_backend_query_nlp_history_row(
                        raw_row=row,
                        query_nlp=query_nlp,
                        embedding=embedding or None,
                    )
                if inserted:
                    inserted_count += 1
                items.append(
                    self._query_nlp_history_item(
                        row,
                        engine,
                        query_nlp,
                        inserted,
                        caveats,
                        embedding_generated=bool(embedding),
                        embedding_retries=embedding_retry_count,
                    )
                )
            except Exception as exc:
                _logger.exception("Query NLP history sync failed for query_id=%s", row_query_id)
                items.append(self._query_nlp_history_item(row, engine, "", False, [str(exc)]))

        return {
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "source_rows": len(source_rows),
            "inserted_rows": inserted_count,
            "embeddings_generated": embeddings_generated,
            "embedding_retries": embedding_retries,
            "rows_failed": len(source_rows) - inserted_count,
            "items": items,
        }

    async def _run_query_task(
        self,
        run_id: str,
        soeid: str,
        sql: str | None,
        prompt: str | None,
        engine: str,
        input_mode: str,
        source_id: str | None,
    ) -> None:
        await self._store.update_run(run_id, status="running", started_at=datetime.now(timezone.utc))
        await self._emit(run_id, "AGENT_STARTED", {"agent_name": "RunOrchestrator"})

        try:
            route = await self._decide_route(
                soeid=soeid,
                submitted_sql=sql or "",
                submitted_prompt=prompt or "",
                input_mode_hint=input_mode,
                run_id=run_id,
            )

            mode = route.get("mode", "sql")
            await self._store.update_run(run_id, route_mode=mode)
            if mode == "nl":
                prompt_text = (prompt or sql or "").strip()
                if not prompt_text:
                    raise ValueError("Prompt is required when router selects NL mode")
                workflow = await self._run_text2sql_workflow(
                    soeid=soeid,
                    prompt=prompt_text,
                    engine=engine,
                    run_id=run_id,
                    source_id=source_id,
                )
                candidate_sql = workflow["final_sql"]
                await self._store.update_run(
                    run_id,
                    final_sql=candidate_sql,
                    source_id=workflow.get("source_id") or source_id,
                    reward_json=workflow.get("reward") or None,
                )
            else:
                candidate_sql = (sql or prompt or "").strip()
                await self._store.update_run(
                    run_id,
                    final_sql=candidate_sql,
                    source_id=source_id,
                )
                await self._emit(
                    run_id,
                    "AGENT_STEP",
                    {
                        "agent_name": "InputRouterAgent",
                        "step_name": "routed_to_sql_direct",
                        "message": route.get("reason", "Direct SQL route selected"),
                    },
                )

            if not candidate_sql:
                raise ValueError("No SQL was produced for execution")

            role = await self._deps.directory.get_user_role(soeid)
            quality = await evaluate_sql_quality(
                deps=self._deps,
                sql=candidate_sql,
                engine=engine,
                role_id=role,
                prompt=prompt or "",
            )
            await self._emit(
                run_id,
                "AGENT_STEP",
                {
                    "agent_name": "DeterministicValidator",
                    "step_name": "validation",
                    "approved": quality["approved"],
                    "risk_score": quality["risk_score"],
                    "issues": quality["issues"],
                },
            )

            if not quality["approved"]:
                await self._store.update_run(
                    run_id,
                    status="failed",
                    ended_at=datetime.now(timezone.utc),
                    error_message="Validation failed before execution",
                )
                await self._emit(
                    run_id,
                    "RUN_FAILED",
                    {
                        "reason": "validation_failed",
                        "findings": quality["policy_findings"],
                        "recommendations": quality["recommendations"],
                    },
                )
                return

            optimized = optimize_sql_with_guardrails(self._settings.default_limit, candidate_sql)
            final_sql = optimized["optimized_sql"]
            await self._store.update_run(run_id, final_sql=final_sql)
            await self._emit(
                run_id,
                "AGENT_STEP",
                {
                    "agent_name": "DeterministicOptimizer",
                    "step_name": "optimization",
                    "changes": optimized["changes"],
                },
            )

            adapter = self._engines.get(engine)
            handle = await adapter.execute_async(final_sql)
            self._handles[run_id] = (engine, handle)
            await self._emit(run_id, "ENGINE_SUBMITTED", {"engine": engine})

            last_state = None
            last_progress = -1
            while True:
                if run_id in self._cancel_requested:
                    await adapter.cancel(handle)

                status = await adapter.get_status(handle)
                progress = int(status.progress_percentage)
                should_emit_progress = status.state == "RUNNING" and progress != last_progress
                if status.state != last_state or should_emit_progress:
                    await self._emit(
                        run_id,
                        "ENGINE_STATE",
                        {
                            "state": status.state,
                            "progressPercentage": progress,
                            "stats": status.stats,
                        },
                    )
                    last_state = status.state
                    last_progress = progress

                if status.state == "FAILED":
                    await self._store.update_run(
                        run_id,
                        status="failed",
                        ended_at=datetime.now(timezone.utc),
                        error_message=(status.error or {}).get("message", "Query execution failed"),
                    )
                    await self._emit(run_id, "RUN_FAILED", {"error": status.error or {}})
                    break

                if status.state == "CANCELLED" or run_id in self._cancel_requested:
                    await self._store.update_run(run_id, status="cancelled", ended_at=datetime.now(timezone.utc))
                    await self._emit(run_id, "RUN_CANCELLED", {"status": "cancelled"})
                    break

                if status.done:
                    results = await adapter.fetch_results(handle)
                    natural_language_query = self._natural_language_query_from_inputs(
                        sql=sql,
                        prompt=prompt,
                        input_mode=input_mode,
                    )
                    await self._store.update_run(
                        run_id,
                        status="succeeded",
                        ended_at=datetime.now(timezone.utc),
                        schema=results.schema,
                        rows=results.rows,
                        stats=status.stats,
                        natural_language_query=natural_language_query,
                    )
                    await self._persist_successful_query_embedding(
                        run_id=run_id,
                        natural_language_query=natural_language_query,
                    )
                    await self._emit(
                        run_id,
                        "RUN_SUCCEEDED",
                        {"row_count": len(results.rows), "schema": results.schema},
                    )
                    break

                await asyncio.sleep(0.25)

        except Exception as exc:
            await self._store.update_run(
                run_id,
                status="failed",
                ended_at=datetime.now(timezone.utc),
                error_message=str(exc),
            )
            await self._emit(run_id, "RUN_FAILED", {"error": {"message": str(exc)}})
        finally:
            self._handles.pop(run_id, None)
            await self._emit(run_id, "AGENT_COMPLETED", {"agent_name": "RunOrchestrator"})

    async def _persist_successful_query_embedding(
        self,
        *,
        run_id: str,
        natural_language_query: str | None,
    ) -> None:
        query_text = (natural_language_query or "").strip()
        if not query_text:
            return
        try:
            embedding, retries = await self._embeddings.embed_query(query_text)
        except Exception as exc:
            _logger.warning("Natural-language query embedding failed for run_id=%s: %s", run_id, exc)
            return
        if not embedding:
            _logger.warning("Natural-language query embedding was not generated for run_id=%s", run_id)
            return
        await self._store.update_run(
            run_id,
            natural_language_query=query_text,
            embedding=embedding,
        )
        await self._emit(
            run_id,
            "AGENT_STEP",
            {
                "agent_name": "RunOrchestrator",
                "step_name": "natural_language_embedding_persisted",
                "embedding_retries": retries,
            },
        )

    @staticmethod
    def _natural_language_query_from_inputs(
        *,
        sql: str | None,
        prompt: str | None,
        input_mode: str,
    ) -> str | None:
        prompt_text = (prompt or "").strip()
        if prompt_text:
            if input_mode == "nl" or sql:
                return prompt_text
            if not AdkDataAssistRuntime._looks_like_executable_sql(prompt_text):
                return prompt_text

        sql_text = (sql or "").strip()
        if input_mode == "nl" and sql_text and not AdkDataAssistRuntime._looks_like_executable_sql(sql_text):
            return sql_text
        return None

    @staticmethod
    def _looks_like_executable_sql(value: str) -> bool:
        return bool(re.match(r"^\s*(SELECT|WITH|EXPLAIN|VALUES|TABLE)\b", value or "", flags=re.IGNORECASE))

    async def _run_adk_events(
        self,
        runner: Any,
        *,
        label: str,
        user_id: str,
        session_id: str,
        new_message: Any,
        state_delta: dict[str, Any] | None = None,
        on_event: Callable[[Any], Awaitable[None]] | None = None,
        includes_query_generator: bool = False,
    ) -> None:
        retry_config = self._adk_retry_config(
            includes_query_generator=includes_query_generator
        )

        @adk_retry(retry_config, label=label)
        async def _consume() -> None:
            kwargs: dict[str, Any] = {
                "user_id": user_id,
                "session_id": session_id,
                "new_message": new_message,
            }
            if state_delta is not None:
                kwargs["state_delta"] = state_delta

            async for event in runner.run_async(**kwargs):
                if on_event is not None:
                    await on_event(event)

        await _consume()

    def _adk_retry_config(self, *, includes_query_generator: bool = False) -> AdkRetryConfig:
        max_retries = self._settings.adk_model_max_retries
        backoff_initial = self._settings.adk_model_retry_backoff_initial_seconds
        backoff_max = self._settings.adk_model_retry_backoff_max_seconds

        if includes_query_generator:
            max_retries = max(max_retries, self._settings.query_generator_model_max_retries)
            backoff_initial = self._min_positive(
                backoff_initial,
                self._settings.query_generator_model_retry_backoff_initial_seconds,
            )
            backoff_max = max(
                backoff_max,
                self._settings.query_generator_model_retry_backoff_max_seconds,
            )

        return AdkRetryConfig(
            max_retries=max_retries,
            backoff_initial_seconds=backoff_initial,
            backoff_max_seconds=backoff_max,
        )

    @staticmethod
    def _min_positive(left: float, right: float) -> float:
        values = [value for value in (float(left), float(right)) if value > 0]
        return min(values) if values else 0.0

    async def _decide_route(
        self,
        soeid: str,
        submitted_sql: str,
        submitted_prompt: str,
        input_mode_hint: str,
        run_id: str | None,
    ) -> dict[str, str]:
        hint = (input_mode_hint or "auto").strip().lower()
        sql_text = submitted_sql.strip()
        prompt_text = submitted_prompt.strip()

        if hint == "sql":
            return {"mode": "sql", "reason": "input_mode=sql short-circuited to direct SQL execution"}
        if hint == "nl":
            return {"mode": "nl", "reason": "input_mode=nl forces NLP workflow"}
        if sql_text:
            return {"mode": "sql", "reason": "explicit SQL payload short-circuited to direct SQL execution"}
        if self._looks_like_sql_text(prompt_text):
            return {"mode": "sql", "reason": "prompt detected as SQL; bypassing context and metadata workflow"}

        session_id = f"route-{run_id or uuid.uuid4()}"
        await self._ensure_session(self.ROUTER_APP, soeid, session_id)
        session = await self._session_service.get_session(app_name=self.ROUTER_APP, user_id=soeid, session_id=session_id)

        message = self._Content(role="user", parts=[self._Part(text="Route this request to SQL direct mode or NL workflow mode.")])
        async def _on_event(event: Any) -> None:
            if run_id:
                await self._emit(run_id, "AGENT_STEP", self._event_payload(event))

        await self._run_adk_events(
            self._router_runner,
            label="input_router",
            user_id=soeid,
            session_id=session_id,
            new_message=message,
            state_delta={
                "submitted_sql": submitted_sql,
                "submitted_prompt": submitted_prompt,
                "input_mode_hint": input_mode_hint,
            },
            on_event=_on_event,
        )

        session = await self._session_service.get_session(app_name=self.ROUTER_APP, user_id=soeid, session_id=session_id)
        if not session:
            raise RuntimeError("Router session not found")
        await self._persist_session_memory(session)

        decision = self._schema_state(session.state, "route_decision_json", RouteDecisionOutput, default={})
        mode = str(decision.get("mode") or "").lower()
        if mode not in {"sql", "nl"}:
            mode = "sql" if submitted_sql.strip() else "nl"
            decision = {"mode": mode, "reason": "Fallback deterministic router"}

        if run_id:
            await self._emit(
                run_id,
                "AGENT_STEP",
                {
                    "agent_name": "InputRouterAgent",
                    "step_name": "decision",
                    "mode": decision["mode"],
                    "reason": decision.get("reason", ""),
                },
            )

        return {"mode": decision["mode"], "reason": str(decision.get("reason", ""))}

    @staticmethod
    def _looks_like_sql_text(value: str) -> bool:
        if not value:
            return False
        kind = statement_kind(value)
        return kind in {"SELECT", "WITH", "UNION", "INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "DROP", "ALTER"}

    async def _run_text2sql_workflow(
        self,
        soeid: str,
        prompt: str,
        engine: str,
        run_id: str | None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        session_id = f"text2sql-{run_id or uuid.uuid4()}"
        await self._ensure_session(self.TEXT2SQL_APP, soeid, session_id)
        session = await self._session_service.get_session(app_name=self.TEXT2SQL_APP, user_id=soeid, session_id=session_id)

        message = self._Content(role="user", parts=[self._Part(text=prompt)])
        _logger.debug(f"Message: {message}")
        _logger.debug("Prompt: {prompt}")
        event_state: dict[str, Any] = {}
        async def _on_event(event: Any) -> None:
            payload = self._event_payload(event)
            self._merge_event_state(event_state, payload)
            if run_id:
                await self._emit(run_id, "AGENT_STEP", payload)

        await self._run_adk_events(
            self._text2sql_runner,
            label="text2sql_workflow",
            user_id=soeid,
            session_id=session_id,
            new_message=message,
            state_delta={
                "soeid": soeid,
                "user_prompt": prompt,
                "submitted_prompt": prompt,
                "submitted_sql": "",
                "engine": engine,
                "input_mode_hint": "nl",
                "source_id": source_id or "",
            },
            on_event=_on_event,
            includes_query_generator=True,
        )

        session = await self._session_service.get_session(app_name=self.TEXT2SQL_APP, user_id=soeid, session_id=session_id)
        if not session:
            raise RuntimeError("Text2SQL workflow session not found")
        await self._persist_session_memory(session)

        state = dict(session.state or {})
        self._fill_missing_state(state, event_state)
        context_bundle = self._schema_state(state, "context_bundle_json", ContextBuilderOutput, default={})
        skillbank_context = self._json_state(state, "skillbank_context_json", default={})
        draft_package = self._schema_state(state, "draft_package_json", DraftPackageOutput, default={})
        critic_package = self._schema_state(state, "critic_package_json", CriticPackageOutput, default={})
        refinement_package = self._schema_state(state, "refinement_package_json", RefinementPackageOutput, default={})
        validation_package = self._schema_state(state, "validation_package_json", ValidationPackageOutput, default={})
        optimization_package = self._schema_state(state, "optimization_package_json", OptimizationPackageOutput, default={})
        optimization_tool_payload = self._json_state(state, "optimization_payload_json", default={})
        verifier_reward = self._json_state(state, "verifier_reward_json", default={})

        final_sql = str(
            optimization_package.get("final_sql")
            or optimization_tool_payload.get("optimized_sql")
            or state.get("final_sql")
            or refinement_package.get("refined_sql")
            or state.get("refined_sql")
            or draft_package.get("draft_sql")
            or state.get("generated_sql")
            or ""
        ).strip()
        if not final_sql:
            raise RuntimeError("Text2SQL workflow did not produce final SQL")

        warnings: list[str] = []
        warnings.extend(list(draft_package.get("warnings") or []))
        warnings.extend(list(critic_package.get("issues") or []))
        for finding in validation_package.get("policy_findings", []):
            message = finding.get("message")
            if message:
                warnings.append(str(message))

        context_refs = context_bundle.get("examples") or context_bundle.get("queries") or []

        return {
            "final_sql": final_sql,
            "source_id": (
                str(state.get("source_id") or "")
                or str(skillbank_context.get("source_id") or "")
                or None
            ),
            "reward": verifier_reward or None,
            "explanation": str(draft_package.get("explanation") or "Generated by Text2SQL workflow"),
            "warnings": sorted({item for item in warnings if item}),
            "context_refs": context_refs,
            "confidence": float(draft_package.get("confidence", 0.0)),
            "assumptions": list(draft_package.get("assumptions") or []),
            "refinement": refinement_package,
            "validation": validation_package,
            "optimization": optimization_package or optimization_tool_payload,
        }

    async def _run_table_description_agent(
        self,
        *,
        engine: str,
        catalog: str,
        schema_name: str,
        table_name: str,
        sample_size: int,
    ) -> dict[str, Any]:
        _logger.debug(f"Running table desc agent with engine: {engine}, catalog {catalog}, "
                      f"schema {schema_name}, table_name {table_name}")
        user_id = "metadata-maintenance"
        session_id = f"table-description-{uuid.uuid4()}"
        initial_state = {
            "engine": engine,
            "catalog": catalog,
            "schema_name": schema_name,
            "table_name": table_name,
            "sample_size": sample_size,
        }
        await self._ensure_session(self.TABLE_DESCRIPTION_APP, user_id, session_id, state=initial_state)

        message = self._Content(
            role="user",
            parts=[
                self._Part(
                    text=(
                        "Generate a backend metadata table description for "
                        f"{catalog}.{schema_name}.{table_name}."
                    )
                )
            ],
        )

        await self._run_adk_events(
            self._table_description_runner,
            label="table_description",
            user_id=user_id,
            session_id=session_id,
            new_message=message,
            state_delta=initial_state,
        )

        session = await self._session_service.get_session(
            app_name=self.TABLE_DESCRIPTION_APP,
            user_id=user_id,
            session_id=session_id,
        )
        if not session:
            raise RuntimeError("Table description session not found")
        await self._persist_session_memory(session)

        payload = self._schema_state(session.state, "table_description_json", TableDescriptionOutput, default={})
        _logger.info(f"Description from LLM for table {table_name} with payload {payload}")
        description = str(payload.get("description") or "").strip()
        if not description:
            _logger.error(f"Failed to generate table description for {catalog}.{schema_name}.{table_name}")
            raise RuntimeError("Table description agent did not produce a description")
        return payload

    async def _run_column_description_agent(
        self,
        *,
        engine: str,
        catalog: str,
        schema_name: str,
        table_name: str,
        column_name: str | None,
        column_metadata: list[dict[str, Any]],
        sample_size: int,
    ) -> dict[str, Any]:
        _logger.debug(f"In ADK runtime for _run_column_description agent with engine: {engine}, catalog {catalog}, schema {schema_name}, table_name {table_name}, column metadata {column_metadata}")
        user_id = "metadata-maintenance"
        session_id = f"column-description-{uuid.uuid4()}"
        column_names = [
            str(item.get("column_name") or "").strip()
            for item in column_metadata
            if str(item.get("column_name") or "").strip()
        ]
        _logger.debug(f"Column names: {column_names}")
        initial_state = {
            "engine": engine,
            "catalog": catalog,
            "schema_name": schema_name,
            "table_name": table_name,
            "column_name": column_name or "",
            "column_names": column_names,
            "column_metadata_json": json.dumps(column_metadata, default=str),
            "sample_size": sample_size,
        }
        await self._ensure_session(self.COLUMN_DESCRIPTION_APP, user_id, session_id, state=initial_state)

        message = self._Content(
            role="user",
            parts=[
                self._Part(
                    text=(
                        "Generate backend metadata column descriptions for "
                        f"{catalog}.{schema_name}.{table_name}."
                    )
                )
            ],
        )
        await self._run_adk_events(
            self._column_description_runner,
            label="column_description",
            user_id=user_id,
            session_id=session_id,
            new_message=message,
            state_delta=initial_state,
        )

        session = await self._session_service.get_session(
            app_name=self.COLUMN_DESCRIPTION_APP,
            user_id=user_id,
            session_id=session_id,
        )
        if not session:
            raise RuntimeError("Column description session not found")
        await self._persist_session_memory(session)

        payload = self._schema_state(session.state, "column_description_json", ColumnDescriptionOutput, default={})
        if not payload.get("columns"):
            raise RuntimeError("Column description agent did not produce column descriptions")
        return payload

    async def _run_query_to_nlp_agent(
        self,
        *,
        engine: str,
        raw_history_id: int,
        raw_sql: str,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        user_id = "metadata-maintenance"
        session_id = f"query-to-nlp-{uuid.uuid4()}"
        initial_state = {
            "engine": engine,
            "raw_history_id": raw_history_id,
            "raw_sql": raw_sql,
            "source_id": source_id or "",
        }
        await self._ensure_session(self.QUERY_TO_NLP_APP, user_id, session_id, state=initial_state)

        message = self._Content(
            role="user",
            parts=[
                self._Part(
                    text=(
                        "Convert this raw SQL query-history row into a concise analyst request. "
                        "Use the query history context tool before answering.\n\n"
                        f"BACKEND_QUERY_HISTORY_RAW.ID: {raw_history_id}\n\n"
                        f"{raw_sql}"
                    )
                )
            ],
        )
        await self._run_adk_events(
            self._query_to_nlp_runner,
            label="query_to_nlp",
            user_id=user_id,
            session_id=session_id,
            new_message=message,
            state_delta=initial_state,
        )

        session = await self._session_service.get_session(
            app_name=self.QUERY_TO_NLP_APP,
            user_id=user_id,
            session_id=session_id,
        )
        if not session:
            raise RuntimeError("Query-to-NLP session not found")
        await self._persist_session_memory(session)

        payload = self._schema_state(session.state, "query_to_nlp_json", QueryToNlpOutput, default={})
        if not str(payload.get("query_nlp") or "").strip():
            raise RuntimeError("Query-to-NLP agent did not produce query_nlp")
        return payload

    def _query_nlp_history_item(
        self,
        row: dict[str, Any],
        fallback_engine: str,
        query_nlp: str,
        inserted: bool,
        caveats: list[Any],
        *,
        embedding_generated: bool = False,
        embedding_retries: int = 0,
    ) -> dict[str, Any]:
        return {
            "raw_query_history_id": int(self._row_value(row, "ID") or 0),
            "engine": str(self._row_value(row, "ENGINE") or fallback_engine),
            "query_id": str(self._row_value(row, "QUERY_ID") or ""),
            "raw_sql": str(self._row_value(row, "RAW_SQL") or ""),
            "query_nlp": query_nlp,
            "embedding_generated": embedding_generated,
            "embedding_retries": embedding_retries,
            "inserted": inserted,
            "caveats": [str(item) for item in caveats if str(item).strip()],
        }

    def _column_description_item(
        self,
        row: dict[str, Any],
        fallback_engine: str,
        description: str,
        updated: bool,
        caveats: list[Any],
        output_item: dict[str, Any] | None = None,
        *,
        embedding_generated: bool = False,
        embedding_retries: int = 0,
    ) -> dict[str, Any]:
        output = output_item or {}
        return {
            "engine": str(self._row_value(row, "ENGINE") or fallback_engine),
            "catalog": str(self._row_value(row, "CATALOG_NAME") or ""),
            "schema_name": str(self._row_value(row, "SCHEMA_NAME") or ""),
            "table_name": str(self._row_value(row, "TABLE_NAME") or ""),
            "column_name": str(self._row_value(row, "COLUMN_NAME") or ""),
            "data_type": str(self._row_value(row, "DATA_TYPE") or ""),
            "description": description,
            "confidence": float(output.get("confidence") or 0.0),
            "semantic_type": str(output.get("semantic_type") or ""),
            "sample_values": [str(value) for value in output.get("sample_values") or []],
            "embedding_generated": embedding_generated,
            "embedding_retries": embedding_retries,
            "updated": updated,
            "caveats": [str(item) for item in caveats if str(item).strip()],
        }

    async def _ensure_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        state: dict[str, Any] | None = None,
    ) -> None:
        session = await self._session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
        if session:
            return
        await self._session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=state or {},
        )

    @staticmethod
    def _json_state(state: dict[str, Any], key: str, default: Any) -> Any:
        raw = state.get(key)
        if raw is None:
            return default
        if isinstance(raw, BaseModel):
            return raw.model_dump()
        if isinstance(raw, (dict, list)):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    candidate = raw[start : end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return default
                return default
        return default

    @classmethod
    def _schema_state(
        cls,
        state: dict[str, Any],
        key: str,
        schema: type[BaseModel],
        default: dict[str, Any],
    ) -> dict[str, Any]:
        payload = cls._json_state(state, key, default={})
        if not isinstance(payload, dict):
            return default
        try:
            return schema.model_validate(payload).model_dump()
        except ValidationError:
            return default

    @staticmethod
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

    @staticmethod
    def _metadata_name_key(value: Any) -> str:
        return str(value or "").strip().strip('"').strip("`").lower()

    @staticmethod
    def _event_payload(event: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "author": getattr(event, "author", None),
            "invocation_id": getattr(event, "invocation_id", None),
            "id": getattr(event, "id", None),
        }

        texts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        function_responses: list[dict[str, Any]] = []

        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts is None:
            parts = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)

            func_call = getattr(part, "function_call", None)
            if func_call:
                function_calls.append(
                    {
                        "name": getattr(func_call, "name", None),
                        "args": getattr(func_call, "args", None),
                    }
                )

            func_resp = getattr(part, "function_response", None)
            if func_resp:
                function_responses.append(
                    {
                        "name": getattr(func_resp, "name", None),
                        "response": getattr(func_resp, "response", None),
                    }
                )

        if texts:
            payload["text"] = "\n".join(texts)
        if function_calls:
            payload["function_calls"] = function_calls
        if function_responses:
            payload["function_responses"] = function_responses

        actions = getattr(event, "actions", None)
        state_delta = getattr(actions, "state_delta", None) if actions else None
        if state_delta:
            payload["state_delta"] = state_delta

        is_final = False
        is_final_fn = getattr(event, "is_final_response", None)
        if callable(is_final_fn):
            try:
                is_final = bool(is_final_fn())
            except Exception:
                is_final = False
        payload["is_final_response"] = is_final

        return payload

    @classmethod
    def _merge_event_state(cls, state: dict[str, Any], event_payload: dict[str, Any]) -> None:
        state_delta = event_payload.get("state_delta")
        if isinstance(state_delta, dict):
            state.update(state_delta)

        for call in event_payload.get("function_calls") or []:
            if not isinstance(call, dict) or call.get("name") != "set_model_response":
                continue
            args = cls._coerce_mapping(call.get("args"))
            if not args:
                continue

            if "mode" in args:
                state["route_decision_json"] = json.dumps(args, default=str)
            if "UserDirectoryInformation" in args or "directory_summary" in args:
                state["directory_agent_output_json"] = json.dumps(args, default=str)
            if "context_pack" in args or "table_context" in args or "backend_search" in args:
                state["context_bundle_json"] = json.dumps(args, default=str)
            if "draft_sql" in args:
                state["draft_package_json"] = json.dumps(args, default=str)
                state.setdefault("generated_sql", args.get("draft_sql"))
                state.setdefault("submitted_sql", args.get("draft_sql"))
            if "approved" in args and "recommendations" in args:
                state["critic_package_json"] = json.dumps(args, default=str)
            if "refined_sql" in args:
                refined_sql = str(args.get("refined_sql") or "").strip()
                state["refinement_package_json"] = json.dumps(args, default=str)
                if refined_sql:
                    state["refined_sql"] = refined_sql
                    state["submitted_sql"] = refined_sql
            if "is_valid" in args:
                state["validation_package_json"] = json.dumps(args, default=str)
            if "final_sql" in args:
                state["optimization_package_json"] = json.dumps(args, default=str)
                state["final_sql"] = args.get("final_sql")
                state["submitted_sql"] = args.get("final_sql")

        for response in event_payload.get("function_responses") or []:
            if not isinstance(response, dict):
                continue
            response_name = response.get("name")
            payload = cls._coerce_mapping(response.get("response"))
            if response_name == "retrieve_skill_context" and payload:
                state["skillbank_context_json"] = json.dumps(payload, default=str)
                if payload.get("source_id"):
                    state.setdefault("source_id", payload.get("source_id"))
                continue
            if response_name == "compute_verifier_reward" and payload:
                state["verifier_reward_json"] = json.dumps(payload, default=str)
                continue
            if response_name != "optimize_sql":
                continue
            optimized_sql = str(payload.get("optimized_sql") or "").strip()
            if optimized_sql:
                state["optimization_payload_json"] = json.dumps(payload, default=str)
                state.setdefault("final_sql", optimized_sql)
                state.setdefault("submitted_sql", optimized_sql)

    @staticmethod
    def _fill_missing_state(state: dict[str, Any], fallback: dict[str, Any]) -> None:
        for key, value in fallback.items():
            if AdkDataAssistRuntime._empty_state_value(state.get(key)):
                state[key] = value

    @staticmethod
    def _empty_state_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (dict, list, tuple, set)):
            return not value
        return False

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "items"):
            try:
                return dict(value.items())
            except Exception:
                return {}
        return {}

    def _configure_model_env(self) -> None:
        vertex_configured = (
            uses_vertex_provider(self._settings.adk_model_provider)
            or uses_vertex_provider(self._settings.query_generator_model_provider)
            or uses_vertex_provider(self._settings.embedding_provider)
        )
        if not vertex_configured:
            return

        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        if self._settings.vertex_project_id:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self._settings.vertex_project_id)
        if self._settings.vertex_location:
            os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self._settings.vertex_location)

    @staticmethod
    def _validate_pg_identifier(value: str) -> str:
        text = (value or "").strip()
        if not text:
            return "public"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
            raise ValueError(f"Invalid PostgreSQL schema identifier: {value}")
        return text

    @classmethod
    def _build_adk_state_db_url(cls, dsn: str, schema: str) -> str:
        """Build a SQLAlchemy async URL for ADK's DatabaseSessionService.

        ADK uses SQLAlchemy with asyncpg under the hood. The URL must:
          - use the ``postgresql+asyncpg://`` driver scheme
          - avoid libpq-only query params such as ``options``; asyncpg receives
            search_path through ``connect_args.server_settings`` instead.

        Accepts plain ``postgresql://`` (libpq) or ``postgresql+asyncpg://``
        (SQLAlchemy) input; normalises to asyncpg in both cases.
        """
        raw = (dsn or "").strip()
        if not raw:
            raise ValueError(
                "ADK_SESSION_DSN is required for ADK session persistence. "
                "Set it in .env: ADK_SESSION_DSN=postgresql://user:pw@host/db"
            )

        parsed = urlsplit(raw)
        if not parsed.scheme.startswith("postgresql"):
            raise ValueError(
                f"ADK_SESSION_DSN must be a PostgreSQL URL (got scheme={parsed.scheme!r}). "
                "Example: ADK_SESSION_DSN=postgresql://user:pw@host/db"
            )

        # ADK / SQLAlchemy requires the asyncpg driver regardless of what was in .env
        scheme = "postgresql+asyncpg"

        params = parse_qsl(parsed.query, keep_blank_values=True)
        remaining_params = [(k, v) for k, v in params if k != "options"]

        url = urlunsplit(
            (
                scheme,
                parsed.netloc,
                parsed.path,
                urlencode(remaining_params, doseq=True),
                parsed.fragment,
            )
        )
        _logger.debug(f"ADK state DB URL built (schema={schema}, driver=asyncpg)")
        return url

    @staticmethod
    def _build_adk_state_connect_args(schema: str) -> dict[str, Any]:
        return {"server_settings": {"search_path": f"{schema},public"}}

    @classmethod
    def _ensure_postgres_schema(cls, dsn: str, schema: str) -> None:
        """Create the ADK Postgres schema if it does not already exist.

        Uses a synchronous psycopg connection so it can run during
        __init__ before the async event loop starts.
        """
        raw = (dsn or "").strip()
        if not raw:
            raise ValueError(
                "ADK_SESSION_DSN is required for ADK session persistence."
            )

        try:
            import psycopg  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "PostgreSQL ADK session persistence requires psycopg. Install with `pip install psycopg[binary]`."
            ) from exc

        # psycopg.connect requires a libpq DSN — strip the SQLAlchemy driver prefix.
        libpq_dsn = raw.replace("postgresql+asyncpg://", "postgresql://").replace("postgresql+psycopg://", "postgresql://")
        schema_ident = f'"{schema.replace(chr(34), chr(34) * 2)}"'
        with psycopg.connect(libpq_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_ident}")
            conn.commit()
        _logger.info(f"ADK Postgres schema ensured: {schema_ident}")

    @staticmethod
    def _load_adk_components():
        try:
            from google.adk.runners import Runner
            from google.adk.sessions.database_session_service import DatabaseSessionService
            from google.genai.types import Content, Part
        except ImportError as exc:
            raise RuntimeError(
                "Google ADK runtime dependencies are missing. Install 'google-adk' and 'google-genai'."
            ) from exc

        return Runner, DatabaseSessionService, Content, Part

    async def _persist_session_memory(self, session: Any) -> None:
        try:
            await self._memory_service.add_session_to_memory(session)
        except Exception as exc:  # pragma: no cover - runtime persistence path
            _logger.warning("Failed to persist ADK memory for session %s: %s", getattr(session, "id", ""), exc)

    async def _emit(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        await self._event_bus.publish(RunEvent(run_id=run_id, event_type=event_type, payload=payload))
