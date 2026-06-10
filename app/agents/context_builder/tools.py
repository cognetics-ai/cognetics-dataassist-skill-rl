from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import app.services.catalog as catalog_svc
from app.agents.common.runtime_context import AgentDependencies

if TYPE_CHECKING:
    from google.adk.tools import ToolContext
else:  # pragma: no cover - runtime import is optional in tests
    ToolContext = Any


def build_tools(deps: AgentDependencies) -> list:
    """Build context-builder tools bound to backend discovery dependencies.

    Args:
        deps: Shared runtime dependencies used by agent tools.

    Returns:
        List of callable tools for backend table/query context assembly.
    """

    async def build_backend_context(
        prompt: str,
        soeid: str = "",
        engine: str = "starburst",
        catalog: str | None = None,
        database_name: str | None = None,
        schema_name: str | None = None,
        table_top_k: int = 10,
        column_top_k: int = 10,
        query_top_k: int = 5,
        tool_context: Any | None = None,
    ) -> dict[str, Any]:
        """Build Text2SQL context from backend metadata and backend query NLP history.

        Args:
            prompt: Current natural-language user question.
            soeid: Authenticated user identifier, retained for role metadata only.
            engine: Backend engine name for metadata/query-history scope.
            catalog: Logical catalog name for metadata/query-history scope.
            database_name: Physical database name for Snowflake-style sources.
            schema_name: Schema name for metadata/query-history scope.
            table_top_k: Number of table-search candidates to retain.
            column_top_k: Number of column-search table candidates to retain.
            query_top_k: Number of similar backend query examples to retain.
            tool_context: ADK context used to persist the assembled context bundle.

        Returns:
            Context payload with backend tables, columns, and similar query examples.
        """

        normalized_prompt = str(prompt or "").strip() or _state_text(tool_context, "user_prompt")
        if not normalized_prompt:
            normalized_prompt = _state_text(tool_context, "submitted_prompt")
        if not normalized_prompt:
            raise ValueError("prompt is required")

        result = await catalog_svc.build_backend_context(
            deps.settings,
            deps.store,
            deps.embeddings,
            deps.engines,
            normalized_prompt,
            engine=engine or "starburst",
            catalog=catalog,
            database_name=database_name,
            schema_name=schema_name,
            table_top_k=table_top_k,
            column_top_k=column_top_k,
            query_top_k=query_top_k,
        )

        role = _state_text(tool_context, "role_id") or deps.settings.directory_default_role
        segment_scope = _segment_scope(tool_context)
        business_title = _business_title(tool_context)
        payload = {
            "role": role,
            "business_title": business_title,
            "segment_scope": segment_scope,
            "queries": result.get("queries", []),
            "examples": result.get("examples", []),
            "tables": result.get("tables", []),
            "table_context": result.get("table_context", []),
            "similar_queries": result.get("similar_queries", []),
            "metadata": result.get("metadata", {}),
            "metadata_summary": result.get("metadata_summary", {}),
            "context_pack": result.get("context_pack", {}),
            "context_text": str(result.get("metadata_summary", {}).get("context_text") or ""),
            "backend_search": result.get("backend_search", {}),
        }

        if tool_context:
            tool_context.state["role_id"] = role
            tool_context.state["context_bundle_json"] = json.dumps(payload)
        return payload

    async def retrieve_skill_context(
        prompt: str,
        schema_k: int = 15,
        skill_k: int = 6,
        query_k: int = 5,
        source_id: str | None = None,
        engine: str | None = None,
        catalog: str | None = None,
        database_name: str | None = None,
        schema_name: str | None = None,
        tool_context: Any | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant SQL skills and schema docs from the SkillSQL-RL catalog.

        This tool augments the backend context assembled by build_backend_context
        with two types of structured knowledge:
          - SqlSkillBank entries (general SQL patterns + Snowflake dialect heuristics
            + task-specific failure-repair rules) retrieved by Equation 6.
          - Semantic schema docs (table and column descriptions + embeddings) from
            the pgvector catalog, retrieved by cosine similarity to the question.

        When the catalog is unavailable (dev mode, no Postgres), returns empty strings
        so the workflow continues without interruption.

        Args:
            prompt:    The current natural-language question.
            schema_k:  Number of schema documents to retrieve (default 15).
            skill_k:   Max specific skills retrieved per query (default 6).
            query_k:   Max FINISHED query-history examples to retrieve.
            tool_context: ADK context used to persist the skill/schema blocks.

        Returns:
            Payload with skills_text (formatted skill block), catalog_text (schema
            docs block), and counts for observability.
        """
        if not deps.has_catalog:
            return {
                "skills_text": "",
                "catalog_text": "",
                "skills_count": 0,
                "catalog_docs_count": 0,
                "source_id": None,
            }

        normalized = str(prompt or "").strip()
        if not normalized and tool_context:
            normalized = _state_text(tool_context, "user_prompt")

        try:
            from skillsql.skillbank.retrieval import format_skills_for_prompt, retrieve_skills

            res = deps.skillsql_resources
            source_id_text = str(source_id or _state_text(tool_context, "source_id") or "").strip()
            source_uuid = uuid.UUID(source_id_text) if source_id_text else None
            skills = retrieve_skills(
                normalized,
                _dialect(deps),
                repo=res.repo,
                embedder=res.embedder,
                source_id=source_uuid,
                k=skill_k,
            )
            skill_text = format_skills_for_prompt(skills)
            context = catalog_svc.generate_context(
                normalized,
                source_id=source_id_text or None,
                engine=engine or _state_text(tool_context, "engine") or None,
                catalog=catalog or _state_text(tool_context, "catalog") or None,
                database_name=database_name or _state_text(tool_context, "database_name") or None,
                schema_name=schema_name or _state_text(tool_context, "schema_name") or None,
                schema_k=schema_k,
                query_k=query_k,
            )
            payload = {
                "skills_text": skill_text,
                "catalog_text": context["context"],
                "skills_count": skill_text.count("###"),
                "catalog_docs_count": int(context.get("docs_retrieved") or 0),
                "query_examples_count": int(context.get("query_examples_retrieved") or 0),
                "source_id": context.get("source_id"),
            }
        except Exception as e:  # noqa: BLE001 — catalog errors must not abort the workflow
            payload = {
                "skills_text": "",
                "catalog_text": "",
                "skills_count": 0,
                "catalog_docs_count": 0,
                "source_id": None,
                "error": str(e),
            }

        if tool_context:
            tool_context.state["skillbank_context_json"] = json.dumps(payload)
        return payload

    return [build_backend_context, retrieve_skill_context]


def _dialect(deps: AgentDependencies) -> str:
    try:
        if deps.has_catalog and deps.skillsql_resources is not None:
            return deps.skillsql_resources.connector.dialect
    except Exception:  # noqa: BLE001
        pass
    engine = getattr(deps.settings, "default_engine", "snowflake")
    return "snowflake" if "snow" in engine.lower() else engine


def _state_json(tool_context: Any | None, key: str, default: Any) -> Any:
    """Read JSON state from ADK tool context.

    Args:
        tool_context: ADK tool context object.
        key: State key to parse as JSON.
        default: Value returned when key does not exist or is unparsable.

    Returns:
        Parsed JSON object or `default`.
    """

    if not tool_context:
        return default
    raw = tool_context.state.get(key)
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default
    return default


def _state_text(tool_context: Any | None, key: str) -> str:
    if not tool_context:
        return ""
    return str(tool_context.state.get(key) or "").strip()


def _segment_scope(tool_context: Any | None) -> list[str]:
    payload = _state_json(tool_context, "managed_segment_last_two_levels_json", default=[])
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]

    # Fallback path for callers that only have full directory payload in state.
    directory_payload = _state_json(tool_context, "user_directory_information_json", default={})
    info = (
        directory_payload.get("UserDirectoryInformation", {})
        if isinstance(directory_payload, dict)
        else {}
    )
    work = info.get("UsersWorkInformation", {}) if isinstance(info, dict) else {}
    last_two = work.get("ManagedSegmentLastTwoLevels", []) if isinstance(work, dict) else []
    if isinstance(last_two, list):
        return [str(item).strip() for item in last_two if str(item).strip()]
    return []


def _business_title(tool_context: Any | None) -> str:
    directory_payload = _state_json(tool_context, "user_directory_information_json", default={})
    info = (
        directory_payload.get("UserDirectoryInformation", {})
        if isinstance(directory_payload, dict)
        else {}
    )
    work = info.get("UsersWorkInformation", {}) if isinstance(info, dict) else {}
    return str(work.get("BusinessTitle") or "").strip()


def _normalized_table(table_name: str) -> str:
    text = table_name.strip().replace('"', "")
    if "." in text:
        text = text.split(".")[-1]
    return text.upper()


def _dedupe_tables(tables: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in tables:
        value = str(item or "").strip()
        if not value:
            continue
        key = _normalized_table(value)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _extract_tables_from_payload(payload: Any) -> list[str]:
    tables: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "tables" and isinstance(value, list):
                tables.extend(str(item).strip() for item in value if str(item).strip())
                continue
            if key in {"table_details", "table_descriptions", "columns"} and isinstance(
                value, dict
            ):
                tables.extend(str(item).strip() for item in value if str(item).strip())
            if isinstance(value, (dict, list)):
                tables.extend(_extract_tables_from_payload(value))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)):
                tables.extend(_extract_tables_from_payload(item))
    return tables


def _lookup_table_details(table_name: str, table_details: dict[str, Any]) -> dict[str, Any]:
    direct = table_details.get(table_name)
    if isinstance(direct, dict):
        return direct
    normalized = _normalized_table(table_name)
    for key, value in table_details.items():
        if _normalized_table(str(key)) == normalized and isinstance(value, dict):
            return value
    return {}
