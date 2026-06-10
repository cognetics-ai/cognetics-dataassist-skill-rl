from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any
import uuid

from google import genai
from google.genai import types
from google.genai.types import GenerationConfig
from tenacity import stop_after_attempt, retry_if_exception_type, retry,wait_exponential

from app.config import Settings, settings
from app.core.sql_utils import extract_tables
from app.models import QueryHistoryEntry, QueryRun, User, UserDirectoryInformation, UsersWorkInformation

logger = logging.getLogger(__name__)

_FTS_STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "from",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "again",
    "then",
    "once",
    "and",
    "but",
    "or",
    "nor",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "not",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "me",
    "my",
    "i",
    "we",
    "our",
    "you",
    "your",
    "it",
    "its",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "show",
    "find",
    "get",
    "give",
    "tell",
    "list",
    "all",
    "any",
    "column",
    "columns",
    "contain",
    "contained",
    "containing",
    "contains",
    "database",
    "databases",
    "db",
    "field",
    "fields",
    "metadata",
    "query",
    "queries",
    "related",
    "schema",
    "schemas",
    "table",
    "tables",
}


class SQLStore:
    """SQL-backed persistence store with SQLite and PostgreSQL support."""

    def __init__(
        self,
        backend: str = "sqlite",
        sqlite_path: str = "data/data_assist.db",
        postgres_dsn: str = "",
        postgres_schema: str = "public",
        embedding_dimension: int = 768,
    ):
        self._backend = (backend or "sqlite").strip().lower()
        if self._backend not in {"sqlite", "postgres"}:
            raise ValueError(f"Unsupported store backend: {backend}")

        self._users: dict[str, User] = {}
        self._directory: dict[str, UserDirectoryInformation] = {}
        self._history: dict[str, QueryHistoryEntry] = {}
        self._runs: dict[str, QueryRun] = {}
        self._lock = asyncio.Lock()
        self._embedding_dimension = max(1, int(embedding_dimension or 768))
        self._pg_schema = self._validate_pg_identifier(postgres_schema or "public")
        self._pgvector_schema = "extensions"

        if self._backend == "sqlite":
            db_path = Path(sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._db = conn
        else:
            dsn = postgres_dsn.strip()
            if not dsn:
                raise ValueError("DATA_ASSIST_POSTGRES_DSN is required when DATA_ASSIST_DB_BACKEND=postgres")
            try:
                import psycopg  # type: ignore
            except ImportError as exc:  # pragma: no cover - runtime dependency path
                raise RuntimeError(
                    "PostgreSQL backend requires psycopg. Install with `pip install psycopg[binary]`."
                ) from exc

            print(f"DSN=  {dsn}")
            self._db = psycopg.connect(dsn)
            self._init_postgres_schema()

        self._init_db()
        self._load_runs_from_db()

    @staticmethod
    def _validate_pg_identifier(value: str) -> str:
        text = (value or "").strip()
        if not text:
            return "public"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
            raise ValueError(f"Invalid PostgreSQL schema identifier: {value}")
        return text

    @staticmethod
    def _quote_ident(value: str) -> str:
        return f'"{value.replace(chr(34), chr(34) * 2)}"'

    def _init_postgres_schema(self) -> None:
        if self._backend != "postgres":
            return
        schema_ident = self._quote_ident(self._pg_schema)
        print(f"Schema ident: {schema_ident}")
        self._execute(f"CREATE SCHEMA IF NOT EXISTS {schema_ident}")
        self._execute('CREATE SCHEMA IF NOT EXISTS "extensions"')
        self._execute(f"SET search_path TO {schema_ident}, public")
        try:
            self._execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions")
        except Exception as exc:  # pragma: no cover - DB runtime path
            raise RuntimeError(
                "Failed to enable pgvector extension. Install/enable extension `vector` in PostgreSQL."
            ) from exc

        ext_row = self._query_one(
            """
            SELECT n.nspname AS schema_name
            FROM pg_extension e
            JOIN pg_namespace n ON n.oid = e.extnamespace
            WHERE e.extname = 'vector'
            """
        )
        if not ext_row or not ext_row.get("schema_name"):
            raise RuntimeError("pgvector extension is not installed or not discoverable in PostgreSQL.")
        self._pgvector_schema = self._validate_pg_identifier(str(ext_row["schema_name"]))
        self._execute(f"SET search_path TO {schema_ident}, public, {self._quote_ident(self._pgvector_schema)}")
        self._db.commit()

    def _sql(self, statement: str) -> str:
        if self._backend == "postgres":
            return statement.replace("?", "%s")
        return statement

    def _execute(self, statement: str, params: tuple[Any, ...] = ()) -> None:
        cursor = self._db.cursor()
        try:
            if self._backend == "postgres":
                if params:
                    cursor.execute(self._sql(statement), params)
                else:
                    cursor.execute(statement)
            else:
                cursor.execute(self._sql(statement), params)
        finally:
            cursor.close()

    def _query_all(self, statement: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        cursor = self._db.cursor()
        try:
            if self._backend == "postgres":
                if params:
                    cursor.execute(self._sql(statement), params)
                else:
                    cursor.execute(statement)
            else:
                cursor.execute(self._sql(statement), params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in (cursor.description or [])]
        finally:
            cursor.close()

        return [self._to_dict_row(row, columns) for row in rows]

    def _query_one(self, statement: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        cursor = self._db.cursor()
        try:
            if self._backend == "postgres":
                if params:
                    cursor.execute(self._sql(statement), params)
                else:
                    cursor.execute(statement)
            else:
                cursor.execute(self._sql(statement), params)
            row = cursor.fetchone()
            columns = [desc[0] for desc in (cursor.description or [])]
        finally:
            cursor.close()

        if row is None:
            return None
        return self._to_dict_row(row, columns)

    def _ensure_backend_query_nlp_history_table(self, identity: str, embedding_type: str) -> None:
        if self._backend_query_nlp_history_needs_recreate():
            self._execute("DROP TABLE IF EXISTS BACKEND_QUERY_NLP_HISTORY")

        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS BACKEND_QUERY_NLP_HISTORY (
                ID {identity},
                ENGINE TEXT NOT NULL,
                QUERY_ID TEXT NOT NULL,
                CATALOG_NAME TEXT,
                SCHEMA_NAME TEXT,
                QUERY_STATE TEXT,
                QUERY_TYPE TEXT,
                USER_EMAIL TEXT,
                ROLE_NAME TEXT,
                CLUSTER_NAME TEXT,
                SOURCE TEXT,
                CREATED_AT TEXT,
                ENDED_AT TEXT,
                RAW_SQL TEXT NOT NULL,
                QUERY_NLP TEXT NOT NULL,
                EMBEDDING {embedding_type},
                SCHEMA_TABLE TEXT,
                TABLES_JSON TEXT,
                METRICS_JSON TEXT NOT NULL,
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                UNIQUE(ENGINE, QUERY_ID)
            )
            """
        )

    def _backend_query_nlp_history_needs_recreate(self) -> bool:
        if self._backend == "sqlite":
            columns = {
                str(row.get("name") or "").upper(): row
                for row in self._query_all("PRAGMA table_info(BACKEND_QUERY_NLP_HISTORY)")
            }
            return bool(columns) and "EMBEDDING" not in columns

        rows = self._query_all(
            """
            SELECT column_name, udt_name, data_type
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            """,
            (self._pg_schema, "backend_query_nlp_history"),
        )
        columns = {str(self._row_value(row, "column_name") or "").upper(): row for row in rows}
        if not columns:
            return False
        embedding_col = columns.get("EMBEDDING")
        if not embedding_col:
            return True
        return str(self._row_value(embedding_col, "udt_name") or "").lower() != "vector"

    @staticmethod
    def _to_dict_row(row: Any, columns: list[str]) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        if hasattr(row, "keys"):
            return {str(key): row[key] for key in row.keys()}
        if isinstance(row, tuple):
            return {columns[idx]: row[idx] for idx in range(min(len(columns), len(row)))}
        return {}

    def _init_db(self) -> None:
        identity = "INTEGER PRIMARY KEY AUTOINCREMENT" if self._backend == "sqlite" else "BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY"
        vector_schema_prefix = f"{self._pgvector_schema}." if self._backend == "postgres" else ""
        embedding_type = "TEXT" if self._backend == "sqlite" else f"{vector_schema_prefix}vector({self._embedding_dimension})"
        nlp_table_name = (
            f"{self._quote_ident(self._pg_schema)}.DATA_USAGE_NLP_QUERIES"
            if self._backend == "postgres"
            else "DATA_USAGE_NLP_QUERIES"
        )

        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS query_runs (
                run_id TEXT PRIMARY KEY,
                soeid TEXT NOT NULL,
                engine TEXT NOT NULL,
                submitted_text TEXT NOT NULL,
                input_mode TEXT NOT NULL,
                route_mode TEXT,
                submitted_sql TEXT,
                submitted_prompt TEXT,
                natural_language_query TEXT,
                embedding {embedding_type},
                source_id TEXT,
                reward_json TEXT,
                final_sql TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                error_message TEXT,
                stats_json TEXT NOT NULL,
                schema_json TEXT NOT NULL,
                rows_json TEXT NOT NULL
            )
            """
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS DISCOVERY_MASTER_TABLE (
                ID {identity},
                ZONE TEXT,
                TABLE_NAME TEXT NOT NULL,
                TABLE_DESCRIPTION TEXT,
                DOMAIN TEXT,
                STANDARDIZED_DOMAIN TEXT,
                SOURCE_SYSTEM TEXT,
                REGION TEXT,
                COUNTRY TEXT,
                TARGET_DB TEXT,
                PII TEXT,
                CRITICAL_DATA_ELEMENT TEXT,
                ASSET_ID TEXT,
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                UNIQUE(TABLE_NAME, TARGET_DB, ASSET_ID)
            )
            """
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS DATA_USAGE_COMMON_QUERIES (
                ID {identity},
                SOEID TEXT,
                EMAIL TEXT,
                NAME TEXT,
                QUERY TEXT NOT NULL,
                QUERY_HASH TEXT NOT NULL,
                TOOL TEXT,
                SCHEMA_TABLE TEXT,
                ALL_QUERY_TABLES TEXT,
                SUCCESS_PERCENTAGE DOUBLE PRECISION,
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                UNIQUE(SOEID, QUERY_HASH, TOOL, SCHEMA_TABLE)
            )
            """
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS COLUMN_DETAILS (
                ID {identity},
                DOMAIN TEXT,
                TARGET_TABLE_NAME TEXT NOT NULL,
                TARGET_COLUMN_NAME TEXT NOT NULL,
                TARGET_COLUMN_NAME_DESC TEXT,
                CRITICAL_DATA_ELEMENT TEXT,
                PII TEXT,
                PRIMARY_FOREIGN_KEY TEXT,
                TARGET_DB TEXT,
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                UNIQUE(TARGET_TABLE_NAME, TARGET_COLUMN_NAME, TARGET_DB)
            )
            """
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {nlp_table_name} (
                ID {identity},
                EMAIL TEXT NOT NULL,
                SOEID TEXT,
                MANAGED_SEGMENT_L1 TEXT,
                MANAGED_SEGMENT_L2 TEXT,
                MANAGED_SEGMENT_L3 TEXT,
                MANAGED_SEGMENT_L4 TEXT,
                MANAGED_SEGMENT_L5 TEXT,
                MANAGED_SEGMENT_L6 TEXT,
                MANAGED_SEGMENT_L7 TEXT,
                MANAGED_SEGMENT_L8 TEXT,
                MANAGED_SEGMENT_L9 TEXT,
                MANAGED_SEGMENT_L10 TEXT,
                MANAGED_SEGMENT_L11 TEXT,
                MANAGED_SEGMENT_L12 TEXT,
                BUSINESS_TITLE TEXT,
                QUERY TEXT NOT NULL,
                QUERY_IN_NLP TEXT NOT NULL,
                SCHEMA_TABLE TEXT,
                ALL_QUERY_TABLES TEXT,
                EMBEDDINGS {embedding_type},
                UPDATED_AT TEXT NOT NULL,
                UNIQUE(EMAIL, QUERY, SCHEMA_TABLE)
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS BACKEND_METADATA_CATALOGS (
                ENGINE TEXT NOT NULL,
                CATALOG_ID TEXT NOT NULL,
                CATALOG_NAME TEXT,
                DESCRIPTION TEXT,
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                PRIMARY KEY (ENGINE, CATALOG_ID)
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS BACKEND_METADATA_SCHEMAS (
                ENGINE TEXT NOT NULL,
                CATALOG_ID TEXT NOT NULL,
                CATALOG_NAME TEXT,
                SCHEMA_ID TEXT NOT NULL,
                SCHEMA_NAME TEXT,
                DESCRIPTION TEXT,
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                PRIMARY KEY (ENGINE, CATALOG_ID, SCHEMA_ID)
            )
            """
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS BACKEND_METADATA_TABLES (
                ENGINE TEXT NOT NULL,
                CATALOG_ID TEXT NOT NULL,
                CATALOG_NAME TEXT,
                SCHEMA_ID TEXT NOT NULL,
                SCHEMA_NAME TEXT,
                TABLE_ID TEXT NOT NULL,
                TABLE_NAME TEXT,
                TABLE_TYPE TEXT,
                DESCRIPTION TEXT,
                EMBEDDING {embedding_type},
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                PRIMARY KEY (ENGINE, CATALOG_ID, SCHEMA_ID, TABLE_ID)
            )
            """
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS BACKEND_QUERY_HISTORY_RAW (
                ID {identity},
                ENGINE TEXT NOT NULL,
                QUERY_ID TEXT NOT NULL,
                CATALOG_NAME TEXT,
                SCHEMA_NAME TEXT,
                QUERY_STATE TEXT,
                QUERY_TYPE TEXT,
                USER_EMAIL TEXT,
                ROLE_NAME TEXT,
                CLUSTER_NAME TEXT,
                SOURCE TEXT,
                CREATED_AT TEXT,
                ENDED_AT TEXT,
                RAW_SQL TEXT NOT NULL,
                SCHEMA_TABLE TEXT,
                TABLES_JSON TEXT,
                METRICS_JSON TEXT NOT NULL,
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                UNIQUE(ENGINE, QUERY_ID)
            )
            """
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS CATALOG_QUERY_HISTORY (
                ID {identity},
                SOURCE_ID TEXT NOT NULL,
                ENGINE TEXT NOT NULL,
                QUERY_ID TEXT NOT NULL,
                CATALOG_NAME TEXT,
                SCHEMA_NAME TEXT,
                QUERY_STATE TEXT,
                QUERY_TYPE TEXT,
                USER_EMAIL TEXT,
                ROLE_NAME TEXT,
                CLUSTER_NAME TEXT,
                SOURCE TEXT,
                CREATED_AT TEXT,
                ENDED_AT TEXT,
                RAW_SQL TEXT NOT NULL,
                SCHEMA_TABLE TEXT,
                TABLES_JSON TEXT,
                METRICS_JSON TEXT NOT NULL,
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                UNIQUE(SOURCE_ID, ENGINE, QUERY_ID)
            )
            """
        )
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS CATALOG_QUERY_HISTORY_NLP (
                ID {identity},
                RAW_QUERY_HISTORY_ID INTEGER,
                SOURCE_ID TEXT NOT NULL,
                ENGINE TEXT NOT NULL,
                QUERY_ID TEXT NOT NULL,
                CATALOG_NAME TEXT,
                SCHEMA_NAME TEXT,
                QUERY_STATE TEXT,
                QUERY_TYPE TEXT,
                USER_EMAIL TEXT,
                ROLE_NAME TEXT,
                CLUSTER_NAME TEXT,
                SOURCE TEXT,
                CREATED_AT TEXT,
                ENDED_AT TEXT,
                RAW_SQL TEXT NOT NULL,
                NLP_TEXT TEXT NOT NULL,
                EMBEDDINGS {embedding_type},
                SCHEMA_TABLE TEXT,
                TABLES_JSON TEXT,
                METRICS_JSON TEXT NOT NULL,
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                UNIQUE(SOURCE_ID, ENGINE, QUERY_ID)
            )
            """
        )
        self._ensure_backend_query_nlp_history_table(identity, embedding_type)
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS BACKEND_METADATA_COLUMNS (
                ENGINE TEXT NOT NULL,
                CATALOG_ID TEXT NOT NULL,
                CATALOG_NAME TEXT,
                SCHEMA_ID TEXT NOT NULL,
                SCHEMA_NAME TEXT,
                TABLE_ID TEXT NOT NULL,
                TABLE_NAME TEXT,
                COLUMN_ID TEXT NOT NULL,
                COLUMN_NAME TEXT,
                ORDINAL_POSITION INTEGER,
                DATA_TYPE TEXT,
                NULLABLE BOOLEAN,
                DESCRIPTION TEXT,
                EMBEDDING {embedding_type},
                RAW_JSON TEXT NOT NULL,
                UPDATED_AT TEXT NOT NULL,
                PRIMARY KEY (ENGINE, CATALOG_ID, SCHEMA_ID, TABLE_ID, COLUMN_ID)
            )
            """
        )
        self._migrate_backend_metadata_embedding_columns(embedding_type)
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS BACKEND_QUERY_NLP (
                ID {identity},
                ENGINE TEXT NOT NULL,
                QUERY_ID TEXT NOT NULL,
                RAW_QUERY_HISTORY_ID INTEGER,
                EMAIL TEXT,
                SOEID TEXT,
                BUSINESS_TITLE TEXT,
                QUERY TEXT NOT NULL,
                QUERY_IN_NLP TEXT NOT NULL,
                SCHEMA_TABLE TEXT,
                ALL_QUERY_TABLES TEXT,
                EMBEDDINGS {embedding_type},
                UPDATED_AT TEXT NOT NULL,
                UNIQUE(ENGINE, QUERY_ID)
            )
            """
        )

        self._migrate_query_runs_columns()
        self._migrate_data_usage_common_columns()
        self._migrate_data_usage_nlp_columns()

        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_query_runs_soed_created
            ON query_runs(soeid, created_at DESC)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_discovery_master_table_name
            ON DISCOVERY_MASTER_TABLE(TABLE_NAME)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_data_usage_common_queries_soeid
            ON DATA_USAGE_COMMON_QUERIES(SOEID)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_column_details_table_name
            ON COLUMN_DETAILS(TARGET_TABLE_NAME)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backend_metadata_tables_name
            ON BACKEND_METADATA_TABLES(ENGINE, TABLE_NAME)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backend_metadata_columns_table
            ON BACKEND_METADATA_COLUMNS(ENGINE, TABLE_NAME)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backend_query_history_created
            ON BACKEND_QUERY_HISTORY_RAW(ENGINE, CREATED_AT)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_catalog_query_history_source_created
            ON CATALOG_QUERY_HISTORY(SOURCE_ID, ENGINE, CREATED_AT)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_catalog_query_history_scope
            ON CATALOG_QUERY_HISTORY(SOURCE_ID, ENGINE, CATALOG_NAME, SCHEMA_NAME)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backend_query_history_user
            ON BACKEND_QUERY_HISTORY_RAW(USER_EMAIL)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backend_query_nlp_history_created
            ON BACKEND_QUERY_NLP_HISTORY(ENGINE, CREATED_AT)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_catalog_query_history_nlp_source_created
            ON CATALOG_QUERY_HISTORY_NLP(SOURCE_ID, ENGINE, CREATED_AT)
            """
        )
        self._execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backend_query_nlp_email
            ON BACKEND_QUERY_NLP(EMAIL)
            """
        )
        self._execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_data_usage_nlp_soeid
            ON {nlp_table_name}(SOEID)
            """
        )
        self._execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_data_usage_nlp_business_title
            ON {nlp_table_name}(BUSINESS_TITLE)
            """
        )
        if self._backend == "postgres":
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_query_runs_embedding_cosine
                ON query_runs
                USING ivfflat (embedding {vector_schema_prefix}vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_data_usage_nlp_embeddings_cosine
                ON {nlp_table_name}
                USING ivfflat (EMBEDDINGS {vector_schema_prefix}vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_data_usage_nlp_query_in_nlp_fts
                ON {nlp_table_name}
                USING GIN (to_tsvector('english', COALESCE(QUERY_IN_NLP, '')))
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_backend_query_nlp_embeddings_cosine
                ON BACKEND_QUERY_NLP
                USING ivfflat (EMBEDDINGS {vector_schema_prefix}vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            self._execute(
                """
                CREATE INDEX IF NOT EXISTS idx_backend_query_nlp_query_in_nlp_fts
                ON BACKEND_QUERY_NLP
                USING GIN (to_tsvector('english', COALESCE(QUERY_IN_NLP, '')))
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_backend_query_nlp_history_embedding_cosine
                ON BACKEND_QUERY_NLP_HISTORY
                USING ivfflat (EMBEDDING {vector_schema_prefix}vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_backend_query_nlp_history_fts
                ON BACKEND_QUERY_NLP_HISTORY
                USING GIN (to_tsvector('english', {self._backend_query_nlp_history_fts_document()}))
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_catalog_query_history_nlp_embeddings_cosine
                ON CATALOG_QUERY_HISTORY_NLP
                USING ivfflat (EMBEDDINGS {vector_schema_prefix}vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_catalog_query_history_nlp_fts
                ON CATALOG_QUERY_HISTORY_NLP
                USING GIN (to_tsvector('english', {self._catalog_query_history_nlp_fts_document()}))
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_backend_metadata_tables_embedding_cosine
                ON BACKEND_METADATA_TABLES
                USING ivfflat (EMBEDDING {vector_schema_prefix}vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_backend_metadata_tables_fts
                ON BACKEND_METADATA_TABLES
                USING GIN (to_tsvector('english', {self._backend_metadata_table_fts_document()}))
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_backend_metadata_columns_embedding_cosine
                ON BACKEND_METADATA_COLUMNS
                USING ivfflat (EMBEDDING {vector_schema_prefix}vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            self._execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_backend_metadata_columns_fts
                ON BACKEND_METADATA_COLUMNS
                USING GIN (to_tsvector('english', {self._backend_metadata_column_fts_document()}))
                """
            )

        self._db.commit()

    def _migrate_query_runs_columns(self) -> None:
        if self._backend == "sqlite":
            cols = {str(row.get("name")).lower() for row in self._query_all("PRAGMA table_info(query_runs)")}
        else:
            rows = self._query_all(
                """
                SELECT column_name, udt_name
                FROM information_schema.columns
                WHERE table_schema=%s AND table_name='query_runs'
                """,
                (self._pg_schema,),
            )
            cols = {str(row.get("column_name")).lower() for row in rows}

        if "soeid" not in cols and "soed_id" in cols:
            self._execute("ALTER TABLE query_runs ADD COLUMN soeid TEXT")
            self._execute("UPDATE query_runs SET soeid = soed_id WHERE soeid IS NULL")
            cols.add("soeid")

        if "natural_language_query" not in cols:
            self._execute("ALTER TABLE query_runs ADD COLUMN natural_language_query TEXT")
            self._execute(
                """
                UPDATE query_runs
                SET natural_language_query = NULLIF(TRIM(COALESCE(submitted_prompt, '')), '')
                WHERE natural_language_query IS NULL
                """
            )
            cols.add("natural_language_query")

        if "embedding" not in cols:
            if self._backend == "sqlite":
                self._execute("ALTER TABLE query_runs ADD COLUMN embedding TEXT")
            else:
                self._execute(
                    f"ALTER TABLE query_runs "
                    f"ADD COLUMN embedding {self._pgvector_schema}.vector({self._embedding_dimension})"
                )
            cols.add("embedding")

        if "source_id" not in cols:
            self._execute("ALTER TABLE query_runs ADD COLUMN source_id TEXT")
            cols.add("source_id")

        if "reward_json" not in cols:
            self._execute("ALTER TABLE query_runs ADD COLUMN reward_json TEXT")
            cols.add("reward_json")

    def _migrate_data_usage_common_columns(self) -> None:
        if self._backend == "sqlite":
            cols = {str(row.get("name")).lower() for row in self._query_all("PRAGMA table_info(DATA_USAGE_COMMON_QUERIES)")}
            if "all_query_tables" not in cols:
                self._execute("ALTER TABLE DATA_USAGE_COMMON_QUERIES ADD COLUMN ALL_QUERY_TABLES TEXT")
            if "query_hash" not in cols:
                self._execute("ALTER TABLE DATA_USAGE_COMMON_QUERIES ADD COLUMN QUERY_HASH TEXT")
            null_hash_rows = self._query_all(
                """
                SELECT ID, QUERY
                FROM DATA_USAGE_COMMON_QUERIES
                WHERE QUERY_HASH IS NULL
                """
            )
            for row in null_hash_rows:
                self._execute(
                    "UPDATE DATA_USAGE_COMMON_QUERIES SET QUERY_HASH = ? WHERE ID = ?",
                    (
                        self._query_hash(str(row.get("QUERY") or "")),
                        row.get("ID"),
                    ),
                )
            self._execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_data_usage_common_queries_dedupe
                ON DATA_USAGE_COMMON_QUERIES(SOEID, QUERY_HASH, TOOL, SCHEMA_TABLE)
                """
            )
            return

        rows = self._query_all(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name='data_usage_common_queries'
            """,
            (self._pg_schema,),
        )
        cols = {str(row.get("column_name")).lower() for row in rows}
        if "all_query_tables" not in cols:
            self._execute("ALTER TABLE DATA_USAGE_COMMON_QUERIES ADD COLUMN ALL_QUERY_TABLES TEXT")
        if "query_hash" not in cols:
            self._execute("ALTER TABLE DATA_USAGE_COMMON_QUERIES ADD COLUMN QUERY_HASH TEXT")
        self._execute(
            """
            UPDATE DATA_USAGE_COMMON_QUERIES
            SET QUERY_HASH = md5(COALESCE(QUERY, ''))
            WHERE QUERY_HASH IS NULL
            """
        )
        self._execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_data_usage_common_queries_dedupe
            ON DATA_USAGE_COMMON_QUERIES(SOEID, QUERY_HASH, TOOL, SCHEMA_TABLE)
            """
        )

        legacy_constraints = self._query_all(
            """
            SELECT c.conname
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN unnest(c.conkey) WITH ORDINALITY AS cols(attnum, ordinality) ON TRUE
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = cols.attnum
            WHERE n.nspname=%s
              AND t.relname='data_usage_common_queries'
              AND c.contype='u'
            GROUP BY c.conname
            HAVING array_agg(a.attname::text ORDER BY cols.ordinality)
                = ARRAY['soeid', 'query', 'tool', 'schema_table']::text[]
            """,
            (self._pg_schema,),
        )
        for row in legacy_constraints:
            constraint_name = str(row.get("conname") or "").strip()
            if constraint_name:
                self._execute(
                    f"ALTER TABLE DATA_USAGE_COMMON_QUERIES DROP CONSTRAINT IF EXISTS {self._quote_ident(constraint_name)}"
                )

    def _migrate_data_usage_nlp_columns(self) -> None:
        if self._backend == "sqlite":
            cols = {str(row.get("name")).lower() for row in self._query_all("PRAGMA table_info(DATA_USAGE_NLP_QUERIES)")}
            if "embeddings" not in cols:
                self._execute("ALTER TABLE DATA_USAGE_NLP_QUERIES ADD COLUMN EMBEDDINGS TEXT")
            if "business_title" not in cols:
                self._execute("ALTER TABLE DATA_USAGE_NLP_QUERIES ADD COLUMN BUSINESS_TITLE TEXT")
            if "all_query_tables" not in cols:
                self._execute("ALTER TABLE DATA_USAGE_NLP_QUERIES ADD COLUMN ALL_QUERY_TABLES TEXT")
            return

        rows = self._query_all(
            """
            SELECT column_name, udt_name
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name='data_usage_nlp_queries'
            """,
            (self._pg_schema,),
        )
        cols = {str(row.get("column_name")).lower(): str(row.get("udt_name")).lower() for row in rows}
        if "embeddings" not in cols:
            self._execute(
                f"ALTER TABLE DATA_USAGE_NLP_QUERIES "
                f"ADD COLUMN EMBEDDINGS {self._pgvector_schema}.vector({self._embedding_dimension})"
            )
            cols["embeddings"] = "vector"
        if "business_title" not in cols:
            self._execute("ALTER TABLE DATA_USAGE_NLP_QUERIES ADD COLUMN BUSINESS_TITLE TEXT")
        if "all_query_tables" not in cols:
            self._execute("ALTER TABLE DATA_USAGE_NLP_QUERIES ADD COLUMN ALL_QUERY_TABLES TEXT")

        if cols.get("embeddings") != "vector":
            raise RuntimeError(
                "DATA_USAGE_NLP_QUERIES.embeddings must be vector type in PostgreSQL. "
                "Please migrate the column to pgvector."
            )

        dim_row = self._query_one(
            """
            SELECT
                a.atttypmod AS dim_typmod,
                format_type(a.atttypid, a.atttypmod) AS formatted_type
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname=%s
              AND c.relname='data_usage_nlp_queries'
              AND a.attname='embeddings'
              AND a.attnum > 0
            """,
            (self._pg_schema,),
        )
        if dim_row:
            existing_dimension = self._extract_vector_dimension(dim_row)
            if existing_dimension and existing_dimension != self._embedding_dimension:
                raise RuntimeError(
                    "Configured embedding dimension does not match DATA_USAGE_NLP_QUERIES.embeddings. "
                    f"expected={self._embedding_dimension}, existing={existing_dimension}"
                )

    def _migrate_backend_metadata_embedding_columns(self, embedding_type: str) -> None:
        for table_name in ("BACKEND_METADATA_TABLES", "BACKEND_METADATA_COLUMNS"):
            if self._backend == "sqlite":
                cols = {
                    str(row.get("name") or "").lower()
                    for row in self._query_all(f"PRAGMA table_info({table_name})")
                }
                if "embedding" not in cols:
                    self._execute(f"ALTER TABLE {table_name} ADD COLUMN EMBEDDING TEXT")
                continue

            rows = self._query_all(
                """
                SELECT column_name, udt_name
                FROM information_schema.columns
                WHERE table_schema = ?
                  AND table_name = ?
                """,
                (self._pg_schema, table_name.lower()),
            )
            cols = {str(row.get("column_name") or "").lower(): str(row.get("udt_name") or "").lower() for row in rows}
            if "embedding" not in cols:
                self._execute(f"ALTER TABLE {table_name} ADD COLUMN EMBEDDING {embedding_type}")
                continue
            if cols.get("embedding") != "vector":
                self._execute(f"ALTER TABLE {table_name} DROP COLUMN EMBEDDING")
                self._execute(f"ALTER TABLE {table_name} ADD COLUMN EMBEDDING {embedding_type}")
                continue

            dim_row = self._query_one(
                """
                SELECT
                    a.atttypmod AS dim_typmod,
                    format_type(a.atttypid, a.atttypmod) AS formatted_type
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = ?
                  AND c.relname = ?
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                """,
                (self._pg_schema, table_name.lower()),
            )
            if dim_row:
                existing_dimension = self._extract_vector_dimension(dim_row)
                if existing_dimension and existing_dimension != self._embedding_dimension:
                    self._execute(f"ALTER TABLE {table_name} DROP COLUMN EMBEDDING")
                    self._execute(f"ALTER TABLE {table_name} ADD COLUMN EMBEDDING {embedding_type}")

    @staticmethod
    def _to_iso(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    @staticmethod
    def _from_iso(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _row_to_run(self, row: dict[str, Any]) -> QueryRun:
        return QueryRun(
            run_id=str(row["run_id"]),
            soeid=str(row["soeid"]),
            engine=str(row["engine"]),
            submitted_text=str(row["submitted_text"]),
            input_mode=str(row["input_mode"]),
            route_mode=row.get("route_mode"),
            submitted_sql=row.get("submitted_sql"),
            submitted_prompt=row.get("submitted_prompt"),
            natural_language_query=row.get("natural_language_query"),
            embedding=self._deserialize_embeddings(row.get("embedding")),
            source_id=row.get("source_id"),
            reward_json=json.loads(str(row.get("reward_json") or "null")),
            final_sql=str(row["final_sql"]),
            status=str(row["status"]),
            created_at=self._from_iso(row.get("created_at")) or datetime.now(timezone.utc),
            started_at=self._from_iso(row.get("started_at")),
            ended_at=self._from_iso(row.get("ended_at")),
            error_message=row.get("error_message"),
            stats=json.loads(str(row.get("stats_json") or "{}")),
            schema=json.loads(str(row.get("schema_json") or "[]")),
            rows=json.loads(str(row.get("rows_json") or "[]")),
        )

    def _persist_run(self, run: QueryRun) -> None:
        self._execute(
            """
            INSERT INTO query_runs(
                run_id, soeid, engine, submitted_text, input_mode, route_mode,
                submitted_sql, submitted_prompt, natural_language_query, embedding,
                source_id, reward_json,
                final_sql, status, created_at, started_at, ended_at, error_message,
                stats_json, schema_json, rows_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                soeid=excluded.soeid,
                engine=excluded.engine,
                submitted_text=excluded.submitted_text,
                input_mode=excluded.input_mode,
                route_mode=excluded.route_mode,
                submitted_sql=excluded.submitted_sql,
                submitted_prompt=excluded.submitted_prompt,
                natural_language_query=excluded.natural_language_query,
                embedding=excluded.embedding,
                source_id=excluded.source_id,
                reward_json=excluded.reward_json,
                final_sql=excluded.final_sql,
                status=excluded.status,
                created_at=excluded.created_at,
                started_at=excluded.started_at,
                ended_at=excluded.ended_at,
                error_message=excluded.error_message,
                stats_json=excluded.stats_json,
                schema_json=excluded.schema_json,
                rows_json=excluded.rows_json
            """,
            (
                run.run_id,
                run.soeid,
                run.engine,
                run.submitted_text,
                run.input_mode,
                run.route_mode,
                run.submitted_sql,
                run.submitted_prompt,
                run.natural_language_query,
                self._serialize_embeddings(run.embedding),
                run.source_id,
                json.dumps(run.reward_json) if run.reward_json is not None else None,
                run.final_sql,
                run.status,
                self._to_iso(run.created_at),
                self._to_iso(run.started_at),
                self._to_iso(run.ended_at),
                run.error_message,
                json.dumps(run.stats or {}),
                json.dumps(run.schema or []),
                json.dumps(run.rows or []),
            ),
        )
        self._db.commit()

    def _load_runs_from_db(self) -> None:
        rows = self._query_all("SELECT * FROM query_runs")
        for row in rows:
            run = self._row_to_run(row)
            self._runs[run.run_id] = run

    async def seed(self) -> None:
        """No-op placeholder for runtime initialization.

        Legacy mock seed data has been removed for production-grade directory-backed auth.
        """
        return None

    async def get_user(self, soeid: str) -> User | None:
        return self._users.get(soeid)

    async def get_directory_information(self, soeid: str) -> UserDirectoryInformation | None:
        return self._directory.get(soeid)

    async def list_queries_by_role(self, role_id: str, limit: int = 10) -> list[QueryHistoryEntry]:
        queries = [item for item in self._history.values() if item.role_id == role_id and item.status == "succeeded"]
        queries.sort(key=lambda item: item.created_at, reverse=True)
        return queries[:limit]

    async def create_run(
        self,
        soeid: str,
        engine: str,
        submitted_text: str,
        input_mode: str,
        submitted_sql: str | None = None,
        submitted_prompt: str | None = None,
        natural_language_query: str | None = None,
        source_id: str | None = None,
    ) -> QueryRun:
        async with self._lock:
            run_id = str(uuid.uuid4())
            run = QueryRun(
                run_id=run_id,
                soeid=soeid,
                engine=engine,
                submitted_text=submitted_text,
                input_mode=input_mode,
                submitted_sql=submitted_sql,
                submitted_prompt=submitted_prompt,
                natural_language_query=natural_language_query,
                source_id=source_id,
                final_sql=submitted_sql or "",
            )
            self._runs[run_id] = run
            self._persist_run(run)
            return replace(run)

    async def get_run(self, run_id: str) -> QueryRun | None:
        run = self._runs.get(run_id)
        if run:
            return replace(run)
        row = self._query_one("SELECT * FROM query_runs WHERE run_id = ?", (run_id,))
        if not row:
            return None
        hydrated = self._row_to_run(row)
        self._runs[run_id] = hydrated
        return replace(hydrated)

    async def update_run(self, run_id: str, **kwargs: Any) -> QueryRun | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            for key, value in kwargs.items():
                setattr(run, key, value)
            self._runs[run_id] = run
            self._persist_run(run)
            return replace(run)

    async def list_runs_for_user(self, soeid: str, limit: int = 100) -> list[QueryRun]:
        runs = [item for item in self._runs.values() if item.soeid == soeid]
        runs.sort(key=lambda item: item.started_at or item.ended_at or item.created_at, reverse=True)
        return [replace(item) for item in runs[:limit]]

    async def list_runs_for_skill_evolution(
        self,
        *,
        limit: int = 100,
        statuses: list[str] | None = None,
    ) -> list[QueryRun]:
        status_set = {str(item).strip().lower() for item in (statuses or ["succeeded", "failed"])}
        runs = []
        for run in self._runs.values():
            if status_set and run.status.lower() not in status_set:
                continue
            if not (run.natural_language_query or run.submitted_prompt):
                continue
            if not (run.final_sql or run.submitted_sql):
                continue
            runs.append(run)
        runs.sort(key=lambda item: item.ended_at or item.started_at or item.created_at, reverse=True)
        return [replace(item) for item in runs[:limit]]

    async def list_users_by_role(self, role_id: str) -> list[User]:
        return [replace(user) for user in self._users.values() if user.role_id == role_id]

    async def upsert_discovery_master_rows(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        async with self._lock:
            for row in rows:
                table_name = str(row.get("table_name") or "").strip()
                if not table_name:
                    continue
                target_db = str(row.get("target_db") or "").strip()
                asset_id = str(row.get("asset_id") or "").strip()
                self._execute(
                    """
                    INSERT INTO DISCOVERY_MASTER_TABLE(
                        ZONE, TABLE_NAME, TABLE_DESCRIPTION, DOMAIN, STANDARDIZED_DOMAIN, SOURCE_SYSTEM,
                        REGION, COUNTRY, TARGET_DB, PII, CRITICAL_DATA_ELEMENT, ASSET_ID, RAW_JSON, UPDATED_AT
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(TABLE_NAME, TARGET_DB, ASSET_ID) DO UPDATE SET
                        ZONE=excluded.ZONE,
                        TABLE_DESCRIPTION=excluded.TABLE_DESCRIPTION,
                        DOMAIN=excluded.DOMAIN,
                        STANDARDIZED_DOMAIN=excluded.STANDARDIZED_DOMAIN,
                        SOURCE_SYSTEM=excluded.SOURCE_SYSTEM,
                        REGION=excluded.REGION,
                        COUNTRY=excluded.COUNTRY,
                        PII=excluded.PII,
                        CRITICAL_DATA_ELEMENT=excluded.CRITICAL_DATA_ELEMENT,
                        RAW_JSON=excluded.RAW_JSON,
                        UPDATED_AT=excluded.UPDATED_AT
                    """,
                    (
                        row.get("zone"),
                        table_name,
                        row.get("table_description"),
                        row.get("domain"),
                        row.get("standardized_domain"),
                        row.get("source_system"),
                        row.get("region"),
                        row.get("country"),
                        target_db,
                        row.get("pii"),
                        row.get("critical_data_element"),
                        asset_id,
                        json.dumps(row.get("raw") or {}),
                        now,
                    ),
                )
                inserted += 1
            self._db.commit()
        return inserted

    async def upsert_common_query_rows(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        async with self._lock:
            for row in rows:
                query_text = str(row.get("query") or "").strip()
                if not query_text:
                    continue
                query_hash = self._query_hash(query_text)
                soeid = str(row.get("soeid") or "").strip()
                tool = str(row.get("tool") or "").strip()
                schema_table = str(row.get("schema_table") or "").strip()
                all_query_tables = self._coerce_table_list(row.get("all_query_tables"))
                if not all_query_tables:
                    all_query_tables = extract_tables(query_text)
                if not schema_table and all_query_tables:
                    schema_table = all_query_tables[0]
                self._execute(
                    """
                    INSERT INTO DATA_USAGE_COMMON_QUERIES(
                        SOEID, EMAIL, NAME, QUERY, QUERY_HASH, TOOL, SCHEMA_TABLE, ALL_QUERY_TABLES, SUCCESS_PERCENTAGE, RAW_JSON, UPDATED_AT
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(SOEID, QUERY_HASH, TOOL, SCHEMA_TABLE) DO UPDATE SET
                        EMAIL=excluded.EMAIL,
                        NAME=excluded.NAME,
                        QUERY=excluded.QUERY,
                        ALL_QUERY_TABLES=excluded.ALL_QUERY_TABLES,
                        SUCCESS_PERCENTAGE=excluded.SUCCESS_PERCENTAGE,
                        RAW_JSON=excluded.RAW_JSON,
                        UPDATED_AT=excluded.UPDATED_AT
                    """,
                    (
                        soeid,
                        row.get("email"),
                        row.get("name"),
                        query_text,
                        query_hash,
                        tool,
                        schema_table,
                        json.dumps(all_query_tables) if all_query_tables else None,
                        row.get("success_percentage"),
                        json.dumps(row.get("raw") or {}),
                        now,
                    ),
                )
                inserted += 1
            self._db.commit()
        return inserted

    async def upsert_column_detail_rows(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        async with self._lock:
            for row in rows:
                table_name = str(row.get("target_table_name") or "").strip()
                column_name = str(row.get("target_column_name") or "").strip()
                if not table_name or not column_name:
                    continue
                target_db = str(row.get("target_db") or "").strip()
                self._execute(
                    """
                    INSERT INTO COLUMN_DETAILS(
                        DOMAIN, TARGET_TABLE_NAME, TARGET_COLUMN_NAME, TARGET_COLUMN_NAME_DESC,
                        CRITICAL_DATA_ELEMENT, PII, PRIMARY_FOREIGN_KEY, TARGET_DB, RAW_JSON, UPDATED_AT
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(TARGET_TABLE_NAME, TARGET_COLUMN_NAME, TARGET_DB) DO UPDATE SET
                        DOMAIN=excluded.DOMAIN,
                        TARGET_COLUMN_NAME_DESC=excluded.TARGET_COLUMN_NAME_DESC,
                        CRITICAL_DATA_ELEMENT=excluded.CRITICAL_DATA_ELEMENT,
                        PII=excluded.PII,
                        PRIMARY_FOREIGN_KEY=excluded.PRIMARY_FOREIGN_KEY,
                        RAW_JSON=excluded.RAW_JSON,
                        UPDATED_AT=excluded.UPDATED_AT
                    """,
                    (
                        row.get("domain"),
                        table_name,
                        column_name,
                        row.get("target_column_name_desc"),
                        row.get("critical_data_element"),
                        row.get("pii"),
                        row.get("primary_foreign_key"),
                        target_db,
                        json.dumps(row.get("raw") or {}),
                        now,
                    ),
                )
                inserted += 1
            self._db.commit()
        return inserted

    async def upsert_backend_metadata_records(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"catalogs": 0, "schemas": 0, "tables": 0, "columns": 0}
        if not rows:
            return counts

        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            for row in rows:
                entity_type = str(row.get("entity_type") or "").strip().lower()
                engine = str(row.get("engine") or "").strip().lower()
                catalog_id = str(row.get("catalog_id") or "").strip()
                if not engine or not catalog_id:
                    continue

                raw_json = json.dumps(row.get("raw") or {}, default=str)
                if entity_type == "catalog":
                    self._execute(
                        """
                        INSERT INTO BACKEND_METADATA_CATALOGS(
                            ENGINE, CATALOG_ID, CATALOG_NAME, DESCRIPTION, RAW_JSON, UPDATED_AT
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ENGINE, CATALOG_ID) DO UPDATE SET
                            CATALOG_NAME=excluded.CATALOG_NAME,
                            DESCRIPTION=excluded.DESCRIPTION,
                            RAW_JSON=excluded.RAW_JSON,
                            UPDATED_AT=excluded.UPDATED_AT
                        """,
                        (
                            engine,
                            catalog_id,
                            row.get("catalog_name"),
                            row.get("description"),
                            raw_json,
                            now,
                        ),
                    )
                    counts["catalogs"] += 1
                    continue

                schema_id = str(row.get("schema_id") or "").strip()
                if not schema_id:
                    continue

                if entity_type == "schema":
                    self._execute(
                        """
                        INSERT INTO BACKEND_METADATA_SCHEMAS(
                            ENGINE, CATALOG_ID, CATALOG_NAME, SCHEMA_ID, SCHEMA_NAME, DESCRIPTION, RAW_JSON, UPDATED_AT
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ENGINE, CATALOG_ID, SCHEMA_ID) DO UPDATE SET
                            CATALOG_NAME=excluded.CATALOG_NAME,
                            SCHEMA_NAME=excluded.SCHEMA_NAME,
                            DESCRIPTION=excluded.DESCRIPTION,
                            RAW_JSON=excluded.RAW_JSON,
                            UPDATED_AT=excluded.UPDATED_AT
                        """,
                        (
                            engine,
                            catalog_id,
                            row.get("catalog_name"),
                            schema_id,
                            row.get("schema_name"),
                            row.get("description"),
                            raw_json,
                            now,
                        ),
                    )
                    counts["schemas"] += 1
                    continue

                table_id = str(row.get("table_id") or "").strip()
                if not table_id:
                    continue

                if entity_type == "table":
                    self._execute(
                        """
                        INSERT INTO BACKEND_METADATA_TABLES(
                            ENGINE, CATALOG_ID, CATALOG_NAME, SCHEMA_ID, SCHEMA_NAME,
                            TABLE_ID, TABLE_NAME, TABLE_TYPE, DESCRIPTION, RAW_JSON, UPDATED_AT
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ENGINE, CATALOG_ID, SCHEMA_ID, TABLE_ID) DO UPDATE SET
                            CATALOG_NAME=excluded.CATALOG_NAME,
                            SCHEMA_NAME=excluded.SCHEMA_NAME,
                            TABLE_NAME=excluded.TABLE_NAME,
                            TABLE_TYPE=excluded.TABLE_TYPE,
                            DESCRIPTION=excluded.DESCRIPTION,
                            RAW_JSON=excluded.RAW_JSON,
                            UPDATED_AT=excluded.UPDATED_AT
                        """,
                        (
                            engine,
                            catalog_id,
                            row.get("catalog_name"),
                            schema_id,
                            row.get("schema_name"),
                            table_id,
                            row.get("table_name"),
                            row.get("object_type"),
                            row.get("description"),
                            raw_json,
                            now,
                        ),
                    )
                    counts["tables"] += 1
                    continue

                column_id = str(row.get("column_id") or "").strip()
                if entity_type == "column" and column_id:
                    self._execute(
                        """
                        INSERT INTO BACKEND_METADATA_COLUMNS(
                            ENGINE, CATALOG_ID, CATALOG_NAME, SCHEMA_ID, SCHEMA_NAME,
                            TABLE_ID, TABLE_NAME, COLUMN_ID, COLUMN_NAME, ORDINAL_POSITION,
                            DATA_TYPE, NULLABLE, DESCRIPTION, RAW_JSON, UPDATED_AT
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ENGINE, CATALOG_ID, SCHEMA_ID, TABLE_ID, COLUMN_ID) DO UPDATE SET
                            CATALOG_NAME=excluded.CATALOG_NAME,
                            SCHEMA_NAME=excluded.SCHEMA_NAME,
                            TABLE_NAME=excluded.TABLE_NAME,
                            COLUMN_NAME=excluded.COLUMN_NAME,
                            ORDINAL_POSITION=excluded.ORDINAL_POSITION,
                            DATA_TYPE=excluded.DATA_TYPE,
                            NULLABLE=excluded.NULLABLE,
                            DESCRIPTION=excluded.DESCRIPTION,
                            RAW_JSON=excluded.RAW_JSON,
                            UPDATED_AT=excluded.UPDATED_AT
                        """,
                        (
                            engine,
                            catalog_id,
                            row.get("catalog_name"),
                            schema_id,
                            row.get("schema_name"),
                            table_id,
                            row.get("table_name"),
                            column_id,
                            row.get("column_name"),
                            row.get("ordinal_position"),
                            row.get("data_type"),
                            row.get("nullable"),
                            row.get("description"),
                            raw_json,
                            now,
                        ),
                    )
                    counts["columns"] += 1
            self._db.commit()
        return counts

    async def upsert_backend_query_history_rows(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        async with self._lock:
            for row in rows:
                engine = str(row.get("engine") or "").strip().lower()
                query_id = str(row.get("query_id") or "").strip()
                raw_sql = str(row.get("raw_sql") or "").strip()
                if not engine or not query_id or not raw_sql:
                    continue

                tables = self._coerce_table_list(row.get("tables"))
                schema_table = str(row.get("schema_table") or "").strip()
                if not schema_table and tables:
                    schema_table = tables[0]

                self._execute(
                    """
                    INSERT INTO BACKEND_QUERY_HISTORY_RAW(
                        ENGINE, QUERY_ID, CATALOG_NAME, SCHEMA_NAME, QUERY_STATE, QUERY_TYPE,
                        USER_EMAIL, ROLE_NAME, CLUSTER_NAME, SOURCE, CREATED_AT, ENDED_AT,
                        RAW_SQL, SCHEMA_TABLE, TABLES_JSON, METRICS_JSON, RAW_JSON, UPDATED_AT
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ENGINE, QUERY_ID) DO UPDATE SET
                        CATALOG_NAME=excluded.CATALOG_NAME,
                        SCHEMA_NAME=excluded.SCHEMA_NAME,
                        QUERY_STATE=excluded.QUERY_STATE,
                        QUERY_TYPE=excluded.QUERY_TYPE,
                        USER_EMAIL=excluded.USER_EMAIL,
                        ROLE_NAME=excluded.ROLE_NAME,
                        CLUSTER_NAME=excluded.CLUSTER_NAME,
                        SOURCE=excluded.SOURCE,
                        CREATED_AT=excluded.CREATED_AT,
                        ENDED_AT=excluded.ENDED_AT,
                        RAW_SQL=excluded.RAW_SQL,
                        SCHEMA_TABLE=excluded.SCHEMA_TABLE,
                        TABLES_JSON=excluded.TABLES_JSON,
                        METRICS_JSON=excluded.METRICS_JSON,
                        RAW_JSON=excluded.RAW_JSON,
                        UPDATED_AT=excluded.UPDATED_AT
                    """,
                    (
                        engine,
                        query_id,
                        row.get("catalog_name"),
                        row.get("schema_name"),
                        row.get("query_state"),
                        row.get("query_type"),
                        row.get("user_email"),
                        row.get("role_name"),
                        row.get("cluster_name"),
                        row.get("source"),
                        self._coerce_iso_text(row.get("created_at")),
                        self._coerce_iso_text(row.get("ended_at")),
                        raw_sql,
                        schema_table or None,
                        json.dumps(tables, default=str) if tables else None,
                        json.dumps(row.get("metrics") or {}, default=str),
                        json.dumps(row.get("raw") or {}, default=str),
                        now,
                    ),
                )
                inserted += 1
            self._db.commit()
        return inserted

    async def upsert_catalog_query_history_rows(
        self,
        *,
        source_id: str,
        rows: list[dict[str, Any]],
    ) -> int:
        normalized_source_id = str(source_id or "").strip()
        if not normalized_source_id or not rows:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        async with self._lock:
            for row in rows:
                engine = str(row.get("engine") or "").strip().lower()
                query_id = str(row.get("query_id") or "").strip()
                raw_sql = str(row.get("raw_sql") or "").strip()
                if not engine or not query_id or not raw_sql:
                    continue

                tables = self._coerce_table_list(row.get("tables"))
                schema_table = str(row.get("schema_table") or "").strip()
                if not schema_table and tables:
                    schema_table = tables[0]

                self._execute(
                    """
                    INSERT INTO CATALOG_QUERY_HISTORY(
                        SOURCE_ID, ENGINE, QUERY_ID, CATALOG_NAME, SCHEMA_NAME, QUERY_STATE, QUERY_TYPE,
                        USER_EMAIL, ROLE_NAME, CLUSTER_NAME, SOURCE, CREATED_AT, ENDED_AT,
                        RAW_SQL, SCHEMA_TABLE, TABLES_JSON, METRICS_JSON, RAW_JSON, UPDATED_AT
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(SOURCE_ID, ENGINE, QUERY_ID) DO UPDATE SET
                        CATALOG_NAME=excluded.CATALOG_NAME,
                        SCHEMA_NAME=excluded.SCHEMA_NAME,
                        QUERY_STATE=excluded.QUERY_STATE,
                        QUERY_TYPE=excluded.QUERY_TYPE,
                        USER_EMAIL=excluded.USER_EMAIL,
                        ROLE_NAME=excluded.ROLE_NAME,
                        CLUSTER_NAME=excluded.CLUSTER_NAME,
                        SOURCE=excluded.SOURCE,
                        CREATED_AT=excluded.CREATED_AT,
                        ENDED_AT=excluded.ENDED_AT,
                        RAW_SQL=excluded.RAW_SQL,
                        SCHEMA_TABLE=excluded.SCHEMA_TABLE,
                        TABLES_JSON=excluded.TABLES_JSON,
                        METRICS_JSON=excluded.METRICS_JSON,
                        RAW_JSON=excluded.RAW_JSON,
                        UPDATED_AT=excluded.UPDATED_AT
                    """,
                    (
                        normalized_source_id,
                        engine,
                        query_id,
                        row.get("catalog_name"),
                        row.get("schema_name"),
                        row.get("query_state"),
                        row.get("query_type"),
                        row.get("user_email"),
                        row.get("role_name"),
                        row.get("cluster_name"),
                        row.get("source"),
                        self._coerce_iso_text(row.get("created_at")),
                        self._coerce_iso_text(row.get("ended_at")),
                        raw_sql,
                        schema_table or None,
                        json.dumps(tables, default=str) if tables else None,
                        json.dumps(row.get("metrics") or {}, default=str),
                        json.dumps(row.get("raw") or {}, default=str),
                        now,
                    ),
                )
                inserted += 1
            self._db.commit()
        return inserted

    async def list_backend_query_history_raw_for_nlp_history(
        self,
        *,
        limit: int = 100,
        engine: str | None = None,
        ids: list[int] | None = None,
        raw_sql: str | None = None,
        missing_only: bool = True,
    ) -> list[dict[str, Any]]:
        where = ["h.RAW_SQL IS NOT NULL", "TRIM(h.RAW_SQL) <> ''"]
        params: list[Any] = []
        normalized_engine = str(engine or "").strip().lower()
        if normalized_engine:
            where.append("h.ENGINE = ?")
            params.append(normalized_engine)
        normalized_ids = sorted({int(item) for item in (ids or []) if int(item) > 0})
        if normalized_ids:
            placeholders = ",".join("?" for _ in normalized_ids)
            where.append(f"h.ID IN ({placeholders})")
            params.extend(normalized_ids)
        if raw_sql:
            where.append("(h.RAW_SQL = ? OR TRIM(h.RAW_SQL) = ?)")
            params.extend([raw_sql, str(raw_sql).strip()])
        if missing_only:
            where.append("n.QUERY_ID IS NULL")

        effective_limit = len(normalized_ids) if normalized_ids else max(1, int(limit))
        params.append(effective_limit)
        async with self._lock:
            return self._query_all(
                f"""
                SELECT
                    h.ID AS ID,
                    h.ENGINE AS ENGINE,
                    h.QUERY_ID AS QUERY_ID,
                    h.CATALOG_NAME AS CATALOG_NAME,
                    h.SCHEMA_NAME AS SCHEMA_NAME,
                    h.QUERY_STATE AS QUERY_STATE,
                    h.QUERY_TYPE AS QUERY_TYPE,
                    h.USER_EMAIL AS USER_EMAIL,
                    h.ROLE_NAME AS ROLE_NAME,
                    h.CLUSTER_NAME AS CLUSTER_NAME,
                    h.SOURCE AS SOURCE,
                    h.CREATED_AT AS CREATED_AT,
                    h.ENDED_AT AS ENDED_AT,
                    h.RAW_SQL AS RAW_SQL,
                    h.SCHEMA_TABLE AS SCHEMA_TABLE,
                    h.TABLES_JSON AS TABLES_JSON,
                    h.METRICS_JSON AS METRICS_JSON,
                    h.RAW_JSON AS RAW_JSON,
                    h.UPDATED_AT AS UPDATED_AT
                FROM BACKEND_QUERY_HISTORY_RAW h
                LEFT JOIN BACKEND_QUERY_NLP_HISTORY n
                  ON n.ENGINE = h.ENGINE
                 AND n.QUERY_ID = h.QUERY_ID
                WHERE {" AND ".join(where)}
                ORDER BY COALESCE(h.CREATED_AT, h.UPDATED_AT) DESC
                LIMIT ?
                """,
                tuple(params),
            )

    async def list_catalog_query_history_for_nlp(
        self,
        *,
        source_id: str | None = None,
        limit: int = 100,
        engine: str | None = None,
        ids: list[int] | None = None,
        raw_sql: str | None = None,
        missing_only: bool = True,
    ) -> list[dict[str, Any]]:
        where = ["h.RAW_SQL IS NOT NULL", "TRIM(h.RAW_SQL) <> ''"]
        params: list[Any] = []
        normalized_source_id = str(source_id or "").strip()
        if normalized_source_id:
            where.append("h.SOURCE_ID = ?")
            params.append(normalized_source_id)
        normalized_engine = str(engine or "").strip().lower()
        if normalized_engine:
            where.append("h.ENGINE = ?")
            params.append(normalized_engine)
        normalized_ids = sorted({int(item) for item in (ids or []) if int(item) > 0})
        if normalized_ids:
            placeholders = ",".join("?" for _ in normalized_ids)
            where.append(f"h.ID IN ({placeholders})")
            params.extend(normalized_ids)
        if raw_sql:
            where.append("(h.RAW_SQL = ? OR TRIM(h.RAW_SQL) = ?)")
            params.extend([raw_sql, str(raw_sql).strip()])
        if missing_only:
            where.append("n.QUERY_ID IS NULL")

        effective_limit = len(normalized_ids) if normalized_ids else max(1, int(limit))
        params.append(effective_limit)
        async with self._lock:
            return self._query_all(
                f"""
                SELECT
                    h.ID AS ID,
                    h.SOURCE_ID AS SOURCE_ID,
                    h.ENGINE AS ENGINE,
                    h.QUERY_ID AS QUERY_ID,
                    h.CATALOG_NAME AS CATALOG_NAME,
                    h.SCHEMA_NAME AS SCHEMA_NAME,
                    h.QUERY_STATE AS QUERY_STATE,
                    h.QUERY_TYPE AS QUERY_TYPE,
                    h.USER_EMAIL AS USER_EMAIL,
                    h.ROLE_NAME AS ROLE_NAME,
                    h.CLUSTER_NAME AS CLUSTER_NAME,
                    h.SOURCE AS SOURCE,
                    h.CREATED_AT AS CREATED_AT,
                    h.ENDED_AT AS ENDED_AT,
                    h.RAW_SQL AS RAW_SQL,
                    h.SCHEMA_TABLE AS SCHEMA_TABLE,
                    h.TABLES_JSON AS TABLES_JSON,
                    h.METRICS_JSON AS METRICS_JSON,
                    h.RAW_JSON AS RAW_JSON,
                    h.UPDATED_AT AS UPDATED_AT
                FROM CATALOG_QUERY_HISTORY h
                LEFT JOIN CATALOG_QUERY_HISTORY_NLP n
                  ON n.SOURCE_ID = h.SOURCE_ID
                 AND n.ENGINE = h.ENGINE
                 AND n.QUERY_ID = h.QUERY_ID
                WHERE {" AND ".join(where)}
                ORDER BY COALESCE(h.CREATED_AT, h.UPDATED_AT) DESC
                LIMIT ?
                """,
                tuple(params),
            )

    async def get_backend_query_history_context_by_raw_sql(
        self,
        *,
        raw_sql: str | None = None,
        raw_history_id: int | None = None,
        engine: str | None = None,
    ) -> dict[str, Any]:
        raw_text = str(raw_sql or "")
        text = raw_text.strip()
        normalized_id = int(raw_history_id or 0)
        if not text and normalized_id <= 0:
            return {}

        where: list[str] = []
        params: list[Any] = []
        if normalized_id > 0:
            where.append("ID = ?")
            params.append(normalized_id)
        else:
            where.append("(RAW_SQL = ? OR TRIM(RAW_SQL) = ?)")
            params.extend([raw_text, text])
        normalized_engine = str(engine or "").strip().lower()
        if normalized_engine:
            where.append("ENGINE = ?")
            params.append(normalized_engine)

        async with self._lock:
            row = self._query_one(
                f"""
                SELECT ID, ENGINE, QUERY_ID, CATALOG_NAME, SCHEMA_NAME, QUERY_STATE,
                       QUERY_TYPE, USER_EMAIL, ROLE_NAME, CLUSTER_NAME, SOURCE,
                       CREATED_AT, ENDED_AT, RAW_SQL, SCHEMA_TABLE, TABLES_JSON,
                       METRICS_JSON, RAW_JSON, UPDATED_AT
                FROM BACKEND_QUERY_HISTORY_RAW
                WHERE {" AND ".join(where)}
                ORDER BY COALESCE(CREATED_AT, UPDATED_AT) DESC
                LIMIT 1
                """,
                tuple(params),
            )
            if not row:
                return {}

            table_names = self._coerce_table_list(self._row_value(row, "TABLES_JSON"))
            if not table_names:
                table_names = extract_tables(text)

            tables: list[dict[str, Any]] = []
            row_engine = str(self._row_value(row, "ENGINE") or normalized_engine or "").strip()
            for table_name in table_names:
                ref = self._parse_backend_table_ref(table_name)
                metadata = self._backend_table_metadata(row_engine, ref)
                columns = self._backend_column_metadata(row_engine, ref)
                tables.append(
                    {
                        "name": table_name,
                        "catalog": ref.get("catalog", ""),
                        "schema_name": ref.get("schema_name", ""),
                        "table_name": ref.get("table_name", ""),
                        "description": self._row_value(metadata or {}, "DESCRIPTION", "") or "",
                        "columns": columns,
                    }
                )

        return {
            "history": row,
            "raw_sql": text,
            "tables_json": table_names,
            "tables": tables,
        }

    async def get_catalog_query_history_context_by_raw_sql(
        self,
        *,
        source_id: str | None = None,
        raw_sql: str | None = None,
        raw_history_id: int | None = None,
        engine: str | None = None,
    ) -> dict[str, Any]:
        raw_text = str(raw_sql or "")
        text = raw_text.strip()
        normalized_id = int(raw_history_id or 0)
        if not text and normalized_id <= 0:
            return {}

        where: list[str] = []
        params: list[Any] = []
        normalized_source_id = str(source_id or "").strip()
        if normalized_source_id:
            where.append("SOURCE_ID = ?")
            params.append(normalized_source_id)
        if normalized_id > 0:
            where.append("ID = ?")
            params.append(normalized_id)
        else:
            where.append("(RAW_SQL = ? OR TRIM(RAW_SQL) = ?)")
            params.extend([raw_text, text])
        normalized_engine = str(engine or "").strip().lower()
        if normalized_engine:
            where.append("ENGINE = ?")
            params.append(normalized_engine)

        async with self._lock:
            row = self._query_one(
                f"""
                SELECT ID, SOURCE_ID, ENGINE, QUERY_ID, CATALOG_NAME, SCHEMA_NAME, QUERY_STATE,
                       QUERY_TYPE, USER_EMAIL, ROLE_NAME, CLUSTER_NAME, SOURCE,
                       CREATED_AT, ENDED_AT, RAW_SQL, SCHEMA_TABLE, TABLES_JSON,
                       METRICS_JSON, RAW_JSON, UPDATED_AT
                FROM CATALOG_QUERY_HISTORY
                WHERE {" AND ".join(where)}
                ORDER BY COALESCE(CREATED_AT, UPDATED_AT) DESC
                LIMIT 1
                """,
                tuple(params),
            )
            if not row:
                return {}

            table_names = self._coerce_table_list(self._row_value(row, "TABLES_JSON"))
            if not table_names:
                table_names = extract_tables(text)

            tables: list[dict[str, Any]] = []
            row_engine = str(self._row_value(row, "ENGINE") or normalized_engine or "").strip()
            for table_name in table_names:
                ref = self._parse_backend_table_ref(table_name)
                metadata = self._backend_table_metadata(row_engine, ref)
                columns = self._backend_column_metadata(row_engine, ref)
                tables.append(
                    {
                        "name": table_name,
                        "catalog": ref.get("catalog", ""),
                        "schema_name": ref.get("schema_name", ""),
                        "table_name": ref.get("table_name", ""),
                        "description": self._row_value(metadata or {}, "DESCRIPTION", "") or "",
                        "columns": columns,
                    }
                )

        return {
            "history": row,
            "raw_sql": text,
            "tables_json": table_names,
            "tables": tables,
        }

    async def list_backend_table_context(
        self,
        tables: list[str],
        *,
        engine: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized = self._coerce_table_list(tables)
        if not normalized:
            return []

        normalized_engine = str(engine or "").strip().lower()
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        async with self._lock:
            for table in normalized:
                ref = self._parse_backend_table_ref(table)
                if not ref.get("table_name"):
                    continue

                metadata = self._backend_table_metadata(normalized_engine, ref) or {}
                columns = self._backend_column_metadata(normalized_engine, ref)
                row_engine = str(self._row_value(metadata, "ENGINE") or normalized_engine or "").strip()
                catalog = str(self._row_value(metadata, "CATALOG_NAME") or ref.get("catalog") or "").strip()
                schema_name = str(self._row_value(metadata, "SCHEMA_NAME") or ref.get("schema_name") or "").strip()
                table_name = str(self._row_value(metadata, "TABLE_NAME") or ref.get("table_name") or "").strip()
                if not table_name:
                    continue

                display_name = self._format_backend_table_name(catalog, schema_name, table_name)
                key = ".".join(
                    part.lower()
                    for part in (row_engine, catalog, schema_name, table_name)
                    if str(part or "").strip()
                )
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "name": display_name,
                        "engine": row_engine,
                        "catalog": catalog,
                        "schema_name": schema_name,
                        "table_name": table_name,
                        "table_type": self._row_value(metadata, "TABLE_TYPE", "") or "",
                        "description": self._row_value(metadata, "DESCRIPTION", "") or "",
                        "columns": columns,
                    }
                )
        return results

    async def upsert_backend_query_nlp_history_row(
        self,
        *,
        raw_row: dict[str, Any],
        query_nlp: str,
        embedding: list[float] | None = None,
    ) -> bool:
        normalized_query_nlp = str(query_nlp or "").strip()
        if not normalized_query_nlp:
            return False

        engine = str(self._row_value(raw_row, "ENGINE") or "").strip().lower()
        query_id = str(self._row_value(raw_row, "QUERY_ID") or "").strip()
        raw_sql = str(self._row_value(raw_row, "RAW_SQL") or "").strip()
        if not engine or not query_id or not raw_sql:
            return False

        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            self._execute(
                """
                INSERT INTO BACKEND_QUERY_NLP_HISTORY(
                    ID, ENGINE, QUERY_ID, CATALOG_NAME, SCHEMA_NAME, QUERY_STATE, QUERY_TYPE,
                    USER_EMAIL, ROLE_NAME, CLUSTER_NAME, SOURCE, CREATED_AT, ENDED_AT,
                    RAW_SQL, QUERY_NLP, EMBEDDING, SCHEMA_TABLE, TABLES_JSON, METRICS_JSON, RAW_JSON, UPDATED_AT
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ENGINE, QUERY_ID) DO UPDATE SET
                    CATALOG_NAME=excluded.CATALOG_NAME,
                    SCHEMA_NAME=excluded.SCHEMA_NAME,
                    QUERY_STATE=excluded.QUERY_STATE,
                    QUERY_TYPE=excluded.QUERY_TYPE,
                    USER_EMAIL=excluded.USER_EMAIL,
                    ROLE_NAME=excluded.ROLE_NAME,
                    CLUSTER_NAME=excluded.CLUSTER_NAME,
                    SOURCE=excluded.SOURCE,
                    CREATED_AT=excluded.CREATED_AT,
                    ENDED_AT=excluded.ENDED_AT,
                    RAW_SQL=excluded.RAW_SQL,
                    QUERY_NLP=excluded.QUERY_NLP,
                    EMBEDDING=excluded.EMBEDDING,
                    SCHEMA_TABLE=excluded.SCHEMA_TABLE,
                    TABLES_JSON=excluded.TABLES_JSON,
                    METRICS_JSON=excluded.METRICS_JSON,
                    RAW_JSON=excluded.RAW_JSON,
                    UPDATED_AT=excluded.UPDATED_AT
                """,
                (
                    self._row_value(raw_row, "ID"),
                    engine,
                    query_id,
                    self._row_value(raw_row, "CATALOG_NAME"),
                    self._row_value(raw_row, "SCHEMA_NAME"),
                    self._row_value(raw_row, "QUERY_STATE"),
                    self._row_value(raw_row, "QUERY_TYPE"),
                    self._row_value(raw_row, "USER_EMAIL"),
                    self._row_value(raw_row, "ROLE_NAME"),
                    self._row_value(raw_row, "CLUSTER_NAME"),
                    self._row_value(raw_row, "SOURCE"),
                    self._row_value(raw_row, "CREATED_AT"),
                    self._row_value(raw_row, "ENDED_AT"),
                    raw_sql,
                    normalized_query_nlp,
                    self._serialize_embeddings(embedding),
                    self._row_value(raw_row, "SCHEMA_TABLE"),
                    self._row_value(raw_row, "TABLES_JSON"),
                    self._row_value(raw_row, "METRICS_JSON") or "{}",
                    self._row_value(raw_row, "RAW_JSON") or "{}",
                    now,
                ),
            )
            self._db.commit()
        return True

    async def upsert_catalog_query_history_nlp_row(
        self,
        *,
        raw_row: dict[str, Any],
        nlp_text: str,
        embedding: list[float] | None = None,
    ) -> bool:
        normalized_nlp_text = str(nlp_text or "").strip()
        if not normalized_nlp_text:
            return False

        source_id = str(self._row_value(raw_row, "SOURCE_ID") or "").strip()
        engine = str(self._row_value(raw_row, "ENGINE") or "").strip().lower()
        query_id = str(self._row_value(raw_row, "QUERY_ID") or "").strip()
        raw_sql = str(self._row_value(raw_row, "RAW_SQL") or "").strip()
        if not source_id or not engine or not query_id or not raw_sql:
            return False

        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            self._execute(
                """
                INSERT INTO CATALOG_QUERY_HISTORY_NLP(
                    RAW_QUERY_HISTORY_ID, SOURCE_ID, ENGINE, QUERY_ID, CATALOG_NAME, SCHEMA_NAME,
                    QUERY_STATE, QUERY_TYPE, USER_EMAIL, ROLE_NAME, CLUSTER_NAME, SOURCE,
                    CREATED_AT, ENDED_AT, RAW_SQL, NLP_TEXT, EMBEDDINGS, SCHEMA_TABLE,
                    TABLES_JSON, METRICS_JSON, RAW_JSON, UPDATED_AT
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(SOURCE_ID, ENGINE, QUERY_ID) DO UPDATE SET
                    RAW_QUERY_HISTORY_ID=excluded.RAW_QUERY_HISTORY_ID,
                    CATALOG_NAME=excluded.CATALOG_NAME,
                    SCHEMA_NAME=excluded.SCHEMA_NAME,
                    QUERY_STATE=excluded.QUERY_STATE,
                    QUERY_TYPE=excluded.QUERY_TYPE,
                    USER_EMAIL=excluded.USER_EMAIL,
                    ROLE_NAME=excluded.ROLE_NAME,
                    CLUSTER_NAME=excluded.CLUSTER_NAME,
                    SOURCE=excluded.SOURCE,
                    CREATED_AT=excluded.CREATED_AT,
                    ENDED_AT=excluded.ENDED_AT,
                    RAW_SQL=excluded.RAW_SQL,
                    NLP_TEXT=excluded.NLP_TEXT,
                    EMBEDDINGS=excluded.EMBEDDINGS,
                    SCHEMA_TABLE=excluded.SCHEMA_TABLE,
                    TABLES_JSON=excluded.TABLES_JSON,
                    METRICS_JSON=excluded.METRICS_JSON,
                    RAW_JSON=excluded.RAW_JSON,
                    UPDATED_AT=excluded.UPDATED_AT
                """,
                (
                    self._row_value(raw_row, "ID"),
                    source_id,
                    engine,
                    query_id,
                    self._row_value(raw_row, "CATALOG_NAME"),
                    self._row_value(raw_row, "SCHEMA_NAME"),
                    self._row_value(raw_row, "QUERY_STATE"),
                    self._row_value(raw_row, "QUERY_TYPE"),
                    self._row_value(raw_row, "USER_EMAIL"),
                    self._row_value(raw_row, "ROLE_NAME"),
                    self._row_value(raw_row, "CLUSTER_NAME"),
                    self._row_value(raw_row, "SOURCE"),
                    self._row_value(raw_row, "CREATED_AT"),
                    self._row_value(raw_row, "ENDED_AT"),
                    raw_sql,
                    normalized_nlp_text,
                    self._serialize_embeddings(embedding),
                    self._row_value(raw_row, "SCHEMA_TABLE"),
                    self._row_value(raw_row, "TABLES_JSON"),
                    self._row_value(raw_row, "METRICS_JSON") or "{}",
                    self._row_value(raw_row, "RAW_JSON") or "{}",
                    now,
                ),
            )
            self._db.commit()
        return True

    async def list_backend_query_nlp_history_by_embedding(
        self,
        embedding: list[float],
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if not embedding or self._backend != "postgres":
            return []

        effective_limit = max(1, int(limit))
        select_columns = self._backend_query_nlp_history_select_columns()
        where, params = self._backend_query_nlp_history_scope_where(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
        )
        where.append("EMBEDDING IS NOT NULL")
        where_sql = f"WHERE {' AND '.join(where)}"
        vector_text = self._vector_literal(embedding)
        vector_op = f"OPERATOR({self._pgvector_schema}.<=>)"
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT
                    {", ".join(select_columns)},
                    (1 - (EMBEDDING {vector_op} CAST(? AS {self._pgvector_schema}.vector))) AS COSINE_SIMILARITY
                FROM BACKEND_QUERY_NLP_HISTORY
                {where_sql}
                ORDER BY EMBEDDING {vector_op} CAST(? AS {self._pgvector_schema}.vector)
                LIMIT ?
                """,
                (vector_text, *params, vector_text, effective_limit),
            )
        return rows

    async def list_catalog_query_history_nlp_by_embedding(
        self,
        embedding: list[float],
        *,
        source_id: str | None = None,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if not embedding or self._backend != "postgres":
            return []

        effective_limit = max(1, int(limit))
        select_columns = self._catalog_query_history_nlp_select_columns()
        where, params = self._catalog_query_history_nlp_scope_where(
            source_id=source_id,
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
        )
        where.append("EMBEDDINGS IS NOT NULL")
        where_sql = f"WHERE {' AND '.join(where)}"
        vector_text = self._vector_literal(embedding)
        vector_op = f"OPERATOR({self._pgvector_schema}.<=>)"
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT
                    {", ".join(select_columns)},
                    (1 - (EMBEDDINGS {vector_op} CAST(? AS {self._pgvector_schema}.vector))) AS COSINE_SIMILARITY
                FROM CATALOG_QUERY_HISTORY_NLP
                {where_sql}
                ORDER BY EMBEDDINGS {vector_op} CAST(? AS {self._pgvector_schema}.vector)
                LIMIT ?
                """,
                (vector_text, *params, vector_text, effective_limit),
            )
        return rows

    async def list_backend_query_nlp_history_by_full_text(
        self,
        prompt: str,
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        normalized = str(prompt or "").strip()
        if not normalized or self._backend != "postgres":
            return []

        effective_limit = max(1, int(limit))
        select_columns = self._backend_query_nlp_history_select_columns()
        where, params = self._backend_query_nlp_history_scope_where(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
        )
        document = self._backend_query_nlp_history_fts_document()
        tsquery_function, fts_query = await self._prepare_postgres_fts_query(normalized)
        where.append(self._postgres_fts_match(document, tsquery_function))
        where_sql = f"WHERE {' AND '.join(where)}"
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT
                    {", ".join(select_columns)},
                    {self._postgres_fts_rank(document, tsquery_function)} AS FTS_SCORE
                FROM BACKEND_QUERY_NLP_HISTORY
                {where_sql}
                ORDER BY FTS_SCORE DESC, COALESCE(CREATED_AT, UPDATED_AT) DESC
                LIMIT ?
                """,
                (fts_query, *params, fts_query, effective_limit),
            )
        return rows

    async def list_catalog_query_history_nlp_by_full_text(
        self,
        prompt: str,
        *,
        source_id: str | None = None,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        normalized = str(prompt or "").strip()
        if not normalized:
            return []
        if self._backend != "postgres":
            return await self._list_catalog_query_history_nlp_by_like(
                normalized,
                source_id=source_id,
                engine=engine,
                catalog=catalog,
                schema_name=schema_name,
                limit=limit,
            )

        effective_limit = max(1, int(limit))
        select_columns = self._catalog_query_history_nlp_select_columns()
        where, params = self._catalog_query_history_nlp_scope_where(
            source_id=source_id,
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
        )
        document = self._catalog_query_history_nlp_fts_document()
        tsquery_function, fts_query = await self._prepare_postgres_fts_query(normalized)
        where.append(self._postgres_fts_match(document, tsquery_function))
        where_sql = f"WHERE {' AND '.join(where)}"
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT
                    {", ".join(select_columns)},
                    {self._postgres_fts_rank(document, tsquery_function)} AS FTS_SCORE
                FROM CATALOG_QUERY_HISTORY_NLP
                {where_sql}
                ORDER BY FTS_SCORE DESC, COALESCE(CREATED_AT, UPDATED_AT) DESC
                LIMIT ?
                """,
                (fts_query, *params, fts_query, effective_limit),
            )
        return rows

    async def _list_catalog_query_history_nlp_by_like(
        self,
        prompt: str,
        *,
        source_id: str | None = None,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        terms = [term for term in re.split(r"\s+", prompt.strip()) if term]
        if not terms:
            return []
        where, params = self._catalog_query_history_nlp_scope_where(
            source_id=source_id,
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
        )
        match_clauses: list[str] = []
        for term in terms:
            pattern = f"%{term}%"
            match_clauses.extend(
                [
                    "NLP_TEXT LIKE ?",
                    "RAW_SQL LIKE ?",
                    "SCHEMA_TABLE LIKE ?",
                    "TABLES_JSON LIKE ?",
                ]
            )
            params.extend([pattern, pattern, pattern, pattern])
        where.append(f"({' OR '.join(match_clauses)})")
        params.append(max(1, int(limit)))
        async with self._lock:
            return self._query_all(
                f"""
                SELECT {", ".join(self._catalog_query_history_nlp_select_columns())}
                FROM CATALOG_QUERY_HISTORY_NLP
                WHERE {" AND ".join(where)}
                ORDER BY COALESCE(CREATED_AT, UPDATED_AT) DESC
                LIMIT ?
                """,
                tuple(params),
            )

    async def list_backend_query_history_for_nlp(
        self,
        *,
        limit: int = 500,
        engine: str | None = None,
    ) -> list[dict[str, Any]]:
        where = [
            "h.RAW_SQL IS NOT NULL",
            "n.QUERY_ID IS NULL",
        ]
        params: list[Any] = []
        normalized_engine = str(engine or "").strip().lower()
        if normalized_engine:
            where.append("h.ENGINE = ?")
            params.append(normalized_engine)

        params.append(max(1, int(limit)))
        async with self._lock:
            return self._query_all(
                f"""
                SELECT
                    h.ID AS ID,
                    h.ENGINE AS ENGINE,
                    h.QUERY_ID AS QUERY_ID,
                    h.USER_EMAIL AS EMAIL,
                    h.RAW_SQL AS QUERY,
                    h.SCHEMA_TABLE AS SCHEMA_TABLE,
                    h.TABLES_JSON AS ALL_QUERY_TABLES,
                    h.QUERY_STATE AS QUERY_STATE,
                    h.QUERY_TYPE AS QUERY_TYPE
                FROM BACKEND_QUERY_HISTORY_RAW h
                LEFT JOIN BACKEND_QUERY_NLP n
                  ON n.ENGINE = h.ENGINE
                 AND n.QUERY_ID = h.QUERY_ID
                WHERE {" AND ".join(where)}
                ORDER BY COALESCE(h.CREATED_AT, h.UPDATED_AT) DESC
                LIMIT ?
                """,
                tuple(params),
            )

    async def upsert_backend_query_nlp_rows(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        async with self._lock:
            for row in rows:
                engine = str(row.get("engine") or "").strip().lower()
                query_id = str(row.get("query_id") or "").strip()
                query_text = str(row.get("query") or "").strip()
                query_in_nlp = str(row.get("query_in_nlp") or "").strip()
                if not engine or not query_id or not query_text or not query_in_nlp:
                    continue

                all_query_tables = self._coerce_table_list(row.get("all_query_tables"))
                if not all_query_tables:
                    all_query_tables = extract_tables(query_text)
                schema_table = str(row.get("schema_table") or "").strip()
                if not schema_table and all_query_tables:
                    schema_table = all_query_tables[0]

                self._execute(
                    """
                    INSERT INTO BACKEND_QUERY_NLP(
                        ENGINE, QUERY_ID, RAW_QUERY_HISTORY_ID, EMAIL, SOEID,
                        BUSINESS_TITLE, QUERY, QUERY_IN_NLP, SCHEMA_TABLE,
                        ALL_QUERY_TABLES, EMBEDDINGS, UPDATED_AT
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ENGINE, QUERY_ID) DO UPDATE SET
                        RAW_QUERY_HISTORY_ID=excluded.RAW_QUERY_HISTORY_ID,
                        EMAIL=excluded.EMAIL,
                        SOEID=excluded.SOEID,
                        BUSINESS_TITLE=excluded.BUSINESS_TITLE,
                        QUERY=excluded.QUERY,
                        QUERY_IN_NLP=excluded.QUERY_IN_NLP,
                        SCHEMA_TABLE=excluded.SCHEMA_TABLE,
                        ALL_QUERY_TABLES=excluded.ALL_QUERY_TABLES,
                        EMBEDDINGS=excluded.EMBEDDINGS,
                        UPDATED_AT=excluded.UPDATED_AT
                    """,
                    (
                        engine,
                        query_id,
                        row.get("raw_query_history_id"),
                        row.get("email"),
                        row.get("soeid"),
                        row.get("business_title"),
                        query_text,
                        query_in_nlp,
                        schema_table,
                        json.dumps(all_query_tables, default=str) if all_query_tables else None,
                        self._serialize_embeddings(row.get("embeddings")),
                        now,
                    ),
                )
                inserted += 1
            self._db.commit()
        return inserted

    async def list_discovery_master(self, limit: int = 5000) -> list[dict[str, Any]]:
        async with self._lock:
            rows = self._query_all(
                """
                SELECT ZONE, TABLE_NAME, TABLE_DESCRIPTION, DOMAIN, STANDARDIZED_DOMAIN, SOURCE_SYSTEM,
                       REGION, COUNTRY, TARGET_DB, PII, CRITICAL_DATA_ELEMENT, ASSET_ID, UPDATED_AT
                FROM DISCOVERY_MASTER_TABLE
                ORDER BY UPDATED_AT DESC
                LIMIT ?
                """,
                (limit,),
            )
        return rows

    async def list_discovery_master_by_tables(self, tables: list[str]) -> list[dict[str, Any]]:
        normalized = [table.strip().upper() for table in tables if table.strip()]
        if not normalized:
            return []
        placeholders = ",".join("?" for _ in normalized)
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT ZONE, TABLE_NAME, TABLE_DESCRIPTION, DOMAIN, STANDARDIZED_DOMAIN, SOURCE_SYSTEM,
                       REGION, COUNTRY, TARGET_DB, PII, CRITICAL_DATA_ELEMENT, ASSET_ID
                FROM DISCOVERY_MASTER_TABLE
                WHERE UPPER(TABLE_NAME) IN ({placeholders})
                """,
                tuple(normalized),
            )
        return rows

    async def list_column_details_by_tables(self, tables: list[str]) -> dict[str, list[dict[str, Any]]]:
        normalized = [table.strip().upper() for table in tables if table.strip()]
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT DOMAIN, TARGET_TABLE_NAME, TARGET_COLUMN_NAME, TARGET_COLUMN_NAME_DESC,
                       CRITICAL_DATA_ELEMENT, PII, PRIMARY_FOREIGN_KEY, TARGET_DB
                FROM COLUMN_DETAILS
                WHERE UPPER(TARGET_TABLE_NAME) IN ({placeholders})
                ORDER BY TARGET_TABLE_NAME, TARGET_COLUMN_NAME
                """,
                tuple(normalized),
            )

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            table_name = str(self._row_value(row, "TARGET_TABLE_NAME") or "")
            if not table_name:
                continue
            grouped.setdefault(table_name, []).append(
                {
                    "name": self._row_value(row, "TARGET_COLUMN_NAME"),
                    "type": "string",
                    "description": self._row_value(row, "TARGET_COLUMN_NAME_DESC"),
                    "domain": self._row_value(row, "DOMAIN"),
                    "criticalDataElement": self._row_value(row, "CRITICAL_DATA_ELEMENT"),
                    "pii": self._row_value(row, "PII"),
                    "primary_foreign_key": self._row_value(row, "PRIMARY_FOREIGN_KEY"),
                    "target_db": self._row_value(row, "TARGET_DB"),
                }
            )
        return grouped

    async def list_backend_metadata_tables_by_tables(self, tables: list[str]) -> list[dict[str, Any]]:
        normalized = [table.strip().upper() for table in tables if table.strip()]
        if not normalized:
            return []
        placeholders = ",".join("?" for _ in normalized)
        async with self._lock:
            return self._query_all(
                f"""
                SELECT ENGINE, CATALOG_NAME, SCHEMA_NAME, TABLE_NAME, TABLE_TYPE, DESCRIPTION, UPDATED_AT
                FROM BACKEND_METADATA_TABLES
                WHERE UPPER(TABLE_NAME) IN ({placeholders})
                ORDER BY UPDATED_AT DESC
                """,
                tuple(normalized),
            )

    async def list_backend_metadata_tables_by_embedding(
        self,
        embedding: list[float],
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if not embedding:
            return []

        effective_limit = max(1, int(limit))
        select_columns = self._backend_metadata_table_select_columns()
        where, params = self._backend_metadata_table_scope_where(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
        )

        if self._backend != "postgres":
            where_sql = f"WHERE {' AND '.join(where)}" if where else ""
            async with self._lock:
                rows = self._query_all(
                    f"""
                    SELECT {", ".join(select_columns)}
                    FROM BACKEND_METADATA_TABLES
                    {where_sql}
                    ORDER BY UPDATED_AT DESC
                    LIMIT ?
                    """,
                    tuple(params) + (effective_limit,),
                )
            for row in rows:
                row["COSINE_SIMILARITY"] = 0.0
            return rows

        where.append("EMBEDDING IS NOT NULL")
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        vector_text = self._vector_literal(embedding)
        vector_op = f"OPERATOR({self._pgvector_schema}.<=>)"
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT
                    {", ".join(select_columns)},
                    (1 - (EMBEDDING {vector_op} CAST(? AS {self._pgvector_schema}.vector))) AS COSINE_SIMILARITY
                FROM BACKEND_METADATA_TABLES
                {where_sql}
                ORDER BY EMBEDDING {vector_op} CAST(? AS {self._pgvector_schema}.vector)
                LIMIT ?
                """,
                (vector_text, *params, vector_text, effective_limit),
            )
        return rows

    async def list_backend_metadata_tables_by_full_text(
        self,
        prompt: str,
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        normalized = str(prompt or "").strip()
        logger.debug(f"Normalized prompt: {normalized}")
        if not normalized or self._backend != "postgres":
            return []

        effective_limit = max(1, int(limit))
        select_columns = self._backend_metadata_table_select_columns()
        logger.debug(f"Select columns: {select_columns}")
        where, params = self._backend_metadata_table_scope_where(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
        )
        logger.debug(f"Where: {where}, params: {params}")

        document = self._backend_metadata_table_fts_document()
        logger.debug(f"Document: {document}")
        tsquery_function, fts_query = await self._prepare_postgres_fts_query(normalized)

        where.append(self._postgres_fts_match(document, tsquery_function))
        where_sql = f"WHERE {' AND '.join(where)}"
        logger.debug(f"Where_sql: {where_sql}")
        query =  f"""
                SELECT
                    {", ".join(select_columns)},
                    {self._postgres_fts_rank(document, tsquery_function)} AS FTS_SCORE
                FROM BACKEND_METADATA_TABLES
                {where_sql}
                ORDER BY FTS_SCORE DESC, UPDATED_AT DESC
                LIMIT ?
                """
        logger.debug(f"Query: {query}")
        logger.debug(f"Params: fts_query: {fts_query}, params: {params}, effective_limit: {effective_limit}")
        async with self._lock:
            rows = self._query_all(
                query,
                (fts_query, *params, fts_query, effective_limit),
            )
        return rows

    async def list_backend_metadata_columns_by_embedding(
        self,
        embedding: list[float],
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if not embedding:
            return []

        effective_limit = max(1, int(limit))
        select_columns = self._backend_metadata_column_select_columns()
        where, params = self._backend_metadata_column_scope_where(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
            table_name=table_name,
        )

        if self._backend != "postgres":
            where_sql = f"WHERE {' AND '.join(where)}" if where else ""
            async with self._lock:
                rows = self._query_all(
                    f"""
                    SELECT {", ".join(select_columns)}
                    FROM BACKEND_METADATA_COLUMNS
                    {where_sql}
                    ORDER BY UPDATED_AT DESC
                    LIMIT ?
                    """,
                    tuple(params) + (effective_limit,),
                )
            for row in rows:
                row["COSINE_SIMILARITY"] = 0.0
            return rows

        where.append("EMBEDDING IS NOT NULL")
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        vector_text = self._vector_literal(embedding)
        vector_op = f"OPERATOR({self._pgvector_schema}.<=>)"
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT
                    {", ".join(select_columns)},
                    (1 - (EMBEDDING {vector_op} CAST(? AS {self._pgvector_schema}.vector))) AS COSINE_SIMILARITY
                FROM BACKEND_METADATA_COLUMNS
                {where_sql}
                ORDER BY EMBEDDING {vector_op} CAST(? AS {self._pgvector_schema}.vector)
                LIMIT ?
                """,
                (vector_text, *params, vector_text, effective_limit),
            )
        return rows

    async def list_backend_metadata_columns_by_full_text(
        self,
        prompt: str,
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        normalized = str(prompt or "").strip()
        if not normalized or self._backend != "postgres":
            return []

        effective_limit = max(1, int(limit))
        select_columns = self._backend_metadata_column_select_columns()
        where, params = self._backend_metadata_column_scope_where(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
            table_name=table_name,
        )
        document = self._backend_metadata_column_fts_document()
        logger.debug(f"Selected document: {document}")
        tsquery_function, fts_query = await self._prepare_postgres_fts_query(normalized)
        where.append(self._postgres_fts_match(document, tsquery_function))
        where_sql = f"WHERE {' AND '.join(where)}"
        logger.debug(f"Column full text search Where: {where_sql}")
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT
                    {", ".join(select_columns)},
                    {self._postgres_fts_rank(document, tsquery_function)} AS FTS_SCORE
                FROM BACKEND_METADATA_COLUMNS
                {where_sql}
                ORDER BY FTS_SCORE DESC, UPDATED_AT DESC
                LIMIT ?
                """,
                (fts_query, *params, fts_query, effective_limit),
            )
        return rows

    async def list_backend_metadata_tables_for_description(
        self,
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        missing_only: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []

        if engine:
            where.append("UPPER(ENGINE) = UPPER(?)")
            params.append(engine)
        if catalog:
            where.append("UPPER(CATALOG_NAME) = UPPER(?)")
            params.append(catalog)
        if schema_name:
            where.append("UPPER(SCHEMA_NAME) = UPPER(?)")
            params.append(schema_name)
        if table_name:
            where.append("UPPER(TABLE_NAME) = UPPER(?)")
            params.append(table_name)
        if missing_only:
            where.append("(DESCRIPTION IS NULL OR TRIM(DESCRIPTION) = '' OR EMBEDDING IS NULL)")

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(max(1, int(limit)))
        async with self._lock:
            query = f"""
                SELECT ENGINE, CATALOG_ID, CATALOG_NAME, SCHEMA_ID, SCHEMA_NAME,
                       TABLE_ID, TABLE_NAME, TABLE_TYPE, DESCRIPTION
                FROM BACKEND_METADATA_TABLES
                {where_sql}
                ORDER BY CATALOG_NAME, SCHEMA_NAME, TABLE_NAME
                LIMIT ?
                """
            logger.debug(f"Query: {query} with params: {params}")
            return self._query_all(query, tuple(params),)

    async def update_backend_metadata_table_description(
        self,
        *,
        engine: str,
        catalog_id: str,
        schema_id: str,
        table_id: str,
        description: str,
        embedding: list[float] | None = None,
    ) -> bool:
        normalized_description = str(description or "").strip()
        if not normalized_description:
            return False

        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            cursor = self._db.cursor()
            try:
                cursor.execute(
                    self._sql(
                        """
                        UPDATE BACKEND_METADATA_TABLES
                        SET DESCRIPTION = ?, EMBEDDING = ?, UPDATED_AT = ?
                        WHERE ENGINE = ?
                          AND CATALOG_ID = ?
                          AND SCHEMA_ID = ?
                          AND TABLE_ID = ?
                        """
                    ),
                    (
                        normalized_description,
                        self._serialize_embeddings(embedding),
                        now,
                        engine,
                        catalog_id,
                        schema_id,
                        table_id,
                    ),
                )
                changed = int(cursor.rowcount or 0)
                self._db.commit()
            finally:
                cursor.close()
        return changed > 0

    async def list_backend_metadata_columns_for_description(
        self,
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        column_name: str | None = None,
        missing_only: bool = True,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []

        if engine:
            where.append("UPPER(ENGINE) = UPPER(?)")
            params.append(engine)
        if catalog:
            where.append("UPPER(CATALOG_NAME) = UPPER(?)")
            params.append(catalog)
        if schema_name:
            where.append("UPPER(SCHEMA_NAME) = UPPER(?)")
            params.append(schema_name)
        if table_name:
            where.append("UPPER(TABLE_NAME) = UPPER(?)")
            params.append(table_name)
        if column_name:
            where.append("UPPER(COLUMN_NAME) = UPPER(?)")
            params.append(column_name)
        if missing_only:
            where.append("(DESCRIPTION IS NULL OR TRIM(DESCRIPTION) = '' OR EMBEDDING IS NULL)")

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(max(1, int(limit)))
        async with self._lock:
            return self._query_all(
                f"""
                SELECT ENGINE, CATALOG_ID, CATALOG_NAME, SCHEMA_ID, SCHEMA_NAME,
                       TABLE_ID, TABLE_NAME, COLUMN_ID, COLUMN_NAME, ORDINAL_POSITION,
                       DATA_TYPE, NULLABLE, DESCRIPTION
                FROM BACKEND_METADATA_COLUMNS
                {where_sql}
                ORDER BY CATALOG_NAME, SCHEMA_NAME, TABLE_NAME, ORDINAL_POSITION, COLUMN_NAME
                LIMIT ?
                """,
                tuple(params),
            )

    async def update_backend_metadata_column_description(
        self,
        *,
        engine: str,
        catalog_id: str,
        schema_id: str,
        table_id: str,
        column_id: str,
        description: str,
        embedding: list[float] | None = None,
    ) -> bool:
        normalized_description = str(description or "").strip()
        if not normalized_description:
            return False

        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            cursor = self._db.cursor()
            try:
                cursor.execute(
                    self._sql(
                        """
                        UPDATE BACKEND_METADATA_COLUMNS
                        SET DESCRIPTION = ?, EMBEDDING = ?, UPDATED_AT = ?
                        WHERE ENGINE = ?
                          AND CATALOG_ID = ?
                          AND SCHEMA_ID = ?
                          AND TABLE_ID = ?
                          AND COLUMN_ID = ?
                        """
                    ),
                    (
                        normalized_description,
                        self._serialize_embeddings(embedding),
                        now,
                        engine,
                        catalog_id,
                        schema_id,
                        table_id,
                        column_id,
                    ),
                )
                changed = int(cursor.rowcount or 0)
                self._db.commit()
            finally:
                cursor.close()
        return changed > 0

    async def list_backend_column_details_by_tables(self, tables: list[str]) -> dict[str, list[dict[str, Any]]]:
        normalized = [table.strip().upper() for table in tables if table.strip()]
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT ENGINE, CATALOG_NAME, SCHEMA_NAME, TABLE_NAME, COLUMN_NAME,
                       DATA_TYPE, DESCRIPTION, ORDINAL_POSITION, NULLABLE
                FROM BACKEND_METADATA_COLUMNS
                WHERE UPPER(TABLE_NAME) IN ({placeholders})
                ORDER BY TABLE_NAME, ORDINAL_POSITION, COLUMN_NAME
                """,
                tuple(normalized),
            )

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            table_name = str(self._row_value(row, "TABLE_NAME") or "")
            if not table_name:
                continue
            grouped.setdefault(table_name, []).append(
                {
                    "name": self._row_value(row, "COLUMN_NAME"),
                    "type": self._row_value(row, "DATA_TYPE") or "string",
                    "description": self._row_value(row, "DESCRIPTION"),
                    "domain": self._row_value(row, "CATALOG_NAME"),
                    "target_db": self._row_value(row, "SCHEMA_NAME"),
                    "engine": self._row_value(row, "ENGINE"),
                    "nullable": self._row_value(row, "NULLABLE"),
                }
            )
        return grouped

    async def list_common_queries_for_users(self, soeids: list[str], limit: int = 200) -> list[dict[str, Any]]:
        normalized = [soeid.strip() for soeid in soeids if soeid.strip()]
        if not normalized:
            return []
        placeholders = ",".join("?" for _ in normalized)
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT ID, SOEID, EMAIL, NAME, QUERY, TOOL, SCHEMA_TABLE, ALL_QUERY_TABLES, SUCCESS_PERCENTAGE, UPDATED_AT
                FROM DATA_USAGE_COMMON_QUERIES
                WHERE SOEID IN ({placeholders})
                ORDER BY UPDATED_AT DESC
                LIMIT ?
                """,
                tuple(normalized) + (limit,),
            )
        return rows

    async def list_nlp_queries_by_segments(self, segment_values: list[str], limit: int = 200) -> list[dict[str, Any]]:
        normalized = []
        seen: set[str] = set()
        for value in segment_values:
            text = value.strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)

        if not normalized:
            return []

        segment_columns = [f"MANAGED_SEGMENT_L{idx}" for idx in range(1, 13)]
        where_clauses: list[str] = []
        params: list[Any] = []
        for segment in normalized:
            clause = "(" + " OR ".join(f"UPPER(COALESCE({col}, '')) = UPPER(?)" for col in segment_columns) + ")"
            where_clauses.append(clause)
            params.extend([segment] * len(segment_columns))

        where_sql = " AND ".join(where_clauses)
        select_columns = [
            "ID",
            "EMAIL",
            "SOEID",
            "BUSINESS_TITLE",
            "QUERY",
            "QUERY_IN_NLP",
            "SCHEMA_TABLE",
            "ALL_QUERY_TABLES",
            "EMBEDDINGS",
            "UPDATED_AT",
            *segment_columns,
        ]
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT {", ".join(select_columns)}
                FROM DATA_USAGE_NLP_QUERIES
                WHERE {where_sql}
                ORDER BY UPDATED_AT DESC
                LIMIT ?
                """,
                tuple(params) + (limit,),
            )
        return rows

    async def list_nlp_queries_by_embedding(self, embedding: list[float], limit: int = 200) -> list[dict[str, Any]]:
        if not embedding:
            return []

        effective_limit = max(1, int(limit))
        segment_columns = [f"MANAGED_SEGMENT_L{idx}" for idx in range(1, 13)]
        select_columns = [
            "ID",
            "EMAIL",
            "SOEID",
            "BUSINESS_TITLE",
            "QUERY",
            "QUERY_IN_NLP",
            "SCHEMA_TABLE",
            "ALL_QUERY_TABLES",
            "EMBEDDINGS",
            "UPDATED_AT",
            *segment_columns,
        ]

        if self._backend != "postgres":
            async with self._lock:
                rows = self._query_all(
                    f"""
                    SELECT {", ".join(select_columns)}
                    FROM DATA_USAGE_NLP_QUERIES
                    WHERE QUERY_IN_NLP IS NOT NULL
                    ORDER BY UPDATED_AT DESC
                    LIMIT ?
                    """,
                    (effective_limit,),
                )
            for row in rows:
                row["COSINE_SIMILARITY"] = 0.0
            return rows

        vector_text = self._vector_literal(embedding)
        vector_op = f"OPERATOR({self._pgvector_schema}.<=>)"
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT
                    {", ".join(select_columns)},
                    (1 - (EMBEDDINGS {vector_op} CAST(? AS {self._pgvector_schema}.vector))) AS COSINE_SIMILARITY
                FROM DATA_USAGE_NLP_QUERIES
                WHERE EMBEDDINGS IS NOT NULL
                ORDER BY EMBEDDINGS {vector_op} CAST(? AS {self._pgvector_schema}.vector)
                LIMIT ?
                """,
                (vector_text, vector_text, effective_limit),
            )
        return rows

    async def list_nlp_queries_by_full_text(self, prompt: str, limit: int = 200) -> list[dict[str, Any]]:
        normalized = str(prompt or "").strip()
        if not normalized or self._backend != "postgres":
            return []

        effective_limit = max(1, int(limit))
        segment_columns = [f"MANAGED_SEGMENT_L{idx}" for idx in range(1, 13)]
        select_columns = [
            "ID",
            "EMAIL",
            "SOEID",
            "BUSINESS_TITLE",
            "QUERY",
            "QUERY_IN_NLP",
            "SCHEMA_TABLE",
            "ALL_QUERY_TABLES",
            "EMBEDDINGS",
            "UPDATED_AT",
            *segment_columns,
        ]
        tsquery_function, fts_query = await self._prepare_postgres_fts_query(normalized)
        document = "COALESCE(QUERY_IN_NLP, '')"
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT
                    {", ".join(select_columns)},
                    {self._postgres_fts_rank(document, tsquery_function)} AS FTS_SCORE
                FROM DATA_USAGE_NLP_QUERIES
                WHERE QUERY_IN_NLP IS NOT NULL
                  AND {self._postgres_fts_match(document, tsquery_function)}
                ORDER BY FTS_SCORE DESC, UPDATED_AT DESC
                LIMIT ?
                """,
                (fts_query, fts_query, effective_limit),
            )
        return rows

    async def list_nlp_queries_by_business_title(self, business_title: str, limit: int = 200) -> list[dict[str, Any]]:
        normalized = str(business_title or "").strip()
        if not normalized:
            return []

        segment_columns = [f"MANAGED_SEGMENT_L{idx}" for idx in range(1, 13)]
        select_columns = [
            "ID",
            "EMAIL",
            "SOEID",
            "BUSINESS_TITLE",
            "QUERY",
            "QUERY_IN_NLP",
            "SCHEMA_TABLE",
            "ALL_QUERY_TABLES",
            "EMBEDDINGS",
            "UPDATED_AT",
            *segment_columns,
        ]
        async with self._lock:
            rows = self._query_all(
                f"""
                SELECT {", ".join(select_columns)}
                FROM DATA_USAGE_NLP_QUERIES
                WHERE UPPER(COALESCE(BUSINESS_TITLE, '')) = UPPER(?)
                ORDER BY UPDATED_AT DESC
                LIMIT ?
                """,
                (normalized, max(1, int(limit))),
            )
        for row in rows:
            row["COSINE_SIMILARITY"] = float(self._row_value(row, "COSINE_SIMILARITY", 0.0) or 0.0)
        return rows

    async def query_rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        async with self._lock:
            return self._query_all(sql, params)

    async def upsert_data_usage_nlp_rows(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        async with self._lock:
            for row in rows:
                email = str(row.get("email") or "").strip()
                query_text = str(row.get("query") or "").strip()
                schema_table = str(row.get("schema_table") or "").strip()
                all_query_tables = self._coerce_table_list(row.get("all_query_tables"))
                query_in_nlp = str(row.get("query_in_nlp") or "").strip()
                if not email or not query_text or not query_in_nlp:
                    continue
                if not all_query_tables:
                    all_query_tables = extract_tables(query_text)
                if not schema_table and all_query_tables:
                    schema_table = all_query_tables[0]

                self._execute(
                    """
                    INSERT INTO DATA_USAGE_NLP_QUERIES(
                        EMAIL, SOEID, MANAGED_SEGMENT_L1, MANAGED_SEGMENT_L2, MANAGED_SEGMENT_L3, MANAGED_SEGMENT_L4,
                        MANAGED_SEGMENT_L5, MANAGED_SEGMENT_L6, MANAGED_SEGMENT_L7, MANAGED_SEGMENT_L8, MANAGED_SEGMENT_L9,
                        MANAGED_SEGMENT_L10, MANAGED_SEGMENT_L11, MANAGED_SEGMENT_L12, BUSINESS_TITLE,
                        QUERY, QUERY_IN_NLP, SCHEMA_TABLE, ALL_QUERY_TABLES,
                        EMBEDDINGS, UPDATED_AT
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(EMAIL, QUERY, SCHEMA_TABLE) DO UPDATE SET
                        SOEID=excluded.SOEID,
                        MANAGED_SEGMENT_L1=excluded.MANAGED_SEGMENT_L1,
                        MANAGED_SEGMENT_L2=excluded.MANAGED_SEGMENT_L2,
                        MANAGED_SEGMENT_L3=excluded.MANAGED_SEGMENT_L3,
                        MANAGED_SEGMENT_L4=excluded.MANAGED_SEGMENT_L4,
                        MANAGED_SEGMENT_L5=excluded.MANAGED_SEGMENT_L5,
                        MANAGED_SEGMENT_L6=excluded.MANAGED_SEGMENT_L6,
                        MANAGED_SEGMENT_L7=excluded.MANAGED_SEGMENT_L7,
                        MANAGED_SEGMENT_L8=excluded.MANAGED_SEGMENT_L8,
                        MANAGED_SEGMENT_L9=excluded.MANAGED_SEGMENT_L9,
                        MANAGED_SEGMENT_L10=excluded.MANAGED_SEGMENT_L10,
                        MANAGED_SEGMENT_L11=excluded.MANAGED_SEGMENT_L11,
                        MANAGED_SEGMENT_L12=excluded.MANAGED_SEGMENT_L12,
                        BUSINESS_TITLE=excluded.BUSINESS_TITLE,
                        QUERY_IN_NLP=excluded.QUERY_IN_NLP,
                        ALL_QUERY_TABLES=excluded.ALL_QUERY_TABLES,
                        EMBEDDINGS=excluded.EMBEDDINGS,
                        UPDATED_AT=excluded.UPDATED_AT
                    """,
                    (
                        email,
                        row.get("soeid"),
                        row.get("managed_segment_l1"),
                        row.get("managed_segment_l2"),
                        row.get("managed_segment_l3"),
                        row.get("managed_segment_l4"),
                        row.get("managed_segment_l5"),
                        row.get("managed_segment_l6"),
                        row.get("managed_segment_l7"),
                        row.get("managed_segment_l8"),
                        row.get("managed_segment_l9"),
                        row.get("managed_segment_l10"),
                        row.get("managed_segment_l11"),
                        row.get("managed_segment_l12"),
                        row.get("business_title"),
                        query_text,
                        query_in_nlp,
                        schema_table,
                        json.dumps(all_query_tables) if all_query_tables else None,
                        self._serialize_embeddings(row.get("embeddings")),
                        now,
                    ),
                )
                inserted += 1
            self._db.commit()
        return inserted

    def _serialize_embeddings(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, list):
            try:
                floats = [float(item) for item in value]
            except (TypeError, ValueError):
                return None
            if not floats:
                return None
            if self._backend == "postgres":
                return self._vector_literal(floats)
            return json.dumps(floats)
        return None

    @staticmethod
    def _deserialize_embeddings(value: Any) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, list):
            try:
                floats = [float(item) for item in value]
            except (TypeError, ValueError):
                return None
            return floats or None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [item for item in text.strip("[]").split(",") if item.strip()]
        if not isinstance(parsed, list):
            return None
        try:
            floats = [float(item) for item in parsed]
        except (TypeError, ValueError):
            return None
        return floats or None

    @staticmethod
    def _coerce_iso_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        text = str(value).strip()
        return text or None

    @staticmethod
    def _query_hash(value: str) -> str:
        return hashlib.md5(str(value).encode("utf-8")).hexdigest()

    @staticmethod
    def _vector_literal(values: list[float]) -> str:
        serialized = ", ".join(f"{float(item):.12g}" for item in values)
        return f"[{serialized}]"

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=False
    )

    async def _extract_fts_keywords_with_llm(user_query: str, model_nm: str):
        """Fetch keywords with retry logic and timeout."""

        client = genai.Client(
            vertexai=True,
            project=settings.vertex_project_id,
            location=settings.vertex_location,
        )
        contents = f"""Extract the key search entities from \n {user_query} \nfor a database full-text search.
                    Return ONLY a comma-separated list of important common nouns (avoid proper nouns), technical terms, and domain-specific words.
                    Exclude: stop words, verbs like 'show/find/get', articles, prepositions, and filler words.
                    Preserve compound terms as single tokens where meaningful (e.g. 'machine_learning', 'customer_id').

                    Query: "{user_query}"

                    Response format: word1, word2, word3"""
        response = await client.aio.models.generate_content(
            model=model_nm,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction="You are an expert in extracting keywords from text for a database full-text search.",
                max_output_tokens=256,
                temperature=0.0,
            ),
        )
        # System instructions help keep the model focused


       # response = await model.generate_content_async(prompt, generation_config=config)

        if not response.text:
            return []

        tokens =  [
            kw.strip().replace(" ", "_")
            for kw in response.text.strip().split(",")
            if kw.strip()
        ]
        keywords: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            keyword = token.strip("_")
            if not keyword or keyword in _FTS_STOP_WORDS or len(keyword) <= 2:
                continue
            if keyword in seen:
                continue
            seen.add(keyword)
            keywords.append(keyword)
        logger.debug(f"Keywords for full index search...: {keywords}")
        return keywords

    @staticmethod
    async def _extract_fts_keywords(prompt: str, use_llm=False) -> list[str]:
        if use_llm:
            logger.debug(f"Using LLM to identify keywords....")
            response = await SQLStore._extract_fts_keywords_with_llm(user_query=prompt, model_nm=settings.vertex_model)
            return response
        tokens = re.findall(r"[A-Za-z0-9_]+", str(prompt or "").lower())
        keywords: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            keyword = token.strip("_")
            if not keyword or keyword in _FTS_STOP_WORDS or len(keyword) <= 2:
                continue
            if keyword in seen:
                continue
            seen.add(keyword)
            keywords.append(keyword)
        return keywords

    @staticmethod
    def _build_fts_or_tsquery(keywords: list[str]) -> str:
        sanitized = [re.sub(r"[^\w]", "", keyword) for keyword in keywords]
        return " | ".join(keyword for keyword in sanitized if keyword)

    @staticmethod
    async def _prepare_postgres_fts_query(prompt: str, use_llm: bool = False) -> tuple[str, str]:
        normalized = str(prompt or "").strip()
        resp = await SQLStore._extract_fts_keywords(normalized, use_llm=use_llm)
        tsquery = SQLStore._build_fts_or_tsquery(resp)
        if tsquery:
            return "to_tsquery", tsquery
        return "websearch_to_tsquery", normalized

    @staticmethod
    def _postgres_fts_match(document: str, tsquery_function: str) -> str:
        if tsquery_function not in {"to_tsquery", "websearch_to_tsquery"}:
            raise ValueError(f"Unsupported PostgreSQL FTS function: {tsquery_function}")
        return f"to_tsvector('english', {document}) @@ {tsquery_function}('english', ?)"

    @staticmethod
    def _postgres_fts_rank(document: str, tsquery_function: str) -> str:
        if tsquery_function not in {"to_tsquery", "websearch_to_tsquery"}:
            raise ValueError(f"Unsupported PostgreSQL FTS function: {tsquery_function}")
        return f"ts_rank_cd(to_tsvector('english', {document}), {tsquery_function}('english', ?))"

    @staticmethod
    def _backend_query_nlp_history_select_columns() -> list[str]:
        return [
            "ID",
            "ENGINE",
            "QUERY_ID",
            "CATALOG_NAME",
            "SCHEMA_NAME",
            "QUERY_STATE",
            "QUERY_TYPE",
            "USER_EMAIL",
            "ROLE_NAME",
            "CLUSTER_NAME",
            "SOURCE",
            "CREATED_AT",
            "ENDED_AT",
            "RAW_SQL",
            "QUERY_NLP",
            "SCHEMA_TABLE",
            "TABLES_JSON",
            "UPDATED_AT",
        ]

    @staticmethod
    def _backend_query_nlp_history_fts_document() -> str:
        return (
            "COALESCE(QUERY_NLP, '') || ' ' || "
            "COALESCE(RAW_SQL, '') || ' ' || "
            "COALESCE(SCHEMA_TABLE, '') || ' ' || "
            "COALESCE(TABLES_JSON, '')"
        )

    @staticmethod
    def _backend_query_nlp_history_scope_where(
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        params: list[Any] = []
        if engine:
            where.append("UPPER(ENGINE) = UPPER(?)")
            params.append(engine)
        if catalog:
            where.append("UPPER(CATALOG_NAME) = UPPER(?)")
            params.append(catalog)
        if schema_name:
            where.append("UPPER(SCHEMA_NAME) = UPPER(?)")
            params.append(schema_name)
        return where, params

    @staticmethod
    def _catalog_query_history_nlp_select_columns() -> list[str]:
        return [
            "ID",
            "RAW_QUERY_HISTORY_ID",
            "SOURCE_ID",
            "ENGINE",
            "QUERY_ID",
            "CATALOG_NAME",
            "SCHEMA_NAME",
            "QUERY_STATE",
            "QUERY_TYPE",
            "USER_EMAIL",
            "ROLE_NAME",
            "CLUSTER_NAME",
            "SOURCE",
            "CREATED_AT",
            "ENDED_AT",
            "RAW_SQL",
            "NLP_TEXT",
            "SCHEMA_TABLE",
            "TABLES_JSON",
            "UPDATED_AT",
        ]

    @staticmethod
    def _catalog_query_history_nlp_fts_document() -> str:
        return (
            "COALESCE(NLP_TEXT, '') || ' ' || "
            "COALESCE(RAW_SQL, '') || ' ' || "
            "COALESCE(SCHEMA_TABLE, '') || ' ' || "
            "COALESCE(TABLES_JSON, '')"
        )

    @staticmethod
    def _catalog_query_history_nlp_scope_where(
        *,
        source_id: str | None = None,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        params: list[Any] = []
        if source_id:
            where.append("SOURCE_ID = ?")
            params.append(str(source_id).strip())
        if engine:
            where.append("UPPER(ENGINE) = UPPER(?)")
            params.append(engine)
        if catalog:
            where.append("UPPER(CATALOG_NAME) = UPPER(?)")
            params.append(catalog)
        if schema_name:
            where.append("UPPER(SCHEMA_NAME) = UPPER(?)")
            params.append(schema_name)
        return where, params

    @staticmethod
    def _backend_metadata_table_select_columns() -> list[str]:
        return [
            "ENGINE",
            "CATALOG_ID",
            "CATALOG_NAME",
            "SCHEMA_ID",
            "SCHEMA_NAME",
            "TABLE_ID",
            "TABLE_NAME",
            "TABLE_TYPE",
            "DESCRIPTION",
            "UPDATED_AT",
        ]

    @staticmethod
    def _backend_metadata_table_fts_document() -> str:
        return (
            "COALESCE(CATALOG_NAME, '') || ' ' || "
            "COALESCE(SCHEMA_NAME, '') || ' ' || "
            "COALESCE(TABLE_NAME, '') || ' ' || "
            "COALESCE(TABLE_TYPE, '') || ' ' || "
            "COALESCE(DESCRIPTION, '')"
        )

    @staticmethod
    def _backend_metadata_table_scope_where(
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        params: list[Any] = []
        if engine:
            where.append("UPPER(ENGINE) = UPPER(?)")
            params.append(engine)
        if catalog:
            where.append("UPPER(CATALOG_NAME) = UPPER(?)")
            params.append(catalog)
        if schema_name:
            where.append("UPPER(SCHEMA_NAME) = UPPER(?)")
            params.append(schema_name)
        return where, params

    @staticmethod
    def _backend_metadata_column_select_columns() -> list[str]:
        return [
            "ENGINE",
            "CATALOG_ID",
            "CATALOG_NAME",
            "SCHEMA_ID",
            "SCHEMA_NAME",
            "TABLE_ID",
            "TABLE_NAME",
            "COLUMN_ID",
            "COLUMN_NAME",
            "ORDINAL_POSITION",
            "DATA_TYPE",
            "NULLABLE",
            "DESCRIPTION",
            "UPDATED_AT",
        ]

    @staticmethod
    def _backend_metadata_column_fts_document() -> str:
        return (
            "COALESCE(CATALOG_NAME, '') || ' ' || "
            "COALESCE(SCHEMA_NAME, '') || ' ' || "
            "COALESCE(TABLE_NAME, '') || ' ' || "
            "COALESCE(COLUMN_NAME, '') || ' ' || "
            "COALESCE(DATA_TYPE, '') || ' ' || "
            "COALESCE(DESCRIPTION, '')"
        )

    @staticmethod
    def _backend_metadata_column_scope_where(
        *,
        engine: str | None = None,
        catalog: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        where, params = SQLStore._backend_metadata_table_scope_where(
            engine=engine,
            catalog=catalog,
            schema_name=schema_name,
        )
        if table_name:
            where.append("UPPER(TABLE_NAME) = UPPER(?)")
            params.append(table_name)
        return where, params

    @staticmethod
    def _extract_vector_dimension(dim_row: dict[str, Any]) -> int | None:
        formatted = str(SQLStore._row_value(dim_row, "formatted_type", "") or "").strip().lower()
        match = re.search(r"vector\((\d+)\)", formatted)
        if match:
            return int(match.group(1))

        typmod = int(SQLStore._row_value(dim_row, "dim_typmod", -1) or -1)
        candidates: list[int] = []
        if typmod > 0:
            candidates.append(typmod)
        if typmod > 4:
            candidates.append(typmod - 4)

        for candidate in candidates:
            if candidate > 0:
                return candidate
        return None

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

    def _backend_table_metadata(self, engine: str, ref: dict[str, str]) -> dict[str, Any] | None:
        table_name = ref.get("table_name", "")
        if not table_name:
            return None

        where = ["UPPER(TABLE_NAME) = UPPER(?)"]
        params: list[Any] = [table_name]
        if engine:
            where.append("UPPER(ENGINE) = UPPER(?)")
            params.append(engine)
        if ref.get("catalog"):
            where.append("UPPER(CATALOG_NAME) = UPPER(?)")
            params.append(ref["catalog"])
        if ref.get("schema_name"):
            where.append("UPPER(SCHEMA_NAME) = UPPER(?)")
            params.append(ref["schema_name"])

        return self._query_one(
            f"""
            SELECT ENGINE, CATALOG_NAME, SCHEMA_NAME, TABLE_NAME, DESCRIPTION
            FROM BACKEND_METADATA_TABLES
            WHERE {" AND ".join(where)}
            ORDER BY UPDATED_AT DESC
            LIMIT 1
            """,
            tuple(params),
        )

    def _backend_column_metadata(self, engine: str, ref: dict[str, str]) -> list[dict[str, Any]]:
        table_name = ref.get("table_name", "")
        if not table_name:
            return []

        where = ["UPPER(TABLE_NAME) = UPPER(?)"]
        params: list[Any] = [table_name]
        if engine:
            where.append("UPPER(ENGINE) = UPPER(?)")
            params.append(engine)
        if ref.get("catalog"):
            where.append("UPPER(CATALOG_NAME) = UPPER(?)")
            params.append(ref["catalog"])
        if ref.get("schema_name"):
            where.append("UPPER(SCHEMA_NAME) = UPPER(?)")
            params.append(ref["schema_name"])

        rows = self._query_all(
            f"""
            SELECT COLUMN_NAME, DATA_TYPE, DESCRIPTION, ORDINAL_POSITION
            FROM BACKEND_METADATA_COLUMNS
            WHERE {" AND ".join(where)}
            ORDER BY ORDINAL_POSITION, COLUMN_NAME
            """,
            tuple(params),
        )
        return [
            {
                "column_name": self._row_value(row, "COLUMN_NAME"),
                "data_type": self._row_value(row, "DATA_TYPE"),
                "description": self._row_value(row, "DESCRIPTION"),
                "ordinal_position": self._row_value(row, "ORDINAL_POSITION"),
            }
            for row in rows
        ]

    @staticmethod
    def _parse_backend_table_ref(value: Any) -> dict[str, str]:
        cleaned = str(value or "").strip().replace('"', "").replace("`", "")
        if not cleaned:
            return {"catalog": "", "schema_name": "", "table_name": ""}
        parts = [part.strip() for part in cleaned.split(".") if part.strip()]
        if len(parts) >= 3:
            return {"catalog": parts[-3], "schema_name": parts[-2], "table_name": parts[-1]}
        if len(parts) == 2:
            return {"catalog": "", "schema_name": parts[0], "table_name": parts[1]}
        return {"catalog": "", "schema_name": "", "table_name": parts[0]}

    @staticmethod
    def _format_backend_table_name(catalog: str | None, schema_name: str | None, table_name: str | None) -> str:
        parts = [str(part or "").strip() for part in (catalog, schema_name, table_name)]
        return ".".join(part for part in parts if part)

    @staticmethod
    def _coerce_table_list(value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, list):
            values = value
        elif isinstance(value, str):
            text = value.strip()
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
            values = [value]

        normalized: list[str] = []
        seen: set[str] = set()
        for item in values:
            if isinstance(item, dict):
                parts = [
                    str(item.get(key) or "").strip()
                    for key in ("catalog", "schema", "table")
                ]
                table = ".".join(part for part in parts if part)
            elif isinstance(item, (list, tuple)):
                parts = [str(part or "").strip() for part in item[:3]]
                table = ".".join(part for part in parts if part)
            else:
                table = str(item or "").strip()
            if not table:
                continue
            key = table.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(table)
        return normalized
