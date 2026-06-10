-- =============================================================================
-- SkillSQL-RL :: app-specific catalog schema (raw-SQL mirror of the ORM).
-- The ORM (skillsql.catalog.models) is the source of truth; this file is for
-- DBAs who provision via SQL/migration tooling. Run against APP_CATALOG_DSN.
--   psql "$APP_CATALOG_DSN" -f sql/catalog_schema.sql
-- NOTE: vector(1024) must match EMBEDDING_DIM.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS skillsql_catalog;
SET search_path TO skillsql_catalog, public;

CREATE TABLE IF NOT EXISTS sources (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type        VARCHAR(32) NOT NULL,
    name               VARCHAR(128) NOT NULL,
    database           VARCHAR(256),
    db_schema          VARCHAR(256),
    origin_fingerprint VARCHAR(128),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_type, name, database, db_schema)
);
CREATE INDEX IF NOT EXISTS ix_sources_type ON sources (source_type);

CREATE TABLE IF NOT EXISTS catalog_tables (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id              UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    fqn                    VARCHAR(512) NOT NULL,
    name                   VARCHAR(256) NOT NULL,
    table_type             VARCHAR(64) NOT NULL DEFAULT 'BASE TABLE',
    comment                TEXT,
    row_estimate           INTEGER,
    nl_description         TEXT,
    description_confidence DOUBLE PRECISION,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_catalog_tables_fqn ON catalog_tables (fqn);
CREATE INDEX IF NOT EXISTS ix_catalog_tables_source ON catalog_tables (source_id);

CREATE TABLE IF NOT EXISTS catalog_columns (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_id          UUID NOT NULL REFERENCES catalog_tables(id) ON DELETE CASCADE,
    name              VARCHAR(256) NOT NULL,
    data_type         VARCHAR(128) NOT NULL,
    nullable          BOOLEAN NOT NULL DEFAULT TRUE,
    ordinal           INTEGER NOT NULL DEFAULT 0,
    is_primary_key    BOOLEAN NOT NULL DEFAULT FALSE,
    comment           TEXT,
    sample_values     JSONB,
    null_fraction     DOUBLE PRECISION,
    distinct_estimate INTEGER,
    nl_description    TEXT
);
CREATE INDEX IF NOT EXISTS ix_catalog_columns_table ON catalog_columns (table_id);

CREATE TABLE IF NOT EXISTS schema_docs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    object_type VARCHAR(16) NOT NULL,           -- table | column
    fqn         VARCHAR(512) NOT NULL,
    table_name  VARCHAR(256) NOT NULL,
    column_name VARCHAR(256),
    text        TEXT NOT NULL,
    embedding   vector(1024),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_schema_docs_source ON schema_docs (source_id);
CREATE INDEX IF NOT EXISTS ix_schema_docs_fqn ON schema_docs (fqn);
-- Cosine ANN index (build after bulk load for best results):
CREATE INDEX IF NOT EXISTS ix_schema_docs_embedding
    ON schema_docs USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS skills (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope            VARCHAR(32) NOT NULL,       -- general_sql|dialect|schema_specific|failure_repair|verifier_obligation
    skill_type       VARCHAR(32) NOT NULL DEFAULT 'strategy',
    title            VARCHAR(256) NOT NULL,
    principle        TEXT NOT NULL,
    when_to_apply    TEXT,
    positive_example TEXT,
    negative_example TEXT,
    source_id        UUID REFERENCES sources(id) ON DELETE SET NULL,
    dialect          VARCHAR(32),
    provenance       JSONB,
    status           VARCHAR(16) NOT NULL DEFAULT 'candidate',  -- candidate|promoted
    embedding        vector(1024),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_skills_scope ON skills (scope);
CREATE INDEX IF NOT EXISTS ix_skills_embedding
    ON skills USING hnsw (embedding vector_cosine_ops);
