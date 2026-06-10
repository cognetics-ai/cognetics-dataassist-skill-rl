-- =============================================================================
-- SkillSQL-RL  ·  One-time database initialisation script
-- =============================================================================
-- Run this ONCE against your local Postgres instance (or let Docker Compose
-- mount it as an initdb script).
--
-- Usage (local Postgres, as a superuser):
--   psql -U postgres -f scripts/init_databases.sql
--
-- What this creates:
--   skillsql  role         -- catalog + pgvector owner
--   adk_demo  role         -- ADK sessions store owner
--   skillsql_catalog DB    -- semantic catalog + SkillBank (pgvector)
--   adk_demo_db DB         -- ADK sessions/events (schema: adk_store)
--
-- After running this, execute:
--   skillsql init-db       -- creates the skillsql_catalog schema + all tables
-- The ADK runtime auto-creates the adk_store schema on first startup.
-- =============================================================================

-- ── Roles (idempotent) ──────────────────────────────────────────────────────
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'skillsql') THEN
        CREATE ROLE skillsql WITH LOGIN PASSWORD 'skillsql';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'adk_demo') THEN
        CREATE ROLE adk_demo WITH LOGIN PASSWORD 'adk_demo';
    END IF;
END
$$;

-- ── Databases (idempotent) ──────────────────────────────────────────────────
SELECT 'CREATE DATABASE skillsql_catalog OWNER skillsql'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'skillsql_catalog')\gexec

SELECT 'CREATE DATABASE adk_demo_db OWNER adk_demo'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'adk_demo_db')\gexec

-- ── pgvector extension (catalog DB only) ───────────────────────────────────
-- Must be connected to skillsql_catalog for this to work:
-- \c skillsql_catalog
-- CREATE EXTENSION IF NOT EXISTS vector;
--
-- The `skillsql init-db` command runs this automatically when first executed.
