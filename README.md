# SkillSQL-RL: Recursive Skill-Augmented RL for Enterprise Text-to-SQL

**Version 0.2.0 — unified codebase**

This repository is the companion engineering deliverable for the proposal
*"SkillSQL-RL: Recursive Skill-Augmented Reinforcement Learning for Enterprise Text-to-SQL"*
(v2, June 2026). It combines two previously separate code trees—the FastAPI data-assist
service and the SkillSQL-RL inference/training framework—into a single, cleanly-structured
Python project.

---

## Architecture overview

```
Two-plane ADK 2.0 architecture
─────────────────────────────────────────────────────────────────────────────
Cataloging plane                          Inference / RL plane
──────────────────────                    ─────────────────────────────────
Connector factory                         SQL generator (Arctic-7B via Ollama)
  │  execute / explain / metadata           │  completion-only; no tools
  ▼                                        ▼
Catalog builder                           Verifier (static gates + execute)
  sample · describe · query history · embed │  Equation 12 reward cascade
  │                                        ▼
  ▼                                       SkillBank (pgvector, 5 scopes)
Postgres + pgvector (APP_CATALOG_DSN)       │  retrieve → Equation 6
  source groups → sources → catalog         │
                                           ▼
                                          ADK Runner (ADK_SESSION_DSN)

                                          vLLM / verl (offline, GPU)
                                            GRPO trainer — Algorithm 1
```

### Package layout

```
cognetics-dataassist-skill-rl/
│
├── app/                        FastAPI backend (preserved from working code)
│   ├── main.py                 Entry point; includes catalog + skillbank routes
│   ├── config.py               App settings (Pydantic)
│   ├── adapters/               EngineAdapter abstraction (Starburst, mock, …)
│   │   └── bridge.py           NEW: wraps DataSourceConnector as EngineAdapter
│   ├── adk/                    Google ADK runtime integration
│   ├── api/routes/             FastAPI routes (auth, discovery, query, text2sql,
│   │                                          catalog, skillbank)
│   ├── core/                   Policy, SQL utils, event bus, SQLite/PG store
│   └── services/               Catalog, embeddings, directory, data-usage NLP
│
├── skillsql/                   SkillSQL-RL core framework
│   ├── config/                 Settings (DATASOURCE_TYPE, DSNs, model config)
│   ├── connectors/             Abstract factory + concrete backends
│   │   ├── base.py             DataSourceConnector ABC + DTOs
│   │   ├── factory.py          ConnectorFactory (registry pattern)
│   │   ├── snowflake_connector.py  Snowflake (v1 concrete backend)
│   │   ├── starburst_connector.py  Starburst Galaxy + Trino
│   │   ├── postgres_connector.py   Postgres
│   │   └── oracle_connector.py     Oracle (stub)
│   ├── catalog/                Semantic catalog (pgvector)
│   │   ├── models.py           ORM: SourceGroup, Source, CatalogTable,
│   │   │                           CatalogColumn, SchemaDocRow,
│   │   │                           CatalogQueryHistory, Skill
│   │   ├── builder.py          build_catalog()
│   │   ├── embeddings.py       Pluggable embedder (Ollama, OpenAI, Vertex)
│   │   └── repository.py       CatalogRepository (CRUD + vector search)
│   ├── skillbank/              SqlSkillBank (5 scopes)
│   │   ├── seeds.py            Curated general_sql + Snowflake dialect skills
│   │   └── retrieval.py        retrieve_skills() → Equation 6
│   ├── context/
│   │   └── builder.py          build_context() → schema + skill prompt block
│   ├── verification/           Formal verifier + reward
│   │   ├── static_gates.py     Safe, Parse, Bind, Scope, Join gates (§5.2)
│   │   ├── obligations.py      Semantic obligations Ω(q), ω(y,q) (§5.3)
│   │   ├── equivalence.py      result_equivalent() instance equivalence
│   │   └── reward.py           compute_reward() cascade (Equation 12)
│   ├── rl/                     GRPO training track
│   │   ├── rollout.py          Trajectory collection (Algorithm 1, lines 4-8)
│   │   ├── distillation.py     Skill distillation T+ / T- (Equations 4-5)
│   │   ├── evolution.py        Recursive skill evolution (Equation 10)
│   │   └── grpo.py             GRPO outer loop (Algorithm 1, Equations 9-10)
│   ├── agents/                 ADK agent construction
│   ├── workflow/               ADK 2.0 workflow graph (retrieve → gen → verify)
│   ├── benchmark/              Spider-2.0-Snow driver
│   │   ├── spider2_loader.py   Parse spider2-snow.jsonl
│   │   └── run_benchmark.py    Full benchmark run + evaluator artifacts
│   ├── models/                 Model registry (Arctic, Gemini, Ollama)
│   ├── observability/          Structured logging (structlog, JSON in prod)
│   └── cli.py                  ``skillsql`` CLI (init-db, catalog-build, …)
│
├── data-assist-frontend/       React/TypeScript frontend (preserved as-is)
│
├── scripts/
│   ├── build_catalog.py        Discover + persist datasource catalog
│   ├── run_benchmark.py        Spider-2.0-Snow benchmark runner
│   ├── train_grpo.py           GRPO training driver
│   ├── plot_results.py         Generate paper figures (Fig 1–8)
│   ├── pull_models.sh          Pull Ollama models (Arctic + embedding model)
│   └── init_databases.sql      Postgres database initialization
│
├── sql/
│   └── catalog_schema.sql      DDL for catalog tables (reference)
│
└── tests/                      Test suite
    ├── conftest.py
    ├── _fakes.py               Mock connector for tests
    ├── test_static_gates.py    Verification lattice tests
    ├── test_obligations.py     Semantic obligations tests
    ├── test_equivalence.py     Instance equivalence tests
    ├── test_reward.py          Composite reward cascade tests
    └── test_connectors.py      Read-only enforcement tests
```

---

## Quick start

### 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | |
| Postgres 15+ with pgvector | Local install assumed. No Docker needed. |
| Ollama | Local install assumed, running on `localhost:11434`. |
| Snowflake / Starburst credentials | Or use `DATASOURCE_TYPE=postgres` for local dev. |

**Postgres setup (once):** Create the two databases SkillSQL-RL needs:
```bash
# Run as a Postgres superuser
psql -U postgres -f scripts/init_databases.sql
# or manually:
psql -U postgres -c "CREATE USER skillsql WITH PASSWORD 'skillsql';"
psql -U postgres -c "CREATE DATABASE skillsql_catalog OWNER skillsql;"
psql -U postgres -c "CREATE USER adk_demo WITH PASSWORD 'adk_demo';"
psql -U postgres -c "CREATE DATABASE adk_demo_db OWNER adk_demo;"
```

> **No Docker required.** If you don't have a local Postgres installation,
> `docker compose --profile pg up -d` starts one. Ollama likewise:
> `docker compose --profile models up -d`. Both are opt-in profiles.

### 2. Install

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or pip
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env .env.bak          # back up if needed
# .env is already populated — edit credentials for your environment:
#   SNOWFLAKE_PASSWORD, STARBURST_PASSWORD, STARBURST_CLIENT_SECRET, etc.
# Verify the two Postgres DSNs match your local setup:
#   APP_CATALOG_DSN  = postgresql+psycopg://skillsql:skillsql@localhost:5432/skillsql_catalog
#   ADK_SESSION_DSN  = postgresql://adk_demo:adk_demo@127.0.0.1:5432/adk_demo_db
```

### 4. Check connectivity, pull models, initialize

```bash
make setup-local       # checks Postgres + Ollama, runs init-db, pulls models
# Or step by step:
make check-postgres    # verify local Postgres is reachable
make check-ollama      # verify Ollama is running
make init-db           # create skillsql_catalog schema + pgvector extension
make models            # pull Arctic-7B + snowflake-arctic-embed + llama3.1:8b
```

If you intentionally want to rebuild the semantic catalog from a clean slate,
use the destructive reset target:

```bash
make reset-catalog-db
# equivalent:
.venv/bin/skillsql init-db --reset
```

### 5. Build the catalog, descriptions, and query history

```bash
make catalog-build     # discover datasource schema → semantic catalog
skillsql catalog-build --seed-skills   # also inserts curated SkillBank seeds
```

The current catalog model has two levels of datasource identity:

| Level | Table | Meaning |
|---|---|---|
| Parent | `source_groups` | Logical benchmark or estate, for example `spider2-snow` |
| Child | `sources` | One physical/queryable scope, for example one Snowflake database/schema or one Starburst catalog/schema |

Use `source_group_name` while loading many related schemas so schema docs and
historical query examples can later be retrieved at the parent level. Use
`source_id` when execution must target one exact database/schema. If both are
provided to context generation, `source_id` takes precedence.

CLI example for a grouped Snowflake catalog build:

```bash
skillsql catalog-build \
  --source-type snowflake \
  --source-name spider2-snow-census \
  --source-group-name spider2-snow \
  --catalog-names CENSUS_DB \
  --db-schema PUBLIC \
  --seed-skills
```

API example for a grouped schema sync:

```bash
curl -X POST http://localhost:8000/catalog/metadata/schema/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "snowflake",
    "source_name": "spider2-snow-census",
    "source_group_name": "spider2-snow",
    "catalog": "CENSUS_DB",
    "database_name": "CENSUS_DB",
    "schema_name": "PUBLIC",
    "include_columns": true,
    "describe": false
  }'
```

For Starburst, pass the catalog as `catalog`. The Galaxy REST lookup needs
`name=<catalog>` internally, but the semantic catalog persists only the bare
catalog name, for example `sample`, not `name=sample`.

After metadata sync, generate natural-language table and column descriptions.
The column description endpoint can run for one table or for every table in the
schema when `table_name` is omitted:

```bash
curl -X POST http://localhost:8000/catalog/metadata/table-description/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "starburst",
    "source_name": "starburst",
    "catalog": "sample",
    "database_name": "sample",
    "schema_name": "burstbank",
    "missing_only": true,
    "limit": 500,
    "sample_size": 5
  }'

curl -X POST http://localhost:8000/catalog/metadata/column-description/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "starburst",
    "source_name": "starburst",
    "catalog": "sample",
    "database_name": "sample",
    "schema_name": "burstbank",
    "missing_only": true,
    "limit": 2000,
    "sample_size": 5
  }'
```

Load historical query examples into the catalog namespace, then generate NLP
text and embeddings for finished queries:

```bash
curl -X POST http://localhost:8000/catalog/query-history/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "starburst",
    "source_name": "starburst",
    "source_group_name": "starburst-sample",
    "catalog": "sample",
    "database_name": "sample",
    "schema_name": "burstbank",
    "limit": 1000
  }'

curl -X POST http://localhost:8000/catalog/query-history/nlp-history/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "starburst",
    "source_group_name": "starburst-sample",
    "catalog": "sample",
    "database_name": "sample",
    "schema_name": "burstbank",
    "limit": 1000,
    "missing_only": true
  }'
```

The SQL prompt context is built from both `schema_docs` and
`catalog_query_history_nlp`. Schema docs are grouped by table and include table
description plus relevant column names, types, and column descriptions.
Historical examples are limited to `query_state = 'FINISHED'`; failed queries
are preserved for failure analysis in `nlp_text` but filtered out of semantic
and lexical retrieval.

### 6. Run the API server

```bash
make serve             # FastAPI on :8000
# or
make serve-reload      # with hot-reload
```

### 7. Run a question end-to-end

```bash
make run Q="What is total revenue by product category for Q1 2024?"
# or
skillsql run "What is total revenue by product category for Q1 2024?"
```

For parent-level retrieval through the API, pass `source_group_id`; list current
groups with `GET /catalog/sources`. For exact execution scope, pass `source_id`:

```bash
curl -X POST http://localhost:8000/sql/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is population of New Jersey as per the census?",
    "source_group_id": "<source-group-uuid>"
  }'
```

---

## Benchmark (Spider-2.0-Snow)

Download `spider2-snow.jsonl` from the [Spider 2.0 repository](https://github.com/xlang-ai/spider2)
and point `SPIDER2_SNOW_JSONL` at it:

```bash
export SPIDER2_SNOW_JSONL=./data/spider2-snow.jsonl

# Non-oracle run (default; comparable to the leaderboard)
make benchmark

# Oracle-table run (flagged in manifest; never mix with leaderboard)
make benchmark-oracle

# Inspect results
cat outputs/spider2_snow/manifest.json
```

> ⚠️ Oracle-table runs inflate scores; the manifest records the flag and must not
> be compared against the standard leaderboard (proposal Section 8.1).

---

## Training (GRPO)

Training is deliberately separate from production inference.

The policy model for GRPO is the SQL generator: `QUERY_GENERATOR_MODEL`,
currently defaulting to `a-kore/Arctic-Text2SQL-R1-7B:latest`. The general ADK
chat agents (`ADK_MODEL`, locally `ollama_chat/llama3.1:8b`) and the embedding
model (`EMBEDDING_MODEL`, locally `snowflake-arctic-embed:l`) are not updated by
GRPO. They provide orchestration, descriptions, context building, and retrieval.

Practically, do not train the Ollama tag directly. Ollama is the local inference
runtime for development. Weight updates should use a trainable Hugging
Face-compatible checkpoint for the same Arctic Text-to-SQL policy family, then
serve the trained checkpoint or LoRA adapter through vLLM or another
OpenAI-compatible endpoint and point `QUERY_GENERATOR_MODEL_PROVIDER` /
`QUERY_GENERATOR_MODEL_API_BASE` / `QUERY_GENERATOR_MODEL` at that endpoint.

Current implementation status:

| Path | What works now |
|---|---|
| `GRPO_POLICY_BACKEND=noop` | Builds prompts, samples candidates, executes/verifies SQL, computes rewards, evolves skills, and writes GRPO JSONL artifacts |
| `GRPO_POLICY_BACKEND=verl` | Training boundary exists and validates that `verl` is installed, but the actual verl trainer wiring is still a placeholder |

### Mac vs. GPU host

Use macOS for catalog loading, prompt/context debugging, local rollouts, reward
scoring, skill evolution, and artifact generation. Docker on macOS does not
provide an NVIDIA CUDA runtime to containers, so it is not sufficient for
vLLM/verl weight updates. Run weight updates on a Linux host or container with
an NVIDIA CUDA GPU, for example a Linux workstation, cloud GPU instance, or a
remote Docker host with the NVIDIA container toolkit.

Check the current host:

```bash
make check-training-env
```

### Stage 1: local dry-run and artifact generation

Before attempting weight updates, prove that catalog context, execution, reward,
and artifact generation work end to end:

```bash
export SPIDER2_SNOW_JSONL=./data/spider2-snow.jsonl

python scripts/train_grpo.py \
  --jsonl "$SPIDER2_SNOW_JSONL" \
  --epochs 1 \
  --group-size 4 \
  --limit 50 \
  --policy-backend noop \
  --output-dir ./outputs/checkpoints

# Same via make; writes artifacts under ./outputs/checkpoints
GRPO_EPOCHS=1 GRPO_GROUP_SIZE=4 GRPO_POLICY_BACKEND=noop make train-grpo
```

Each epoch writes:

| Artifact | Purpose |
|---|---|
| `grpo_batch_epoch_0001.jsonl` | Prompt, sampled SQL response, reward, normalized advantage, task/source metadata |
| `grpo_batch_epoch_0001.manifest.json` | Batch size and reward summary |
| `training_metrics.jsonl` | Epoch-level reward, execution accuracy, and skill-evolution metrics |
| `manifest.json` | Final training run summary |

### Stage 2: prepare a GPU training environment

On the Linux CUDA host:

```bash
git clone <repo-url>
cd cognetics-dataassist-skill-rl
python -m venv .venv
. .venv/bin/activate
pip install -e ".[training]"

# Install the heavy GPU stack in this environment.
# Pin versions according to the CUDA/PyTorch image being used.
pip install vllm verl

python scripts/check_training_env.py --require-gpu --require-verl
```

Use a CUDA-ready base image or host environment that already has matching
NVIDIA driver, CUDA runtime, PyTorch, vLLM, and verl versions. The project keeps
`vllm` and `verl` out of default dependencies because they are large,
Linux/GPU-specific, and brittle on developer laptops.

### Stage 3: wire the real verl update

The code boundary for actual weight updates is
`skillsql/rl/policy_update.py::_build_verl_policy_update`. That function should
replace the placeholder with these steps:

1. Convert the GRPO batch rows into verl data records containing `prompt`,
   generated `response`/`sql`, scalar `reward`, group id, and normalized
   `advantage`.
2. Load the trainable Arctic policy checkpoint as the actor model.
3. Load a frozen reference copy of the same base checkpoint for KL control.
4. Use vLLM for rollout/logprob throughput, or generate candidates with the
   actor server and compute actor/reference logprobs inside the trainer.
5. Apply the GRPO clipped objective with the configured KL penalty.
6. Save a checkpoint or LoRA adapter under `outputs/checkpoints`.
7. Emit update metrics such as loss, KL, reward mean, grad norm, checkpoint
   path, and `weights_updated=true`.

The repository already computes the GRPO inputs:

```text
G candidates per task
R_i = verifier reward
A_i = (R_i - mean(R_group)) / (std(R_group) + epsilon)
```

The missing production piece is only the trainer-specific conversion from this
batch schema into verl's dataset/DataProto format and checkpoint save/load
configuration.

### Stage 4: run GRPO weight updates

Once the verl update function is wired:

```bash
export SPIDER2_SNOW_JSONL=./data/spider2-snow.jsonl
export GRPO_POLICY_BACKEND=verl
export GRPO_EPOCHS=3
export GRPO_GROUP_SIZE=8

make train-grpo
```

After training, serve the new checkpoint for inference. For example, if the
checkpoint is exposed by an OpenAI-compatible vLLM server:

```bash
export QUERY_GENERATOR_MODEL_PROVIDER=openai
export QUERY_GENERATOR_MODEL_API_BASE=http://gpu-host:8001/v1
export QUERY_GENERATOR_MODEL_API_KEY=dummy
export QUERY_GENERATOR_MODEL=skillsql-arctic-grpo
```

Production inference should continue to use `/sql/generate` or the ADK
Text2SQL workflow. The training loop should consume benchmark tasks and query
run feedback, write artifacts/checkpoints, then deploy a promoted SQL-generator
checkpoint back into inference.

---

## Generating paper figures

```bash
pip install -e ".[plotting]"
make plot-results
# Figures written to outputs/figures/fig{1..8}.<pdf|png>
```

Figure inventory:
- **Fig 1** — Execution Accuracy: SkillSQL-RL vs. baselines (bar)
- **Fig 2** — EX by task category (grouped bars)
- **Fig 3** — Training reward + EX over epochs (dual-axis line)
- **Fig 4** — Diagnostic metrics: static validity / exec success / retrieval hit
- **Fig 5** — Skill-evolution gain ΔEX per category (before/after)
- **Fig 6** — Prompt footprint: schema / skills / task token breakdown (stacked bars)
- **Fig 7** — Reward–result correlation scatter (Property 1 verification)
- **Fig 8** — Ablation analysis A1–A4 vs. full SkillSQL-RL

---

## Key design decisions

### Semantic catalog and source groups

The catalog has a parent/child source model:

```text
source_groups
  └── sources
        ├── catalog_tables
        │     └── catalog_columns
        ├── schema_docs
        ├── catalog_query_history
        └── catalog_query_history_nlp
```

`source_groups` lets one benchmark or enterprise estate contain many physical
query scopes. For Spider-2.0-Snow, use a parent like `spider2-snow` and create
one child `source` per Snowflake database/schema that was loaded. For Starburst,
use the bare catalog name in persistence, even though the Galaxy REST API lookup
uses `name=<catalog>` internally.

Context generation first retrieves schema docs, groups them by table, and emits
table descriptions plus relevant columns, data types, and column descriptions.
It then retrieves top-k semantically similar rows from
`catalog_query_history_nlp` where the raw query state is `FINISHED`, formats them
as in-context SQL/NL examples, and merges their table/column references back
into the schema context.

Use `source_group_id` for broad retrieval across a benchmark/estate. Use
`source_id` for an exact executable scope. When both are supplied, `source_id`
wins.

### Connector abstraction (proposal §6.1)

All datasource access goes through `DataSourceConnector` (abstract factory).
The verifier, catalog builder, and benchmark runner **never** import a vendor SDK directly.
Starburst is fully async (aiohttp, Galaxy REST + Trino REST); Snowflake and Postgres use
synchronous DB-API wrapped in `asyncio.to_thread()` via `SyncConnectorMixin`.

```
ConnectorFactory.create("snowflake", cfg) → SnowflakeConnector  (async via SyncConnectorMixin)
ConnectorFactory.create("starburst", cfg) → StarburstConnector  (natively async — Galaxy API + Trino REST)
ConnectorFactory.create("postgres",  cfg) → PostgresConnector   (async via SyncConnectorMixin)
ConnectorFactory.create("oracle",    cfg) → OracleConnector     (stub — extension point)
```

### Two DSNs (proposal §6.4)

| Env var           | Database            | Schema              | Driver     | Purpose                          |
|-------------------|---------------------|---------------------|------------|----------------------------------|
| `APP_CATALOG_DSN` | `skillsql_catalog`  | `skillsql_catalog`  | psycopg    | Catalog, SchemaDocRow, Skill ORM |
| `ADK_SESSION_DSN` | `adk_demo_db`       | `adk_store`         | asyncpg    | ADK sessions + events            |

`ADK_SESSION_DSN` uses a plain `postgresql://` URL in `.env`; the runtime converts it to
`postgresql+asyncpg://` and passes `search_path=adk_store,public` through asyncpg
`server_settings` automatically.
The `adk_store` schema is created on first startup (no manual SQL needed).

### Composite reward (proposal Eq. 12)

```
R(τ) = -1.00              ← not Safe
       -0.60              ← not Parse
       -0.35              ← not Bind
       -0.25 + 0.15·ω    ← not Exec
       +1.00 + 0.15·ω    ← Exec + match (Property 1: correct always > incorrect)
       +0.10 + 0.25·ω    ← Exec, no gold
```

### SqlSkillBank scopes (proposal Table 1)

| Scope                   | Retrieval              | Example principle                          |
|-------------------------|------------------------|--------------------------------------------|
| `general_sql`           | Always injected        | Declare grain before aggregating           |
| `dialect`               | Always (for dialect)   | Use QUALIFY not WHERE for window filters   |
| `schema_specific`       | Embedding + source_id  | Customer balances: monthly net, then CTE   |
| `failure_repair`        | Error sig + embedding  | Invalid identifier → copy from schema      |
| `verifier_obligation`   | From obligation extractor | Date spine required for absent periods  |

---

## Environment variables

All configuration lives in `.env` at the project root. Critical variables:

| Variable                | Example value                                        | Description                              |
|-------------------------|------------------------------------------------------|------------------------------------------|
| `DATASOURCE_TYPE`       | `snowflake`                                          | Active backend (snowflake\|starburst\|postgres) |
| `APP_CATALOG_DSN`       | `postgresql+psycopg://skillsql:skillsql@localhost/skillsql_catalog` | Catalog + SkillBank (psycopg sync) |
| `APP_CATALOG_SCHEMA`    | `skillsql_catalog`                                   | Postgres schema for catalog tables       |
| `ADK_SESSION_DSN`       | `postgresql://adk_demo:adk_demo@127.0.0.1/adk_demo_db` | ADK sessions (plain URL; runtime adds asyncpg driver) |
| `ADK_SESSION_SCHEMA`    | `adk_store`                                          | Postgres schema for ADK sessions/events  |
| `ADK_APP_NAME`          | `skillsql_rl`                                        | ADK application namespace                |
| `SNOWFLAKE_ACCOUNT`     | `RSRSBDK-YDB67606`                                   | Snowflake account identifier             |
| `SNOWFLAKE_USER`        | —                                                    | Snowflake username                       |
| `SNOWFLAKE_PASSWORD`    | —                                                    | Snowflake password (use secrets in prod) |
| `STARBURST_HOST`        | `acme.galaxy.starburst.io`                           | Galaxy API host                          |
| `STARBURST_TRINO_HOST`  | `acme.trino.galaxy.starburst.io`                     | Trino cluster host                       |
| `STARBURST_CLIENT_ID`   | —                                                    | Galaxy OAuth client ID                   |
| `STARBURST_CLIENT_SECRET` | —                                                  | Galaxy OAuth client secret               |
| `OLLAMA_API_BASE`       | `http://localhost:11434`                             | Ollama server URL (local)                |
| `ADK_MODEL_PROVIDER`    | `ollama`                                             | Provider for workflow agents             |
| `ADK_MODEL`             | `ollama_chat/llama3.1:8b`                            | Model for router, catalog, and workflow agents |
| `QUERY_GENERATOR_MODEL_PROVIDER` | `ollama`                                     | Provider for the SQL generator agent     |
| `QUERY_GENERATOR_MODEL` | `a-kore/Arctic-Text2SQL-R1-7B:latest`                | Text2SQL model used only by query generation |
| `ADK_MODEL_API_BASE`    | —                                                    | Optional LiteLLM/OpenAI-compatible base URL |
| `ADK_MODEL_API_KEY`     | —                                                    | Optional LiteLLM/OpenAI-compatible API key |
| `ADK_MODEL_TIMEOUT_SECONDS` | `1800`                                           | LiteLLM timeout for non-query-generator agents |
| `ADK_MODEL_MAX_RETRIES` | `1`                                                  | Retry count after the initial ADK model attempt |
| `ADK_MODEL_RETRY_BACKOFF_INITIAL_SECONDS` | `2`                              | Initial ADK runner retry backoff |
| `ADK_MODEL_RETRY_BACKOFF_MAX_SECONDS` | `30`                                  | Maximum ADK runner retry backoff |
| `QUERY_GENERATOR_MODEL_TIMEOUT_SECONDS` | `1800`                               | LiteLLM timeout for the query-generator model |
| `QUERY_GENERATOR_MODEL_MAX_RETRIES` | `1`                                      | Retry count after the initial query-generator attempt |
| `QUERY_GENERATOR_MODEL_RETRY_BACKOFF_INITIAL_SECONDS` | `2`                  | Initial query-generator runner retry backoff |
| `QUERY_GENERATOR_MODEL_RETRY_BACKOFF_MAX_SECONDS` | `30`                      | Maximum query-generator runner retry backoff |
| `EMBEDDING_PROVIDER`    | `ollama`                                             | Embedding provider (`ollama`, `openai`, `vertex`, `litellm`) |
| `EMBEDDING_MODEL`       | `snowflake-arctic-embed:l`                           | Embedding model name                     |
| `EMBEDDING_DIM`         | `1024`                                               | Must match pgvector column dimension     |
| `EMBEDDING_API_BASE`    | —                                                    | Optional embedding endpoint base URL     |
| `EMBEDDING_API_KEY`     | —                                                    | Optional embedding API key               |
| `SPIDER2_SNOW_JSONL`    | `./data/spider2-snow.jsonl`                          | Benchmark tasks file                     |
| `GRPO_POLICY_BACKEND`   | `noop`                                               | `noop` for artifact-only runs; `verl` after GPU trainer wiring |

---

## Development

```bash
# Run tests
make test

# Lint
make lint

# Type check
make typecheck

# Stop and clean infrastructure
make infra-down
make clean
```

---

## Evaluation protocol (proposal §8)

| Metric                    | Definition                                                         |
|---------------------------|--------------------------------------------------------------------|
| Execution accuracy (EX)   | `κ([[y]]_D) = κ([[y*]]_D)` as multisets (instance equivalence)    |
| EX Pass@k                 | Any of k candidates correct                                        |
| Static validity rate      | Fraction passing all gates in §5.2                                 |
| Execution success rate    | Fraction that complete without runtime error                       |
| Retrieval hit rate        | Gold-relevant tables/columns in top-k schema context              |
| Reward–result correlation | Pearson r between R(τ) and EX (Property 1 verification)           |
| Skill-evolution gain      | ΔEX from evolved vs. frozen SqlSkillBank                          |
| Prompt footprint          | Tokens: schema / skills / task                                     |

Ablations mirror the four analysis questions of Xia et al. [2026]:
- **A1**: Remove hierarchy (specific-only skills)
- **A2**: Replace SqlSkillBank with raw trajectories
- **A3**: Remove cold-start SFT
- **A4**: Remove recursive evolution

---

## References

- Xia et al. (2026). SkillRL: Evolving agents via recursive skill-augmented RL. arXiv:2602.08234.
- Yu et al. (2025). Arctic-Text2SQL-R1. arXiv:2505.20315. Snowflake.
- Lei et al. (2024). Spider 2.0. arXiv:2411.07763. ICLR 2025.
- Shao et al. (2024). DeepSeekMath / GRPO. arXiv:2402.03300.
