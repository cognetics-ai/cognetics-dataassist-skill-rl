# SkillSQL-RL -- unified development Makefile
# Usage: make <target>
#
# All commands use the project-local .venv to guarantee the correct version
# of skillsql / uvicorn / python is called, regardless of what is active in
# the shell's PATH or conda environment.

.DEFAULT_GOAL := help
PYTHON        := .venv/bin/python
UV            := uv
SKILLSQL      := .venv/bin/skillsql   # always use the project-local install
PORT          ?= 8000

.PHONY: help install install-dev install-training install-plotting \
        setup-local check-postgres check-ollama \
        infra-pg infra-pg-down infra-ollama models \
        init-db reset-catalog-db catalog-build schema-context \
        serve serve-reload \
        generate verify score run \
        benchmark benchmark-oracle \
        train-grpo check-training-env plot-results \
        test lint typecheck \
        clean

# ──────────────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "SkillSQL-RL Makefile targets"
	@echo "────────────────────────────────────────────────────────"
	@echo "  install            Install core dependencies"
	@echo "  install-dev        Install + dev extras (test, lint)"
	@echo "  install-training   Install + vLLM/verl training extras"
	@echo "  install-plotting   Install + matplotlib/seaborn"
	@echo ""
	@echo "  setup-local        First-time local setup: check PG/Ollama, init-db, pull models"
	@echo "  check-postgres     Verify local Postgres is reachable"
	@echo "  check-ollama       Verify local Ollama is reachable at OLLAMA_API_BASE"
	@echo ""
	@echo "  models             Pull Ollama models (Arctic + embeddings)"
	@echo "  infra-pg           Start Postgres via Docker (only if no local Postgres)"
	@echo "  infra-pg-down      Stop Postgres Docker container"
	@echo ""
	@echo "  init-db            Create catalog schema + pgvector extension"
	@echo "  reset-catalog-db   Drop/recreate catalog tables (destructive)"
	@echo "  catalog-build      Discover datasource and persist catalog"
	@echo "  schema-context     Show retrieved schema for a question"
	@echo ""
	@echo "  serve              Start FastAPI (production)"
	@echo "  serve-reload       Start FastAPI with hot-reload (dev)"
	@echo ""
	@echo "  generate Q='...'   Single-shot Arctic SQL generation"
	@echo "  verify SQL='...'   Run static gates + execution check"
	@echo "  score Q='...' SQL='...'  Composite verifier reward"
	@echo "  run Q='...' [SOURCE_ID='uuid']  End-to-end Text-to-SQL workflow"
	@echo ""
	@echo "  benchmark          Spider-2.0-Snow benchmark (non-oracle)"
	@echo "  benchmark-oracle   Spider-2.0-Snow with oracle tables (flagged)"
	@echo ""
	@echo "  train-grpo         Run GRPO rollouts; verl backend requires GPU"
	@echo "  check-training-env Show whether this host can run verl/vLLM training"
	@echo "  plot-results       Generate paper graphs from benchmark outputs"
	@echo ""
	@echo "  test               Run test suite"
	@echo "  lint               Ruff lint"
	@echo "  typecheck          Mypy type check"
	@echo "  clean              Remove build artifacts"
	@echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Installation
# ──────────────────────────────────────────────────────────────────────────────
install:
	$(UV) pip install -e .

install-dev:
	$(UV) pip install -e ".[dev]"

install-training:
	$(UV) pip install -e ".[training]"

install-plotting:
	$(UV) pip install -e ".[plotting]"

# ──────────────────────────────────────────────────────────────────────────────
# Local setup (no Docker required — Postgres and Ollama assumed local)
# ──────────────────────────────────────────────────────────────────────────────

check-postgres:
	@echo "Checking local Postgres..."
	@pg_isready -h localhost -p 5432 && echo "  Postgres OK" || \
	  (echo "  Postgres not reachable on localhost:5432."; \
	   echo "  Start it locally or run: make infra-pg"; exit 1)

check-ollama:
	@OLLAMA_BASE=$${OLLAMA_API_BASE:-http://localhost:11434}; \
	echo "Checking Ollama at $${OLLAMA_BASE}..."; \
	curl -sf "$${OLLAMA_BASE}/api/tags" > /dev/null && echo "  Ollama OK" || \
	  (echo "  Ollama not reachable at $${OLLAMA_BASE}."; \
	   echo "  Start it: ollama serve  (or install from https://ollama.com)"; exit 1)

setup-local: check-postgres check-ollama
	@echo ""
	@echo "1. Initialising catalog schema..."
	$(MAKE) init-db
	@echo ""
	@echo "2. Pulling Ollama models..."
	$(MAKE) models
	@echo ""
	@echo "Local setup complete. Next: make catalog-build"

# ── Optional Docker targets (only needed when local Postgres/Ollama unavailable)
infra-pg:
	@echo "Starting Postgres via Docker (profile: pg)..."
	docker compose --profile pg up -d postgres

infra-pg-down:
	docker compose --profile pg down

infra-ollama:
	@echo "Starting Ollama via Docker (profile: models)..."
	docker compose --profile models up -d ollama

models:
	@bash scripts/pull_models.sh

# ──────────────────────────────────────────────────────────────────────────────
# Catalog
# ──────────────────────────────────────────────────────────────────────────────
init-db:
	$(SKILLSQL) init-db

reset-catalog-db:
	$(SKILLSQL) init-db --reset

catalog-build:
	$(SKILLSQL) catalog-build

schema-context:
	@if [ -z "$(Q)" ]; then echo "Usage: make schema-context Q='your question'"; exit 1; fi
	$(SKILLSQL) schema-context "$(Q)"

# ──────────────────────────────────────────────────────────────────────────────
# API server
# ──────────────────────────────────────────────────────────────────────────────
serve:
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

serve-reload:
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
generate:
	@if [ -z "$(Q)" ]; then echo "Usage: make generate Q='your question'"; exit 1; fi
	$(SKILLSQL) generate "$(Q)"

verify:
	@if [ -z "$(SQL)" ]; then echo "Usage: make verify SQL='SELECT ...'"; exit 1; fi
	$(SKILLSQL) verify "$(SQL)"

score:
	@if [ -z "$(Q)" ] || [ -z "$(SQL)" ]; then echo "Usage: make score Q='...' SQL='...'"; exit 1; fi
	$(SKILLSQL) score "$(Q)" "$(SQL)"

run:
	@if [ -z "$(Q)" ]; then echo "Usage: make run Q='your question' [SOURCE_ID='uuid']"; exit 1; fi
	$(SKILLSQL) run "$(Q)" $(if $(SOURCE_ID),--source-id "$(SOURCE_ID)",)

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────────────
benchmark:
	$(PYTHON) scripts/run_benchmark.py \
		--jsonl $${SPIDER2_SNOW_JSONL:-./data/spider2-snow.jsonl} \
		--output-dir ./outputs/spider2_snow \
		--group-size $${BENCH_GROUP_SIZE:-8}

benchmark-oracle:
	$(PYTHON) scripts/run_benchmark.py \
		--jsonl $${SPIDER2_SNOW_JSONL:-./data/spider2-snow.jsonl} \
		--output-dir ./outputs/spider2_snow_oracle \
		--group-size $${BENCH_GROUP_SIZE:-8} \
		--oracle-tables

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
train-grpo:
	$(PYTHON) scripts/train_grpo.py \
		--epochs $${GRPO_EPOCHS:-3} \
		--group-size $${GRPO_GROUP_SIZE:-8} \
		--policy-backend $${GRPO_POLICY_BACKEND:-noop} \
		--output-dir ./outputs/checkpoints

check-training-env:
	$(PYTHON) scripts/check_training_env.py

# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────
plot-results:
	$(PYTHON) scripts/plot_results.py \
		--results-dir ./outputs \
		--output-dir ./outputs/figures

# ──────────────────────────────────────────────────────────────────────────────
# Quality
# ──────────────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	ruff check app skillsql tests scripts

typecheck:
	mypy app skillsql

# ──────────────────────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -type d -name "*.egg-info" | xargs rm -rf
	find . -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache dist build
