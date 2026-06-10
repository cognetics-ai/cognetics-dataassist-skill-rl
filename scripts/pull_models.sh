#!/usr/bin/env bash
# =============================================================================
# SkillSQL-RL  ·  Pull required Ollama models
# =============================================================================
# Prerequisites:  Ollama must be installed and running locally.
#   Install:  https://ollama.com/download
#   Start:    ollama serve   (or the desktop app auto-starts it)
#
# Usage:
#   ./scripts/pull_models.sh
#
# Models pulled:
#   a-kore/Arctic-Text2SQL-R1-7B  -- SQL generation (completion-only, no tools)
#   snowflake-arctic-embed:l      -- schema/skill embedding (1024 dims)
#   llama3.1:8b                   -- tool-capable agent (catalog/verify assist)
# =============================================================================
set -euo pipefail

OLLAMA_BASE="${OLLAMA_API_BASE:-http://localhost:11434}"

# Check Ollama is reachable before attempting pulls
if ! curl -sf "${OLLAMA_BASE}/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Ollama is not reachable at ${OLLAMA_BASE}"
    echo "  Install Ollama: https://ollama.com/download"
    echo "  Then start it:  ollama serve"
    exit 1
fi

echo "Ollama is running at ${OLLAMA_BASE}"
echo ""

echo "1/3  Pulling Arctic Text2SQL-R1-7B (SQL generator)..."
ollama pull a-kore/Arctic-Text2SQL-R1-7B

echo "2/3  Pulling snowflake-arctic-embed:l (embedding model, 1024 dims)..."
ollama pull snowflake-arctic-embed:l

echo "3/3  Pulling llama3.1:8b (tool-capable agent model)..."
ollama pull llama3.1:8b

echo ""
echo "All models ready. Verify with:  ollama list"
