#!/usr/bin/env python3
"""Build (or refresh) the semantic catalog for a datasource.

Usage:
    python scripts/build_catalog.py [OPTIONS]

Options:
    --source-type   Override DATASOURCE_TYPE from .env (snowflake|starburst|postgres)
    --source-name   Human label for this catalog entry (default: "default")
    --no-profile    Skip column sampling (faster for large schemas)
    --describe      Add LLM NL descriptions (slower; requires a tool-capable model)
    --seed-skills   Insert curated general_sql / dialect seed skills after build
    --k             Number of schema docs to retrieve for a sample question (test)
    --test-q        Optional test question to verify retrieval after build

Examples:
    python scripts/build_catalog.py
    python scripts/build_catalog.py --source-type snowflake --seed-skills
    python scripts/build_catalog.py --test-q "What is total revenue by month in 2024?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to Python path when run directly (python scripts/build_catalog.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Env + logging bootstrap ────────────────────────────────────────────────────
# IMPORTANT: load_root_env() MUST be called before configure_logging() so that
# LOG_LEVEL, LOG_JSON, and SKILLSQL_ENV from .env are visible to the logger.
# configure_logging(force=True) tears down any module-import-time configuration
# that may have been established at INFO before .env was loaded.
from skillsql.config.env_loader import load_root_env

load_root_env()

from skillsql.observability.logging import configure_logging, get_logger

configure_logging(force=True)  # force=True: re-reads LOG_LEVEL now that .env is loaded
log = get_logger(__name__)     # __name__ (no quotes) = "build_catalog", not the string "__name__"

from skillsql.config.settings import get_settings



def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic catalog for a datasource")
    parser.add_argument("--source-type", default=None, help="Override DATASOURCE_TYPE")
    parser.add_argument("--source-name", default="default", help="Label for this catalog entry")
    parser.add_argument("--no-profile", action="store_true", help="Skip column value sampling")
    parser.add_argument("--describe", action="store_true", help="Add LLM NL descriptions")
    parser.add_argument("--seed-skills", action="store_true", help="Insert seed SqlSkillBank skills")
    parser.add_argument("--k", type=int, default=5, help="Top-K docs for retrieval test")
    parser.add_argument("--test-q", default=None, help="Test question for retrieval verification")
    args = parser.parse_args()

    s = get_settings()
    source_type = args.source_type or s.DATASOURCE_TYPE
    log.info("catalog_build_start....", source_type=source_type, source_name=args.source_name)
    log.debug("catalog_build_start", level=s.LOG_LEVEL if hasattr(s, "LOG_LEVEL") else "DEBUG")

    # 1. Initialize schema
    from app.services.catalog import init_db
    init_result = init_db()
    log.info("init_db", **init_result)
    print(json.dumps({"step": "init_db", **init_result}, indent=2))

    # 2. Build catalog
    from app.services.catalog import build_catalog_sync as build_catalog_api
    result = build_catalog_api(
        source_type=source_type,
        source_name=args.source_name,
        profile=not args.no_profile,
        describe=args.describe,
    )
    print(json.dumps({"step": "catalog_build", **result}, indent=2, default=str))
    log.info("catalog_built", tables=result.get("tables"), docs=result.get("docs"))

    # 3. Seed SkillBank
    if args.seed_skills:
        from skillsql.resources import get_resources
        from skillsql.skillbank.seeds import load_seeds
        n = load_seeds(get_resources().repo)
        print(json.dumps({"step": "seed_skills", "inserted": n}, indent=2))
        log.info("seeds_loaded", inserted=n)

    # 4. Verify retrieval
    if args.test_q:
        from app.services.catalog import get_schema_context as get_schema_context_api
        ctx = get_schema_context_api(args.test_q, k=args.k)
        print(json.dumps({"step": "retrieval_test", "question": args.test_q, **ctx}, indent=2))
        log.info("retrieval_ok", docs_retrieved=len(ctx.get("docs", [])))

    print("\nCatalog build complete.")


if __name__ == "__main__":
    main()
