"""SkillSQL-RL command-line interface.

Commands
--------
    skillsql connectors             list registered datasource connectors
    skillsql init-db                create catalog schema + pgvector
    skillsql catalog-build          discover + persist a datasource catalog
    skillsql schema-context "<q>"   show retrieved schema for a question
    skillsql generate "<q>"         Arctic single-shot SQL generation
    skillsql verify "<sql>"         static gates (+ optional execution)
    skillsql score "<q>" "<sql>"    composite verifier reward (optional --gold)
    skillsql run "<q>"              end-to-end SkillSQL-RL workflow
    skillsql benchmark              run Spider-2.0-Snow benchmark
    skillsql serve                  start the main FastAPI server (app.main:app)

All business-logic calls route through ``app.services.catalog`` so that the CLI,
the HTTP API, and scripts all exercise exactly the same code paths.
"""

from __future__ import annotations

import json

import typer
from app.observability.logging import configure_logging, get_logger

logger = get_logger(__name__)

app = typer.Typer(add_completion=False, help="Enterprise Text-to-SQL on Google ADK 2.0.")


def _echo(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.callback()
def _main() -> None:
    configure_logging()


# ── Framework inspection ───────────────────────────────────────────────────────

@app.command("connectors")
def connectors() -> None:
    """List registered datasource connectors (abstract factory registry)."""
    from .connectors.factory import ConnectorFactory, _ensure_registered

    _ensure_registered()
    _echo({"connectors": ConnectorFactory.available()})


# ── Cataloging plane ───────────────────────────────────────────────────────────

@app.command("init-db")
def init_db(reset: bool = typer.Option(False, help="Drop and recreate catalog tables.")) -> None:
    """Create the catalog schema, pgvector extension, and all tables (idempotent)."""
    from app.services.catalog import init_db as _init_db

    _echo(_init_db(reset=reset))


@app.command("catalog-build")
def catalog_build(
    source_type: str = typer.Option(None, help="Override DATASOURCE_TYPE."),
    source_name: str = typer.Option("default"),
    source_group_id: str = typer.Option(None, help="Existing source group UUID."),
    source_group_name: str = typer.Option(None, help="Logical source group name."),
    catalog_names: str = typer.Option(
        None,
        help="Comma-separated catalog names (Starburst; default: all).",
    ),
    db_schema: str = typer.Option(None, help="Restrict to one schema."),
    profile: bool = typer.Option(True, help="Sample distinct column values."),
    describe: bool = typer.Option(
        False,
        help=(
            "Deprecated for generic catalog builds; use app table/column description "
            "sync endpoints for NL descriptions."
        ),
    ),
    seed_skills: bool = typer.Option(
        False,
        help="Insert curated general_sql and dialect SkillBank seeds after build.",
    ),
) -> None:
    """Discover a datasource and persist its semantic catalog to Postgres."""
    from app.services.catalog import build_catalog_sync, seed_skillbank

    print("Building catalog...")

    cats = [c.strip() for c in catalog_names.split(",")] if catalog_names else None
    _echo(build_catalog_sync(
        source_type=source_type,
        source_name=source_name,
        source_group_id=source_group_id,
        source_group_name=source_group_name,
        catalog_names=cats,
        db_schema=db_schema,
        profile=profile,
        describe=describe,
    ))
    if seed_skills:
        _echo({"step": "seed_skills", **seed_skillbank()})


@app.command("schema-context")
def schema_context(
    question: str,
    k: int = 15,
    source_id: str = typer.Option(None),
) -> None:
    """Show the top-k schema docs retrieved for a question."""
    from app.services.catalog import get_schema_context

    _echo(get_schema_context(question, source_id=source_id, k=k))


# ── Inference plane ────────────────────────────────────────────────────────────

@app.command("generate")
def generate(question: str, source_id: str = typer.Option(None)) -> None:
    """Generate a single SQL candidate with the Arctic agent."""
    import asyncio

    from app.services.catalog import generate_sql

    _echo(asyncio.run(generate_sql(question, source_id=source_id)))


@app.command("verify")
def verify(
    sql: str,
    source_id: str = typer.Option(None),
    execute: bool = typer.Option(True),
) -> None:
    """Run static-lattice gates (and optionally execute) a SQL candidate."""
    from app.services.catalog import verify_sql

    _echo(verify_sql(sql, source_id=source_id, execute=execute))


@app.command("score")
def score(
    question: str,
    sql: str,
    gold: str = typer.Option(None, help="Gold SQL for equivalence check."),
    source_id: str = typer.Option(None),
) -> None:
    """Compute composite verifier reward R(τ) for a candidate SQL."""
    import asyncio

    from app.services.catalog import score_sql

    _echo(asyncio.run(score_sql(question, sql, gold_sql=gold, source_id=source_id)))


@app.command("run")
def run(question: str, source_id: str = typer.Option(None)) -> None:
    """Run the end-to-end SkillSQL-RL workflow (retrieve → generate → verify/select)."""
    import asyncio

    from app.services.catalog import run_text2sql

    _echo(asyncio.run(run_text2sql(question, source_id=source_id)))


# ── Benchmark ─────────────────────────────────────────────────────────────────

@app.command("benchmark")
def benchmark(
    jsonl: str = typer.Option(None, help="Path to spider2-snow.jsonl."),
    limit: int = typer.Option(None, help="Cap the number of tasks (debugging)."),
    group_size: int = typer.Option(None, help="Candidates per question (G)."),
    oracle_tables: bool = typer.Option(
        False,
        help="Flag oracle-table runs (never compare to leaderboard).",
    ),
    output_dir: str = typer.Option("./outputs/spider2_snow"),
) -> None:
    """Run the Spider-2.0-Snow benchmark and write evaluator artifacts."""
    from .benchmark.run_benchmark import run_benchmark_sync

    _echo(run_benchmark_sync(
        jsonl_path=jsonl,
        limit=limit,
        group_size=group_size,
        oracle_tables=oracle_tables,
        output_dir=output_dir,
    ))


# ── Server ─────────────────────────────────────────────────────────────────────

@app.command("serve")
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = typer.Option(False, help="Hot-reload (dev only)."),
) -> None:
    """Start the FastAPI server (app.main:app — all routes)."""
    import uvicorn
    logger.info(f"Serving FastAPI at http://{host}:{port}") 
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
