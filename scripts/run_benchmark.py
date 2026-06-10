#!/usr/bin/env python3
"""Spider-2.0-Snow benchmark runner (proposal Section 8).

Runs the Text-to-SQL workflow on all 547 Spider-2.0-Snow tasks and writes:
  outputs/spider2_snow/predictions.jsonl      -- {instance_id, sql}
  outputs/spider2_snow/sql/<id>.sql           -- one file per prediction
  outputs/spider2_snow/results.jsonl          -- per-task EX (when gold known)
  outputs/spider2_snow/manifest.json          -- model/catalog/skillbank versions
  outputs/spider2_snow/figures/               -- plots (if --plot is given)

Usage:
    python scripts/run_benchmark.py [OPTIONS]

Options:
    --jsonl PATH        Path to spider2-snow.jsonl (default: $SPIDER2_SNOW_JSONL)
    --output-dir DIR    Write artifacts here (default: ./outputs/spider2_snow)
    --limit N           Process only the first N tasks (debugging)
    --group-size G      Candidates per question (default: $BENCH_GROUP_SIZE or 8)
    --oracle-tables     Flag this run as oracle-table assisted (never mix with leaderboard)
    --plot              Generate result figures after the run

Examples:
    python scripts/run_benchmark.py --limit 10
    python scripts/run_benchmark.py --oracle-tables --output-dir ./outputs/oracle
    python scripts/run_benchmark.py --plot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillsql.config.env_loader import load_root_env
load_root_env()
from skillsql.observability.logging import configure_logging, get_logger
configure_logging(force=True)
log = get_logger("run_benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spider-2.0-Snow benchmark")
    p.add_argument("--jsonl", default=None, help="Path to spider2-snow.jsonl")
    p.add_argument("--output-dir", default="./outputs/spider2_snow")
    p.add_argument("--limit", type=int, default=None, help="Cap number of tasks")
    p.add_argument("--group-size", type=int, default=None, help="Candidates per task (G)")
    p.add_argument("--oracle-tables", action="store_true", help="Flag oracle-table run")
    p.add_argument("--plot", action="store_true", help="Generate figures after run")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    from skillsql.benchmark.run_benchmark import run_benchmark

    result = await run_benchmark(
        jsonl_path=args.jsonl,
        limit=args.limit,
        group_size=args.group_size,
        oracle_tables=args.oracle_tables if args.oracle_tables else None,
        output_dir=args.output_dir,
    )
    manifest = result.get("manifest", {})
    print(json.dumps({"benchmark_result": manifest}, indent=2, default=str))

    if manifest.get("oracle_tables"):
        print("\n⚠  ORACLE-TABLE RUN: do NOT compare these scores against the standard leaderboard.")

    if args.plot:
        from scripts.plot_results import generate_figures
        generate_figures(Path(args.output_dir).parent, Path(args.output_dir) / "figures")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
