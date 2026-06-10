#!/usr/bin/env python3
"""GRPO training driver (Algorithm 1, proposal Section 4.4).

Orchestrates the full SkillSQL-RL training pipeline:
  1. Build / verify the semantic catalog
  2. Load training tasks (Spider-2.0-Snow or a local JSONL)
  3. Cold-start SFT data generation (teacher generates skill-augmented traces)
  4. GRPO loop with recursive skill evolution
  5. Write training metrics and the evolved SqlSkillBank manifest

Requires: Ollama (SQL generator), Postgres + pgvector (catalog), and
optionally vLLM + verl for GPU-accelerated weight updates.  Without verl, the
policy update is a no-op (useful for testing the reward/evolution pipeline).

Usage:
    python scripts/train_grpo.py [OPTIONS]

Options:
    --epochs N          Training epochs (default: 3)
    --group-size G      Candidates per task (G, default: 8)
    --limit N           Cap tasks for debugging
    --output-dir DIR    Write metrics + manifests here
    --no-weight-update  Skip policy gradient (eval-only / dry-run)
    --policy-backend    Policy updater: noop or verl
    --jsonl PATH        Training tasks JSONL (default: $SPIDER2_SNOW_JSONL)

Examples:
    python scripts/train_grpo.py --epochs 1 --group-size 4 --limit 50 --no-weight-update
    python scripts/train_grpo.py --epochs 3 --group-size 8 --output-dir ./outputs/training
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillsql.config.env_loader import load_root_env
load_root_env()
from skillsql.observability.logging import configure_logging, get_logger
configure_logging(force=True)
log = get_logger("train_grpo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training for SkillSQL-RL")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Cap training tasks")
    p.add_argument("--output-dir", default="./outputs/checkpoints")
    p.add_argument("--no-weight-update", action="store_true", help="Dry-run without gradient step")
    p.add_argument("--policy-backend", choices=["noop", "verl"], default=None)
    p.add_argument("--jsonl", default=None, help="Training tasks JSONL path")
    return p.parse_args()


def _make_generate_fn():
    """Return a prompt-conditioned generate_fn backed by the SQL generator model.

    In the training track (vLLM/verl), replace this with a vLLM rollout engine.
    """
    import asyncio

    from app.adk.skillsql_runner import run_agent_once
    from skillsql.agents.sql_generator.agent import get_agent
    from skillsql.workflow._adk import clean_sql

    def generate(question: str, prompt: str) -> str:
        try:
            raw = asyncio.run(run_agent_once(get_agent(), prompt))
            return clean_sql(raw)
        except Exception as e:  # noqa: BLE001
            log.warning("generate_failed", error=str(e))
            return ""

    return generate


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from skillsql.benchmark.spider2_loader import load_spider2_snow
    from skillsql.catalog.source_resolver import resolve_source_for_database
    from skillsql.config.settings import get_settings
    from skillsql.resources import get_resources, set_active_source
    from skillsql.rl.grpo import GRPOConfig, run_grpo
    from skillsql.rl.policy_update import build_policy_update_fn
    from skillsql.rl.rollout import TaskSpec
    from skillsql.verification.reward import RewardConfig

    s = get_settings()
    jsonl_path = args.jsonl or s.SPIDER2_SNOW_JSONL
    log.info("training_start", epochs=args.epochs, group_size=args.group_size, jsonl=jsonl_path)

    # Load tasks
    tasks_raw = load_spider2_snow(jsonl_path, limit=args.limit)
    log.info("tasks_loaded", n=len(tasks_raw))

    # Build context prompts for all tasks upfront
    res = get_resources()
    from skillsql.context.builder import build_context

    task_specs: list[TaskSpec] = []
    for t in tasks_raw:
        source_id = (
            resolve_source_for_database(res.repo, t.db_id, source_type=s.DATASOURCE_TYPE)
            or resolve_source_for_database(res.repo, t.db_id)
        )
        if source_id:
            set_active_source(source_id)
        try:
            ctx = build_context(
                t.question,
                res.connector.dialect,
                repo=res.repo,
                embedder=res.embedder,
                source_id=source_id,
            )
            prompt = ctx["full_prompt"]
        except Exception as e:  # noqa: BLE001
            log.warning("context_build_failed", task=t.instance_id, error=str(e))
            prompt = f"Question: {t.question}\n\nWrite a SQL query:"
        task_specs.append(TaskSpec(
            task_id=t.instance_id,
            question=t.question,
            prompt=prompt,
            gold_sql=t.gold_sql,
            source_id=source_id,
        ))

    generate_fn = _make_generate_fn()
    policy_backend = "noop" if args.no_weight_update else (args.policy_backend or s.GRPO_POLICY_BACKEND)
    update_fn = build_policy_update_fn(
        backend=policy_backend,
        output_dir=args.output_dir,
    )

    grpo_cfg = GRPOConfig(
        group_size=args.group_size,
        epochs=args.epochs,
    )

    t0 = time.perf_counter()
    epoch_stats = run_grpo(
        task_specs,
        generate_fn,
        policy_update_fn=update_fn,
        grpo_config=grpo_cfg,
        reward_config=RewardConfig(),
    )
    elapsed = time.perf_counter() - t0

    # Write metrics
    metrics = [
        {
            "epoch": s.epoch,
            "mean_reward": round(s.mean_reward, 4),
            "execution_accuracy": round(s.execution_accuracy, 4),
            "elapsed_s": round(s.elapsed_s, 1),
            "skills_added": s.evolution.skills_promoted if s.evolution else 0,
            "update_loss": s.update_metrics.get("loss"),
            "weights_updated": s.update_metrics.get("weights_updated", False),
            "artifact_batch_path": s.update_metrics.get("artifact_batch_path"),
            "artifact_manifest_path": s.update_metrics.get("artifact_manifest_path"),
        }
        for s in epoch_stats
    ]
    metrics_path = out / "training_metrics.jsonl"
    with metrics_path.open("w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")

    # Write manifest
    final = epoch_stats[-1] if epoch_stats else None
    manifest = {
        "epochs": args.epochs,
        "group_size": args.group_size,
        "tasks": len(task_specs),
        "total_elapsed_s": round(elapsed, 1),
        "final_execution_accuracy": round(final.execution_accuracy, 4) if final else None,
        "final_mean_reward": round(final.mean_reward, 4) if final else None,
        "weight_updates_applied": any(s.update_metrics.get("weights_updated") for s in epoch_stats),
        "policy_backend": policy_backend,
        "metrics_file": str(metrics_path),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))
    log.info("training_done", **manifest)


if __name__ == "__main__":
    main()
