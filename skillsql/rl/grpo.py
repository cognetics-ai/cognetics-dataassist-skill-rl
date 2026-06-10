"""GRPO training loop (Algorithm 1, proposal Section 4.4, Equations 9–10).

This module implements the GRPO outer loop:

    for epoch in 1..N:
        for each task d:
            S_ret = Retrieve(d, SqlSkillBank)        [Equation 6]
            {τ_i}^G ~ π_θ(· | d, S_g, S_ret)        [sample G candidates]
            R_i = R(τ_i)                             [verifier reward, Eq. 12]
            A_i = group_normalize(R)                  [Equation 2]
            θ ← θ + ∇ J_GRPO(θ)                      [Equation 9]
        if validation epoch:
            T-_val ← failed validation trajectories
            S_new ← M_T(T-_val, SqlSkillBank)         [Equation 10]
            SqlSkillBank ← SqlSkillBank ∪ S_new

The weight update (Equation 9) requires vLLM/verl with GPU. This module
orchestrates all *logic* that is not weight-update-specific so that:
  (a) the full loop runs as-is on the verl trainer (see scripts/train_grpo.py),
  (b) the rollout / reward / evolution logic can be tested in isolation.

``PolicyUpdateFn`` is the seam: for production training it wraps verl's update;
for testing or dry runs it can be a no-op or a PPO stub.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..catalog.repository import CatalogRepository
from ..connectors.base import DataSourceConnector
from ..observability.logging import get_logger
from ..resources import get_resources
from ..verification.reward import RewardConfig, compute_reward_sync
from .distillation import RuleBasedTeacher
from .evolution import EvolutionResult, evolve_skillbank
from .rollout import GenerateFn, TaskSpec

log = get_logger(__name__)

# A function that receives a batch of (prompt, sql, reward) tuples and performs
# a gradient step. Returns updated model metrics (loss, grad_norm, etc.).
PolicyUpdateFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


def _noop_update(batch: list[dict[str, Any]]) -> dict[str, float]:
    """No-op placeholder used for dry-runs and offline evaluation."""
    return {"loss": float("nan"), "n_samples": len(batch)}


@dataclass
class GRPOConfig:
    """GRPO hyperparameters (proposal notation)."""

    group_size: int = 8          # G: candidates per task
    epochs: int = 3              # N: training epochs
    epsilon: float = 0.2         # PPO clip ratio ε
    beta_kl: float = 0.01        # KL penalty coefficient β
    evolution_threshold: float = 0.5  # δ_evo: evolve when acc(C) < this
    evolution_interval: int = 1  # validate after every N epochs
    max_tasks_per_epoch: int | None = None  # cap for debugging
    timeout_s: int = 60
    row_cap: int = 5000


@dataclass
class EpochStats:
    epoch: int
    n_tasks: int = 0
    mean_reward: float = 0.0
    execution_accuracy: float = 0.0
    evolution: EvolutionResult | None = None
    update_metrics: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0


def group_normalize_rewards(rewards: list[float]) -> list[float]:
    """Group-normalized advantages A_i (Equation 2).

    A_i = (R_i - mean(R)) / (std(R) + ε)
    """
    arr = np.array(rewards, dtype=np.float64)
    mu = arr.mean()
    sigma = arr.std()
    if sigma < 1e-8:
        return [0.0] * len(rewards)
    return ((arr - mu) / (sigma + 1e-8)).tolist()


def build_grpo_batch(
    tasks: list[TaskSpec],
    generate_fn: GenerateFn,
    connector: DataSourceConnector,
    group_size: int,
    reward_config: RewardConfig,
    context_fn: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """Sample G candidates per task and compute group-normalized advantages.

    Returns a flat list of sample dicts, each with:
      prompt, sql, reward, advantage, task_id, group_id, candidate_index
    """
    batch: list[dict[str, Any]] = []
    for task in tasks:
        prompt = context_fn(task.question) if context_fn else task.prompt
        group_id = str(task.task_id or task.question)
        candidates: list[str] = [generate_fn(task.question, prompt) for _ in range(group_size)]

        # Execute gold once to reuse across all G candidates.
        gold_exec = None
        if task.gold_sql:
            gold_exec = asyncio.run(connector.execute(
                task.gold_sql, read_only=True,
                timeout_s=reward_config.reward_match and 60 or 60,
                row_cap=reward_config.reward_match and 5000 or 5000,
            ))

        rewards = [
            compute_reward_sync(
                question=task.question,
                sql=sql,
                connector=connector,
                gold=gold_exec,
                config=reward_config,
            ).total
            for sql in candidates
        ]
        advantages = group_normalize_rewards(rewards)
        for idx, (sql, rew, adv) in enumerate(zip(candidates, rewards, advantages, strict=False)):
            batch.append({
                "task_id": task.task_id,
                "group_id": group_id,
                "candidate_index": idx,
                "question": task.question,
                "prompt": prompt,
                "sql": sql,
                "reward": rew,
                "advantage": adv,
                "source_id": task.source_id,
            })
    return batch


def run_grpo(
    tasks: list[TaskSpec],
    generate_fn: GenerateFn,
    policy_update_fn: PolicyUpdateFn | None = None,
    *,
    repo: CatalogRepository | None = None,
    connector: DataSourceConnector | None = None,
    grpo_config: GRPOConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> list[EpochStats]:
    """Run the full GRPO loop (Algorithm 1) and return per-epoch statistics.

    For production training, pass a ``policy_update_fn`` that wraps the verl
    trainer. For dry runs or inference-only benchmarking, omit it (defaults to
    a no-op that returns NaN loss).
    """
    cfg = grpo_config or GRPOConfig()
    rcfg = reward_config or RewardConfig()
    update_fn = policy_update_fn or _noop_update
    res = get_resources()
    connector = connector or res.connector
    repo = repo or res.repo

    stats: list[EpochStats] = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        epoch_tasks = tasks[:cfg.max_tasks_per_epoch] if cfg.max_tasks_per_epoch else tasks
        log.info("grpo_epoch_start", epoch=epoch, tasks=len(epoch_tasks))

        # Build GRPO batch (G candidates per task, group-normalized rewards)
        batch = build_grpo_batch(
            epoch_tasks, generate_fn, connector, cfg.group_size, rcfg
        )

        # Policy gradient update via the injected update function
        update_metrics = update_fn(batch)

        # Compute epoch-level stats
        rewards = [s["reward"] for s in batch]
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        n_correct = sum(1 for s in batch if s["reward"] >= rcfg.tau_success)
        exec_acc = n_correct / len(batch) if batch else 0.0

        # Recursive skill evolution at validation epoch
        evolution_result = None
        if epoch % cfg.evolution_interval == 0:
            log.info("grpo_evolution", epoch=epoch)
            # Collect failures from this epoch's batch
            failed_trajs = [
                s for s in batch if s["reward"] <= rcfg.tau_fail
            ]
            if exec_acc < cfg.evolution_threshold and failed_trajs:
                from .rollout import Trajectory

                # Wrap batch entries as minimal Trajectory objects for evolution
                traj_failures = [
                    Trajectory(
                        task_id=s["task_id"],
                        question=str(s.get("question") or ""),
                        prompt=s["prompt"],
                        sql=s["sql"],
                        reward=type("R", (), {  # minimal duck-typed reward
                            "is_success": False,
                            "total": s["reward"],
                            "stage": "exec_fail",
                            "gate_report": None,
                            "exec_result": None,
                        })(),
                        source_id=s.get("source_id"),
                    )
                    for s in failed_trajs
                ]
                evolution_result = evolve_skillbank(
                    traj_failures, repo, teacher=RuleBasedTeacher()
                )

        elapsed = time.perf_counter() - t0
        ep_stats = EpochStats(
            epoch=epoch,
            n_tasks=len(epoch_tasks),
            mean_reward=mean_reward,
            execution_accuracy=exec_acc,
            evolution=evolution_result,
            update_metrics=update_metrics,
            elapsed_s=elapsed,
        )
        stats.append(ep_stats)
        log.info(
            "grpo_epoch_done",
            epoch=epoch,
            mean_reward=round(mean_reward, 4),
            exec_acc=round(exec_acc, 4),
            elapsed_s=round(elapsed, 1),
            skills_added=evolution_result.skills_promoted if evolution_result else 0,
        )

    return stats
