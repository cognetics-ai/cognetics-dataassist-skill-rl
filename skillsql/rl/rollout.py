"""Trajectory collection (Algorithm 1, lines 4–8).

Rolls out ``base_policy`` (or the current policy) over a pool of tasks and
separates the resulting trajectories into successes (T+) and failures (T-) using
the thresholds ``tau_success`` and ``tau_fail`` from :class:`RewardConfig`.

This module is backend-agnostic: it calls the connector and verifier directly and
does NOT import any LLM-generation library -- the caller supplies a ``generate_fn``
that wraps whatever inference backend is in use (Ollama, vLLM, etc.).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from ..connectors.base import DataSourceConnector, ExecResult
from ..resources import get_resources
from ..verification.reward import RewardBreakdown, RewardConfig, compute_reward_sync

# Type alias: (question, prompt) -> SQL string
GenerateFn = Callable[[str, str], str]


@dataclass
class Trajectory:
    """One complete Text-to-SQL attempt."""

    task_id: str
    question: str
    prompt: str
    sql: str
    reward: RewardBreakdown
    source_id: uuid.UUID | None = None
    gold_exec: ExecResult | None = None

    @property
    def is_success(self) -> bool:
        return self.reward.is_success

    @property
    def reward_total(self) -> float:
        return self.reward.total


@dataclass
class RolloutResult:
    """Outcome of a full rollout epoch over a task pool."""

    successes: list[Trajectory] = field(default_factory=list)  # T+
    failures: list[Trajectory] = field(default_factory=list)   # T-
    other: list[Trajectory] = field(default_factory=list)      # between thresholds

    @property
    def all_trajectories(self) -> list[Trajectory]:
        return self.successes + self.failures + self.other

    def summary(self) -> dict[str, float]:
        total = len(self.all_trajectories)
        if total == 0:
            return {"success_rate": 0.0, "failure_rate": 0.0, "n": 0}
        return {
            "n": total,
            "success_rate": len(self.successes) / total,
            "failure_rate": len(self.failures) / total,
            "mean_reward": sum(t.reward_total for t in self.all_trajectories) / total,
        }


@dataclass
class TaskSpec:
    """Minimal specification of a Text-to-SQL training task."""

    task_id: str
    question: str
    prompt: str                         # assembled by context.builder
    gold_sql: str | None = None
    source_id: uuid.UUID | None = None


def rollout_tasks(
    tasks: list[TaskSpec],
    generate_fn: GenerateFn,
    connector: DataSourceConnector | None = None,
    config: RewardConfig | None = None,
) -> RolloutResult:
    """Run one trajectory per task and split into T+ / T-.

    In the training track, the caller will have already assembled the prompts
    (via context.builder); here we just invoke the generator, run the verifier,
    and classify the outcome.
    """
    cfg = config or RewardConfig()
    if connector is None:
        connector = get_resources().connector

    result = RolloutResult()
    for task in tasks:
        sql = generate_fn(task.question, task.prompt)
        gold_exec: ExecResult | None = None
        if task.gold_sql:
            gold_exec = asyncio.run(connector.execute(
                task.gold_sql, read_only=True, timeout_s=cfg.reward_match and 60 or 60
            ))
        breakdown = compute_reward_sync(
            question=task.question,
            sql=sql,
            connector=connector,
            gold=gold_exec,
            config=cfg,
        )
        traj = Trajectory(
            task_id=task.task_id,
            question=task.question,
            prompt=task.prompt,
            sql=sql,
            reward=breakdown,
            source_id=task.source_id,
            gold_exec=gold_exec,
        )
        if breakdown.total >= cfg.tau_success:
            result.successes.append(traj)
        elif breakdown.total <= cfg.tau_fail:
            result.failures.append(traj)
        else:
            result.other.append(traj)
    return result
