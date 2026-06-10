"""Composite, staged verifier reward (proposal Section 5.4, Eq. 12).

Cascade (first matching case):

    -1.00                         not Safe
    -0.60                         Safe, not Parse
    -0.35                         Parse, not Bind
    -0.25 + 0.15*omega            Bind, not Exec
     1.00*[equiv] + 0.15*omega + eta*Eff   Exec, gold available
     0.10 + 0.25*omega + rho*SC   Exec, no gold

clipped to [-1.0, 1.2].

**Exact-match dominance (Property 1).** Because ``omega, Eff in [0,1]`` and
``eta <= 0.1``, the additive shaping is bounded by 0.25 < 1.0, so a correct query
(>= 1.0) always out-scores an incorrect executed one. Obligation/efficiency terms
only break ties among equally-(in)correct queries -- this is what makes the reward
resistant to hacking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..connectors.base import DataSourceConnector, ExecResult
from .equivalence import result_equivalent
from .obligations import score_obligations
from .static_gates import GateReport, run_static_lattice


@dataclass
class RewardConfig:
    obligation_weight_exec_fail: float = 0.15
    obligation_weight_match: float = 0.15
    obligation_weight_nogold: float = 0.25
    efficiency_weight: float = 0.10        # eta
    self_consistency_weight: float = 0.10  # rho
    reward_match: float = 1.0
    reward_exec_nogold_base: float = 0.10
    penalty_unsafe: float = -1.0
    penalty_parse: float = -0.6
    penalty_bind: float = -0.35
    penalty_exec_fail: float = -0.25
    clip_low: float = -1.0
    clip_high: float = 1.2
    # rollout pool thresholds for distillation (proposal Section 4.1)
    tau_success: float = 0.99
    tau_fail: float = -0.20


@dataclass
class RewardBreakdown:
    total: float
    stage: str  # unsafe|parse|bind|exec_fail|matched|exec_nogold
    obligation_score: float = 0.0
    efficiency: float = 0.0
    self_consistency: float = 0.0
    equivalent: bool | None = None
    gate_report: GateReport | None = None
    exec_result: ExecResult | None = None
    components: dict[str, float] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.equivalent is True


def _efficiency(exec_result: ExecResult, timeout_s: int) -> float:
    """Cheap normalized efficiency in [0,1] from latency vs. the timeout budget.

    A production setup should prefer bytes-scanned / partitions-pruned from the
    warehouse query profile; latency is a portable proxy used here.
    """
    if not exec_result.ok or timeout_s <= 0:
        return 0.0
    budget_ms = timeout_s * 1000.0
    return max(0.0, min(1.0, 1.0 - (exec_result.elapsed_ms / budget_ms)))


async def compute_reward(
    *,
    question: str,
    sql: str,
    connector: DataSourceConnector,
    gold: ExecResult | None = None,
    known_tables: set[str] | None = None,
    group_results: list[ExecResult] | None = None,
    timeout_s: int = 60,
    row_cap: int = 5000,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score one candidate SQL against the verifier cascade (async).

    ``gold`` is the executed reference result (None when unavailable, e.g.
    Spider tasks without a released gold answer). ``group_results`` enables the
    self-consistency bonus when no gold is present.
    """
    cfg = config or RewardConfig()
    dialect = connector.dialect

    report = run_static_lattice(sql, dialect, known_tables=known_tables)
    if report.first_failure == "safe":
        return RewardBreakdown(total=cfg.penalty_unsafe, stage="unsafe", gate_report=report)
    if report.first_failure == "parse":
        return RewardBreakdown(total=cfg.penalty_parse, stage="parse", gate_report=report)
    if report.first_failure == "bind":
        return RewardBreakdown(total=cfg.penalty_bind, stage="bind", gate_report=report)

    omega = score_obligations(question, sql, dialect).score
    exec_result = await connector.execute(sql, read_only=True, timeout_s=timeout_s, row_cap=row_cap)

    if not exec_result.ok:
        total = cfg.penalty_exec_fail + cfg.obligation_weight_exec_fail * omega
        return _finalize(
            cfg, total, "exec_fail", omega=omega, report=report, exec_result=exec_result
        )

    if gold is not None:
        equivalent = result_equivalent(exec_result, gold)
        eff = _efficiency(exec_result, timeout_s)
        total = (
            cfg.reward_match * (1.0 if equivalent else 0.0)
            + cfg.obligation_weight_match * omega
            + cfg.efficiency_weight * eff
        )
        return _finalize(
            cfg, total, "matched", omega=omega, eff=eff, equivalent=equivalent,
            report=report, exec_result=exec_result,
        )

    # No gold: bounded positive + obligations + self-consistency.
    sc = _self_consistency(exec_result, group_results)
    total = (
        cfg.reward_exec_nogold_base
        + cfg.obligation_weight_nogold * omega
        + cfg.self_consistency_weight * sc
    )
    return _finalize(
        cfg, total, "exec_nogold", omega=omega, sc=sc, report=report, exec_result=exec_result
    )


def _self_consistency(this: ExecResult, group: list[ExecResult] | None) -> float:
    if not group:
        return 0.0
    matches = sum(1 for other in group if other is not this and result_equivalent(this, other))
    denom = max(1, len(group) - 1)
    return matches / denom


def _finalize(
    cfg: RewardConfig,
    total: float,
    stage: str,
    *,
    omega: float = 0.0,
    eff: float = 0.0,
    sc: float = 0.0,
    equivalent: bool | None = None,
    report: GateReport | None = None,
    exec_result: ExecResult | None = None,
) -> RewardBreakdown:
    clipped = max(cfg.clip_low, min(cfg.clip_high, total))
    return RewardBreakdown(
        total=clipped,
        stage=stage,
        obligation_score=omega,
        efficiency=eff,
        self_consistency=sc,
        equivalent=equivalent,
        gate_report=report,
        exec_result=exec_result,
        components={"raw_total": total, "omega": omega, "efficiency": eff, "self_consistency": sc},
    )


def compute_reward_sync(**kwargs) -> RewardBreakdown:
    """Synchronous wrapper for use in scripts, CLIs, and non-async tests.

    Runs ``compute_reward`` in a new event loop.  Do NOT call from inside an
    already-running event loop (use ``await compute_reward(...)`` there).
    """
    import asyncio
    return asyncio.run(compute_reward(**kwargs))
