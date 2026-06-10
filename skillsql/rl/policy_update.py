"""Policy update boundary for GRPO training.

Candidate generation and reward computation are useful without GPU training.
This module isolates the actual actor update so dry-runs, benchmark rollouts,
and future verl/vLLM integration do not leak into production inference code.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from ..observability.logging import get_logger
from .artifacts import write_grpo_batch_artifact

PolicyBackend = Literal["noop", "verl"]
PolicyUpdateFn = Callable[[list[dict[str, Any]]], dict[str, float | int | str | bool]]

log = get_logger(__name__)


def build_policy_update_fn(
    *,
    backend: str = "noop",
    output_dir: str | None = None,
) -> PolicyUpdateFn:
    """Build the policy update function for GRPO.

    ``noop`` is the default and exercises rollout/reward/skill-evolution without
    touching model weights. ``verl`` currently validates that verl is installed
    and records a placeholder update result; the actual trainer wiring belongs
    here, not in scripts or production agents.
    """
    normalized = (backend or "noop").strip().lower()
    if normalized == "noop":
        return _with_artifact_writer(
            backend="noop",
            output_dir=output_dir,
            updater=noop_policy_update,
        )
    if normalized == "verl":
        return _with_artifact_writer(
            backend="verl",
            output_dir=output_dir,
            updater=_build_verl_policy_update(output_dir=output_dir),
        )
    raise ValueError(f"Unsupported GRPO policy backend: {backend}")


def noop_policy_update(batch: list[dict[str, Any]]) -> dict[str, float | int | str | bool]:
    return {
        "backend": "noop",
        "loss": float("nan"),
        "n_samples": len(batch),
        "weights_updated": False,
    }


def _build_verl_policy_update(*, output_dir: str | None) -> PolicyUpdateFn:
    try:
        import verl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional GPU stack
        raise RuntimeError(
            "GRPO_POLICY_BACKEND=verl requires the optional verl/vLLM training stack. "
            "Install/configure verl before enabling weight updates."
        ) from exc

    checkpoint_dir = Path(output_dir or "./outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def update(batch: list[dict[str, Any]]) -> dict[str, float | int | str | bool]:
        # Real integration point:
        # 1. Convert batch rows into verl DataProto / dataset records.
        # 2. Run actor rollout logprobs + reference KL.
        # 3. Apply GRPO update and save checkpoints.
        log.info(
            "verl_policy_update_placeholder",
            n_samples=len(batch),
            output_dir=str(checkpoint_dir),
        )
        return {
            "backend": "verl",
            "loss": 0.0,
            "n_samples": len(batch),
            "weights_updated": False,
            "note": "verl installed; trainer wiring placeholder only",
            "checkpoint_dir": str(checkpoint_dir),
        }

    return update


def _with_artifact_writer(
    *,
    backend: PolicyBackend,
    output_dir: str | None,
    updater: PolicyUpdateFn,
) -> PolicyUpdateFn:
    epoch = 0

    def update(batch: list[dict[str, Any]]) -> dict[str, float | int | str | bool]:
        nonlocal epoch
        epoch += 1
        artifact_metrics: dict[str, float | int | str | bool] = {}
        if output_dir:
            summary = write_grpo_batch_artifact(
                batch,
                output_dir=output_dir,
                epoch=epoch,
                metadata={"policy_backend": backend},
            )
            artifact_metrics = {
                "artifact_batch_path": summary.batch_path,
                "artifact_manifest_path": summary.manifest_path,
                "artifact_records": summary.records,
                "artifact_reward_mean": summary.reward_mean,
            }

        metrics = updater(batch)
        return {**metrics, **artifact_metrics}

    return update
