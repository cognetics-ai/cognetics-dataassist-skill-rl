"""GRPO artifact serialization.

These artifacts are the boundary between local rollout/reward collection and a
GPU training job. They are intentionally plain JSONL so they can be inspected on
macOS, copied to a Linux GPU host, and consumed by verl/vLLM tooling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "skillsql-grpo-batch-v1"


@dataclass
class GrpoArtifactSummary:
    schema_version: str
    epoch: int
    records: int
    batch_path: str
    manifest_path: str
    reward_mean: float
    reward_min: float
    reward_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_grpo_record(row: dict[str, Any], *, epoch: int) -> dict[str, Any]:
    """Normalize one GRPO batch row into the stable JSONL artifact schema."""
    reward = _float_value(row.get("reward"), default=0.0)
    advantage = _float_value(row.get("advantage"), default=0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "task_id": str(row.get("task_id") or ""),
        "group_id": str(row.get("group_id") or row.get("task_id") or ""),
        "candidate_index": int(row.get("candidate_index") or 0),
        "source_id": str(row.get("source_id") or "") or None,
        "question": str(row.get("question") or ""),
        "prompt": str(row.get("prompt") or ""),
        "response": str(row.get("sql") or row.get("response") or ""),
        "sql": str(row.get("sql") or row.get("response") or ""),
        "reward": reward,
        "advantage": advantage,
        "metadata": dict(row.get("metadata") or {}),
    }


def write_grpo_batch_artifact(
    batch: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    epoch: int,
    metadata: dict[str, Any] | None = None,
) -> GrpoArtifactSummary:
    """Write batch JSONL plus a compact manifest and return paths/stats."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_path = out_dir / f"grpo_batch_epoch_{epoch:04d}.jsonl"
    manifest_path = out_dir / f"grpo_batch_epoch_{epoch:04d}.manifest.json"

    records = [normalize_grpo_record(row, epoch=epoch) for row in batch]
    with batch_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str) + "\n")

    rewards = [float(record["reward"]) for record in records]
    summary = GrpoArtifactSummary(
        schema_version=SCHEMA_VERSION,
        epoch=epoch,
        records=len(records),
        batch_path=str(batch_path),
        manifest_path=str(manifest_path),
        reward_mean=(sum(rewards) / len(rewards)) if rewards else 0.0,
        reward_min=min(rewards) if rewards else 0.0,
        reward_max=max(rewards) if rewards else 0.0,
    )
    manifest = {
        **summary.to_dict(),
        "metadata": metadata or {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return summary


def _float_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
