from __future__ import annotations

import sys

import pytest


def test_noop_policy_update_boundary_reports_no_weight_update():
    from skillsql.rl.policy_update import build_policy_update_fn

    update = build_policy_update_fn(backend="noop")
    result = update([{"reward": 0.1}, {"reward": -0.2}])

    assert result["backend"] == "noop"
    assert result["n_samples"] == 2
    assert result["weights_updated"] is False


def test_noop_policy_update_writes_artifact_when_output_dir_is_provided(tmp_path):
    from skillsql.rl.policy_update import build_policy_update_fn

    update = build_policy_update_fn(backend="noop", output_dir=str(tmp_path))
    result = update([
        {
            "task_id": "task-1",
            "group_id": "task-1",
            "candidate_index": 0,
            "question": "q",
            "prompt": "p",
            "sql": "SELECT 1",
            "reward": 0.1,
            "advantage": 0.0,
        }
    ])

    assert result["backend"] == "noop"
    assert result["artifact_records"] == 1
    assert result["artifact_reward_mean"] == 0.1
    assert (tmp_path / "grpo_batch_epoch_0001.jsonl").exists()
    assert (tmp_path / "grpo_batch_epoch_0001.manifest.json").exists()


def test_verl_policy_update_requires_optional_stack(monkeypatch):
    from skillsql.rl.policy_update import build_policy_update_fn

    monkeypatch.setitem(sys.modules, "verl", None)

    with pytest.raises(RuntimeError, match="requires the optional verl/vLLM"):
        build_policy_update_fn(backend="verl")
