from __future__ import annotations

import uuid

from tests._fakes import FakeConnector


def test_rollout_tasks_uses_sync_reward_wrapper():
    from skillsql.rl.rollout import TaskSpec, rollout_tasks
    from skillsql.verification.reward import RewardBreakdown

    result = rollout_tasks(
        [TaskSpec(task_id="t1", question="q", prompt="p")],
        generate_fn=lambda question, prompt: "SELECT 1",
        connector=FakeConnector(),
    )

    assert len(result.all_trajectories) == 1
    assert isinstance(result.all_trajectories[0].reward, RewardBreakdown)
    assert isinstance(result.all_trajectories[0].reward.total, float)


def test_build_grpo_batch_uses_sync_reward_wrapper():
    from skillsql.rl.grpo import build_grpo_batch
    from skillsql.rl.rollout import TaskSpec
    from skillsql.verification.reward import RewardConfig

    source_id = uuid.uuid4()
    batch = build_grpo_batch(
        [TaskSpec(task_id="t1", question="q", prompt="p", source_id=source_id)],
        generate_fn=lambda question, prompt: "SELECT 1",
        connector=FakeConnector(),
        group_size=2,
        reward_config=RewardConfig(),
    )

    assert [row["reward"] for row in batch] == [batch[0]["reward"], batch[1]["reward"]]
    assert all(isinstance(row["reward"], float) for row in batch)
    assert all(row["question"] == "q" for row in batch)
    assert all(row["source_id"] == source_id for row in batch)
    assert [row["group_id"] for row in batch] == ["t1", "t1"]
    assert [row["candidate_index"] for row in batch] == [0, 1]
