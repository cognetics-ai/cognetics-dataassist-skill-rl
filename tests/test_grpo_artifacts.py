from __future__ import annotations

import json
import uuid


def test_normalize_grpo_record_uses_stable_artifact_schema():
    from skillsql.rl.artifacts import SCHEMA_VERSION, normalize_grpo_record

    source_id = uuid.uuid4()
    record = normalize_grpo_record(
        {
            "task_id": "task-1",
            "question": "How many accounts?",
            "prompt": "prompt text",
            "sql": "SELECT COUNT(*) FROM account",
            "reward": "0.5",
            "advantage": "1.25",
            "source_id": source_id,
            "candidate_index": 2,
        },
        epoch=3,
    )

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["epoch"] == 3
    assert record["task_id"] == "task-1"
    assert record["group_id"] == "task-1"
    assert record["candidate_index"] == 2
    assert record["source_id"] == str(source_id)
    assert record["response"] == "SELECT COUNT(*) FROM account"
    assert record["sql"] == "SELECT COUNT(*) FROM account"
    assert record["reward"] == 0.5
    assert record["advantage"] == 1.25


def test_write_grpo_batch_artifact_writes_jsonl_and_manifest(tmp_path):
    from skillsql.rl.artifacts import write_grpo_batch_artifact

    summary = write_grpo_batch_artifact(
        [
            {
                "task_id": "task-1",
                "group_id": "task-1",
                "candidate_index": 0,
                "question": "q",
                "prompt": "p",
                "sql": "SELECT 1",
                "reward": 1.0,
                "advantage": 0.5,
            },
            {
                "task_id": "task-1",
                "group_id": "task-1",
                "candidate_index": 1,
                "question": "q",
                "prompt": "p",
                "sql": "SELECT 2",
                "reward": -0.5,
                "advantage": -0.5,
            },
        ],
        output_dir=tmp_path,
        epoch=1,
        metadata={"policy_backend": "noop"},
    )

    batch_path = tmp_path / "grpo_batch_epoch_0001.jsonl"
    manifest_path = tmp_path / "grpo_batch_epoch_0001.manifest.json"

    assert summary.records == 2
    assert summary.batch_path == str(batch_path)
    assert summary.manifest_path == str(manifest_path)
    assert summary.reward_mean == 0.25
    assert batch_path.exists()
    assert manifest_path.exists()

    records = [json.loads(line) for line in batch_path.read_text().splitlines()]
    assert [row["candidate_index"] for row in records] == [0, 1]
    assert [row["sql"] for row in records] == ["SELECT 1", "SELECT 2"]

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "skillsql-grpo-batch-v1"
    assert manifest["records"] == 2
    assert manifest["metadata"]["policy_backend"] == "noop"
