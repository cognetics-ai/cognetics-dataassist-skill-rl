"""Single benchmark entry point (proposal Section 8).

For each Spider-2.0-Snow task: map the task DB to a catalog source, run the ADK
Text-to-SQL workflow to get a predicted query, and (when gold is available)
execute both to compute Execution Accuracy via instance equivalence. Writes:

    <out>/predictions.jsonl       {instance_id, sql}            (evaluator input)
    <out>/sql/<instance_id>.sql   one file per prediction       (evaluator input)
    <out>/results.jsonl           per-task EX (when gold known)
    <out>/manifest.json           model / catalog / skillbank / prompt versions,
                                   group size, and the ORACLE-TABLES flag

Oracle-table runs are flagged in the manifest and must never be compared against
the standard (non-oracle) leaderboard.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config.settings import get_settings
from ..catalog.source_resolver import resolve_source_for_database
from ..models.registry import model_spec_for
from ..observability.logging import get_logger
from ..resources import get_resources, set_active_source
from ..verification.equivalence import result_equivalent
from .spider2_loader import Spider2Task, load_spider2_snow

log = get_logger(__name__)


def _map_db_to_source(db_id: str) -> uuid.UUID | None:
    """Find a catalog source whose database or name matches the task's db."""
    res = get_resources()
    settings = get_settings()
    return (
        resolve_source_for_database(res.repo, db_id, source_type=settings.DATASOURCE_TYPE)
        or resolve_source_for_database(res.repo, db_id)
    )


def _manifest(tasks: int, group_size: int, oracle: bool) -> dict[str, Any]:
    s = get_settings()
    skill_count = 0
    try:
        from ..catalog.models import Skill

        with get_resources().repo.session() as sess:
            skill_count = sess.query(Skill).filter(Skill.status == "promoted").count()
    except Exception:  # noqa: BLE001
        pass
    return {
        "run_id": uuid.uuid4().hex,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasource_type": s.DATASOURCE_TYPE,
        "sql_generator_model": model_spec_for("sql_generator"),
        "embedding_model": s.EMBEDDING_MODEL,
        "embedding_dim": s.EMBEDDING_DIM,
        "group_size": group_size,
        "promoted_skill_count": skill_count,
        "prompt_version": "sql_generator/v1",
        "oracle_tables": oracle,
        "tasks": tasks,
        "metric": "execution_accuracy (instance equivalence; extra columns tolerated)",
    }


async def run_benchmark(
    *,
    jsonl_path: str | None = None,
    limit: int | None = None,
    group_size: int | None = None,
    oracle_tables: bool | None = None,
    output_dir: str = "./outputs/spider2_snow",
) -> dict[str, Any]:
    """Run the benchmark and write evaluator-compatible artifacts + a manifest."""
    from app.adk.skillsql_runner import build_runner, run_text2sql

    s = get_settings()
    jsonl_path = jsonl_path or s.SPIDER2_SNOW_JSONL
    group_size = group_size or s.BENCH_GROUP_SIZE
    oracle = oracle_tables if oracle_tables is not None else s.BENCH_ORACLE_TABLES

    out = Path(output_dir)
    (out / "sql").mkdir(parents=True, exist_ok=True)

    tasks: list[Spider2Task] = load_spider2_snow(jsonl_path, limit=limit)
    log.info("benchmark_start", tasks=len(tasks), group_size=group_size, oracle=oracle)

    runner = build_runner()  # one Runner reused across tasks
    res = get_resources()

    preds_fh = (out / "predictions.jsonl").open("w", encoding="utf-8")
    results_fh = (out / "results.jsonl").open("w", encoding="utf-8")
    n_scored = n_correct = 0
    try:
        for i, task in enumerate(tasks, 1):
            sid = _map_db_to_source(task.db_id)
            if sid:
                set_active_source(sid)
            else:
                log.warning("no_source_for_db", db=task.db_id, instance=task.instance_id)

            try:
                result = await run_text2sql(task.question, runner=runner)
                sql = result.get("sql") or ""
            except Exception as e:  # noqa: BLE001 -- one task must not abort the run
                log.error("task_failed", instance=task.instance_id, error=str(e))
                sql = ""

            preds_fh.write(json.dumps({"instance_id": task.instance_id, "sql": sql}) + "\n")
            (out / "sql" / f"{task.instance_id or i}.sql").write_text(sql, encoding="utf-8")

            record: dict[str, Any] = {"instance_id": task.instance_id, "db": task.db_id}
            if task.gold_sql and sql:
                pred_res = await res.connector.execute(
                    sql, read_only=True, timeout_s=s.SQL_STATEMENT_TIMEOUT_S, row_cap=s.SQL_ROW_CAP
                )
                gold_res = await res.connector.execute(
                    task.gold_sql, read_only=True, timeout_s=s.SQL_STATEMENT_TIMEOUT_S,
                    row_cap=s.SQL_ROW_CAP,
                )
                ex = bool(result_equivalent(pred_res, gold_res))
                record["execution_accuracy"] = ex
                n_scored += 1
                n_correct += int(ex)
            results_fh.write(json.dumps(record) + "\n")
            if i % 10 == 0:
                log.info("benchmark_progress", done=i, total=len(tasks))
    finally:
        preds_fh.close()
        results_fh.close()

    manifest = _manifest(len(tasks), group_size, oracle)
    if n_scored:
        manifest["execution_accuracy"] = n_correct / n_scored
        manifest["scored_tasks"] = n_scored
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("benchmark_done", output_dir=str(out), scored=n_scored, correct=n_correct)
    return {"output_dir": str(out), "manifest": manifest}


def run_benchmark_sync(**kwargs: Any) -> dict[str, Any]:
    import asyncio

    return asyncio.run(run_benchmark(**kwargs))
