from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import create_engine

from skillsql.catalog.models import Skill
from skillsql.catalog.repository import CatalogRepository
from skillsql.config.settings import get_settings
from skillsql.rl.live_feedback import distill_live_query_runs, query_run_to_trajectory


def _repo() -> CatalogRepository:
    repo = CatalogRepository(
        settings=SimpleNamespace(
            APP_CATALOG_SCHEMA="",
            SQLALCHEMY_HIDE_PARAMETERS=True,
            APP_CATALOG_DSN="sqlite:///:memory:",
            EMBEDDING_DIM=4,
        ),
        engine=create_engine("sqlite:///:memory:", future=True),
    )
    repo.init_schema()
    return repo


def test_query_run_to_trajectory_uses_reward_and_source_id():
    repo = _repo()
    source_id = repo.upsert_source(
        "snowflake",
        "census",
        catalog_name="CENSUS_DB",
        database="CENSUS_DB",
        db_schema="PUBLIC",
    )
    run = SimpleNamespace(
        run_id="run-1",
        natural_language_query="What is population of New Jersey?",
        submitted_prompt=None,
        final_sql="select missing_col from census.public.state_population",
        submitted_sql=None,
        source_id=str(source_id),
        reward_json={"total": -0.35, "stage": "bind"},
        status="failed",
        error_message="unknown column missing_col",
        engine="snowflake",
    )

    trajectory = query_run_to_trajectory(run)

    assert trajectory is not None
    assert trajectory.task_id == "run-1"
    assert trajectory.question == "What is population of New Jersey?"
    assert trajectory.source_id == source_id
    assert trajectory.reward.total == -0.35
    assert trajectory.reward.stage == "bind"


def test_distill_live_query_runs_inserts_candidate_skill():
    repo = _repo()
    source_id = repo.upsert_source(
        "snowflake",
        "census",
        catalog_name="CENSUS_DB",
        database="CENSUS_DB",
        db_schema="PUBLIC",
    )
    run = SimpleNamespace(
        run_id="run-1",
        natural_language_query="What is population of New Jersey?",
        submitted_prompt=None,
        final_sql="select missing_col from census.public.state_population",
        submitted_sql=None,
        source_id=str(source_id),
        reward_json={"total": -0.35, "stage": "bind"},
        status="failed",
        error_message="unknown column missing_col",
        engine="snowflake",
    )

    result = distill_live_query_runs(
        [run],
        repo,
        embedder=lambda texts: [[0.1] * get_settings().EMBEDDING_DIM for _ in texts],
    )

    assert result.skills_inserted == 1
    with repo.session() as session:
        skill = session.query(Skill).one()
        assert skill.status == "candidate"
        assert skill.scope == "failure_repair"
        assert skill.source_id == source_id
        assert skill.provenance["source"] == "live_query_runs"
        assert skill.provenance["task_id"] == "run-1"
