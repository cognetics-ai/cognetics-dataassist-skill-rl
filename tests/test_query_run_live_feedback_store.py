from __future__ import annotations


async def test_query_runs_persist_live_feedback_metadata(tmp_path):
    from app.core.store import SQLStore

    store = SQLStore(
        backend="sqlite",
        sqlite_path=str(tmp_path / "runs.db"),
        embedding_dimension=4,
    )
    run = await store.create_run(
        soeid="user1",
        engine="snowflake",
        submitted_text="What is population of New Jersey?",
        input_mode="nl",
        submitted_prompt="What is population of New Jersey?",
        natural_language_query="What is population of New Jersey?",
        source_id="290b30e8-1833-41a8-a4bd-c2909184d7b3",
    )
    await store.update_run(
        run.run_id,
        final_sql="select population from census.public.state_population",
        status="failed",
        error_message="unknown column population",
        reward_json={"total": -0.35, "stage": "bind"},
    )

    hydrated = await store.get_run(run.run_id)
    candidates = await store.list_runs_for_skill_evolution(limit=10)

    assert hydrated is not None
    assert hydrated.source_id == "290b30e8-1833-41a8-a4bd-c2909184d7b3"
    assert hydrated.reward_json == {"total": -0.35, "stage": "bind"}
    assert [candidate.run_id for candidate in candidates] == [run.run_id]
