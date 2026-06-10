from __future__ import annotations

from types import SimpleNamespace

from skillsql.connectors.base import ExecResult


class _Ctx:
    state: dict = {}


def test_workflow_nodes_accept_state_as_node_input():
    from skillsql.workflow.nodes import retrieve_schema_node, verify_and_select_node

    retrieve_state = {
        "question": "What is population of New Jersey as per the census?",
        "dialect": "starburst",
        "source_id": None,
    }
    verify_state = {"question": "q", "dialect": "starburst", "candidates": []}

    assert retrieve_schema_node._bind_parameters(_Ctx(), retrieve_state) == {
        "node_input": retrieve_state
    }
    assert verify_and_select_node._bind_parameters(_Ctx(), verify_state) == {
        "node_input": verify_state
    }


def test_retrieve_schema_node_uses_catalog_service_generated_context(monkeypatch):
    from app.services import catalog as catalog_svc
    from skillsql.workflow import nodes

    calls: dict[str, object] = {}

    class FakeRepo:
        def general_and_dialect_skills(self, dialect):
            calls["skills_dialect"] = dialect
            return [SimpleNamespace(title="Limit scope", principle="Use provided schema only")]

    monkeypatch.setattr(
        nodes,
        "get_resources",
        lambda: SimpleNamespace(repo=FakeRepo()),
    )

    def fake_generate_context(question, **kwargs):
        calls["context"] = {"question": question, **kwargs}
        return {
            "context": "## In-Context SQL Examples\nSQL example\n\n## Relevant Tables and Columns\nTable docs",
            "docs_retrieved": 4,
            "query_examples_retrieved": 1,
            "tables": [{"name": "account"}, {"name": "customer"}],
        }

    monkeypatch.setattr(catalog_svc, "generate_context", fake_generate_context)

    state = {
        "question": "What is population of New Jersey as per the census?",
        "dialect": "trino",
        "source_id": "290b30e8-1833-41a8-a4bd-c2909184d7b3",
        "top_k": 12,
        "query_k": 3,
    }

    result = nodes.retrieve_schema_node._func(state)

    assert result["schema_context"].startswith("## In-Context SQL Examples")
    assert result["schema_context_stats"] == {
        "docs_retrieved": 4,
        "query_examples_retrieved": 1,
        "tables": 2,
    }
    assert result["skills"] == "- Limit scope: Use provided schema only"
    assert calls["context"] == {
        "question": "What is population of New Jersey as per the census?",
        "source_id": "290b30e8-1833-41a8-a4bd-c2909184d7b3",
        "engine": None,
        "catalog": None,
        "database_name": None,
        "schema_name": None,
        "schema_k": 12,
        "query_k": 3,
    }
    assert calls["skills_dialect"] == "trino"


async def test_verify_and_select_node_awaits_async_reward(monkeypatch):
    from skillsql.workflow import nodes

    calls: dict[str, object] = {"executed": [], "rewarded": []}

    class FakeConnector:
        dialect = "trino"

        async def execute(self, sql, **kwargs):
            calls["executed"].append(sql)
            return ExecResult(columns=["c"], rows=[[sql]], row_count=1, dialect="trino")

    monkeypatch.setattr(
        nodes,
        "get_resources",
        lambda: SimpleNamespace(connector=FakeConnector()),
    )
    monkeypatch.setattr(nodes, "_known_tables", lambda source_id: None)

    async def fake_compute_reward(**kwargs):
        calls["rewarded"].append(kwargs)
        total = 0.7 if kwargs["sql"] == "SELECT 2" else 0.2
        return SimpleNamespace(
            total=total,
            stage="exec_nogold",
            equivalent=None,
            gate_report=SimpleNamespace(messages=[]),
        )

    monkeypatch.setattr(nodes, "compute_reward", fake_compute_reward)

    result = await nodes.verify_and_select_node._func(
        {
            "question": "q",
            "source_id": None,
            "candidates": ["SELECT 1", "SELECT 2"],
        }
    )

    assert result["best_sql"] == "SELECT 2"
    assert result["best_reward"] == 0.7
    assert calls["executed"] == ["SELECT 1", "SELECT 2"]
    assert [call["sql"] for call in calls["rewarded"]] == ["SELECT 1", "SELECT 2"]
    assert all(
        isinstance(result, ExecResult)
        for call in calls["rewarded"]
        for result in call["group_results"]
    )
