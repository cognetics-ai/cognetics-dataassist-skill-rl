from __future__ import annotations

from app.adk.runtime import AdkDataAssistRuntime
from app.adk.skillsql_runner import _adk_state_connect_args, _adk_state_db_url


def test_runtime_adk_state_url_strips_libpq_options_and_uses_connect_args():
    url = AdkDataAssistRuntime._build_adk_state_db_url(
        "postgresql://user:pw@localhost:5432/adk_db?options=-csearch_path=old,public",
        "adk_store",
    )

    assert url == "postgresql+asyncpg://user:pw@localhost:5432/adk_db"
    assert AdkDataAssistRuntime._build_adk_state_connect_args("adk_store") == {
        "server_settings": {"search_path": "adk_store,public"}
    }


def test_skillsql_runner_adk_state_url_strips_options_and_uses_connect_args():
    url = _adk_state_db_url(
        "postgresql+psycopg://user:pw@localhost/adk_db?ssl=true&options=-csearch_path=old",
        "workflow_store",
    )

    assert url == "postgresql+asyncpg://user:pw@localhost/adk_db?ssl=true"
    assert _adk_state_connect_args("workflow_store") == {
        "server_settings": {"search_path": "workflow_store,public"}
    }
