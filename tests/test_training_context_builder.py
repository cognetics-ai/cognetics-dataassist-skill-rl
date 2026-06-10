from __future__ import annotations

import uuid
from types import SimpleNamespace


def test_build_context_prefers_generated_schema_and_query_history_context(monkeypatch):
    import app.services.catalog as catalog_svc
    from skillsql.context import builder

    source_id = uuid.uuid4()
    calls: dict[str, object] = {}

    def fake_generate_context(question, **kwargs):
        calls["context"] = {"question": question, **kwargs}
        return {
            "context": (
                "## In-Context SQL Examples\n"
                "SQL example\n\n"
                "## Relevant Tables and Columns\n"
                "table context"
            )
        }

    monkeypatch.setattr(catalog_svc, "generate_context", fake_generate_context)
    monkeypatch.setattr(builder, "retrieve_skills", lambda *args, **kwargs: [])

    context = builder.build_context(
        "What is population of New Jersey?",
        "snowflake",
        repo=SimpleNamespace(settings=SimpleNamespace()),
        embedder=lambda texts: [[0.0] for _ in texts],
        source_id=source_id,
    )

    assert "## In-Context SQL Examples" in context["schema_context"]
    assert "## Schema Context\n## In-Context SQL Examples" in context["full_prompt"]
    assert calls["context"] == {
        "question": "What is population of New Jersey?",
        "source_id": str(source_id),
        "schema_k": 15,
        "query_k": 5,
    }
