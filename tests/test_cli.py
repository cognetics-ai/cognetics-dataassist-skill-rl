from __future__ import annotations

from skillsql.cli import app
from typer.testing import CliRunner


def test_catalog_build_seed_skills_option(monkeypatch):
    from app.services import catalog as catalog_svc

    calls: dict[str, object] = {}

    def fake_build_catalog_sync(**kwargs):
        calls["build"] = kwargs
        return {"tables": 2, "docs": 4}

    def fake_seed_skillbank():
        calls["seed"] = True
        return {"inserted": 3}

    monkeypatch.setattr(catalog_svc, "build_catalog_sync", fake_build_catalog_sync)
    monkeypatch.setattr(catalog_svc, "seed_skillbank", fake_seed_skillbank)

    result = CliRunner().invoke(
        app,
        [
            "catalog-build",
            "--source-type",
            "starburst",
            "--source-name",
            "starburst",
            "--catalog-names",
            "sample",
            "--db-schema",
            "burstbank",
            "--seed-skills",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["build"] == {
        "source_type": "starburst",
        "source_name": "starburst",
        "source_group_id": None,
        "source_group_name": None,
        "catalog_names": ["sample"],
        "db_schema": "burstbank",
        "profile": True,
        "describe": False,
    }
    assert calls["seed"] is True
    assert '"step": "seed_skills"' in result.output
    assert '"inserted": 3' in result.output
