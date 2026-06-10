from __future__ import annotations

from app.main import app


def test_catalog_query_history_routes_are_registered_under_catalog():
    routes = {(route.path, tuple(sorted(getattr(route, "methods", []) or []))) for route in app.routes}
    paths = {path for path, _methods in routes}

    assert "/catalog/query-history/sync" in paths
    assert "/catalog/query-history/nlp-history/sync" in paths
    assert "/catalog/live-feedback/skills/sync" in paths
    assert "/context/backend-metadata/query-history/sync" not in paths
    assert "/context/backend-metadata/query-history/nlp-history/sync" not in paths
