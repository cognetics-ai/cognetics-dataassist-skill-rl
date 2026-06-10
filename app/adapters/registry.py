from __future__ import annotations

from app.adapters.base import EngineAdapter
from app.adapters.connector_engine_adapter import connector_adapter_from_settings
from app.adapters.mock import MockEngineAdapter
from app.adapters.starburst_trino_adapter import StarburstTrinoAdapter
from app.config import Settings


class EngineRegistry:
    def __init__(self, settings: Settings):
        starburst = StarburstTrinoAdapter(settings)
        postgres = connector_adapter_from_settings("postgres", settings)

        self._adapters: dict[str, EngineAdapter] = {
            "mock": MockEngineAdapter(),
            "starburst": starburst,
            "trino": starburst,
            "snowflake": connector_adapter_from_settings("snowflake", settings),
            "postgres": postgres,
            "sql": postgres,
        }

    def get(self, engine_name: str) -> EngineAdapter:
        key = (engine_name or "").strip().lower()
        if key not in self._adapters:
            raise KeyError(f"Unsupported engine '{engine_name}'.")
        return self._adapters[key]

    def available(self) -> list[str]:
        return sorted(self._adapters)
