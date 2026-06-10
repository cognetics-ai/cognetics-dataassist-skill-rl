from __future__ import annotations

import os
from typing import Any

import pytest
import skillsql.models.registry as registry
from skillsql.models.providers import build_model


class FakeLiteLlm:
    def __init__(self, *, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs


def test_plain_ollama_model_is_wrapped_for_litellm(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434")

    model = build_model(
        "nemotron-3-ultra:cloud",
        provider="ollama",
        timeout_seconds=42,
        max_retries=3,
        lite_llm_cls=FakeLiteLlm,
    )

    assert model.model == "ollama_chat/nemotron-3-ultra:cloud"
    assert model.kwargs == {
        "api_base": "http://localhost:11434",
        "timeout": 42.0,
        "num_retries": 3,
    }
    assert os.environ["OLLAMA_API_BASE"] == "http://localhost:11434"


def test_provider_prefixed_model_is_wrapped_for_litellm():
    model = build_model(
        "openai/gpt-4.1",
        api_base="https://llm-proxy.example/v1",
        api_key="test-key",
        timeout_seconds=90,
        max_retries=4,
        lite_llm_cls=FakeLiteLlm,
    )

    assert model.model == "openai/gpt-4.1"
    assert model.kwargs == {
        "api_base": "https://llm-proxy.example/v1",
        "api_key": "test-key",
        "timeout": 90.0,
        "num_retries": 4,
    }


def test_native_google_model_remains_string():
    assert build_model("gemini-2.5-flash", provider="vertex", lite_llm_cls=FakeLiteLlm) == (
        "gemini-2.5-flash"
    )


def test_plain_model_without_provider_fails_fast():
    with pytest.raises(ValueError, match="missing a provider"):
        build_model("nemotron-3-ultra:cloud", lite_llm_cls=FakeLiteLlm)


def test_schema_retriever_uses_general_adk_provider_for_plain_model(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_build_model(spec: str, **kwargs: Any) -> str:
        captured["spec"] = spec
        captured["provider"] = kwargs.get("provider")
        captured["timeout_seconds"] = kwargs.get("timeout_seconds")
        captured["max_retries"] = kwargs.get("max_retries")
        return "model"

    monkeypatch.setenv("ADK_MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("ADK_MODEL", "nemotron-3-ultra:cloud")
    monkeypatch.setenv("ADK_MODEL_TIMEOUT_SECONDS", "123")
    monkeypatch.setenv("ADK_MODEL_MAX_RETRIES", "5")
    monkeypatch.delenv("SCHEMA_RETRIEVER_MODEL", raising=False)
    monkeypatch.setattr(registry, "build_model", fake_build_model)

    resolved = registry.resolve_role("schema_retriever")

    assert resolved.model == "model"
    assert resolved.model_spec == "nemotron-3-ultra:cloud"
    assert resolved.provider == "ollama"
    assert captured == {
        "spec": "nemotron-3-ultra:cloud",
        "provider": "ollama",
        "timeout_seconds": 123.0,
        "max_retries": 5,
    }


def test_sql_generator_uses_query_generator_model_not_general_adk_model(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_build_model(spec: str, **kwargs: Any) -> str:
        captured["spec"] = spec
        captured["provider"] = kwargs.get("provider")
        captured["timeout_seconds"] = kwargs.get("timeout_seconds")
        captured["max_retries"] = kwargs.get("max_retries")
        return "query-model"

    monkeypatch.setenv("ADK_MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("ADK_MODEL", "nemotron-3-ultra:cloud")
    monkeypatch.setenv("QUERY_GENERATOR_MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("QUERY_GENERATOR_MODEL", "a-kore/Arctic-Text2SQL-R1-7B:latest")
    monkeypatch.setenv("QUERY_GENERATOR_MODEL_TIMEOUT_SECONDS", "456")
    monkeypatch.setenv("QUERY_GENERATOR_MODEL_MAX_RETRIES", "6")
    monkeypatch.delenv("SQL_GEN_MODEL", raising=False)
    monkeypatch.setattr(registry, "build_model", fake_build_model)

    resolved = registry.resolve_role("sql_generator")

    assert resolved.model == "query-model"
    assert resolved.model_spec == "a-kore/Arctic-Text2SQL-R1-7B:latest"
    assert resolved.provider == "ollama"
    assert captured == {
        "spec": "a-kore/Arctic-Text2SQL-R1-7B:latest",
        "provider": "ollama",
        "timeout_seconds": 456.0,
        "max_retries": 6,
    }
