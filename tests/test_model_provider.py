from __future__ import annotations

from typing import Any

from app.adk.model_provider import (
    build_adk_model,
    build_query_generator_adk_model,
    query_generator_uses_tools,
    uses_vertex_adk_provider,
)
from app.config import Settings
from app.services.embeddings import EmbeddingService


class FakeLiteLlm:
    def __init__(self, *, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs


def test_default_models_use_ollama_chat_and_text2sql_split(monkeypatch):
    for key in (
        "ADK_MODEL_PROVIDER",
        "ADK_MODEL",
        "DEFAULT_CHAT_MODEL",
        "QUERY_GENERATOR_MODEL_PROVIDER",
        "QUERY_GENERATOR_MODEL",
        "OLLAMA_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    workflow_model = build_adk_model(settings, lite_llm_cls=FakeLiteLlm)
    query_model = build_query_generator_adk_model(settings, lite_llm_cls=FakeLiteLlm)

    assert settings.adk_model_provider == "ollama"
    assert workflow_model.model == "ollama_chat/llama3.1:8b"
    assert query_model.model == "ollama_chat/a-kore/Arctic-Text2SQL-R1-7B:latest"
    assert query_generator_uses_tools(settings) is False
    assert uses_vertex_adk_provider(settings) is False


def test_legacy_default_chat_and_ollama_model_envs_are_resolved(monkeypatch):
    monkeypatch.delenv("ADK_MODEL", raising=False)
    monkeypatch.delenv("QUERY_GENERATOR_MODEL", raising=False)
    monkeypatch.setenv("DEFAULT_CHAT_MODEL", "ollama_chat/llama3.1:8b")
    monkeypatch.setenv("OLLAMA_MODEL", "a-kore/Arctic-Text2SQL-R1-7B:latest")

    settings = Settings(_env_file=None)

    assert settings.adk_model == "ollama_chat/llama3.1:8b"
    assert settings.query_generator_model == "a-kore/Arctic-Text2SQL-R1-7B:latest"


def test_litellm_provider_prefixes_model_and_passes_connection_kwargs():
    settings = Settings(
        _env_file=None,
        adk_model_provider="openai",
        adk_model="gpt-4.1",
        adk_model_api_base="https://llm-proxy.example/v1",
        adk_model_api_key="test-key",
        adk_model_api_version="2026-01-01",
        adk_model_timeout_seconds=1800,
        adk_model_max_retries=1,
    )

    model = build_adk_model(settings, lite_llm_cls=FakeLiteLlm)

    assert model.model == "openai/gpt-4.1"
    assert model.kwargs == {
        "api_base": "https://llm-proxy.example/v1",
        "api_key": "test-key",
        "api_version": "2026-01-01",
        "timeout": 1800.0,
        "num_retries": 1,
    }


def test_query_generator_connection_kwargs_do_not_inherit_across_providers():
    settings = Settings(
        _env_file=None,
        adk_model_provider="openai",
        adk_model="gpt-4.1",
        adk_model_api_base="https://llm-proxy.example/v1",
        query_generator_model_timeout_seconds=1800,
        query_generator_model_max_retries=1,
        query_generator_model_provider="ollama",
        query_generator_model="a-kore/Arctic-Text2SQL-R1-7B:latest",
    )

    model = build_query_generator_adk_model(settings, lite_llm_cls=FakeLiteLlm)

    assert model.model == "ollama_chat/a-kore/Arctic-Text2SQL-R1-7B:latest"
    assert model.kwargs == {
        "api_base": settings.ollama_api_base,
        "timeout": 1800.0,
        "num_retries": 1,
    }


def test_model_timeout_and_retries_are_configurable():
    settings = Settings(
        _env_file=None,
        adk_model_provider="openai",
        adk_model="gpt-4.1",
        adk_model_timeout_seconds=45,
        adk_model_max_retries=3,
        query_generator_model_provider="openai",
        query_generator_model="gpt-4.1-mini",
        query_generator_model_timeout_seconds=90,
        query_generator_model_max_retries=4,
    )

    workflow_model = build_adk_model(settings, lite_llm_cls=FakeLiteLlm)
    query_model = build_query_generator_adk_model(settings, lite_llm_cls=FakeLiteLlm)

    assert workflow_model.kwargs["timeout"] == 45.0
    assert workflow_model.kwargs["num_retries"] == 3
    assert query_model.kwargs["timeout"] == 90.0
    assert query_model.kwargs["num_retries"] == 4


def test_vertex_provider_uses_native_adk_model_string():
    settings = Settings(
        _env_file=None,
        adk_model_provider="vertex",
        adk_model="gemini-2.5-flash",
    )

    assert build_adk_model(settings, lite_llm_cls=FakeLiteLlm) == "gemini-2.5-flash"
    assert uses_vertex_adk_provider(settings) is True


def test_embedding_defaults_and_vertex_dimension():
    local_settings = Settings(_env_file=None)
    vertex_settings = Settings(
        _env_file=None,
        embedding_provider="vertex",
        embedding_model="",
        embedding_dimension=0,
    )
    openai_settings = Settings(
        _env_file=None,
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        embedding_dimension=0,
    )

    assert local_settings.active_embedding_model == "snowflake-arctic-embed:l"
    assert local_settings.active_embedding_dimension == 1024
    assert vertex_settings.active_embedding_model == "text-embedding-005"
    assert vertex_settings.active_embedding_dimension == 768
    assert openai_settings.active_embedding_dimension == 3072


def test_ollama_embedding_uses_embedding_api_base(monkeypatch):
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[float]]:
            return {"embedding": [0.1, 0.2, 0.3]}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr("app.services.embeddings.httpx.post", fake_post)
    settings = Settings(
        _env_file=None,
        embedding_provider="ollama",
        embedding_model="test-embed",
        embedding_dimension=3,
        embedding_api_base="http://embedding-host:11434",
    )

    vector = EmbeddingService(settings)._embed_ollama_sync("sample text")

    assert vector == [0.1, 0.2, 0.3]
    assert calls[0]["url"] == "http://embedding-host:11434/api/embeddings"
    assert calls[0]["json"] == {"model": "test-embed", "prompt": "sample text"}
