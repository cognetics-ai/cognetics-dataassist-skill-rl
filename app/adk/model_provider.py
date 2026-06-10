from __future__ import annotations

import os
from typing import Any, Protocol

from app.config import Settings


class LiteLlmFactory(Protocol):
    def __call__(self, *, model: str, **kwargs: Any) -> Any:
        ...


_VERTEX_PROVIDERS = {"", "vertex", "gemini", "google", "google_vertex"}
_OLLAMA_PROVIDERS = {"ollama", "ollama_chat"}
_LITELLM_PASSTHROUGH_PROVIDERS = {"litellm", "lite_llm"}
_LITELLM_PROVIDER_PREFIXES = {
    "anthropic": "anthropic",
    "azure": "azure",
    "azure_openai": "azure",
    "bedrock": "bedrock",
    "cohere": "cohere",
    "databricks": "databricks",
    "fireworks": "fireworks_ai",
    "fireworks_ai": "fireworks_ai",
    "groq": "groq",
    "mistral": "mistral",
    "openai": "openai",
    "openai_compatible": "openai",
    "together": "together_ai",
    "together_ai": "together_ai",
    "watsonx": "watsonx",
    "xai": "xai",
}
_KNOWN_LITELLM_PREFIXES = set(_LITELLM_PROVIDER_PREFIXES.values()) | {
    "azure",
    "ollama",
    "ollama_chat",
    "openai",
    "vertex_ai",
}


def build_adk_model(settings: Settings, *, lite_llm_cls: LiteLlmFactory | None = None) -> Any:
    """Return an ADK model value for the configured provider."""

    return _build_model_for_provider(
        settings,
        provider=settings.adk_model_provider,
        model=settings.adk_model,
        api_base=settings.adk_model_api_base,
        api_key=settings.adk_model_api_key,
        api_version=settings.adk_model_api_version,
        timeout_seconds=settings.adk_model_timeout_seconds,
        max_retries=settings.adk_model_max_retries,
        lite_llm_cls=lite_llm_cls,
    )


def build_vertex_adk_model(settings: Settings) -> str:
    """Return the legacy Gemini/Vertex model used by older ADK workflow wiring."""

    model = settings.adk_model if uses_vertex_adk_provider(settings) else ""
    return _configured_model_name(model, settings.vertex_model, label="Vertex ADK model")


def build_query_generator_adk_model(
    settings: Settings,
    *,
    lite_llm_cls: LiteLlmFactory | None = None,
) -> Any:
    """Return the model configured specifically for the SQL generator agent."""

    provider = settings.query_generator_model_provider
    model = settings.query_generator_model
    same_provider = _normalize_provider(provider) == _normalize_provider(
        settings.adk_model_provider
    )
    if not model and same_provider:
        model = settings.adk_model
    return _build_model_for_provider(
        settings,
        provider=provider,
        model=model,
        api_base=settings.query_generator_model_api_base
        or (settings.adk_model_api_base if same_provider else ""),
        api_key=settings.query_generator_model_api_key
        or (settings.adk_model_api_key if same_provider else ""),
        api_version=settings.query_generator_model_api_version
        or (settings.adk_model_api_version if same_provider else ""),
        timeout_seconds=settings.query_generator_model_timeout_seconds,
        max_retries=settings.query_generator_model_max_retries,
        lite_llm_cls=lite_llm_cls,
    )


def query_generator_uses_tools(settings: Settings) -> bool:
    if settings.query_generator_use_tools is not None:
        return bool(settings.query_generator_use_tools)
    return not (
        uses_ollama_provider(settings.query_generator_model_provider)
        or uses_ollama_model(settings.query_generator_model)
    )


def uses_vertex_adk_provider(settings: Settings) -> bool:
    return uses_vertex_provider(settings.adk_model_provider)


def uses_vertex_provider(provider: str | None) -> bool:
    return _normalize_provider(provider) in _VERTEX_PROVIDERS


def uses_ollama_provider(provider: str | None) -> bool:
    return _normalize_provider(provider) in _OLLAMA_PROVIDERS


def uses_ollama_model(model: str | None) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith(("ollama/", "ollama_chat/"))


def _build_model_for_provider(
    settings: Settings,
    *,
    provider: str | None,
    model: str | None,
    api_base: str = "",
    api_key: str = "",
    api_version: str = "",
    timeout_seconds: float | int | None = None,
    max_retries: int | None = None,
    lite_llm_cls: LiteLlmFactory | None = None,
) -> Any:
    normalized_provider = _normalize_provider(provider)
    if normalized_provider in _VERTEX_PROVIDERS:
        return _configured_model_name(model, settings.vertex_model, label="Vertex ADK model")
    if normalized_provider in _OLLAMA_PROVIDERS:
        return build_ollama_adk_model(
            settings,
            model=model,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            lite_llm_cls=lite_llm_cls,
        )
    if (
        normalized_provider in _LITELLM_PASSTHROUGH_PROVIDERS
        or normalized_provider in _LITELLM_PROVIDER_PREFIXES
    ):
        return build_litellm_adk_model(
            provider=normalized_provider,
            model=model,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            lite_llm_cls=lite_llm_cls,
        )
    raise ValueError(
        "Unsupported ADK_MODEL_PROVIDER. Expected a Google/Vertex provider, ollama, "
        "or a LiteLLM provider such as openai, azure_openai, anthropic, bedrock, or litellm."
    )


def build_ollama_adk_model(
    settings: Settings,
    *,
    model: str | None = "",
    api_base: str = "",
    api_key: str = "",
    api_version: str = "",
    timeout_seconds: float | int | None = None,
    max_retries: int | None = None,
    lite_llm_cls: LiteLlmFactory | None = None,
) -> Any:
    """Create the ADK LiteLLM wrapper for an Ollama-hosted chat model."""

    resolved_api_base = (api_base or settings.ollama_api_base or "").strip().rstrip("/")
    if not resolved_api_base:
        raise ValueError("OLLAMA_API_BASE is required when ADK_MODEL_PROVIDER=ollama")
    os.environ["OLLAMA_API_BASE"] = resolved_api_base

    raw_model = _configured_model_name(model, settings.ollama_model, label="Ollama ADK model")
    normalized_model = _normalize_ollama_model_name(raw_model)
    cls = lite_llm_cls or _load_lite_llm()
    return cls(
        model=normalized_model,
        **_litellm_kwargs(
            api_base=resolved_api_base,
            api_key=api_key,
            api_version=api_version,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        ),
    )


def build_litellm_adk_model(
    *,
    provider: str,
    model: str | None,
    api_base: str = "",
    api_key: str = "",
    api_version: str = "",
    timeout_seconds: float | int | None = None,
    max_retries: int | None = None,
    lite_llm_cls: LiteLlmFactory | None = None,
) -> Any:
    """Create the ADK LiteLLM wrapper for an enterprise-hosted chat model."""

    raw_model = _configured_model_name(model, "", label=f"{provider} ADK model")
    normalized_model = _normalize_litellm_model_name(provider, raw_model)
    cls = lite_llm_cls or _load_lite_llm()
    return cls(
        model=normalized_model,
        **_litellm_kwargs(
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        ),
    )


def _normalize_provider(provider: str | None) -> str:
    return (provider or "").strip().lower().replace("-", "_")


def _configured_model_name(primary: str | None, fallback: str | None, *, label: str) -> str:
    model = (primary or fallback or "").strip()
    if not model:
        raise ValueError(f"{label} is required")
    return model


def _normalize_ollama_model_name(model: str) -> str:
    if model.startswith("ollama_chat/"):
        return model
    if model.startswith("ollama/"):
        return f"ollama_chat/{model.removeprefix('ollama/')}"
    return f"ollama_chat/{model}"


def _normalize_litellm_model_name(provider: str, model: str) -> str:
    if provider in _LITELLM_PASSTHROUGH_PROVIDERS or _has_litellm_provider_prefix(model):
        return model
    prefix = _LITELLM_PROVIDER_PREFIXES[provider]
    return f"{prefix}/{model}"


def _has_litellm_provider_prefix(model: str) -> bool:
    prefix, separator, _ = model.partition("/")
    return bool(separator) and prefix.strip().lower() in _KNOWN_LITELLM_PREFIXES


def _litellm_kwargs(
    *,
    api_base: str = "",
    api_key: str = "",
    api_version: str = "",
    timeout_seconds: float | int | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if api_base:
        kwargs["api_base"] = api_base.strip().rstrip("/")
    if api_key:
        kwargs["api_key"] = api_key.strip()
    if api_version:
        kwargs["api_version"] = api_version.strip()
    if timeout_seconds is not None and float(timeout_seconds) > 0:
        kwargs["timeout"] = float(timeout_seconds)
    if max_retries is not None and int(max_retries) >= 0:
        kwargs["num_retries"] = int(max_retries)
    return kwargs


def _load_lite_llm() -> LiteLlmFactory:
    try:
        from google.adk.models.lite_llm import LiteLlm
    except ImportError as exc:  # pragma: no cover - depends on installed ADK extras
        raise RuntimeError(
            "Ollama ADK models require the LiteLLM ADK extension. "
            "Install the project dependencies so the `litellm` package is available."
        ) from exc
    return LiteLlm
