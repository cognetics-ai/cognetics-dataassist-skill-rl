"""Model providers -- the single place ADK model objects are constructed.

A model spec is just a string from an agent's ``.env`` (its ``model`` field). The
provider prefix selects the builder:

    ``ollama_chat/<name>``  -> LiteLlm over Ollama's chat API (recommended)
    ``ollama/<name>``       -> LiteLlm over Ollama (discouraged; can loop)
    ``openai/<name>``       -> LiteLlm over an OpenAI-compatible endpoint
    ``gemini-*`` / ``vertex/...`` -> native ADK model string (passed through)

Isolating construction here keeps ADK/LiteLLM specifics out of agent code and
makes the SDK swap a one-file change.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from ..config.settings import get_settings


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


def _ensure_ollama_env(api_base: str | None = None) -> None:
    """LiteLLM relies on ``OLLAMA_API_BASE`` after the completion call, so make sure
    it is present in the process env (mirrors ADK's documented requirement)."""
    s = get_settings()
    resolved_api_base = (api_base or s.OLLAMA_API_BASE or "").strip().rstrip("/")
    if resolved_api_base:
        os.environ["OLLAMA_API_BASE"] = resolved_api_base


def build_model(
    model_spec: str,
    *,
    provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    timeout_seconds: float | int | None = None,
    max_retries: int | None = None,
    lite_llm_cls: LiteLlmFactory | None = None,
) -> Any:
    """Return an ADK-compatible model object (or model string) for ``model_spec``.

    ADK accepts either a bare string (native Google models) or a ``LiteLlm``
    instance (everything else). We import ``LiteLlm`` lazily so the package imports
    without ``google-adk`` installed (e.g. for unit-testing the pure layers).
    """
    spec = model_spec.strip()
    if not spec:
        raise ValueError("Model spec is required for SkillSQL agent construction")

    explicit_prefix = _model_prefix(spec)
    normalized_provider = _normalize_provider(provider)

    if spec.startswith("vertex/"):
        return spec.split("/", 1)[1]

    if explicit_prefix in _OLLAMA_PROVIDERS:
        return _build_litellm_model(
            _normalize_ollama_model_name(spec),
            provider="ollama",
            api_base=api_base or os.getenv("OLLAMA_API_BASE") or get_settings().OLLAMA_API_BASE,
            api_key=api_key,
            api_version=api_version,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            lite_llm_cls=lite_llm_cls,
        )

    if explicit_prefix in _KNOWN_LITELLM_PREFIXES:
        return _build_litellm_model(
            spec,
            provider=explicit_prefix,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            lite_llm_cls=lite_llm_cls,
        )

    if normalized_provider in _OLLAMA_PROVIDERS:
        return _build_litellm_model(
            _normalize_ollama_model_name(spec),
            provider="ollama",
            api_base=api_base or os.getenv("OLLAMA_API_BASE") or get_settings().OLLAMA_API_BASE,
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
        return _build_litellm_model(
            _normalize_litellm_model_name(normalized_provider, spec),
            provider=normalized_provider,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            lite_llm_cls=lite_llm_cls,
        )

    if normalized_provider in _VERTEX_PROVIDERS and _is_native_google_model(spec):
        return spec

    if not normalized_provider and _is_native_google_model(spec):
        return spec

    raise ValueError(
        "SkillSQL agent model is missing a provider. "
        f"model={spec!r}, provider={provider!r}. "
        "Use a provider-prefixed model such as ollama_chat/<model> or openai/<model>, "
        "or set ADK_MODEL_PROVIDER / <ROLE>_MODEL_PROVIDER."
    )


def resolve_model_spec(env_key: str = "AGENT_MODEL", default: str | None = None) -> str:
    """Read a model spec from the (already agent-overridden) environment."""
    s = get_settings()
    return os.environ.get(env_key) or default or s.DEFAULT_CHAT_MODEL


def _build_litellm_model(
    model: str,
    *,
    provider: str,
    api_base: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    timeout_seconds: float | int | None = None,
    max_retries: int | None = None,
    lite_llm_cls: LiteLlmFactory | None = None,
) -> Any:
    if _normalize_provider(provider) in _OLLAMA_PROVIDERS or model.startswith("ollama"):
        _ensure_ollama_env(api_base)
    cls = lite_llm_cls or _load_lite_llm()
    return cls(
        model=model,
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


def _model_prefix(model: str) -> str:
    prefix, separator, _ = model.partition("/")
    return prefix.strip().lower() if separator else ""


def _normalize_ollama_model_name(model: str) -> str:
    if model.startswith("ollama_chat/"):
        return model
    if model.startswith("ollama/"):
        return f"ollama_chat/{model.removeprefix('ollama/')}"
    return f"ollama_chat/{model}"


def _normalize_litellm_model_name(provider: str, model: str) -> str:
    if (
        provider in _LITELLM_PASSTHROUGH_PROVIDERS
        or _model_prefix(model) in _KNOWN_LITELLM_PREFIXES
    ):
        return model
    return f"{_LITELLM_PROVIDER_PREFIXES[provider]}/{model}"


def _is_native_google_model(model: str) -> bool:
    lowered = model.strip().lower()
    return lowered.startswith(("gemini", "models/gemini", "publishers/google/models/"))


def _litellm_kwargs(
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
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
            "SkillSQL non-Google agent models require the LiteLLM ADK extension. "
            "Install the project dependencies so `google.adk.models.lite_llm` is available."
        ) from exc
    return LiteLlm
