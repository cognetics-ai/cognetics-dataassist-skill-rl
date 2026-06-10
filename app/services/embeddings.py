from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from app.config import Settings

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Simple in-process rate limiter enforcing global call spacing."""

    def __init__(self, max_per_second: float):
        self._interval = (1.0 / max_per_second) if max_per_second > 0 else 0.0
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            if self._next_slot > now:
                await asyncio.sleep(self._next_slot - now)
                now = time.monotonic()
            self._next_slot = max(self._next_slot, now) + self._interval


_VERTEX_PROVIDERS = {"vertex", "gemini", "google", "google_vertex"}
_OLLAMA_PROVIDERS = {"ollama", "ollama_chat"}
_OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai_compatible"}
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
    "litellm": "",
    "lite_llm": "",
    "mistral": "mistral",
    "together": "together_ai",
    "together_ai": "together_ai",
    "voyage": "voyage",
    "watsonx": "watsonx",
    "xai": "xai",
}
_KNOWN_LITELLM_PREFIXES = {
    prefix for prefix in _LITELLM_PROVIDER_PREFIXES.values() if prefix
} | {"azure", "openai", "ollama", "ollama_chat", "vertex_ai"}


class EmbeddingService:
    """Pluggable embedding client with retry/backoff and a small in-memory cache."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._rate_limiter = AsyncRateLimiter(max(0.0, settings.vertex_embedding_rps))
        self._cache: dict[str, list[float]] = {}
        self._max_cache_entries = 20_000
        self._request_timeout_seconds = (
            max(1, settings.vertex_embedding_request_timeout_ms) / 1000.0
        )

    @property
    def dimension(self) -> int:
        return self._settings.active_embedding_dimension

    async def embed_document(self, text: str) -> tuple[list[float], int]:
        return await self._embed(text, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> tuple[list[float], int]:
        return await self._embed(text, task_type="RETRIEVAL_QUERY")

    async def _embed(self, text: str, task_type: str) -> tuple[list[float], int]:
        content = (text or "").strip()
        if not content:
            return [], 0

        cache_key = f"{task_type}:{content}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached, 0

        max_attempts = max(1, int(self._settings.vertex_embedding_max_retries) + 1)
        wait = wait_exponential_jitter(
            initial=max(1, self._settings.vertex_embedding_backoff_initial_ms) / 1000.0,
            max=max(1, self._settings.vertex_embedding_backoff_max_ms) / 1000.0,
        )

        retries = 0
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait,
                retry=retry_if_exception(self._is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    retries = attempt.retry_state.attempt_number - 1
                    await self._rate_limiter.wait()
                    logger.debug(
                        "Embedding request provider=%s model=%s task_type=%s",
                        self._provider,
                        self._settings.active_embedding_model,
                        task_type,
                    )
                    try:
                        vector = await asyncio.wait_for(
                            asyncio.to_thread(self._embed_sync, content, task_type),
                            timeout=self._request_timeout_seconds,
                        )
                    except TimeoutError as exc:
                        raise TimeoutError(
                            "Embedding request timed out after "
                            f"{self._request_timeout_seconds:.1f}s."
                        ) from exc
                    if not vector:
                        raise ValueError("Embedding response did not include vector values.")
                    self._cache_put(cache_key, vector)
                    return vector, retries
        except Exception as exc:
            logger.warning("Embedding generation failed after %s retries: %s", retries, exc)
            return [], retries

        return [], retries

    @property
    def _provider(self) -> str:
        return _normalize_provider(self._settings.embedding_provider)

    def _embed_sync(self, text: str, task_type: str) -> list[float]:
        provider = self._provider
        if provider in _OLLAMA_PROVIDERS:
            return self._embed_ollama_sync(text)
        if provider in _OPENAI_COMPATIBLE_PROVIDERS:
            return self._embed_openai_compatible_sync(text)
        if provider in _VERTEX_PROVIDERS:
            return self._embed_vertex_sync(text, task_type)
        if provider in _LITELLM_PROVIDER_PREFIXES or _has_litellm_provider_prefix(
            self._settings.active_embedding_model
        ):
            return self._embed_litellm_sync(text, provider=provider)
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {self._settings.embedding_provider}")

    def _embed_ollama_sync(self, text: str) -> list[float]:
        api_base = (
            self._settings.embedding_api_base or self._settings.ollama_api_base or ""
        ).strip().rstrip("/")
        if not api_base:
            raise ValueError("OLLAMA_API_BASE is required when EMBEDDING_PROVIDER=ollama")

        response = httpx.post(
            f"{api_base}/api/embeddings",
            json={"model": self._settings.active_embedding_model, "prompt": text},
            timeout=self._request_timeout_seconds,
        )
        response.raise_for_status()
        return self._coerce_vector(response.json().get("embedding"))

    def _embed_openai_compatible_sync(self, text: str) -> list[float]:
        api_base = (
            self._settings.embedding_api_base
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        api_key = self._settings.embedding_api_key or os.getenv("OPENAI_API_KEY") or "sk-none"

        response = httpx.post(
            f"{api_base}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": self._settings.active_embedding_model, "input": [text]},
            timeout=self._request_timeout_seconds,
        )
        response.raise_for_status()
        return self._extract_embedding_vector(response.json())

    def _embed_litellm_sync(self, text: str, *, provider: str) -> list[float]:
        from litellm import embedding  # type: ignore

        model = self._normalize_litellm_model_name(provider, self._settings.active_embedding_model)
        kwargs: dict[str, Any] = {"model": model, "input": [text]}
        if self._settings.embedding_api_base:
            kwargs["api_base"] = self._settings.embedding_api_base
        if self._settings.embedding_api_key:
            kwargs["api_key"] = self._settings.embedding_api_key

        response = embedding(**kwargs)
        return self._extract_embedding_vector(response)

    def _embed_vertex_sync(self, text: str, task_type: str) -> list[float]:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        client = genai.Client(
            vertexai=True,
            project=self._settings.vertex_project_id,
            location=self._settings.vertex_location,
        )
        config = genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dimension,
        )
        response = client.models.embed_content(
            model=self._settings.active_embedding_model,
            contents=[text],
            config=config,
        )

        return self._extract_vertex_embedding_vector(response)

    def _extract_vertex_embedding_vector(self, response: Any) -> list[float]:
        candidate: Any = None
        embeddings = getattr(response, "embeddings", None)
        if embeddings:
            candidate = embeddings[0]
        if candidate is None:
            candidate = getattr(response, "embedding", None)

        if candidate is None and isinstance(response, dict):
            raw_embeddings = response.get("embeddings") or []
            candidate = raw_embeddings[0] if raw_embeddings else response.get("embedding")

        if candidate is None:
            return []

        values = getattr(candidate, "values", None)
        if values is None and isinstance(candidate, dict):
            values = candidate.get("values")
        if not values:
            return []

        return self._coerce_vector(values)

    def _extract_embedding_vector(self, response: Any) -> list[float]:
        if isinstance(response, dict):
            data = response.get("data") or []
            if data:
                first = data[0]
                if isinstance(first, dict):
                    return self._coerce_vector(first.get("embedding"))
                return self._coerce_vector(getattr(first, "embedding", None))
            return self._coerce_vector(response.get("embedding"))

        data = getattr(response, "data", None) or []
        if data:
            first = data[0]
            if isinstance(first, dict):
                return self._coerce_vector(first.get("embedding"))
            return self._coerce_vector(getattr(first, "embedding", None))
        return self._coerce_vector(getattr(response, "embedding", None))

    def _coerce_vector(self, values: Any) -> list[float]:
        if not values:
            return []
        vector = [float(value) for value in values]
        expected = self.dimension
        if expected > 0 and len(vector) != expected:
            raise ValueError(
                f"Embedding dimension {len(vector)} does not match configured dimension {expected} "
                f"for model {self._settings.active_embedding_model}"
            )
        return vector

    @staticmethod
    def _normalize_litellm_model_name(provider: str, model: str) -> str:
        if provider in {"litellm", "lite_llm"} or _has_litellm_provider_prefix(model):
            return model
        prefix = _LITELLM_PROVIDER_PREFIXES[provider]
        return f"{prefix}/{model}"

    @staticmethod
    def _is_retryable_exception(exc: BaseException) -> bool:
        text = str(exc).lower()
        retry_signals = (
            "429",
            "rate limit",
            "resource exhausted",
            "quota",
            "temporarily unavailable",
            "timeout",
            "connection reset",
            "internal error",
            "service unavailable",
            "502",
            "bad gateway",
            "gateway",
            "upstream",
        )
        return any(signal in text for signal in retry_signals)

    def _cache_put(self, key: str, value: list[float]) -> None:
        self._cache[key] = value
        if len(self._cache) > self._max_cache_entries:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest, None)


def _normalize_provider(provider: str | None) -> str:
    return (provider or "").strip().lower().replace("-", "_")


def _has_litellm_provider_prefix(model: str | None) -> bool:
    prefix, separator, _ = (model or "").partition("/")
    return bool(separator) and prefix.strip().lower() in _KNOWN_LITELLM_PREFIXES


VertexEmbeddingService = EmbeddingService
