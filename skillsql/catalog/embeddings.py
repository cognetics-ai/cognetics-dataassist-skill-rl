"""Pluggable embedding providers.

The default is a **local Ollama** embedding model (self-contained stack); OpenAI
and Vertex providers are included behind the same interface. The active provider
and dimension come from settings (``EMBEDDING_PROVIDER``, ``EMBEDDING_MODEL``,
``EMBEDDING_DIM``); the dimension must match the ``pgvector`` column.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.settings import Settings, get_settings


class Embedder(abc.ABC):
    def __init__(self, model: str, dim: int) -> None:
        self.model = model
        self.dim = dim

    @abc.abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts)


class OllamaEmbedder(Embedder):
    """Calls a local Ollama server's ``/api/embeddings`` endpoint."""

    def __init__(self, model: str, dim: int, api_base: str) -> None:
        super().__init__(model, dim)
        self.api_base = api_base.rstrip("/")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
    def _one(self, client: httpx.Client, text: str) -> list[float]:
        resp = client.post(
            f"{self.api_base}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60.0,
        )
        resp.raise_for_status()
        vec = resp.json()["embedding"]
        if len(vec) != self.dim:
            raise ValueError(
                f"embedding dim {len(vec)} != configured EMBEDDING_DIM {self.dim} "
                f"for model {self.model}"
            )
        return vec

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        with httpx.Client() as client:
            return [self._one(client, t) for t in texts]


class OpenAIEmbedder(Embedder):
    """OpenAI-compatible embeddings (also works against an Ollama OpenAI shim)."""

    def __init__(self, model: str, dim: int, api_base: str | None, api_key: str | None) -> None:
        super().__init__(model, dim)
        self.api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or "sk-none"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        with httpx.Client() as client:
            resp = client.post(
                f"{self.api_base}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": list(texts)},
                timeout=60.0,
            )
            resp.raise_for_status()
            return [d["embedding"] for d in resp.json()["data"]]


class VertexEmbedder(Embedder):
    """Vertex AI text-embeddings. Imported lazily to avoid a hard GCP dependency."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from vertexai.language_models import TextEmbeddingModel  # lazy

        model = TextEmbeddingModel.from_pretrained(self.model)
        return [e.values for e in model.get_embeddings(list(texts))]


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Construct the configured embedder."""
    s = settings or get_settings()
    provider = s.EMBEDDING_PROVIDER
    if provider == "ollama":
        return OllamaEmbedder(s.EMBEDDING_MODEL, s.EMBEDDING_DIM, s.OLLAMA_API_BASE)
    if provider == "openai":
        return OpenAIEmbedder(s.EMBEDDING_MODEL, s.EMBEDDING_DIM, s.OLLAMA_API_BASE, None)
    if provider == "vertex":
        return VertexEmbedder(s.EMBEDDING_MODEL, s.EMBEDDING_DIM)
    raise ValueError(f"unknown embedding provider: {provider}")
