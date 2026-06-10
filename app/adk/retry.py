from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.observability.logging import get_logger

_logger = get_logger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass(frozen=True)
class AdkRetryConfig:
    max_retries: int = 1
    backoff_initial_seconds: float = 2.0
    backoff_max_seconds: float = 30.0

    @property
    def max_attempts(self) -> int:
        return max(1, int(self.max_retries) + 1)


def adk_retry(config: AdkRetryConfig, *, label: str) -> Callable[[_F], _F]:
    """Build a tenacity retry decorator for transient ADK/LiteLLM failures."""

    max_attempts = config.max_attempts

    def _before_sleep(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        next_action = getattr(retry_state, "next_action", None)
        _logger.warning(
            "adk_agent_retrying",
            label=label,
            attempt=retry_state.attempt_number,
            max_attempts=max_attempts,
            sleep_seconds=getattr(next_action, "sleep", None),
            error_type=type(exc).__name__ if exc else "",
            error=str(exc) if exc else "",
        )

    return retry(
        reraise=True,
        retry=retry_if_exception(is_retryable_adk_exception),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=max(0.0, float(config.backoff_initial_seconds)),
            max=max(0.0, float(config.backoff_max_seconds)),
        ),
        before_sleep=_before_sleep,
    )


def is_retryable_adk_exception(exc: BaseException) -> bool:
    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower()
    retryable_markers = (
        "timeout",
        "timed out",
        "connection refused",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "too many requests",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
        "service unavailable",
        "gateway timeout",
        "ollamaerror",
    )
    return any(marker in text for marker in retryable_markers)
