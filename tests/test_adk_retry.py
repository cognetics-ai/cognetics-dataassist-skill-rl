from __future__ import annotations

import pytest

from app.adk.retry import AdkRetryConfig, adk_retry


@pytest.mark.asyncio
async def test_adk_retry_retries_transient_timeout():
    calls = 0

    @adk_retry(
        AdkRetryConfig(
            max_retries=1,
            backoff_initial_seconds=0,
            backoff_max_seconds=0,
        ),
        label="test",
    )
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timed out")
        return "ok"

    assert await flaky() == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_adk_retry_does_not_retry_non_transient_error():
    calls = 0

    @adk_retry(
        AdkRetryConfig(
            max_retries=3,
            backoff_initial_seconds=0,
            backoff_max_seconds=0,
        ),
        label="test",
    )
    async def invalid() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("bad prompt")

    with pytest.raises(ValueError, match="bad prompt"):
        await invalid()

    assert calls == 1
