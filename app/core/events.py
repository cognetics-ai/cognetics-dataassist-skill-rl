from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import AsyncIterator

from app.models import RunEvent


class EventBus:
    def __init__(self, history_size: int = 200):
        self._subs: dict[str, list[asyncio.Queue[RunEvent]]] = defaultdict(list)
        self._history: dict[str, deque[RunEvent]] = defaultdict(lambda: deque(maxlen=history_size))
        self._lock = asyncio.Lock()

    async def publish(self, event: RunEvent) -> None:
        async with self._lock:
            self._history[event.run_id].append(event)
            queues = list(self._subs.get(event.run_id, []))
        for queue in queues:
            await queue.put(event)

    async def clear_history(self, run_id: str) -> None:
        async with self._lock:
            self._history.pop(run_id, None)

    async def subscribe(self, run_id: str, replay_history: bool = True) -> AsyncIterator[RunEvent]:
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        async with self._lock:
            self._subs[run_id].append(queue)
            history = list(self._history.get(run_id, []))

        try:
            if replay_history:
                for item in history:
                    yield item
            while True:
                event = await queue.get()
                yield event
                if event.event_type in {"RUN_SUCCEEDED", "RUN_FAILED", "RUN_CANCELLED"}:
                    break
        finally:
            async with self._lock:
                subs = self._subs.get(run_id, [])
                if queue in subs:
                    subs.remove(queue)
