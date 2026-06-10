from __future__ import annotations

import asyncio
import uuid

from app.adapters.base import EngineAdapter, EngineHandle, EngineStatus, ExplainResult, ResultPage


class MockEngineAdapter(EngineAdapter):
    name = "mock"

    def __init__(self):
        self._jobs: dict[str, dict] = {}

    async def explain(self, sql: str) -> ExplainResult:
        return ExplainResult(
            ok=True,
            summary={
                "engine": self.name,
                "plan": "Mock logical plan generated",
                "warnings": [] if "*" not in sql else ["SELECT * may scan unnecessary columns"],
            },
        )

    async def execute_async(self, sql: str) -> EngineHandle:
        handle_id = str(uuid.uuid4())
        self._jobs[handle_id] = {
            "sql": sql,
            "state_index": 0,
            "states": ["QUEUED", "PLANNING", "RUNNING", "FINISHING", "FINISHED"],
            "progress": [0, 15, 55, 85, 100],
            "cancelled": False,
            "rows": [["north", 1250], ["south", 980], ["west", 430]],
            "schema": [{"name": "region", "type": "varchar"}, {"name": "revenue", "type": "bigint"}],
        }
        return EngineHandle(handle_id=handle_id, raw={})

    async def get_status(self, handle: EngineHandle) -> EngineStatus:
        job = self._jobs[handle.handle_id]
        if job["cancelled"]:
            return EngineStatus(state="CANCELLED", done=True, progress_percentage=job["progress"][job["state_index"]])

        idx = job["state_index"]
        state = job["states"][idx]
        progress = job["progress"][idx]
        done = state in {"FINISHED", "FAILED", "CANCELLED"}

        if not done:
            await asyncio.sleep(0.4)
            job["state_index"] = min(len(job["states"]) - 1, job["state_index"] + 1)

        return EngineStatus(
            state=state,
            done=done,
            progress_percentage=progress,
            stats={"elapsedTimeMillis": idx * 400},
        )

    async def fetch_results(self, handle: EngineHandle, page_token: str | None = None) -> ResultPage:
        job = self._jobs[handle.handle_id]
        return ResultPage(schema=job["schema"], rows=job["rows"], next_page_token=None)

    async def cancel(self, handle: EngineHandle) -> bool:
        job = self._jobs.get(handle.handle_id)
        if not job:
            return False
        job["cancelled"] = True
        return True
