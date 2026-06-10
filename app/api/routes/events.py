from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.dependencies import AppContext, get_ctx

router = APIRouter(prefix="/events", tags=["events"])


@router.get("/stream")
async def stream_events(run_id: str = Query(...), ctx: AppContext = Depends(get_ctx)) -> StreamingResponse:
    async def gen():
        async for event in ctx.event_bus.subscribe(run_id=run_id, replay_history=True):
            yield event.to_sse()

    return StreamingResponse(gen(), media_type="text/event-stream")
