from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import AppContext, get_ctx
from app.schemas import (
    CancelRequest,
    CancelResponse,
    QueryRunHistoryItem,
    QueryRunHistoryResponse,
    ResultsResponse,
    RunRequest,
    RunResponse,
)

router = APIRouter(prefix="/query", tags=["query"])
CtxDep = Annotated[AppContext, Depends(get_ctx)]


@router.post("/run", response_model=RunResponse)
async def run_query(payload: RunRequest, ctx: CtxDep) -> RunResponse:
    try:
        run_id = await ctx.adk_runtime.run_query(
            soeid=payload.soeid,
            sql=payload.sql,
            engine=payload.engine,
            prompt=payload.prompt,
            input_mode=payload.input_mode.value,
            run_id=payload.run_id,
            source_id=payload.source_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RunResponse(run_id=run_id)


@router.post("/cancel", response_model=CancelResponse)
async def cancel_query(payload: CancelRequest, ctx: CtxDep) -> CancelResponse:
    cancelled = await ctx.adk_runtime.cancel_run(payload.run_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Run not found")
    return CancelResponse(run_id=payload.run_id, cancelled=True)


@router.get("/results", response_model=ResultsResponse)
async def get_results(
    run_id: Annotated[str, Query()],
    ctx: CtxDep,
    page_token: Annotated[str | None, Query()] = None,
) -> ResultsResponse:
    run = await ctx.store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return ResultsResponse(
        run_id=run.run_id,
        status=run.status,
        result_schema=run.schema,
        rows=run.rows,
        next_page_token=page_token,
        error_message=run.error_message,
    )


@router.get("/history", response_model=QueryRunHistoryResponse)
async def list_run_history(
    soeid: Annotated[str, Query()],
    ctx: CtxDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> QueryRunHistoryResponse:
    runs = await ctx.store.list_runs_for_user(soeid=soeid, limit=limit)
    items = [
        QueryRunHistoryItem(
            run_id=run.run_id,
            soeid=run.soeid,
            engine=run.engine,
            input_mode=run.input_mode,
            route_mode=run.route_mode,
            submitted_text=run.submitted_text,
            submitted_sql=run.submitted_sql,
            submitted_prompt=run.submitted_prompt,
            natural_language_query=run.natural_language_query,
            source_id=run.source_id,
            reward_total=(
                float(run.reward_json["total"])
                if isinstance(run.reward_json, dict) and run.reward_json.get("total") is not None
                else None
            ),
            reward_stage=(
                str(run.reward_json["stage"])
                if isinstance(run.reward_json, dict) and run.reward_json.get("stage") is not None
                else None
            ),
            final_sql=run.final_sql,
            status=run.status,
            query_start_time=run.started_at,
            query_end_time=run.ended_at,
            created_at=run.created_at,
            error_message=run.error_message,
            row_count=len(run.rows),
        )
        for run in runs
    ]
    return QueryRunHistoryResponse(soeid=soeid, runs=items)
