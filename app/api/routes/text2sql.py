"""Text-to-SQL routes.

draft    POST /text2sql/draft    -- production workflow (ADK SequentialAgent)
validate POST /text2sql/validate -- validate existing SQL against policy/EXPLAIN
run      POST /text2sql/run      -- SkillSQL-RL training-path workflow (no loop)

``draft`` and ``validate`` route through the full ADK runtime
(``AdkDataAssistRuntime``) which uses the SequentialAgent+LoopAgent workflow
with critic/refiner and skill distillation.

``run`` routes through the SkillSQL training-path workflow (retrieve → generate →
verify/select) — the same workflow the benchmark and GRPO loop use.  It returns
the best candidate and its reward breakdown but does not run the refinement loop.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import AppContext, get_ctx
from app.schemas import DraftRequest, DraftResponse, ValidateRequest, ValidateResponse

router = APIRouter(prefix="/text2sql", tags=["text2sql"])


# ── Production workflow (SequentialAgent + LoopAgent) ─────────────────────────

@router.post("/draft", response_model=DraftResponse,
    summary="Generate + refine SQL via the full production workflow",
    description="Routes through the ADK SequentialAgent workflow: Directory → "
                "ContextBuilder[+SkillBank] → Generator → Critic/Refiner loop → "
                "Validator[+Reward] → Optimizer → SkillDistillation.")
async def draft(payload: DraftRequest, ctx: AppContext = Depends(get_ctx)) -> DraftResponse:
    engine = payload.engine_preference or "starburst"
    data = await ctx.adk_runtime.draft(payload.soeid, payload.prompt, engine=engine)
    return DraftResponse(**data)


@router.post("/validate", response_model=ValidateResponse,
    summary="Validate existing SQL against policy and EXPLAIN")
async def validate(payload: ValidateRequest, ctx: AppContext = Depends(get_ctx)) -> ValidateResponse:
    data = await ctx.adk_runtime.validate(payload.soeid, payload.sql, payload.engine)
    return ValidateResponse(**data)


# ── SkillSQL training-path workflow (schema retrieve → generate → verify) ─────

class RunRequest(BaseModel):
    question: str
    source_id: str | None = None


@router.post("/run",
    summary="SkillSQL-RL inference workflow (retrieve → generate → verify/select)",
    description="Runs the SkillSQL-RL training-path workflow: retrieves schema context "
                "and skills, samples G candidates from the Arctic agent, scores each "
                "with the formal verifier reward, and returns the best SQL with its "
                "reward breakdown. No critic/refiner loop (use /draft for that).")
async def run(req: RunRequest) -> dict:
    from app.services.catalog import run_text2sql as _run
    return await _run(req.question, source_id=req.source_id)
