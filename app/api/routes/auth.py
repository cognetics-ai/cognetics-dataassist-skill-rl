from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import AppContext, get_ctx
from app.schemas import AuthLoginRequest, AuthMeResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=AuthMeResponse)
async def login(payload: AuthLoginRequest, ctx: AppContext = Depends(get_ctx)) -> AuthMeResponse:
    try:
        profile = await ctx.auth_service.authenticate(payload.soeid, payload.password)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Directory lookup failed: {exc}") from exc
    return AuthMeResponse(**profile)


@router.get("/me", response_model=AuthMeResponse)
async def me(
    soed_id: str | None = Query(default=None, alias="soed_id"),
    soeid: str | None = Query(default=None),
    ctx: AppContext = Depends(get_ctx),
) -> AuthMeResponse:
    effective_soeid = soed_id or soeid
    try:
        profile = await ctx.auth_service.me(effective_soeid)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Directory lookup failed: {exc}") from exc
    return AuthMeResponse(**profile)
