"""FastAPI application entry point.

Route groups
------------
/health          -- liveness
/auth            -- authentication
/discovery/*     -- legacy role-based schema/query context
/text2sql/*      -- draft (production workflow), validate, run (training workflow)
/query/*         -- query execution + history
/events/*        -- server-sent events
/catalog/*       -- catalog build, retrieval, liveness
/skillbank/*     -- SkillBank seed + list
/sql/*           -- verify, score, generate (SkillSQL-RL inference tools)
/admin/*         -- DB init

Two-plane architecture (proposal Section 6):
    Cataloging plane  → /catalog/* /skillbank/* /admin/init-db
    Inference plane   → /text2sql/run /sql/generate /sql/verify /sql/score
    Production plane  → /text2sql/draft /text2sql/validate /catalog/query-history/*
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, discovery, events, health, query, text2sql
from app.api.routes.catalog import (
    admin_router,
    catalog_router,
    skillbank_router,
    sql_router,
)
from app.config import settings
from app.dependencies import build_context
from app.observability.logging import get_logger

_logger = get_logger(__name__)

_logger.debug("Starting Data Assist APIs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ctx = await build_context()
    yield


def start() -> None:
    """Entry point for the ``dataassist`` CLI script."""
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description=(
        "Data Assist API — unified FastAPI service combining the production "
        "ADK workflow with the SkillSQL-RL catalog, verification, and inference tools."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Core routes ────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(discovery.router)
app.include_router(text2sql.router)
app.include_router(query.router)
app.include_router(events.router)

# ── SkillSQL-RL routes ─────────────────────────────────────────────────────────
app.include_router(admin_router)     # /admin/init-db
app.include_router(catalog_router)   # /catalog/*
app.include_router(skillbank_router) # /skillbank/*
app.include_router(sql_router)       # /sql/verify|score|generate
