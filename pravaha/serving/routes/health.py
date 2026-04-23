"""Health API — /health and /health/ready."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(raw_request: Request):
    engine = raw_request.app.state.engine
    return {"ready": engine.is_ready, "status": "ok" if engine.is_ready else "loading"}


@router.get("/metrics")
async def metrics(raw_request: Request):
    engine = raw_request.app.state.engine
    return engine.get_stats()
