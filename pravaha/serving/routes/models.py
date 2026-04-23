"""Models API — GET /v1/models."""

from __future__ import annotations

from fastapi import APIRouter, Request

from pravaha.serving.schemas import ModelCard, ModelList

router = APIRouter(tags=["Models"])


@router.get("/models")
async def list_models(raw_request: Request):
    engine = raw_request.app.state.engine
    model_name = engine.config.model.model_path
    return ModelList(data=[ModelCard(id=model_name)])
