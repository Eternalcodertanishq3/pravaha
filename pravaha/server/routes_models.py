"""Models API route — GET /v1/models.

Returns the currently loaded model information,
compatible with the OpenAI Models API specification.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from pravaha.server.schemas import ModelCard, ModelList

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Models"])


@router.get("/models")
async def list_models(raw_request: Request) -> ModelList:
    """List the currently loaded model(s)."""
    engine = raw_request.app.state.engine
    model_path = engine.config.model.model_path

    return ModelList(
        data=[
            ModelCard(id=model_path),
        ]
    )


@router.get("/models/{model_id}")
async def retrieve_model(model_id: str, raw_request: Request) -> ModelCard:
    """Retrieve details of a specific model."""
    engine = raw_request.app.state.engine
    loaded_model = engine.config.model.model_path

    if model_id != loaded_model:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Loaded model: '{loaded_model}'"
        )

    return ModelCard(id=loaded_model)
