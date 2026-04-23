"""Metrics route — GET /metrics (Prometheus format)."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.responses import Response

router = APIRouter(tags=["Metrics"])


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Return metrics in Prometheus exposition format."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response(content="# prometheus_client not installed\n", media_type="text/plain")
