"""Health and Metrics API routes.

Production observability endpoints:
- GET /health       — Basic liveness probe
- GET /health/ready — Readiness probe (model loaded)
- GET /metrics      — Live telemetry snapshot
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Basic liveness probe. Returns 200 if the server process is alive."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness_check(raw_request: Request):
    """Readiness probe. Returns 200 only if the model is loaded and scheduler is ready."""
    engine = getattr(raw_request.app.state, "engine", None)

    if engine is None:
        return {"status": "not_ready", "reason": "Engine not initialized"}

    if not hasattr(engine, "model") or engine.model is None:
        return {"status": "not_ready", "reason": "Model not loaded"}

    return {
        "status": "ready",
        "model": engine.config.model.model_path,
        "quantization": engine.config.model.quantization or "none",
    }


@router.get("/metrics")
async def get_metrics(raw_request: Request):
    """Live telemetry snapshot — GPU, model, and throughput stats."""
    engine = getattr(raw_request.app.state, "engine", None)

    if engine is None:
        return {"status": "engine_not_loaded"}

    # GPU memory stats
    memory = engine.get_memory_stats()

    # Telemetry counters (from engine)
    telemetry = {}
    if hasattr(engine, "get_telemetry"):
        telemetry = engine.get_telemetry()

    return {
        "model": engine.config.model.model_path,
        "quantization": engine.config.model.quantization or "none",
        "gpu_name": memory.get("gpu_name", "N/A"),
        "gpu_utilization_pct": round(memory.get("utilization_pct", 0), 2),
        "vram_allocated_gb": round(memory.get("allocated_gb", 0), 4),
        "vram_reserved_gb": round(memory.get("reserved_gb", 0), 4),
        "vram_total_gb": round(memory.get("total_gb", 0), 2),
        **telemetry,
    }
