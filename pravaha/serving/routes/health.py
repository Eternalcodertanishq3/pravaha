"""Health API — Production readiness and liveness endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["Health"])

_START_TIME = time.time()


@router.get("/health")
async def health():
    """Liveness probe — returns 200 OK if service process is alive."""
    return {
        "status": "ok",
        "service": "pravaha",
        "version": "3.0.0",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
    }


@router.get("/health/ready")
async def ready(raw_request: Request, response: Response):
    """Deep readiness probe — checks engine, model, queue depth, and memory health."""
    engine = raw_request.app.state.engine

    if engine is None or not hasattr(engine, "is_ready") or not engine.is_ready:
        response.status_code = 503
        return {
            "ready": False,
            "status": "loading",
            "message": "Engine is initializing or shut down.",
        }

    stats = engine.get_stats()
    queue_stats = stats.get("queue", {})
    hardware_stats = stats.get("hardware", {})

    waiting = queue_stats.get("waiting", 0)
    running = queue_stats.get("running", 0)
    usage_pct = stats.get("cache_usage_pct", 0.0)

    # Determine health status
    is_overloaded = waiting >= 500 or usage_pct >= 95.0
    status_str = "degraded" if is_overloaded else "ok"

    if is_overloaded:
        response.status_code = 429  # Indicate temporary backpressure

    return {
        "ready": True,
        "status": status_str,
        "subsystems": {
            "model_loaded": True,
            "scheduler_ready": True,
            "gpu_available": hardware_stats.get("gpu", {}).get("available", False),
        },
        "health_indicators": {
            "queue_waiting": waiting,
            "running_requests": running,
            "kv_cache_usage_pct": round(usage_pct, 1),
            "cpu_pct": hardware_stats.get("cpu_pct", 0.0),
            "memory_pct": hardware_stats.get("ram_pct", 0.0),
        },
    }


@router.get("/metrics")
async def metrics(raw_request: Request):
    """Return engine stats and telemetry metrics."""
    engine = raw_request.app.state.engine
    if engine is None:
        return {"error": "Engine not initialized"}
    return engine.get_stats()
