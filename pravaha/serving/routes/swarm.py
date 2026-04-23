"""Swarm API — POST /v1/swarm/run, GET /v1/swarm/agents."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["Swarm"])


class SwarmRequest(BaseModel):
    """Request body for swarm execution."""

    prompt: str
    pipeline: str = "plan-execute-audit"
    max_audit_iterations: int = Field(default=3, ge=1, le=10)
    min_score: float = Field(default=70.0, ge=0, le=100)


@router.post("/swarm/run")
async def run_swarm(request: SwarmRequest, raw_request: Request):
    """Execute a swarm pipeline with the self-healing audit loop."""
    engine = raw_request.app.state.engine
    from pravaha.swarm.orchestrator import SwarmOrchestrator
    from pravaha.swarm.pipeline import BUILTIN_PIPELINES

    orchestrator = SwarmOrchestrator()
    pipeline_def = BUILTIN_PIPELINES.get(request.pipeline)
    if not pipeline_def:
        return {"error": f"Unknown pipeline: {request.pipeline}",
                "available": list(BUILTIN_PIPELINES.keys())}

    result = await orchestrator.execute_with_audit(
        worker_pipeline=pipeline_def.worker_steps,
        task=request.prompt,
        engine=engine,
        max_iterations=request.max_audit_iterations,
        min_score=request.min_score,
    )
    return result


@router.get("/swarm/agents")
async def list_agents():
    """List all loaded swarm agents with stats."""
    from pravaha.swarm.orchestrator import SwarmOrchestrator

    orchestrator = SwarmOrchestrator()
    return {"agents": orchestrator.list_agents(), "total": len(orchestrator._agents)}


@router.get("/swarm/pipelines")
async def list_pipelines():
    """List available built-in pipelines."""
    from pravaha.swarm.pipeline import BUILTIN_PIPELINES

    return {
        "pipelines": [
            {
                "name": p.name,
                "description": p.description,
                "workers": p.worker_steps,
                "auditors": p.audit_steps,
            }
            for p in BUILTIN_PIPELINES.values()
        ]
    }
