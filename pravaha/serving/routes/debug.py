"""Debug routes — replay, step, trace endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Debug"])


class ReplayRequest(BaseModel):
    request_id: str


@router.post("/debug/replay")
async def replay_request(request: ReplayRequest):
    """Replay a recorded request exactly."""
    from pravaha.debug.replayer import RequestReplayer

    replayer = RequestReplayer()
    recording = replayer.get_recording(request.request_id)
    if recording:
        return {"request_id": request.request_id, "replay": recording}
    return {"error": f"No recording found for {request.request_id}"}


@router.get("/debug/step")
async def step_debug(request_id: str, pos: int = 0):
    """Step through inference at a specific token position."""
    from pravaha.debug.step_debugger import TokenStepDebugger

    debugger = TokenStepDebugger()
    info = debugger.get_step_info(request_id, pos)
    return info or {"error": "No debug info available"}


@router.get("/debug/trace")
async def get_trace(request_id: str):
    """Export full token-by-token decision trace."""
    from pravaha.debug.trace_logger import TraceLogger

    logger = TraceLogger()
    trace = logger.get_trace(request_id)
    return {"request_id": request_id, "trace": trace}
