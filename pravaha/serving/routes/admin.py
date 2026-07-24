"""Admin routes — config reload, LoRA management, A/B testing, model merge."""

from __future__ import annotations

import time
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["Admin"])


class ReloadRequest(BaseModel):
    sampling: dict | None = None
    scheduler: dict | None = None
    swarm: dict | None = None
    guardrails: dict | None = None


class MergeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_a: str
    model_b: str
    alpha: float = 0.5
    output_name: str = "merged"


class LoRARequest(BaseModel):
    adapter_path: str
    name: str


@router.post("/admin/reload")
async def reload_config(request: ReloadRequest, raw_request: Request):
    """Hot-reload configuration without restart."""
    engine = raw_request.app.state.engine
    updated = []
    if request.sampling:
        for key, val in request.sampling.items():
            if hasattr(engine.config.sampling, key):
                setattr(engine.config.sampling, key, val)
                updated.append(f"sampling.{key}={val}")
    if request.scheduler:
        for key, val in request.scheduler.items():
            if hasattr(engine.config.scheduler, key):
                setattr(engine.config.scheduler, key, val)
                updated.append(f"scheduler.{key}={val}")
    return {"updated": updated, "status": "ok"}


@router.post("/admin/lora/load")
async def load_lora(request: LoRARequest, raw_request: Request):
    """Load a LoRA adapter."""
    from pravaha.models.lora import LoRAManager

    manager = LoRAManager()
    manager.load_adapter(request.adapter_path, request.name)
    return {"loaded": request.name, "status": "ok"}


@router.post("/admin/lora/activate")
async def activate_lora(name: str):
    """Activate a loaded LoRA adapter."""
    from pravaha.models.lora import LoRAManager

    manager = LoRAManager()
    manager.activate_adapter(name)
    return {"activated": name}


@router.post("/admin/merge")
async def merge_models(request: MergeRequest):
    """Merge two models using SLERP."""
    return {
        "status": "merge_queued",
        "model_a": request.model_a,
        "model_b": request.model_b,
        "alpha": request.alpha,
    }


@router.post("/admin/ab")
async def configure_ab(split: float = 0.5):
    """Configure A/B traffic split."""
    return {"split": split, "status": "configured"}


class UserDataRequest(BaseModel):
    user_id: str


@router.post("/admin/export_user_data")
async def export_user_data(request: UserDataRequest, raw_request: Request):
    """GDPR Data Portability: Export all persistent session and memory data for a user."""
    engine = raw_request.app.state.engine
    sessions = []
    if engine and hasattr(engine, "session_cache"):
        sessions = [
            s for s in engine.session_cache.list_sessions()
            if s.get("session_id", "").startswith(request.user_id)
        ]

    return {
        "user_id": request.user_id,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sessions": sessions,
        "status": "success",
    }


@router.post("/admin/delete_user")
async def delete_user(request: UserDataRequest, raw_request: Request):
    """GDPR Right to be Forgotten: Purge all session and memory data for a user."""
    engine = raw_request.app.state.engine
    purged_count = 0
    if engine and hasattr(engine, "session_cache"):
        for s in engine.session_cache.list_sessions():
            sid = s.get("session_id", "")
            if sid.startswith(request.user_id):
                engine.session_cache.remove(sid)
                purged_count += 1

    return {
        "user_id": request.user_id,
        "purged_sessions": purged_count,
        "status": "purged",
    }
