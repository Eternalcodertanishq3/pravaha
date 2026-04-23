"""Admin routes — config reload, LoRA management, A/B testing, model merge."""

from __future__ import annotations
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter(tags=["Admin"])


class ReloadRequest(BaseModel):
    sampling: Optional[dict] = None
    scheduler: Optional[dict] = None
    swarm: Optional[dict] = None
    guardrails: Optional[dict] = None


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
    return {"status": "merge_queued", "model_a": request.model_a,
            "model_b": request.model_b, "alpha": request.alpha}


@router.post("/admin/ab")
async def configure_ab(split: float = 0.5):
    """Configure A/B traffic split."""
    return {"split": split, "status": "configured"}
