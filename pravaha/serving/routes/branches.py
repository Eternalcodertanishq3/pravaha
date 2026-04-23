"""Branches route — conversation branching endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["Branching"])


class BranchRequest(BaseModel):
    session_id: str
    fork_at: int = Field(ge=0, description="Message index to fork at")
    label: str | None = None


@router.post("/branch")
async def create_branch(request: BranchRequest):
    """Fork a conversation at a specific message."""
    from pravaha.branching.branch_manager import BranchManager

    manager = BranchManager()
    branch = manager.create_branch(request.session_id, request.fork_at, request.label)
    return {"branch_id": branch.branch_id, "label": branch.label, "fork_point": branch.fork_point}


@router.get("/branch/{session_id}")
async def list_branches(session_id: str):
    """List all branches for a session."""
    from pravaha.branching.branch_manager import BranchManager

    manager = BranchManager()
    branches = manager.list_branches(session_id)
    return {
        "branches": [
            {"id": b.branch_id, "label": b.label, "fork_point": b.fork_point} for b in branches
        ]
    }


@router.post("/branch/{branch_id}/checkout")
async def checkout_branch(branch_id: str):
    """Checkout a specific branch."""
    from pravaha.branching.branch_manager import BranchManager

    manager = BranchManager()
    branch = manager.checkout(branch_id)
    if branch:
        return {"branch_id": branch.branch_id, "messages": len(branch.messages)}
    return {"error": "Branch not found"}


@router.delete("/branch/{branch_id}")
async def delete_branch(branch_id: str):
    """Delete a conversation branch."""
    from pravaha.branching.branch_manager import BranchManager

    manager = BranchManager()
    manager.delete_branch(branch_id)
    return {"deleted": branch_id}
