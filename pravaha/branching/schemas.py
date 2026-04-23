"""Branching Schemas — Data models for conversation forking."""

from __future__ import annotations
from dataclasses import dataclass, field
import time
import uuid


@dataclass
class BranchNode:
    """A single node in the conversation tree."""
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    role: str = "user"
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class Branch:
    """A named branch in the conversation tree."""
    branch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "main"
    head_node_id: str = ""
    created_at: float = field(default_factory=time.time)
    parent_branch_id: str | None = None
