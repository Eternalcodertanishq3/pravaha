"""Branch Store — Persist conversation tree state."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pravaha.branching.schemas import Branch, BranchNode

logger = logging.getLogger(__name__)


class BranchStore:
    """In-memory + disk-backed conversation tree storage."""

    def __init__(self, persist_path: str | None = None) -> None:
        self.persist_path = persist_path
        self._nodes: dict[str, BranchNode] = {}
        self._branches: dict[str, Branch] = {}

    def add_node(self, node: BranchNode) -> None:
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> BranchNode | None:
        return self._nodes.get(node_id)

    def add_branch(self, branch: Branch) -> None:
        self._branches[branch.branch_id] = branch

    def get_branch(self, branch_id: str) -> Branch | None:
        return self._branches.get(branch_id)

    def list_branches(self) -> list[Branch]:
        return list(self._branches.values())

    def delete_branch(self, branch_id: str) -> bool:
        """Delete a branch. Returns True if found and deleted."""
        if branch_id in self._branches:
            del self._branches[branch_id]
            return True
        return False

    def get_history(self, node_id: str) -> list[BranchNode]:
        """Walk up the tree from node_id to root, return in chronological order."""
        history = []
        current: str | None = node_id
        while current:
            node = self._nodes.get(current)
            if node is None:
                break
            history.append(node)
            current = node.parent_id
        return list(reversed(history))

    def save(self) -> None:
        if not self.persist_path:
            return
        p = Path(self.persist_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "role": n.role,
                    "content": n.content,
                    "timestamp": n.timestamp,
                }
                for nid, n in self._nodes.items()
            },
            "branches": {
                bid: {
                    "branch_id": b.branch_id,
                    "name": b.name,
                    "head_node_id": b.head_node_id,
                    "parent_branch_id": b.parent_branch_id,
                }
                for bid, b in self._branches.items()
            },
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
