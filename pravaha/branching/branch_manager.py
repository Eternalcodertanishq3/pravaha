"""Branch Manager — Git-like conversation forking and merging."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from pravaha.branching.branch_store import BranchStore
from pravaha.branching.schemas import Branch, BranchNode

logger = logging.getLogger(__name__)


class BranchManager:
    """Manage conversation branches with git-like semantics.

    Exposes the exact API surface expected by serving/routes/branches.py:
    - create_branch(session_id, fork_at, label)
    - list_branches(session_id)
    - checkout(branch_id) -> Branch | None
    - delete_branch(branch_id) -> bool
    """

    def __init__(self, store: BranchStore | None = None) -> None:
        self.store = store or BranchStore()
        main = Branch(name="main")
        self.store.add_branch(main)
        self._current_branch = main.branch_id

    # ── Route-compatible API ──────────────────────────────────────

    def create_branch(
        self,
        session_id: str,
        fork_at: int,
        label: Optional[str] = None,
    ) -> Branch:
        """Fork conversation at message index fork_at."""
        branch = Branch(
            session_id=session_id,
            label=label or f"branch-{uuid.uuid4().hex[:6]}",
            fork_point=fork_at,
            messages=[],
            parent_branch_id=self._current_branch,
        )
        self.store.add_branch(branch)
        logger.info(
            f"Created branch '{branch.label}' for session "
            f"'{session_id}' at fork_point={fork_at}"
        )
        return branch

    def list_branches(self, session_id: str) -> list[Branch]:
        """List all branches for a specific session."""
        return [
            b for b in self.store.list_branches()
            if b.session_id == session_id
        ]

    def checkout(self, branch_id: str) -> Branch | None:
        """Checkout a specific branch. Returns Branch or None."""
        branch = self.store.get_branch(branch_id)
        if branch:
            self._current_branch = branch_id
        return branch

    def delete_branch(self, branch_id: str) -> bool:
        """Delete a branch. Returns True if found and deleted."""
        return self.store.delete_branch(branch_id)

    # ── Legacy API (preserved for backward compat) ────────────────

    def add_message(
        self, role: str, content: str, parent_id: str | None = None
    ) -> BranchNode:
        """Add a message to the current branch."""
        branch = self.store.get_branch(self._current_branch)
        if parent_id is None and branch:
            parent_id = branch.head_node_id or None
        node = BranchNode(parent_id=parent_id, role=role, content=content)
        self.store.add_node(node)
        if branch:
            branch.head_node_id = node.node_id
            branch.messages.append({"role": role, "content": content})
        return node

    def fork(self, name: str, from_node_id: str | None = None) -> Branch:
        """Legacy fork by node ID."""
        current = self.store.get_branch(self._current_branch)
        head = from_node_id or (current.head_node_id if current else "")
        new_branch = Branch(
            name=name,
            head_node_id=head,
            parent_branch_id=self._current_branch,
        )
        self.store.add_branch(new_branch)
        logger.info(f"Forked branch '{name}' from node {head[:8]}")
        return new_branch

    def get_current_history(self) -> list[BranchNode]:
        """Get message history for current branch."""
        branch = self.store.get_branch(self._current_branch)
        if branch and branch.head_node_id:
            return self.store.get_history(branch.head_node_id)
        return []

    @property
    def current_branch(self) -> Branch | None:
        return self.store.get_branch(self._current_branch)
