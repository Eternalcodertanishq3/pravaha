"""Branch Manager — Git-like conversation forking and merging."""

from __future__ import annotations
import logging
from typing import Optional
from pravaha.branching.schemas import Branch, BranchNode
from pravaha.branching.branch_store import BranchStore

logger = logging.getLogger(__name__)


class BranchManager:
    """Manage conversation branches with git-like semantics."""

    def __init__(self, store: Optional[BranchStore] = None) -> None:
        self.store = store or BranchStore()
        main = Branch(name="main")
        self.store.add_branch(main)
        self._current_branch = main.branch_id

    def add_message(self, role: str, content: str, parent_id: Optional[str] = None) -> BranchNode:
        branch = self.store.get_branch(self._current_branch)
        if parent_id is None and branch:
            parent_id = branch.head_node_id or None
        node = BranchNode(parent_id=parent_id, role=role, content=content)
        self.store.add_node(node)
        if branch:
            branch.head_node_id = node.node_id
        return node

    def fork(self, name: str, from_node_id: Optional[str] = None) -> Branch:
        current = self.store.get_branch(self._current_branch)
        head = from_node_id or (current.head_node_id if current else "")
        new_branch = Branch(name=name, head_node_id=head, parent_branch_id=self._current_branch)
        self.store.add_branch(new_branch)
        logger.info(f"Forked branch '{name}' from node {head[:8]}")
        return new_branch

    def checkout(self, branch_id: str) -> None:
        if self.store.get_branch(branch_id):
            self._current_branch = branch_id

    def get_current_history(self) -> list[BranchNode]:
        branch = self.store.get_branch(self._current_branch)
        if branch and branch.head_node_id:
            return self.store.get_history(branch.head_node_id)
        return []

    def list_branches(self) -> list[Branch]:
        return self.store.list_branches()

    @property
    def current_branch(self) -> Optional[Branch]:
        return self.store.get_branch(self._current_branch)
