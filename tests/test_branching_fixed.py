"""Tests for branching API — CRUD operations."""

from __future__ import annotations


class TestBranchManager:
    def test_create_branch(self) -> None:
        from pravaha.branching.branch_manager import BranchManager
        manager = BranchManager()
        branch = manager.create_branch("session-1", fork_at=3, label="test-fork")
        assert branch.session_id == "session-1"
        assert branch.label == "test-fork"
        assert branch.fork_point == 3

    def test_list_branches_by_session(self) -> None:
        from pravaha.branching.branch_manager import BranchManager
        manager = BranchManager()
        manager.create_branch("s1", 0, "b1")
        manager.create_branch("s2", 0, "b2")
        manager.create_branch("s1", 1, "b3")
        s1_branches = manager.list_branches("s1")
        assert len(s1_branches) == 2

    def test_checkout(self) -> None:
        from pravaha.branching.branch_manager import BranchManager
        manager = BranchManager()
        branch = manager.create_branch("s1", 0)
        result = manager.checkout(branch.branch_id)
        assert result is not None
        assert result.branch_id == branch.branch_id

    def test_checkout_nonexistent(self) -> None:
        from pravaha.branching.branch_manager import BranchManager
        manager = BranchManager()
        result = manager.checkout("nonexistent")
        assert result is None

    def test_delete_branch(self) -> None:
        from pravaha.branching.branch_manager import BranchManager
        manager = BranchManager()
        branch = manager.create_branch("s1", 0)
        deleted = manager.delete_branch(branch.branch_id)
        assert deleted is True
