"""Tests for agent memory system."""

from __future__ import annotations

import os
import tempfile

import pytest


class TestMemoryStore:
    def test_put_and_get(self) -> None:
        from pravaha.swarm.memory.memory_store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = MemoryStore(db_path=db_path)
            store.put("coder", "fix_1", "Fixed null pointer", importance=0.8)
            val = store.get("coder", "fix_1")
            assert val == "Fixed null pointer"
            store.close()
        finally:
            os.unlink(db_path)

    def test_get_recent(self) -> None:
        from pravaha.swarm.memory.memory_store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = MemoryStore(db_path=db_path)
            store.put("agent", "k1", "first")
            store.put("agent", "k2", "second")
            recent = store.get_recent("agent", limit=2)
            assert len(recent) == 2
            store.close()
        finally:
            os.unlink(db_path)

    def test_search(self) -> None:
        from pravaha.swarm.memory.memory_store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = MemoryStore(db_path=db_path)
            store.put("agent", "bug_fix", "Fixed segfault in parser")
            store.put("agent", "feature", "Added search functionality")
            results = store.search("agent", "parser")
            assert len(results) >= 1
            assert "parser" in results[0].lower()
            store.close()
        finally:
            os.unlink(db_path)


class TestEpisodicMemory:
    def test_record_and_recall(self) -> None:
        from pravaha.swarm.memory.episodic_memory import EpisodicMemory
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            mem = EpisodicMemory(db_path=db_path)
            mem.record_episode("coder", "fix bug", "patched line 42", "success", True)
            episodes = mem.get_recent("coder", limit=1)
            assert len(episodes) == 1
            assert episodes[0]["success"] is True
            mem.close()
        finally:
            os.unlink(db_path)


class TestSemanticMemory:
    def test_store_and_recall(self) -> None:
        from pravaha.swarm.memory.semantic_memory import SemanticMemory
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            mem = SemanticMemory(db_path=db_path)
            mem.store_fact("agent", "Python supports async generators")
            mem.store_fact("agent", "Rust provides memory safety")
            results = mem.recall("agent", "async generators Python")
            assert len(results) >= 1
            assert results[0]["similarity"] > 0
            mem.close()
        finally:
            os.unlink(db_path)
