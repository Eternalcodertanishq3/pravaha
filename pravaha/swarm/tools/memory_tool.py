"""Memory Tool — Read/write to agent persistent memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pravaha.swarm.memory.memory_store import MemoryStore


class MemoryTool:
    """Store or retrieve facts in agent persistent memory."""

    name = "memory"
    description = "Store or retrieve facts in agent memory"
    arg_schema = '{"action": "store|retrieve|recent", "key": "str", "value": "str"}'

    def __init__(self, store: MemoryStore | None = None, agent_role: str = "default"):
        self._store = store
        self._role = agent_role

    def execute(
        self,
        action: str,
        key: str = "",
        value: str = "",
    ) -> dict:
        """Execute a memory operation."""
        if self._store is None:
            return {"error": "No memory store attached", "success": False}

        if action == "store":
            if not key or not value:
                return {"error": "key and value required for store", "success": False}
            self._store.put(self._role, key, value)
            return {"stored": key, "success": True}

        elif action == "retrieve":
            if not key:
                return {"error": "key required for retrieve", "success": False}
            val = self._store.get(self._role, key)
            return {"key": key, "value": val or "not found", "success": True}

        elif action == "recent":
            limit = int(key) if key.isdigit() else 5
            recent = self._store.get_recent(self._role, limit)
            return {"memories": recent, "success": True}

        return {"error": f"Unknown action: {action}", "success": False}
