"""Memory Store — SQLite-backed persistent memory for agents.

Persists between sessions. Each agent has its own namespace.
This is what transforms agents from stateless prompts into
entities that actually learn from experience.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryStore:
    """SQLite-backed persistent memory for agents.

    Each agent gets its own namespace (agent_role) so memories
    don't leak between agents. Supports importance-weighted
    retrieval and access-time tracking.
    """

    def __init__(self, db_path: str = "data/agent_memory.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_role TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 1,
                UNIQUE(agent_role, namespace, key)
            );
            CREATE INDEX IF NOT EXISTS idx_memories_role
                ON memories(agent_role, accessed_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(agent_role, importance DESC);
        """)
        self._conn.commit()

    def put(
        self,
        agent_role: str,
        key: str,
        value: str,
        importance: float = 0.5,
        namespace: str = "default",
    ) -> None:
        """Store or update a memory."""
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO memories
                (agent_role, namespace, key, value,
                 created_at, accessed_at, importance, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(agent_role, namespace, key) DO UPDATE SET
                value = excluded.value,
                accessed_at = excluded.accessed_at,
                importance = MAX(importance, excluded.importance),
                access_count = access_count + 1
            """,
            (agent_role, namespace, key, value, now, now, importance),
        )
        self._conn.commit()

    def get(self, agent_role: str, key: str) -> str | None:
        """Retrieve a specific memory by key."""
        cursor = self._conn.execute(
            "SELECT value FROM memories WHERE agent_role=? AND key=?",
            (agent_role, key),
        )
        row = cursor.fetchone()
        if row:
            # Update access time
            self._conn.execute(
                """UPDATE memories SET accessed_at=?, access_count=access_count+1
                   WHERE agent_role=? AND key=?""",
                (time.time(), agent_role, key),
            )
            self._conn.commit()
            return row[0]
        return None

    def get_recent(self, agent_role: str, limit: int = 10) -> list[str]:
        """Get most recently accessed memories."""
        cursor = self._conn.execute(
            """
            SELECT key || ': ' || value FROM memories
            WHERE agent_role=?
            ORDER BY accessed_at DESC LIMIT ?
            """,
            (agent_role, limit),
        )
        return [row[0] for row in cursor.fetchall()]

    def get_important(
        self, agent_role: str, min_importance: float = 0.7
    ) -> list[str]:
        """Get high-importance memories."""
        cursor = self._conn.execute(
            """
            SELECT key || ': ' || value FROM memories
            WHERE agent_role=? AND importance >= ?
            ORDER BY importance DESC
            """,
            (agent_role, min_importance),
        )
        return [row[0] for row in cursor.fetchall()]

    def search(self, agent_role: str, query: str, limit: int = 5) -> list[str]:
        """Simple text search across memories (LIKE-based)."""
        cursor = self._conn.execute(
            """
            SELECT key || ': ' || value FROM memories
            WHERE agent_role=? AND (key LIKE ? OR value LIKE ?)
            ORDER BY importance DESC, accessed_at DESC
            LIMIT ?
            """,
            (agent_role, f"%{query}%", f"%{query}%", limit),
        )
        return [row[0] for row in cursor.fetchall()]

    def count(self, agent_role: str) -> int:
        """Count memories for an agent."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE agent_role=?",
            (agent_role,),
        )
        return cursor.fetchone()[0]

    def clear(self, agent_role: str) -> int:
        """Clear all memories for an agent. Returns count deleted."""
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE agent_role=?",
            (agent_role,),
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
