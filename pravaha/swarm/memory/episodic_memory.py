"""Episodic Memory — Records task-result pairs as episodes.

Agents use this to avoid repeating mistakes. Example: CoderAgent
records every bug it fixed. Next time it sees similar code, it
remembers the fix.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """Records every task-result pair as an episode.

    Episodes are stored in SQLite and can be recalled by
    agent role and keyword similarity.
    """

    def __init__(self, db_path: str = "data/agent_memory.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_role TEXT NOT NULL,
                task TEXT NOT NULL,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                success INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                tags TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_role
                ON episodes(agent_role, created_at DESC);
        """)
        self._conn.commit()

    def record_episode(
        self,
        agent_role: str,
        task: str,
        action: str,
        outcome: str,
        success: bool,
        tags: str = "",
    ) -> None:
        """Record a task-action-outcome episode."""
        self._conn.execute(
            """
            INSERT INTO episodes
                (agent_role, task, action, outcome, success, created_at, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (agent_role, task[:500], action[:500], outcome[:500],
             int(success), time.time(), tags),
        )
        self._conn.commit()

    def recall_similar(
        self,
        agent_role: str,
        task: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Recall episodes similar to the given task.

        Uses keyword overlap scoring — lightweight alternative to
        embedding-based similarity. For production, consider
        sqlite-vec extension for true vector search.
        """
        # Extract keywords from task
        words = set(task.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "to", "in", "of", "and", "or"}
        keywords = words - stop_words

        if not keywords:
            return []

        # Build LIKE conditions for keyword matching
        conditions = " OR ".join(
            "task LIKE ?" for _ in keywords
        )
        params = [f"%{kw}%" for kw in keywords]
        params_tuple = (agent_role, *params, top_k)

        cursor = self._conn.execute(
            f"""
            SELECT task, action, outcome, success, created_at
            FROM episodes
            WHERE agent_role=? AND ({conditions})
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params_tuple,
        )

        return [
            {
                "task": row[0],
                "action": row[1],
                "outcome": row[2],
                "success": bool(row[3]),
                "timestamp": row[4],
            }
            for row in cursor.fetchall()
        ]

    def get_recent(self, agent_role: str, limit: int = 5) -> list[dict]:
        """Get most recent episodes."""
        cursor = self._conn.execute(
            """
            SELECT task, action, outcome, success, created_at
            FROM episodes WHERE agent_role=?
            ORDER BY created_at DESC LIMIT ?
            """,
            (agent_role, limit),
        )
        return [
            {
                "task": row[0],
                "action": row[1],
                "outcome": row[2],
                "success": bool(row[3]),
            }
            for row in cursor.fetchall()
        ]

    def get_failures(self, agent_role: str, limit: int = 5) -> list[dict]:
        """Get recent failures to learn from."""
        cursor = self._conn.execute(
            """
            SELECT task, action, outcome FROM episodes
            WHERE agent_role=? AND success=0
            ORDER BY created_at DESC LIMIT ?
            """,
            (agent_role, limit),
        )
        return [
            {"task": row[0], "action": row[1], "outcome": row[2]}
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        self._conn.close()
