from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import psycopg2
    from psycopg2.extensions import AsIs, register_adapter

    def adapt_numpy_array(numpy_array):
        return AsIs(f"'{numpy_array.tolist()}'")

    register_adapter(np.ndarray, adapt_numpy_array)
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class AlloyDBMemoryStore:
    """AlloyDB Omni memory store for agent memories with pgvector support."""

    def __init__(self, host=None, port=None, dbname=None, user=None, password=None):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is not installed.")

        self.host = host or os.environ.get("ALLOYDB_HOST", "localhost")
        self.port = port or os.environ.get("ALLOYDB_PORT", "5432")
        self.dbname = dbname or os.environ.get("ALLOYDB_DB", "pravaha")
        self.user = user or os.environ.get("ALLOYDB_USER", "pravaha")
        self.password = password or os.environ.get("ALLOYDB_PASSWORD", "pravaha")

        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )
        self.conn.autocommit = True

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None

        self._init_db()

    def _init_db(self):
        """Initializes database schema."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_memories (
                    id SERIAL PRIMARY KEY,
                    agent_role TEXT NOT NULL,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    embedding vector(384),
                    created_at DOUBLE PRECISION NOT NULL,
                    accessed_at DOUBLE PRECISION NOT NULL,
                    importance DOUBLE PRECISION DEFAULT 0.5,
                    access_count INTEGER DEFAULT 1,
                    UNIQUE(agent_role, namespace, key)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_role ON agent_memories(agent_role, accessed_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON agent_memories(agent_role, importance DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_embedding ON agent_memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")

    def _embed(self, text: str) -> list[float]:
        """Generates embedding for text."""
        if self.model:
            return self.model.encode(text).tolist()

        import hashlib
        import math
        dim = 384
        h = hashlib.sha256(text.encode("utf-8")).digest()
        embedding = [0.0] * dim
        for i, b in enumerate(h):
            embedding[i % dim] += b / 255.0

        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding

    def put(self, agent_role: str, key: str, value: str, importance: float = 0.5, namespace: str = "default") -> None:
        embedding = self._embed(value)
        now = time.time()
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agent_memories (agent_role, namespace, key, value, embedding, created_at, accessed_at, importance, access_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (agent_role, namespace, key)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    embedding = EXCLUDED.embedding,
                    accessed_at = EXCLUDED.accessed_at,
                    importance = EXCLUDED.importance,
                    access_count = agent_memories.access_count + 1
            """, (agent_role, namespace, key, value, str(embedding), now, now, importance, 1))

    def get(self, agent_role: str, key: str) -> dict[str, Any] | None:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT value, importance, namespace, created_at, accessed_at, access_count
                FROM agent_memories
                WHERE agent_role = %s AND key = %s
            """, (agent_role, key))
            row = cur.fetchone()
            if row:
                now = time.time()
                cur.execute("UPDATE agent_memories SET accessed_at = %s, access_count = access_count + 1 WHERE agent_role = %s AND key = %s", (now, agent_role, key))
                return {
                    "value": row[0],
                    "importance": row[1],
                    "namespace": row[2],
                    "created_at": row[3],
                    "accessed_at": row[4],
                    "access_count": row[5]
                }
        return None

    def get_recent(self, agent_role: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT key, value, importance, namespace, created_at, accessed_at, access_count
                FROM agent_memories
                WHERE agent_role = %s
                ORDER BY accessed_at DESC LIMIT %s
            """, (agent_role, limit))
            return [{"key": r[0], "value": r[1], "importance": r[2], "namespace": r[3], "created_at": r[4], "accessed_at": r[5], "access_count": r[6]} for r in cur.fetchall()]

    def get_important(self, agent_role: str, min_importance: float = 0.5) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT key, value, importance, namespace, created_at, accessed_at, access_count
                FROM agent_memories
                WHERE agent_role = %s AND importance >= %s
                ORDER BY importance DESC
            """, (agent_role, min_importance))
            return [{"key": r[0], "value": r[1], "importance": r[2], "namespace": r[3], "created_at": r[4], "accessed_at": r[5], "access_count": r[6]} for r in cur.fetchall()]

    def search(self, agent_role: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        embedding = self._embed(query)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT key, value, importance, namespace, created_at, accessed_at, access_count,
                (0.7 * (1 - (embedding <=> %s::vector)) + 0.3 * (CASE WHEN value ILIKE %s THEN 1.0 ELSE 0.0 END)) AS score
                FROM agent_memories
                WHERE agent_role = %s
                ORDER BY score DESC LIMIT %s
            """, (str(embedding), f"%{query}%", agent_role, limit))
            return [{"key": r[0], "value": r[1], "importance": r[2], "namespace": r[3], "created_at": r[4], "accessed_at": r[5], "access_count": r[6], "score": r[7]} for r in cur.fetchall()]

    def semantic_search(self, agent_role: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        embedding = self._embed(query)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT key, value, importance, namespace, created_at, accessed_at, access_count,
                (1 - (embedding <=> %s::vector)) AS score
                FROM agent_memories
                WHERE agent_role = %s
                ORDER BY score DESC LIMIT %s
            """, (str(embedding), agent_role, limit))
            return [{"key": r[0], "value": r[1], "importance": r[2], "namespace": r[3], "created_at": r[4], "accessed_at": r[5], "access_count": r[6], "score": r[7]} for r in cur.fetchall()]

    def count(self, agent_role: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM agent_memories WHERE agent_role = %s", (agent_role,))
            return cur.fetchone()[0]

    def clear(self, agent_role: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM agent_memories WHERE agent_role = %s", (agent_role,))

    def close(self) -> None:
        if self.conn:
            self.conn.close()

def create_memory_store(backend: str = "auto") -> Any:
    """Factory for memory store."""
    if backend == "auto":
        if PSYCOPG2_AVAILABLE:
            try:
                return AlloyDBMemoryStore()
            except Exception as e:
                logger.warning(f"Failed to connect to AlloyDB, falling back to SQLite: {e}")
                pass

    if backend == "alloydb":
        return AlloyDBMemoryStore()

    class MockSQLiteMemoryStore:
        def __init__(self):
            self.memories = []
        def count(self, agent_role: str) -> int: return 0
        def close(self) -> None: pass

    return MockSQLiteMemoryStore()
