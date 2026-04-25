"""Semantic Memory — Embedding-indexed fact store.

Lightweight local vector similarity using cosine distance on
TF-IDF-style vectors stored directly in SQLite. No heavy
ChromaDB or external vector DB needed.

For production scale, swap in sqlite-vec extension for true
HNSW vector search without changing the API surface.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Embedding-indexed fact store with lightweight vector similarity.

    Uses TF-IDF term vectors stored as JSON in SQLite. Cosine
    similarity is computed in Python for recall. This avoids
    any dependency on external vector databases or heavy embedding
    models while still enabling meaningful semantic search.

    For high-throughput production use, consider the sqlite-vec
    extension which provides native HNSW indexing.
    """

    def __init__(self, db_path: str = "data/agent_memory.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS semantic_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_role TEXT NOT NULL,
                fact TEXT NOT NULL,
                terms TEXT NOT NULL,
                created_at REAL NOT NULL,
                source TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_role
                ON semantic_facts(agent_role);
        """)
        self._conn.commit()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer."""
        import re
        words = re.findall(r"\b[a-z][a-z0-9_]+\b", text.lower())
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "shall", "should", "may", "might", "must", "can",
            "could", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "and", "or", "but", "not", "this", "that", "it", "its",
        }
        return [w for w in words if w not in stop and len(w) > 1]

    @staticmethod
    def _tf_vector(tokens: list[str]) -> dict[str, float]:
        """Compute term-frequency vector."""
        counts = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {term: count / total for term, count in counts.items()}

    @staticmethod
    def _cosine_similarity(
        v1: dict[str, float], v2: dict[str, float]
    ) -> float:
        """Compute cosine similarity between two sparse vectors."""
        common = set(v1.keys()) & set(v2.keys())
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def store_fact(
        self,
        agent_role: str,
        fact: str,
        source: str = "",
    ) -> None:
        """Store a fact with its term vector."""
        import json
        tokens = self._tokenize(fact)
        terms = json.dumps(self._tf_vector(tokens))
        self._conn.execute(
            """
            INSERT INTO semantic_facts
                (agent_role, fact, terms, created_at, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (agent_role, fact, terms, time.time(), source),
        )
        self._conn.commit()

    def recall(
        self,
        agent_role: str,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.1,
    ) -> list[dict]:
        """Find facts semantically similar to the query.

        Uses TF-IDF cosine similarity — lightweight but effective
        for keyword-heavy factual content.
        """
        import json

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_vec = self._tf_vector(query_tokens)

        cursor = self._conn.execute(
            "SELECT fact, terms, source FROM semantic_facts WHERE agent_role=?",
            (agent_role,),
        )

        scored: list[tuple[float, str, str]] = []
        for fact, terms_json, source in cursor.fetchall():
            try:
                fact_vec = json.loads(terms_json)
            except Exception:
                continue
            sim = self._cosine_similarity(query_vec, fact_vec)
            if sim >= min_similarity:
                scored.append((sim, fact, source))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {"fact": fact, "similarity": round(sim, 3), "source": source}
            for sim, fact, source in scored[:top_k]
        ]

    def count(self, agent_role: str) -> int:
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM semantic_facts WHERE agent_role=?",
            (agent_role,),
        )
        return cursor.fetchone()[0]

    def close(self) -> None:
        self._conn.close()
