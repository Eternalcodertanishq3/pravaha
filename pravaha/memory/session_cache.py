"""Session KV-Cache Persistence — Reuse KV cache across HTTP requests.

Feature 2: When a user sends 10 messages in a conversation, each message
reuses the KV-cache from all previous turns. vLLM recomputes from scratch
per request. Pravaha doesn't.

Storage: in-memory LRU dict keyed by session_id.
Eviction: LRU by last-access time.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Persisted KV-cache state for a conversation session.

    Attributes:
        session_id: Unique session identifier.
        block_table: Physical block IDs holding the cached KV state.
        context_len: Number of tokens already in the cache.
        message_count: Number of messages processed in this session.
        last_access: Timestamp of last use (for LRU eviction).
        created_at: Timestamp of session creation.
    """

    session_id: str
    block_table: list[int] = field(default_factory=list)
    context_len: int = 0
    message_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)


class SessionKVCache:
    """Persist KV cache blocks across HTTP requests in the same session.

    Why this matters: In a multi-turn conversation, the model needs the
    KV cache from all previous turns. Without session persistence, every
    new message requires re-computing the entire conversation history
    through the model (expensive prefill). With session persistence,
    we skip straight to generating the response.

    Thread-safety: All operations are protected by a lock since multiple
    HTTP requests may access sessions concurrently.
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        ttl_seconds: int = 3600,
    ) -> None:
        """Initialize the session cache.

        Args:
            max_sessions: Maximum concurrent sessions before LRU eviction.
            ttl_seconds: Time-to-live for sessions in seconds.
        """
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = threading.Lock()

        logger.info(f"SessionKVCache initialized: max_sessions={max_sessions}, ttl={ttl_seconds}s")

    def save(
        self,
        session_id: str,
        block_table: list[int],
        context_len: int,
    ) -> None:
        """Save or update KV-cache state for a session.

        If the session already exists, its state is updated. If it's new,
        a new entry is created. LRU eviction is triggered if max_sessions
        is exceeded.

        Args:
            session_id: Unique session identifier.
            block_table: Physical block IDs holding the cached state.
            context_len: Number of tokens in the cache.
        """
        with self._lock:
            now = time.time()

            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.block_table = block_table
                session.context_len = context_len
                session.message_count += 1
                session.last_access = now
                # Move to end (most recently used)
                self._sessions.move_to_end(session_id)
            else:
                self._sessions[session_id] = SessionState(
                    session_id=session_id,
                    block_table=block_table,
                    context_len=context_len,
                    message_count=1,
                    last_access=now,
                    created_at=now,
                )

            # Evict if over capacity
            self._evict_if_needed()

    def load(self, session_id: str) -> tuple[list[int], int] | None:
        """Load cached KV state for a session.

        Args:
            session_id: Session to look up.

        Returns:
            Tuple of (block_table, context_len) if found, None otherwise.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            # Check TTL
            if time.time() - session.last_access > self.ttl_seconds:
                logger.info(f"Session {session_id} expired (TTL).")
                self._remove_session(session_id)
                return None

            # Update access time and move to end
            session.last_access = time.time()
            self._sessions.move_to_end(session_id)

            return session.block_table.copy(), session.context_len

    def remove(self, session_id: str) -> bool:
        """Explicitly remove a session.

        Args:
            session_id: Session to remove.

        Returns:
            True if the session existed and was removed.
        """
        with self._lock:
            return self._remove_session(session_id)

    def _remove_session(self, session_id: str) -> bool:
        """Internal: remove a session without locking.

        Args:
            session_id: Session to remove.

        Returns:
            True if found and removed.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def _evict_if_needed(self) -> None:
        """Evict oldest sessions if we exceed max_sessions."""
        while len(self._sessions) > self.max_sessions:
            # Pop the first (oldest/LRU) entry
            evicted_id, evicted = self._sessions.popitem(last=False)
            logger.info(
                f"SessionKVCache: evicted session {evicted_id} "
                f"(age={time.time() - evicted.created_at:.0f}s, "
                f"msgs={evicted.message_count})"
            )

    def evict_expired(self) -> int:
        """Remove all sessions that have exceeded their TTL.

        Returns:
            Number of sessions evicted.
        """
        with self._lock:
            now = time.time()
            expired = [
                sid
                for sid, state in self._sessions.items()
                if now - state.last_access > self.ttl_seconds
            ]
            for sid in expired:
                self._remove_session(sid)

            if expired:
                logger.info(f"SessionKVCache: evicted {len(expired)} expired sessions.")
            return len(expired)

    def list_sessions(self) -> list[dict[str, object]]:
        """List all active sessions with metadata.

        Returns:
            List of session info dictionaries.
        """
        with self._lock:
            now = time.time()
            return [
                {
                    "session_id": state.session_id,
                    "context_len": state.context_len,
                    "message_count": state.message_count,
                    "age_seconds": round(now - state.created_at, 1),
                    "idle_seconds": round(now - state.last_access, 1),
                    "num_blocks": len(state.block_table),
                }
                for state in self._sessions.values()
            ]

    def get_stats(self) -> dict[str, int | float]:
        """Return cache statistics.

        Returns:
            Dictionary with session count, capacity, and block usage.
        """
        with self._lock:
            total_blocks = sum(len(s.block_table) for s in self._sessions.values())
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "total_cached_blocks": total_blocks,
                "capacity_pct": round(len(self._sessions) / self.max_sessions * 100, 1)
                if self.max_sessions > 0
                else 0.0,
            }
