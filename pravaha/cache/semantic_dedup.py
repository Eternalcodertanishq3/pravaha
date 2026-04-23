"""Semantic Deduplication — Detect and serve duplicate requests from cache."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SemanticDedup:
    """Cache responses for semantically identical requests.

    Uses exact prompt hash matching with TTL-based expiration.
    """

    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 300) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _hash(self, prompt: str, params_hash: str = "") -> str:
        return hashlib.sha256(f"{prompt}|{params_hash}".encode()).hexdigest()

    def get(self, prompt: str, params_hash: str = "") -> str | None:
        key = self._hash(prompt, params_hash)
        entry = self._cache.get(key)
        if entry is not None:
            response, ts = entry
            if time.time() - ts < self.ttl_seconds:
                self._hits += 1
                self._cache.move_to_end(key)
                return response
            del self._cache[key]
        self._misses += 1
        return None

    def put(self, prompt: str, response: str, params_hash: str = "") -> None:
        key = self._hash(prompt, params_hash)
        self._cache[key] = (response, time.time())
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def get_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, total) * 100, 1),
        }
