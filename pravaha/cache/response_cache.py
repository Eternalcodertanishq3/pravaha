"""Response Cache — Full response caching with TTL."""

from __future__ import annotations
import logging
from pravaha.cache.semantic_dedup import SemanticDedup

logger = logging.getLogger(__name__)

# ResponseCache is an alias for SemanticDedup with different defaults
class ResponseCache(SemanticDedup):
    """Full response cache optimized for longer TTLs."""

    def __init__(self, max_entries: int = 5000, ttl_seconds: int = 3600) -> None:
        super().__init__(max_entries=max_entries, ttl_seconds=ttl_seconds)
