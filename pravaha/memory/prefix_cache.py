"""Prefix Cache — Cross-request prefix sharing for KV blocks.

# Phase 4: Shared prefix optimization.

When multiple requests share the same prompt prefix (e.g., the same system
prompt), they can share the KV cache blocks for that prefix. This dramatically
reduces memory usage in swarm scenarios where all 32 agents share the same
system instructions.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PrefixEntry:
    """A cached prefix with its physical block IDs and reference count.

    Attributes:
        content_hash: SHA-256 of the token ID sequence.
        block_ids: Physical block IDs storing this prefix.
        ref_count: Number of active requests sharing this prefix.
        token_count: Number of tokens in the prefix.
    """

    content_hash: str
    block_ids: list[int]
    ref_count: int = 1
    token_count: int = 0


class PrefixCache:
    """Cross-request prefix sharing for KV cache blocks.

    Why this matters for swarm: When 32 agents all share the same system
    prompt (e.g., 200 tokens), without prefix sharing each agent allocates
    its own KV blocks for those 200 tokens. With prefix sharing, those
    blocks are computed and stored ONCE, then shared via reference counting.

    At 16 tokens/block, a 200-token system prompt uses 13 blocks.
    With 32 agents, that's 32×13=416 blocks without sharing vs 13 with sharing.
    That's a 32x memory savings for the prefix portion.
    """

    def __init__(self, block_size: int = 16) -> None:
        """Initialize the prefix cache.

        Args:
            block_size: Tokens per block (must match the KV cache layout).
        """
        self.block_size = block_size
        # content_hash → PrefixEntry
        self._entries: dict[str, PrefixEntry] = {}
        # block_id → content_hash (reverse lookup for cleanup)
        self._block_to_hash: dict[int, str] = {}

        self._hits: int = 0
        self._misses: int = 0

    def compute_prefix_hash(self, token_ids: list[int], block_idx: int) -> str:
        """Compute a hash for a specific block of tokens within a prefix.

        Args:
            token_ids: Full token ID sequence.
            block_idx: Which block (0-indexed) to hash.

        Returns:
            SHA-256 hex digest of the block's token content.
        """
        start = block_idx * self.block_size
        end = start + self.block_size
        block_tokens = tuple(token_ids[start:end])
        return hashlib.sha256(str(block_tokens).encode()).hexdigest()

    def lookup(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Find shared prefix blocks for the given token sequence.

        Tries to match as many complete blocks as possible from the cache.
        Returns the shared block IDs and how many tokens they cover.

        Args:
            token_ids: Token IDs of the prompt being processed.

        Returns:
            Tuple of (shared_block_ids, tokens_covered).
        """
        shared_blocks: list[int] = []
        tokens_covered = 0
        num_full_blocks = len(token_ids) // self.block_size

        for block_idx in range(num_full_blocks):
            block_hash = self.compute_prefix_hash(token_ids, block_idx)
            entry = self._entries.get(block_hash)

            if entry is not None and entry.ref_count > 0:
                # Found a matching prefix block
                shared_blocks.extend(entry.block_ids)
                entry.ref_count += 1
                tokens_covered = (block_idx + 1) * self.block_size
                self._hits += 1
            else:
                # Prefix sharing breaks at first mismatch
                self._misses += 1
                break

        return shared_blocks, tokens_covered

    def register(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_idx: int,
    ) -> None:
        """Register a newly computed block for future prefix sharing.

        Args:
            token_ids: Full token ID sequence.
            block_ids: Physical block IDs for this block.
            block_idx: Which block in the sequence (0-indexed).
        """
        block_hash = self.compute_prefix_hash(token_ids, block_idx)

        if block_hash not in self._entries:
            self._entries[block_hash] = PrefixEntry(
                content_hash=block_hash,
                block_ids=block_ids,
                ref_count=1,
                token_count=self.block_size,
            )
            for bid in block_ids:
                self._block_to_hash[bid] = block_hash

    def release(self, token_ids: list[int], num_blocks: int) -> list[int]:
        """Release prefix sharing references when a request completes.

        Decrements ref counts. Returns block IDs that are now fully freed
        (ref_count reached 0) so they can be returned to the allocator.

        Args:
            token_ids: Token IDs of the completing request.
            num_blocks: Number of prefix blocks that were shared.

        Returns:
            List of block IDs that are now free (ref_count = 0).
        """
        freed: list[int] = []

        for block_idx in range(num_blocks):
            block_hash = self.compute_prefix_hash(token_ids, block_idx)
            entry = self._entries.get(block_hash)

            if entry is not None:
                entry.ref_count -= 1
                if entry.ref_count <= 0:
                    freed.extend(entry.block_ids)
                    for bid in entry.block_ids:
                        self._block_to_hash.pop(bid, None)
                    del self._entries[block_hash]

        return freed

    def get_stats(self) -> dict[str, int | float]:
        """Return prefix cache statistics.

        Returns:
            Dictionary with hit rate, entry count, and total shared blocks.
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        total_shared_blocks = sum(len(e.block_ids) for e in self._entries.values())
        total_refs = sum(e.ref_count for e in self._entries.values())

        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 1),
            "total_shared_blocks": total_shared_blocks,
            "total_refs": total_refs,
        }

    def clear(self) -> None:
        """Remove all cached prefixes."""
        self._entries.clear()
        self._block_to_hash.clear()
        self._hits = 0
        self._misses = 0
        logger.info("PrefixCache cleared.")
