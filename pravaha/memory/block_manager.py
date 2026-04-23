"""Block Manager — Wraps the Rust BlockAllocator for Python usage.

# Phase 4: Paged attention memory management.

Provides a Python-friendly interface to the high-performance Rust-based
BlockAllocator. Handles block allocation, reference counting, prefix
sharing, and eviction policies.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


class BlockManager:
    """High-level block management wrapping the Rust BlockAllocator.

    Why this exists: The Rust BlockAllocator is extremely fast but low-level.
    This manager adds prefix sharing logic, eviction policies, and
    Python-friendly APIs on top.

    Attributes:
        num_blocks: Total number of physical blocks in the pool.
        block_size: Number of tokens per block.
        allocator: The Rust-backed BlockAllocator instance.
    """

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        """Initialize the block manager.

        Args:
            num_blocks: Total physical blocks to manage.
            block_size: Tokens per block (must match KV cache layout).
        """
        try:
            from pravaha.pravaha_core import BlockAllocator

            self.allocator = BlockAllocator(num_blocks)
            self._rust_available = True
        except ImportError:
            logger.warning(
                "Rust BlockAllocator not available. Using Python fallback. "
                "Build with `maturin develop` for production performance."
            )
            self.allocator = _PythonBlockAllocator(num_blocks)
            self._rust_available = False

        self.num_blocks = num_blocks
        self.block_size = block_size

        # Prefix sharing: content_hash → block_id
        self.hash_to_block: dict[str, int] = {}

        logger.info(
            f"BlockManager initialized: {num_blocks} blocks × {block_size} tokens/block "
            f"({'Rust' if self._rust_available else 'Python fallback'})"
        )

    def allocate(self, num_required: int) -> list[int]:
        """Allocate contiguous blocks from the free pool.

        Args:
            num_required: Number of blocks to allocate.

        Returns:
            List of allocated physical block IDs.

        Raises:
            RuntimeError: If not enough free blocks are available.
        """
        if self.num_free_blocks() < num_required:
            raise RuntimeError(
                f"Out of memory: need {num_required} blocks but only {self.num_free_blocks()} free"
            )
        return self.allocator.allocate(num_required)

    def free(self, block_id: int) -> None:
        """Release a block back to the free pool.

        Decrements the reference count. The block is only freed when
        the ref count reaches zero (supports prefix sharing).

        Args:
            block_id: Physical block ID to free.
        """
        self.allocator.free(block_id)

    def free_blocks(self, block_ids: list[int]) -> None:
        """Release multiple blocks at once.

        Args:
            block_ids: List of physical block IDs to free.
        """
        for bid in block_ids:
            self.free(bid)

    def num_free_blocks(self) -> int:
        """Return the number of currently free blocks."""
        return self.allocator.num_free_blocks()

    def get_ref_count(self, block_id: int) -> int:
        """Get the reference count of a block.

        Args:
            block_id: Physical block ID.

        Returns:
            Current reference count (0 = free, >1 = shared).
        """
        return self.allocator.get_ref_count(block_id)

    def increment_ref(self, block_id: int) -> None:
        """Increment the reference count for prefix sharing.

        When multiple requests share the same prompt prefix, they share
        physical KV cache blocks. This increments the refcount so the
        block isn't freed until all sharers are done.

        Args:
            block_id: Physical block ID to share.
        """
        self.allocator.increment_ref(block_id)

    def touch(self, block_id: int) -> None:
        """Update the last-accessed timestamp for LRU eviction.

        Args:
            block_id: Physical block ID to touch.
        """
        self.allocator.touch(block_id)

    def swap_out(self, block_id: int) -> None:
        """Mark a block as swapped to CPU.

        Args:
            block_id: Physical block ID.
        """
        self.allocator.swap_out(block_id)

    def swap_in(self, block_id: int) -> None:
        """Mark a block as restored to GPU.

        Args:
            block_id: Physical block ID.
        """
        self.allocator.swap_in(block_id)

    def get_lru_block(self) -> int | None:
        """Find the least recently used GPU-resident block.

        Returns:
            Block ID of the LRU block, or None if no GPU blocks exist.
        """
        return self.allocator.get_lru_block()

    def compute_content_hash(self, token_ids: list[int]) -> str:
        """Hash a sequence of token IDs for prefix sharing lookup.

        Args:
            token_ids: Token IDs representing the block content.

        Returns:
            SHA-256 hex digest of the content.
        """
        content = str(tuple(token_ids))
        return hashlib.sha256(content.encode()).hexdigest()

    def try_share_block(self, token_ids: list[int]) -> int | None:
        """Try to find a shared block with matching content.

        If found, increments the reference count and returns the block ID.

        Args:
            token_ids: Token IDs to match against existing blocks.

        Returns:
            Shared block ID if found, None otherwise.
        """
        content_hash = self.compute_content_hash(token_ids)
        shared_id = self.hash_to_block.get(content_hash)

        if shared_id is not None:
            try:
                if self.get_ref_count(shared_id) > 0:
                    self.increment_ref(shared_id)
                    return shared_id
            except Exception:
                self.hash_to_block.pop(content_hash, None)

        return None

    def register_block_content(self, block_id: int, token_ids: list[int]) -> None:
        """Register a block's content hash for future prefix sharing.

        Args:
            block_id: Physical block ID.
            token_ids: Token IDs stored in this block.
        """
        content_hash = self.compute_content_hash(token_ids)
        self.hash_to_block[content_hash] = block_id

    def get_usage_stats(self) -> dict[str, float]:
        """Return block pool usage statistics.

        Returns:
            Dictionary with usage percentage, free count, and total count.
        """
        free = self.num_free_blocks()
        used = self.num_blocks - free
        pct = (used / self.num_blocks * 100) if self.num_blocks > 0 else 0.0
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": free,
            "used_blocks": used,
            "usage_pct": round(pct, 1),
        }


class _PythonBlockAllocator:
    """Pure-Python fallback for when the Rust extension isn't built.

    Mirrors the Rust BlockAllocator API for development/testing without
    a Rust toolchain.
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))
        self.ref_counts: list[int] = [0] * num_blocks
        self.states: list[str] = ["free"] * num_blocks

    def allocate(self, num_required: int) -> list[int]:
        if len(self.free_blocks) < num_required:
            raise RuntimeError("Out of memory blocks")
        allocated = []
        for _ in range(num_required):
            bid = self.free_blocks.pop()
            self.ref_counts[bid] = 1
            self.states[bid] = "gpu"
            allocated.append(bid)
        return allocated

    def free(self, block_id: int) -> None:
        if block_id >= self.num_blocks:
            raise ValueError(f"Invalid block_id: {block_id}")
        if self.ref_counts[block_id] > 0:
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                self.states[block_id] = "free"
                self.free_blocks.append(block_id)

    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_ref_count(self, block_id: int) -> int:
        return self.ref_counts[block_id]

    def increment_ref(self, block_id: int) -> None:
        if self.states[block_id] == "free":
            raise RuntimeError("Cannot increment ref on free block")
        self.ref_counts[block_id] += 1

    def touch(self, block_id: int) -> None:
        pass  # No LRU tracking in fallback

    def swap_out(self, block_id: int) -> None:
        if self.states[block_id] == "gpu":
            self.states[block_id] = "cpu"

    def swap_in(self, block_id: int) -> None:
        if self.states[block_id] == "cpu":
            self.states[block_id] = "gpu"

    def get_lru_block(self) -> int | None:
        for i in range(self.num_blocks):
            if self.states[i] == "gpu" and self.ref_counts[i] > 0:
                return i
        return None
