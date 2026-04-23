"""GPU ↔ CPU Swap Orchestration.

Manages the transfer of KV-cache blocks between GPU and CPU memory
when the GPU runs out of cache blocks. Coordinates with the scheduler
and block manager to preempt low-priority requests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SwapRecord:
    """Record of a block swap operation.

    Attributes:
        request_id: ID of the request whose blocks were swapped.
        block_ids: Physical block IDs that were moved.
        direction: 'out' (GPU→CPU) or 'in' (CPU→GPU).
        timestamp: When the swap occurred.
    """

    request_id: str
    block_ids: list[int] = field(default_factory=list)
    direction: str = "out"
    timestamp: float = 0.0


class SwapManager:
    """Orchestrates GPU ↔ CPU block swapping.

    When the GPU KV cache runs out of blocks and new requests arrive,
    the swap manager coordinates moving inactive blocks to CPU RAM
    to free GPU memory. When those requests need to resume, blocks
    are swapped back in.

    This enables oversubscription: more concurrent requests than the
    GPU cache can physically hold, at the cost of swap latency.
    """

    def __init__(
        self,
        max_swap_blocks: int = 256,
    ) -> None:
        """Initialize the swap manager.

        Args:
            max_swap_blocks: Maximum number of blocks that can be swapped to CPU.
        """
        self.max_swap_blocks = max_swap_blocks
        self._swapped_blocks: set[int] = set()
        self._swap_history: list[SwapRecord] = []

        logger.info(f"SwapManager initialized: max_swap_blocks={max_swap_blocks}")

    def can_swap_out(self, num_blocks: int) -> bool:
        """Check if there's room to swap out blocks.

        Args:
            num_blocks: Number of blocks to swap out.

        Returns:
            True if there's sufficient CPU swap space.
        """
        return len(self._swapped_blocks) + num_blocks <= self.max_swap_blocks

    def record_swap_out(
        self,
        request_id: str,
        block_ids: list[int],
    ) -> None:
        """Record blocks being swapped out to CPU.

        Args:
            request_id: ID of the owning request.
            block_ids: Block IDs being moved to CPU.
        """
        import time

        self._swapped_blocks.update(block_ids)
        self._swap_history.append(
            SwapRecord(
                request_id=request_id,
                block_ids=block_ids,
                direction="out",
                timestamp=time.time(),
            )
        )

        logger.debug(
            f"SwapManager: swapped out {len(block_ids)} blocks for {request_id}. "
            f"Total swapped: {len(self._swapped_blocks)}"
        )

    def record_swap_in(
        self,
        request_id: str,
        block_ids: list[int],
    ) -> None:
        """Record blocks being swapped back in from CPU.

        Args:
            request_id: ID of the owning request.
            block_ids: Block IDs being moved back to GPU.
        """
        import time

        self._swapped_blocks -= set(block_ids)
        self._swap_history.append(
            SwapRecord(
                request_id=request_id,
                block_ids=block_ids,
                direction="in",
                timestamp=time.time(),
            )
        )

        logger.debug(
            f"SwapManager: swapped in {len(block_ids)} blocks for {request_id}. "
            f"Total swapped: {len(self._swapped_blocks)}"
        )

    def get_swapped_count(self) -> int:
        """Return the number of blocks currently on CPU."""
        return len(self._swapped_blocks)

    def is_block_swapped(self, block_id: int) -> bool:
        """Check if a specific block is on CPU.

        Args:
            block_id: Block ID to check.

        Returns:
            True if the block is currently on CPU.
        """
        return block_id in self._swapped_blocks

    def get_stats(self) -> dict[str, int | float]:
        """Return swap statistics.

        Returns:
            Dictionary with swap counts and capacity.
        """
        swap_outs = sum(1 for r in self._swap_history if r.direction == "out")
        swap_ins = sum(1 for r in self._swap_history if r.direction == "in")

        return {
            "currently_swapped": len(self._swapped_blocks),
            "max_swap_blocks": self.max_swap_blocks,
            "total_swap_outs": swap_outs,
            "total_swap_ins": swap_ins,
            "capacity_pct": round(len(self._swapped_blocks) / self.max_swap_blocks * 100, 1)
            if self.max_swap_blocks > 0
            else 0.0,
        }

    def clear_history(self) -> None:
        """Clear swap history (but not current swap state)."""
        self._swap_history.clear()
