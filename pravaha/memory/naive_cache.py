"""Naive KV-Cache implementation (Phase 2 — educational).

# Phase 2: Pre-allocated KV cache without paging.

A simpler KV cache that pre-allocates a full contiguous buffer per sequence.
Used for educational purposes and as a fallback when paged attention isn't needed.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class NaiveKVCache:
    """Pre-allocated contiguous KV-cache (Phase 2).

    Unlike PagedKVCache, this allocates one large contiguous buffer per
    sequence slot. Simple but wastes memory due to fragmentation when
    sequences have different lengths.

    Why this exists: It's much easier to understand than paged attention.
    Researchers learning about KV caches should start here before moving
    to PagedKVCache.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        max_batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        """Allocate the full KV cache buffer.

        Args:
            num_layers: Number of transformer layers.
            num_kv_heads: Number of key-value attention heads.
            head_dim: Dimension per attention head.
            max_seq_len: Maximum sequence length to support.
            max_batch_size: Maximum number of concurrent sequences.
            dtype: Data type for cache tensors.
            device: Device for allocation.
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.device = device

        # Pre-allocated cache: (num_layers, max_batch, max_seq_len, num_kv_heads, head_dim)
        cache_shape = (num_layers, max_batch_size, max_seq_len, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        # Track how many tokens are stored per slot
        self.seq_lens: list[int] = [0] * max_batch_size

        # Track which slots are in use
        self.active_slots: set[int] = set()

        cache_mb = (
            2
            * num_layers
            * max_batch_size
            * max_seq_len
            * num_kv_heads
            * head_dim
            * torch.finfo(dtype).bits
            // 8
        ) / (1024 * 1024)
        logger.info(
            f"NaiveKVCache allocated: {num_layers}L × {max_batch_size} slots × "
            f"{max_seq_len} tokens = {cache_mb:.1f} MB"
        )

    def allocate_slot(self) -> int:
        """Find and allocate a free cache slot.

        Returns:
            Slot index.

        Raises:
            RuntimeError: If all slots are occupied.
        """
        for slot in range(self.max_batch_size):
            if slot not in self.active_slots:
                self.active_slots.add(slot)
                self.seq_lens[slot] = 0
                return slot
        raise RuntimeError(f"NaiveKVCache: all {self.max_batch_size} slots are occupied")

    def free_slot(self, slot: int) -> None:
        """Release a cache slot.

        Args:
            slot: Slot index to free.
        """
        self.active_slots.discard(slot)
        self.seq_lens[slot] = 0
        # Zero out the slot to prevent stale data
        self.k_cache[:, slot, :, :, :] = 0
        self.v_cache[:, slot, :, :, :] = 0

    def append(
        self,
        layer_idx: int,
        slot: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Append new K/V tokens to a slot.

        Args:
            layer_idx: Transformer layer index.
            slot: Cache slot index.
            k: Key tensor, shape (num_kv_heads, num_new_tokens, head_dim).
            v: Value tensor, same shape as k.
        """
        current_len = self.seq_lens[slot]
        num_new = k.shape[1]

        if current_len + num_new > self.max_seq_len:
            raise RuntimeError(
                f"NaiveKVCache: slot {slot} would exceed max_seq_len "
                f"({current_len} + {num_new} > {self.max_seq_len})"
            )

        # Write into the pre-allocated buffer
        self.k_cache[layer_idx, slot, current_len : current_len + num_new, :, :] = k.transpose(0, 1)
        self.v_cache[layer_idx, slot, current_len : current_len + num_new, :, :] = v.transpose(0, 1)

        if layer_idx == self.num_layers - 1:
            # Only increment seq_len once, after the last layer is written
            self.seq_lens[slot] = current_len + num_new

    def get(
        self,
        layer_idx: int,
        slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V for a slot.

        Args:
            layer_idx: Transformer layer index.
            slot: Cache slot index.

        Returns:
            Tuple of (k, v), each shape (num_kv_heads, seq_len, head_dim).
        """
        seq_len = self.seq_lens[slot]
        k = self.k_cache[layer_idx, slot, :seq_len, :, :].transpose(0, 1)
        v = self.v_cache[layer_idx, slot, :seq_len, :, :].transpose(0, 1)
        return k, v

    def get_batch(
        self,
        layer_idx: int,
        slots: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V for a batch of slots.

        Args:
            layer_idx: Transformer layer index.
            slots: List of cache slot indices.

        Returns:
            Tuple of (k, v), each shape (batch, num_kv_heads, max_seq_len_in_batch, head_dim).
        """
        max_len = max(self.seq_lens[s] for s in slots) if slots else 0
        batch_size = len(slots)

        k_out = torch.zeros(
            (batch_size, self.num_kv_heads, max_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        v_out = torch.zeros_like(k_out)

        for i, slot in enumerate(slots):
            slen = self.seq_lens[slot]
            if slen > 0:
                k_out[i, :, :slen, :] = self.k_cache[layer_idx, slot, :slen, :, :].transpose(0, 1)
                v_out[i, :, :slen, :] = self.v_cache[layer_idx, slot, :slen, :, :].transpose(0, 1)

        return k_out, v_out

    @classmethod
    def from_model_config(
        cls,
        arch_config: object,
        max_seq_len: int = 1024,
        max_batch_size: int = 4,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> NaiveKVCache:
        """Factory: create cache from a ModelArchConfig."""
        return cls(
            num_layers=arch_config.num_layers,  # type: ignore[attr-defined]
            num_kv_heads=arch_config.num_kv_heads,  # type: ignore[attr-defined]
            head_dim=arch_config.head_dim,  # type: ignore[attr-defined]
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            dtype=dtype,
            device=device,
        )
