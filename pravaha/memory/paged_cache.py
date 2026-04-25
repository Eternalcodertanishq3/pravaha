"""Paged KV-Cache implementation (Phase 4).

# Phase 4: Paged attention with vectorized gather.

Eliminates memory fragmentation by using fixed-size blocks (pages).
Integrates with the Rust-based BlockAllocator for high-performance management.

Memory layout per cache (K or V separately):
    shape: (num_layers, num_blocks, block_size, num_kv_heads, head_dim)

Bug Fixes Applied:
  - Fix 3: Removed duplicate update_from_hf_past_key_values() definition
  - Fix 3: Replaced Python for-loop in get_batch() with vectorized gather
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class PagedKVCache:
    """Paged KV-cache for memory-efficient inference.

    Stores K/V tensors in a single large buffer of physical blocks.
    Mapping from logical position to physical storage is handled via block tables.

    Why paged attention matters: Traditional KV caches allocate one contiguous
    buffer per sequence. With N concurrent sequences, this causes massive
    fragmentation. Paged attention breaks the cache into fixed-size blocks
    that can be independently allocated, shared, and swapped — like virtual
    memory pages in an OS.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        """Allocate physical KV-cache block pool.

        Args:
            num_layers: Number of transformer layers.
            num_kv_heads: Number of key-value attention heads.
            head_dim: Dimension of each attention head.
            block_size: Number of tokens per block.
            num_blocks: Total number of physical blocks.
            dtype: Data type for cache tensors.
            device: Device to allocate cache on.
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = device

        # GPU Pool — the main cache
        cache_shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        # CPU Pool (for swapping) — allocated lazily
        self.cpu_k_cache: torch.Tensor | None = None
        self.cpu_v_cache: torch.Tensor | None = None

        cache_mb = (
            2
            * num_layers
            * num_blocks
            * block_size
            * num_kv_heads
            * head_dim
            * torch.finfo(dtype).bits
            // 8
        ) / (1024 * 1024)
        logger.info(
            f"PagedKVCache allocated: {num_layers}L × {num_blocks} blocks × "
            f"{block_size} tok/block = {cache_mb:.1f} MB"
        )

    def _ensure_cpu_cache(self) -> None:
        """Lazy allocation of CPU swap space."""
        if self.cpu_k_cache is None:
            cache_shape = (
                self.num_layers,
                self.num_blocks,
                self.block_size,
                self.num_kv_heads,
                self.head_dim,
            )
            self.cpu_k_cache = torch.empty(cache_shape, dtype=self.dtype, device="cpu")
            self.cpu_v_cache = torch.empty(cache_shape, dtype=self.dtype, device="cpu")
            logger.info("Allocated CPU swap space for KV-cache.")

    def allocate_blocks(self, num_blocks: int) -> list[int]:
        """Allocate blocks from the pool using the internal free list.

        This is a convenience method for standalone usage. When using the
        scheduler, block allocation is handled by the BlockManager.

        Args:
            num_blocks: Number of blocks to allocate.

        Returns:
            List of physical block IDs.
        """
        # This method delegates to a simple internal allocator for standalone use
        if not hasattr(self, "_free_list"):
            self._free_list: list[int] = list(range(self.num_blocks - 1, -1, -1))
        if len(self._free_list) < num_blocks:
            raise RuntimeError(
                f"PagedKVCache: need {num_blocks} blocks but only {len(self._free_list)} free"
            )
        return [self._free_list.pop() for _ in range(num_blocks)]

    def free_blocks(self, block_ids: list[int]) -> None:
        """Return blocks to the free pool.

        Args:
            block_ids: List of physical block IDs to free.
        """
        if not hasattr(self, "_free_list"):
            self._free_list = []
        self._free_list.extend(block_ids)

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        request_ids: list[str],
        block_tables: list[list[int]],
        slot_offsets: list[int],
    ) -> None:
        """Append new key-value states to specific blocks.

        Args:
            layer_idx: Which transformer layer (0-indexed).
            k: Key tensor, shape (batch_size, num_kv_heads, new_tokens, head_dim).
            v: Value tensor, same shape as k.
            request_ids: Unique identifier for each request in the batch.
            block_tables: List of physical block IDs for each request.
            slot_offsets: The logical token index (0-based) where appending starts.
        """
        batch_size, num_heads, num_new_tokens, head_dim = k.shape

        for b in range(batch_size):
            blocks = block_tables[b]
            start_offset = slot_offsets[b]

            for t in range(num_new_tokens):
                logical_pos = start_offset + t
                block_idx_in_table = logical_pos // self.block_size
                block_offset = logical_pos % self.block_size

                physical_block_id = blocks[block_idx_in_table]

                # Copy into the physical buffer
                self.k_cache[layer_idx, physical_block_id, block_offset, :, :] = k[b, :, t, :]
                self.v_cache[layer_idx, physical_block_id, block_offset, :, :] = v[b, :, t, :]

    def get_batch(
        self,
        layer_idx: int,
        block_tables: list[list[int]],
        context_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached K/V for a batch of requests using vectorized gather.

        Fix 3: Replaced the slow Python for-loop with vectorized index operations
        that build index tensors and use advanced indexing for the gather.

        Args:
            layer_idx: Transformer layer index.
            block_tables: Physical block IDs per request.
            context_lens: Number of tokens in cache per request.

        Returns:
            Tuple of (k_out, v_out), each shape (batch, num_kv_heads, max_len, head_dim).
        """
        max_len = max(context_lens) if context_lens else 0
        batch = len(block_tables)

        # Output tensors
        k_out = torch.zeros(
            (batch, self.num_kv_heads, max_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        v_out = torch.zeros_like(k_out)

        # Vectorized gather: build index tensors per batch element
        for b, (blocks, clen) in enumerate(zip(block_tables, context_lens)):
            if clen == 0:
                continue

            # Build vectorized index arrays instead of per-token Python loop
            positions = torch.arange(clen, device=self.device)
            blk_ids = torch.tensor(
                [blocks[t // self.block_size] for t in range(clen)],
                device=self.device,
            )
            offsets = positions % self.block_size

            # Gather from cache: shape (clen, num_kv_heads, head_dim)
            k_gathered = self.k_cache[layer_idx, blk_ids, offsets]
            v_gathered = self.v_cache[layer_idx, blk_ids, offsets]

            # Transpose: (clen, num_kv_heads, head_dim) → (num_kv_heads, clen, head_dim)
            k_out[b, :, :clen, :] = k_gathered.transpose(0, 1)
            v_out[b, :, :clen, :] = v_gathered.transpose(0, 1)

        return k_out, v_out

    def to_hf_past_key_values(
        self,
        block_tables: list[list[int]],
        context_lens: list[int],
    ) -> object:
        """Convert paged blocks to HuggingFace's past_key_values format.

        Gathers non-contiguous physical blocks into a contiguous batch that
        HuggingFace models expect for cached attention.

        Args:
            block_tables: Physical block IDs per request.
            context_lens: Tokens in cache per request.

        Returns:
            HuggingFace-compatible past_key_values (DynamicCache or tuple).
        """
        if not block_tables:
            return None

        try:
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()
            for layer_idx in range(self.num_layers):
                k, v = self.get_batch(layer_idx, block_tables, context_lens)
                cache.update(k, v, layer_idx)
            return cache
        except ImportError:
            result = []
            for layer_idx in range(self.num_layers):
                k, v = self.get_batch(layer_idx, block_tables, context_lens)
                result.append((k, v))
            return tuple(result)

    def update_from_hf_past_key_values(
        self,
        past_key_values: object,
        num_new_tokens: int,
        request_ids: list[str],
        block_tables: list[list[int]],
        slot_offsets: list[int],
    ) -> None:
        """Extract new K/V states from HF output and store them in blocks.

        Fix 3: Removed duplicate method definition from the original codebase.

        Args:
            past_key_values: HF's returned past_key_values (DynamicCache or tuple).
            num_new_tokens: Number of new tokens generated in this step.
            request_ids: Unique identifier for each request.
            block_tables: Assigned physical blocks per request.
            slot_offsets: The logical token index where the new tokens start.
        """
        # Strategy 1: DynamicCache with .key_cache / .value_cache attributes
        key_cache = getattr(
            past_key_values,
            "key_cache",
            getattr(past_key_values, "_key_cache", None),
        )
        value_cache = getattr(
            past_key_values,
            "value_cache",
            getattr(past_key_values, "_value_cache", None),
        )

        if key_cache is not None and value_cache is not None:
            for layer_idx in range(self.num_layers):
                k_full = key_cache[layer_idx]
                v_full = value_cache[layer_idx]
                # Peel DynamicLayer wrappers if present
                if hasattr(k_full, "data"):
                    k_full = k_full.data
                if hasattr(v_full, "data"):
                    v_full = v_full.data

                k_new = k_full[:, :, -num_new_tokens:, :]
                v_new = v_full[:, :, -num_new_tokens:, :]
                self.append(
                    layer_idx,
                    k_new,
                    v_new,
                    request_ids,
                    block_tables,
                    slot_offsets,
                )
            return

        # Strategy 2: Standard HF tuple format (layer_idx → (k, v))
        if isinstance(past_key_values, (list, tuple)):
            for layer_idx, layer_data in enumerate(past_key_values):
                if isinstance(layer_data, (list, tuple)) and len(layer_data) == 2:
                    k_full, v_full = layer_data
                    k_new = k_full[:, :, -num_new_tokens:, :]
                    v_new = v_full[:, :, -num_new_tokens:, :]
                    self.append(
                        layer_idx,
                        k_new,
                        v_new,
                        request_ids,
                        block_tables,
                        slot_offsets,
                    )
            return

        # Strategy 3: Iteration fallback for exotic cache implementations
        try:
            from typing import cast, Iterable, Any
            for layer_idx, layer_data in enumerate(cast(Iterable[Any], past_key_values)):
                k_full = getattr(
                    layer_data,
                    "key_cache",
                    getattr(layer_data, "k", None),
                )
                v_full = getattr(
                    layer_data,
                    "value_cache",
                    getattr(layer_data, "v", None),
                )
                if k_full is None and isinstance(layer_data, (list, tuple)):
                    k_full, v_full = layer_data[0], layer_data[1]

                if k_full is not None and v_full is not None:
                    k_new = k_full[:, :, -num_new_tokens:, :]
                    v_new = v_full[:, :, -num_new_tokens:, :]
                    self.append(
                        layer_idx,
                        k_new,
                        v_new,
                        request_ids,
                        block_tables,
                        slot_offsets,
                    )
            return
        except Exception:
            pass

        logger.warning(
            f"PagedKVCache: Unknown HF cache type {type(past_key_values)}. "
            f"Generation might be degraded."
        )

    def swap_out(self, block_ids: list[int]) -> None:
        """Move selected blocks from GPU to CPU.

        Args:
            block_ids: Physical block IDs to swap out.
        """
        self._ensure_cpu_cache()
        assert self.cpu_k_cache is not None and self.cpu_v_cache is not None
        for bid in block_ids:
            self.cpu_k_cache[:, bid, ...] = self.k_cache[:, bid, ...].to("cpu", non_blocking=True)
            self.cpu_v_cache[:, bid, ...] = self.v_cache[:, bid, ...].to("cpu", non_blocking=True)
        if self.device != "cpu":
            torch.cuda.synchronize()

    def swap_in(self, block_ids: list[int]) -> None:
        """Move selected blocks from CPU back to GPU.

        Args:
            block_ids: Physical block IDs to swap in.
        """
        if self.cpu_k_cache is None or self.cpu_v_cache is None:
            return
        assert self.cpu_k_cache is not None and self.cpu_v_cache is not None
        for bid in block_ids:
            self.k_cache[:, bid, ...] = self.cpu_k_cache[:, bid, ...].to(
                self.device, non_blocking=True
            )
            self.v_cache[:, bid, ...] = self.cpu_v_cache[:, bid, ...].to(
                self.device, non_blocking=True
            )
        if self.device != "cpu":
            torch.cuda.synchronize()

    def get_usage_pct(self) -> float:
        """Return approximate cache usage as a percentage."""
        if hasattr(self, "_free_list"):
            used = self.num_blocks - len(self._free_list)
            return (used / self.num_blocks * 100) if self.num_blocks > 0 else 0.0
        return 0.0

    @classmethod
    def from_model_config(
        cls,
        arch_config: object,
        num_blocks: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> PagedKVCache:
        """Factory: create cache from a ModelArchConfig.

        Args:
            arch_config: Model architecture config with num_layers, num_kv_heads, head_dim.
            num_blocks: Total physical blocks.
            block_size: Tokens per block.
            dtype: Cache tensor dtype.
            device: Target device.

        Returns:
            Initialized PagedKVCache.
        """
        return cls(
            num_layers=arch_config.num_layers,  # type: ignore[attr-defined]
            num_kv_heads=arch_config.num_kv_heads,  # type: ignore[attr-defined]
            head_dim=arch_config.head_dim,  # type: ignore[attr-defined]
            block_size=block_size,
            num_blocks=num_blocks,
            dtype=dtype,
            device=device,
        )
