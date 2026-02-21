"""Paged KV-Cache implementation (Phase 4).

Eliminates memory fragmentation by using fixed-size blocks (pages).
Integrates with the Rust-based BlockAllocator for high-performance management.

Memory layout per cache (K or V separately):
    shape: (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

import torch
from pravaha_core import BlockAllocator

logger = logging.getLogger(__name__)


class PagedKVCache:
    """Paged KV-cache for memory-efficient inference.

    Stores K/V tensors in a single large buffer of physical blocks.
    Mapping from logical position to physical storage is handled via block tables.
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
    ):
        """Allocate physical KV-cache block pool."""
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = device

        # GPU Pool
        cache_shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        # CPU Pool (for swapping) - allocated on demand or as a buffer
        self.cpu_k_cache: Optional[torch.Tensor] = None
        self.cpu_v_cache: Optional[torch.Tensor] = None

        logger.info(
            f"PagedKVCache allocated: {num_layers} layers, {num_blocks} blocks."
        )

    def _ensure_cpu_cache(self):
        """Lazy allocation of CPU swap space."""
        if self.cpu_k_cache is None:
            cache_shape = (self.num_layers, self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim)
            self.cpu_k_cache = torch.empty(cache_shape, dtype=self.dtype, device="cpu")
            self.cpu_v_cache = torch.empty(cache_shape, dtype=self.dtype, device="cpu")
            logger.info("Allocated CPU swap space for KV-cache.")

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        request_ids: List[str],
        block_tables: List[List[int]],
        slot_offsets: List[int],
    ) -> None:
        """Append new key-value states to specific blocks.

        Args:
            layer_idx: Which transformer layer (0-indexed).
            k: Key tensor, shape (batch_size, num_kv_heads, new_tokens, head_dim).
               new_tokens is usually 1 for decode, or prompt_len for prefill.
            v: Value tensor, same shape as k.
            request_ids: Unique identifier for each request in the batch.
            block_tables: List of physical block IDs for each request.
            slot_offsets: The logical token index (0-based) where appending starts.
        """
        batch_size, num_heads, num_new_tokens, head_dim = k.shape

        for b in range(batch_size):
            blocks = block_tables[b]
            start_offset = slot_offsets[b]
            
            # Map each token to its physical block and offset
            for t in range(num_new_tokens):
                logical_pos = start_offset + t
                block_idx_in_table = logical_pos // self.block_size
                block_offset = logical_pos % self.block_size
                
                physical_block_id = blocks[block_idx_in_table]
                
                # Copy into the physical buffer
                # Shape in k: (num_kv_heads, head_dim)
                self.k_cache[layer_idx, physical_block_id, block_offset, :, :] = k[b, :, t, :]
                self.v_cache[layer_idx, physical_block_id, block_offset, :, :] = v[b, :, t, :]

    def get_batch(
        self,
        layer_idx: int,
        block_tables: List[List[int]],
        context_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached K/V for a batch of requests.
        
        This currently performs a naive gather-to-contiguous-buffer.
        In the future, this will be replaced by a PagedAttention kernel.
        """
        max_context_len = max(context_lens) if context_lens else 0
        batch_size = len(block_tables)
        
        # Output shape: (batch_size, num_kv_heads, max_context_len, head_dim)
        k_out = torch.zeros(
            (batch_size, self.num_kv_heads, max_context_len, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        v_out = torch.zeros(
            (batch_size, self.num_kv_heads, max_context_len, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        
        for b in range(batch_size):
            blocks = block_tables[b]
            current_len = context_lens[b]
            
            for t in range(current_len):
                block_idx_in_table = t // self.block_size
                block_offset = t % self.block_size
                physical_block_id = blocks[block_idx_in_table]
                
                k_out[b, :, t, :] = self.k_cache[layer_idx, physical_block_id, block_offset, :, :]
                v_out[b, :, t, :] = self.v_cache[layer_idx, physical_block_id, block_offset, :, :]
                
        return k_out, v_out

    def to_hf_past_key_values(self, block_tables: List[List[int]], context_lens: List[int]):
        """Convert paged blocks to HuggingFace's past_key_values format.

        This gathers non-contiguous physical blocks into a contiguous batch.
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
        past_key_values,
        num_new_tokens: int,
        request_ids: List[str],
        block_tables: List[List[int]],
        slot_offsets: List[int],
    ) -> None:
        """Extract the new K/V states from HF output and store them in blocks.

        Args:
            past_key_values: HF's returned past_key_values.
            num_new_tokens: Number of new tokens generated in this step.
            request_ids: Unique identifier for each request.
            block_tables: Assigned physical blocks per request.
            slot_offsets: The logical token index where the new tokens start.
        """
    def update_from_hf_past_key_values(
        self,
        past_key_values,
        num_new_tokens: int,
        request_ids: List[str],
        block_tables: List[List[int]],
        slot_offsets: List[int],
    ) -> None:
        """Extract the new K/V states from HF output and store them in blocks."""
        # Standardize access to HF Cache structure (DynamicCache, tuple, etc.)
        
        # 1. Primary: Try direct attribute access (DynamicCache standard)
        # Even with DynamicLayer wrappers, .key_cache and .value_cache usually return the raw lists of tensors.
        key_cache = getattr(past_key_values, "key_cache", getattr(past_key_values, "_key_cache", None))
        value_cache = getattr(past_key_values, "value_cache", getattr(past_key_values, "_value_cache", None))

        if key_cache is not None and value_cache is not None:
            for layer_idx in range(self.num_layers):
                k_full = key_cache[layer_idx]
                v_full = value_cache[layer_idx]
                # If these are DynamicLayer objects, we might need to peel them
                if hasattr(k_full, "data"): k_full = k_full.data
                if hasattr(v_full, "data"): v_full = v_full.data
                
                k_new = k_full[:, :, -num_new_tokens:, :]
                v_new = v_full[:, :, -num_new_tokens:, :]
                self.append(layer_idx, k_new, v_new, request_ids, block_tables, slot_offsets)
            return

        # 2. Secondary: Standard HF tuple format (layer_idx -> (k, v))
        if isinstance(past_key_values, (list, tuple)):
            for layer_idx, layer_data in enumerate(past_key_values):
                if isinstance(layer_data, (list, tuple)) and len(layer_data) == 2:
                    k_full, v_full = layer_data
                    k_new = k_full[:, :, -num_new_tokens:, :]
                    v_new = v_full[:, :, -num_new_tokens:, :]
                    self.append(layer_idx, k_new, v_new, request_ids, block_tables, slot_offsets)
            return

        # 3. Last Resort: Iteration (Handle DynamicCache if attributes failed)
        try:
            for layer_idx, layer_data in enumerate(past_key_values):
                # If it's a DynamicLayer, it won't unpack, so we try attributes on it
                k_full = getattr(layer_data, "key_cache", getattr(layer_data, "k", layer_data[0] if isinstance(layer_data, (list, tuple)) else None))
                v_full = getattr(layer_data, "value_cache", getattr(layer_data, "v", layer_data[1] if isinstance(layer_data, (list, tuple)) else None))
                
                if k_full is not None and v_full is not None:
                    k_new = k_full[:, :, -num_new_tokens:, :]
                    v_new = v_full[:, :, -num_new_tokens:, :]
                    self.append(layer_idx, k_new, v_new, request_ids, block_tables, slot_offsets)
            return
        except Exception:
            pass

        logger.warning(
            f"PagedKVCache: Unknown HF cache type {type(past_key_values)}. "
            f"Generation might be degraded."
        )

    def swap_out(self, block_ids: List[int]) -> None:
        """Move selected blocks from GPU to CPU."""
        self._ensure_cpu_cache()
        for bid in block_ids:
            # Shape: (num_layers, block_size, num_kv_heads, head_dim)
            self.cpu_k_cache[:, bid, ...] = self.k_cache[:, bid, ...].to("cpu", non_blocking=True)
            self.cpu_v_cache[:, bid, ...] = self.v_cache[:, bid, ...].to("cpu", non_blocking=True)
        # Non-blocking moves usually need a sync before the allocator marks them as CPU
        torch.cuda.synchronize()

    def swap_in(self, block_ids: List[int]) -> None:
        """Move selected blocks from CPU back to GPU."""
        if self.cpu_k_cache is None:
            return
        for bid in block_ids:
            self.k_cache[:, bid, ...] = self.cpu_k_cache[:, bid, ...].to(self.device, non_blocking=True)
            self.v_cache[:, bid, ...] = self.cpu_v_cache[:, bid, ...].to(self.device, non_blocking=True)
        torch.cuda.synchronize()

    @classmethod
    def from_model_config(
        cls,
        arch_config,
        num_blocks: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> PagedKVCache:
        """Factory: create cache from a ModelArchConfig."""
        return cls(
            num_layers=arch_config.num_layers,
            num_kv_heads=arch_config.num_kv_heads,
            head_dim=arch_config.head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            dtype=dtype,
            device=device,
        )
