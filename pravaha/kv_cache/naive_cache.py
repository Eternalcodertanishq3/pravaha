"""Naive KV-Cache implementation.

Pre-allocates contiguous GPU tensors for key-value states across all transformer
layers. Provides explicit control over cache lifecycle (append, read, clear)
while maintaining compatibility with HuggingFace's past_key_values format.

Memory layout per cache (K or V separately):
    shape: (num_layers, batch_size=1, num_kv_heads, max_seq_len, head_dim)

Phase 2 baseline — replaced by paged KV-cache in Phase 4.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class NaiveKVCache:
    """Pre-allocated KV-cache for batched inference (Phase 3).

    Allocates fixed-size tensors for all layers upfront and manages a write
    pointer per batch slot. This supports continuous batching without
    paged attention complexity.

    Memory layout:
        shape: (num_layers, max_batch_size, num_kv_heads, max_seq_len, head_dim)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        batch_size: int = 1,  # Now represents max_batch_size
    ):
        """Allocate KV-cache tensors for Phase 3 continuous batching.

        Args:
            num_layers: Number of transformer layers.
            num_kv_heads: Number of key-value attention heads.
            head_dim: Dimension per attention head.
            max_seq_len: Maximum sequence length to cache per slot.
            dtype: Data type for cache tensors.
            device: Device to allocate on.
            batch_size: Maximum number of concurrent requests (slots).
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size

        # Pre-allocate separate K and V tensors
        # Shape: (num_layers, max_batch_size, num_kv_heads, max_seq_len, head_dim)
        cache_shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)

        self.k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        # Write pointers: how many tokens are currently cached PER SLOT
        # Shape: (max_batch_size,)
        self._seq_lens = torch.zeros(batch_size, dtype=torch.long, device="cpu")

        total_bytes = self.k_cache.nelement() * self.k_cache.element_size() * 2
        logger.info(
            f"NaiveKVCache allocated: {num_layers} layers, "
            f"max_batch_size={batch_size}, max_seq_len={max_seq_len} | "
            f"{total_bytes / (1024**2):.1f} MB"
        )

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slot_indices: list[int]) -> None:
        """Append new key-value states for specific active slots.

        Args:
            layer_idx: Which transformer layer (0-indexed).
            k: Key tensor, shape (batch_size, num_kv_heads, new_tokens, head_dim).
               Note: `batch_size` here matches `len(slot_indices)`.
            v: Value tensor, same shape as k.
            slot_indices: List of physical slot indices these elements belong to.

        Raises:
            RuntimeError: If cache would overflow max_seq_len for any slot.
        """
        assert k.shape[0] == len(slot_indices), "Batch dim must match number of slots"
        new_tokens = k.shape[2]  # seq_len dimension

        # Check for overflow
        for batch_idx, slot in enumerate(slot_indices):
            current_len = self._seq_lens[slot].item()
            if current_len + new_tokens > self.max_seq_len:
                raise RuntimeError(
                    f"KV-cache overflow: slot {slot} trying to write {new_tokens} tokens "
                    f"at position {current_len}, but max_seq_len={self.max_seq_len}"
                )

        # Write into pre-allocated buffer per slot
        for batch_idx, slot in enumerate(slot_indices):
            start = self._seq_lens[slot].item()
            end = start + new_tokens
            
            self.k_cache[layer_idx, slot, :, start:end, :] = k[batch_idx]
            self.v_cache[layer_idx, slot, :, start:end, :] = v[batch_idx]

        # Increment pointers only on the last layer
        if layer_idx == self.num_layers - 1:
            for slot in slot_indices:
                self._seq_lens[slot] += new_tokens

    def get(self, layer_idx: int, slot_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached K/V for specific slots, padded to max active length.

        To batch forward passes in HuggingFace models, the K/V tensors must be
        uniform rectangles: (batch_size, num_kv_heads, max_cached_len, head_dim).
        Shorter sequences are padded.

        Returns:
            Tuple of (k, v) for the requested slots.
        """
        # Find the maximum sequence length among the active slots
        active_lengths = [self._seq_lens[slot].item() for slot in slot_indices]
        max_active_len = max(active_lengths) if active_lengths else 0

        # Construct padded tensors
        # HF expects (batch_size, num_kv_heads, seq_len, head_dim)
        current_batch_size = len(slot_indices)
        
        # We index the cache: k_cache[layer, slots, :, :max_active_len, :]
        k_padded = self.k_cache[layer_idx, slot_indices, :, :max_active_len, :]
        v_padded = self.v_cache[layer_idx, slot_indices, :, :max_active_len, :]

        return k_padded, v_padded

    def get_seq_lens(self, slot_indices: list[int]) -> list[int]:
        """Return the current number of cached tokens strictly for active slots."""
        return [self._seq_lens[slot].item() for slot in slot_indices]

    def clear_slots(self, slot_indices: list[int]) -> None:
        """Reset the cache pointers for specific slots."""
        for slot in slot_indices:
            self._seq_lens[slot] = 0

    def memory_usage_bytes(self) -> dict:
        """Report memory usage breakdown across all slots."""
        element_size = self.k_cache.element_size()

        # KV-cache: allocated memory (full pre-allocation)
        kv_allocated = self.k_cache.nelement() * element_size * 2

        # KV-cache: actually used memory (sum of all _seq_lens)
        total_used_tokens = self._seq_lens.sum().item()
        kv_used = (
            self.num_layers
            * self.num_kv_heads
            * total_used_tokens
            * self.head_dim
            * element_size
            * 2  # K + V
        )

        return {
            "kv_cache_allocated_bytes": kv_allocated,
            "kv_cache_allocated_mb": kv_allocated / (1024**2),
            "kv_cache_used_bytes": kv_used,
            "kv_cache_used_mb": kv_used / (1024**2),
            "active_slots": (self._seq_lens > 0).sum().item(),
            "max_slots": self.batch_size
        }

    def to_hf_past_key_values(self, slot_indices: list[int]):
        """Convert managed cache to HuggingFace's past_key_values format.

        Returns:
            DynamicCache (or tuple) compatible with model.forward(), or None if empty.
        """
        if not slot_indices:
            return None

        # Check if any slot has tokens
        if sum(self.get_seq_lens(slot_indices)) == 0:
            return None

        try:
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()
            for layer_idx in range(self.num_layers):
                k, v = self.get(layer_idx, slot_indices)
                cache.update(k, v, layer_idx)
            return cache
        except ImportError:
            # Fallback for older transformers: tuple of (k, v) per layer
            result = []
            for layer_idx in range(self.num_layers):
                k, v = self.get(layer_idx, slot_indices)
                result.append((k, v))
            return tuple(result)

    def update_from_hf_past_key_values(
        self, past_key_values, num_new_tokens: int, slot_indices: list[int]
    ) -> None:
        """Extract the new K/V states from HF output and store them.

        Args:
            past_key_values: HF's returned past_key_values.
            num_new_tokens: Number of new tokens generated in this step.
            slot_indices: The physical slots these sequences belong to.
        """
        key_cache = None
        value_cache = None

        if hasattr(past_key_values, "key_cache"):
            key_cache = past_key_values.key_cache
            value_cache = past_key_values.value_cache
        elif hasattr(past_key_values, "_key_cache"):
            key_cache = past_key_values._key_cache
            value_cache = past_key_values._value_cache
        
        if key_cache is not None:
            for layer_idx in range(self.num_layers):
                k_full = key_cache[layer_idx]
                v_full = value_cache[layer_idx]
                k_new = k_full[:, :, -num_new_tokens:, :]
                v_new = v_full[:, :, -num_new_tokens:, :]
                self.append(layer_idx, k_new, v_new, slot_indices)
            return

        if hasattr(past_key_values, "layers"):
            for layer_idx, layer in enumerate(past_key_values.layers):
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                
                if keys is None or values is None or keys.numel() == 0 or values.numel() == 0:
                    continue
                
                k_full = keys
                v_full = values
                k_new = k_full[:, :, -num_new_tokens:, :]
                v_new = v_full[:, :, -num_new_tokens:, :]
                self.append(layer_idx, k_new, v_new, slot_indices)
            return

        if isinstance(past_key_values, (list, tuple)):
            for layer_idx, (k_full, v_full) in enumerate(past_key_values):
                k_new = k_full[:, :, -num_new_tokens:, :]
                v_new = v_full[:, :, -num_new_tokens:, :]
                self.append(layer_idx, k_new, v_new, slot_indices)
            return

        logger.warning(
            f"Unknown cache type: {type(past_key_values)}. "
            f"Attrs: {[a for a in dir(past_key_values) if 'cache' in a.lower() or 'key' in a.lower()]}"
        )
        try:
            for layer_idx in range(self.num_layers):
                k_full, v_full = past_key_values[layer_idx]
                k_new = k_full[:, :, -num_new_tokens:, :]
                v_new = v_full[:, :, -num_new_tokens:, :]
                self.append(layer_idx, k_new, v_new, slot_indices)
        except Exception as e:
            logger.error(f"Failed to extract KV-cache: {e}")

    @classmethod
    def from_model_config(
        cls,
        arch_config,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        batch_size: int = 1,
    ) -> NaiveKVCache:
        """Factory: create cache from a ModelArchConfig.

        Args:
            arch_config: ModelArchConfig with layer/head dimensions.
            max_seq_len: Maximum sequence length to cache.
            dtype: Cache data type.
            device: Device to allocate on.
            batch_size: Maximum batch size (slots) for continuous batching.

        Returns:
            Initialized NaiveKVCache.
        """
        return cls(
            num_layers=arch_config.num_layers,
            num_kv_heads=arch_config.num_kv_heads,
            head_dim=arch_config.head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
        )

    def __repr__(self) -> str:
        return (
            f"NaiveKVCache(layers={self.num_layers}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, seq_len={self._seq_len}/{self.max_seq_len}, "
            f"device={self.device})"
        )
