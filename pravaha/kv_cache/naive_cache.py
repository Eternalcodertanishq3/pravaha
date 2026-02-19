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
    """Pre-allocated KV-cache for single-sequence inference.

    Allocates fixed-size tensors for all layers upfront and manages a write
    pointer (_seq_len) that tracks how many tokens have been cached.

    Usage:
        cache = NaiveKVCache(num_layers=12, num_kv_heads=12, head_dim=64,
                             max_seq_len=1024, dtype=torch.float16, device="cuda")

        # During generation:
        for layer_idx in range(num_layers):
            cache.append(layer_idx, k_state, v_state)  # from model output
            k, v = cache.get(layer_idx)                 # for next model input

        # Convert to HF format for model.forward()
        past = cache.to_hf_past_key_values()
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        """Allocate KV-cache tensors.

        Args:
            num_layers: Number of transformer layers.
            num_kv_heads: Number of key-value attention heads (may differ
                from query heads in GQA models like Llama).
            head_dim: Dimension per attention head.
            max_seq_len: Maximum sequence length to cache.
            dtype: Data type for cache tensors.
            device: Device to allocate on.
            batch_size: Batch dimension (1 for Phase 2 single-sequence).
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size

        # Pre-allocate separate K and V tensors
        # Shape: (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        cache_shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)

        self.k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        # Write pointer — how many tokens are currently cached
        self._seq_len = 0

        total_bytes = self.k_cache.nelement() * self.k_cache.element_size() * 2
        logger.info(
            f"NaiveKVCache allocated: {num_layers} layers, "
            f"{num_kv_heads} kv_heads, head_dim={head_dim}, "
            f"max_seq_len={max_seq_len} | "
            f"{total_bytes / (1024**2):.1f} MB"
        )

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append new key-value states for a given layer.

        Writes K/V at position [_seq_len : _seq_len + new_tokens] and
        increments the pointer on layer 0 (all layers advance together).

        Args:
            layer_idx: Which transformer layer (0-indexed).
            k: Key tensor, shape (batch_size, num_kv_heads, new_tokens, head_dim).
            v: Value tensor, same shape as k.

        Raises:
            RuntimeError: If cache would overflow max_seq_len.
        """
        new_tokens = k.shape[2]  # seq_len dimension

        if self._seq_len + new_tokens > self.max_seq_len:
            raise RuntimeError(
                f"KV-cache overflow: trying to write {new_tokens} tokens "
                f"at position {self._seq_len}, but max_seq_len={self.max_seq_len}"
            )

        # Write into pre-allocated buffer
        start = self._seq_len
        end = start + new_tokens
        self.k_cache[layer_idx, :, :, start:end, :] = k
        self.v_cache[layer_idx, :, :, start:end, :] = v

        # Increment pointer only on the last layer (all layers advance together)
        if layer_idx == self.num_layers - 1:
            self._seq_len = end

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached K/V for a given layer (only the valid portion).

        Returns:
            Tuple of (k, v), each shape (batch_size, num_kv_heads, seq_len, head_dim).
        """
        k = self.k_cache[layer_idx, :, :, : self._seq_len, :]
        v = self.v_cache[layer_idx, :, :, : self._seq_len, :]
        return k, v

    def get_seq_len(self) -> int:
        """Return the current number of cached tokens."""
        return self._seq_len

    def clear(self) -> None:
        """Reset the cache for a new sequence.

        Only resets the write pointer — no need to zero memory since
        subsequent appends will overwrite the relevant positions.
        """
        self._seq_len = 0

    def memory_usage_bytes(self) -> dict:
        """Report memory usage breakdown.

        Returns:
            Dict with kv_cache_bytes, estimated activation_bytes, and total.
        """
        element_size = self.k_cache.element_size()

        # KV-cache: allocated memory (full pre-allocation)
        kv_allocated = self.k_cache.nelement() * element_size * 2

        # KV-cache: actually used memory (up to _seq_len)
        kv_used = (
            self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * self._seq_len
            * self.head_dim
            * element_size
            * 2  # K + V
        )

        # Rough activation estimate per step:
        # batch × seq_len × hidden_size × dtype_bytes × 4 (intermediates)
        hidden_size = self.num_kv_heads * self.head_dim
        activation_est = (
            self.batch_size * max(self._seq_len, 1) * hidden_size * element_size * 4
        )

        return {
            "kv_cache_allocated_bytes": kv_allocated,
            "kv_cache_allocated_mb": kv_allocated / (1024**2),
            "kv_cache_used_bytes": kv_used,
            "kv_cache_used_mb": kv_used / (1024**2),
            "activation_estimate_bytes": activation_est,
            "activation_estimate_mb": activation_est / (1024**2),
            "total_estimate_bytes": kv_allocated + activation_est,
            "total_estimate_mb": (kv_allocated + activation_est) / (1024**2),
        }

    def to_hf_past_key_values(self):
        """Convert managed cache to HuggingFace's past_key_values format.

        Transformers v5+ uses DynamicCache objects. Falls back to
        tuple format for older versions.

        Returns:
            DynamicCache (or tuple) compatible with model.forward(), or None if empty.
        """
        if self._seq_len == 0:
            return None

        try:
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()
            for layer_idx in range(self.num_layers):
                k, v = self.get(layer_idx)
                cache.update(k, v, layer_idx)
            return cache
        except ImportError:
            # Fallback for older transformers: tuple of (k, v) per layer
            result = []
            for layer_idx in range(self.num_layers):
                k, v = self.get(layer_idx)
                result.append((k, v))
            return tuple(result)

    def update_from_hf_past_key_values(
        self, past_key_values, num_new_tokens: int
    ) -> None:
        """Extract the new K/V states from HF output and store them.

        After calling model.forward() with our cache as past_key_values,
        HF returns the *full* cache (old + new). We extract only the new
        tokens (the last `num_new_tokens` positions) and append them.

        Supports both transformers v5 DynamicCache and legacy tuple format.

        Args:
            past_key_values: HF's returned past_key_values (DynamicCache or tuple).
            num_new_tokens: Number of new tokens generated in this step.
        """
        key_cache = None
        value_cache = None

        # Transformers v5+: DynamicCache with .key_cache / .value_cache lists
        if hasattr(past_key_values, "key_cache"):
            key_cache = past_key_values.key_cache
            value_cache = past_key_values.value_cache
        elif hasattr(past_key_values, "_key_cache"):
            key_cache = past_key_values._key_cache
            value_cache = past_key_values._value_cache
        
        # Scenario 1: dynamic cache with explicit key/value lists
        if key_cache is not None:
            for layer_idx in range(self.num_layers):
                k_full = key_cache[layer_idx]
                v_full = value_cache[layer_idx]
                k_new = k_full[:, :, -num_new_tokens:, :]
                v_new = v_full[:, :, -num_new_tokens:, :]
                self.append(layer_idx, k_new, v_new)
            return

        # Scenario 2: dynamic cache with .layers list (transformers main/nightly)
        if hasattr(past_key_values, "layers"):
            for layer_idx, layer in enumerate(past_key_values.layers):
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                
                # Check directly if keys/values are empty tensors (numel == 0) or None
                if keys is None or values is None or keys.numel() == 0 or values.numel() == 0:
                    continue
                
                k_full = keys
                v_full = values
                k_new = k_full[:, :, -num_new_tokens:, :]
                v_new = v_full[:, :, -num_new_tokens:, :]
                self.append(layer_idx, k_new, v_new)
            return

        # Scenario 3: Legacy tuple of (k, v) per layer
        if isinstance(past_key_values, (list, tuple)):
            for layer_idx, (k_full, v_full) in enumerate(past_key_values):
                k_new = k_full[:, :, -num_new_tokens:, :]
                v_new = v_full[:, :, -num_new_tokens:, :]
                self.append(layer_idx, k_new, v_new)
            return

        # Unknown cache type — try to extract via indexing with warning
        logger.warning(
            f"Unknown cache type: {type(past_key_values)}. "
            f"Attrs: {[a for a in dir(past_key_values) if 'cache' in a.lower() or 'key' in a.lower()]}"
        )
        try:
            for layer_idx in range(self.num_layers):
                k_full, v_full = past_key_values[layer_idx]
                k_new = k_full[:, :, -num_new_tokens:, :]
                v_new = v_full[:, :, -num_new_tokens:, :]
                self.append(layer_idx, k_new, v_new)
        except Exception as e:
            logger.error(f"Failed to extract KV-cache: {e}")

    @classmethod
    def from_model_config(
        cls,
        arch_config,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> NaiveKVCache:
        """Factory: create cache from a ModelArchConfig.

        Args:
            arch_config: ModelArchConfig with layer/head dimensions.
            max_seq_len: Maximum sequence length to cache.
            dtype: Cache data type.
            device: Device to allocate on.

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
        )

    def __repr__(self) -> str:
        return (
            f"NaiveKVCache(layers={self.num_layers}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, seq_len={self._seq_len}/{self.max_seq_len}, "
            f"device={self.device})"
        )
