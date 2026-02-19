"""Tests for NaiveKVCache.

Covers: allocation, append, get, clear, memory reporting,
HF format conversion, and equivalence with HF's built-in cache.
"""

import pytest
import torch

from pravaha.kv_cache.naive_cache import NaiveKVCache
from pravaha.models.model_config import ModelArchConfig


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def cache():
    """Small test cache (4 layers, 4 heads, head_dim=8, max_seq=32)."""
    return NaiveKVCache(
        num_layers=4,
        num_kv_heads=4,
        head_dim=8,
        max_seq_len=32,
        dtype=torch.float32,
        device="cpu",
    )


@pytest.fixture
def gpt2_arch_config():
    """GPT-2-like architecture config for factory tests."""
    return ModelArchConfig(
        arch_name="gpt2",
        num_layers=12,
        hidden_size=768,
        num_heads=12,
        num_kv_heads=12,
        intermediate_size=3072,
        vocab_size=50257,
        max_position_embeddings=1024,
    )


# ─── Shape Validation ────────────────────────────────────────────────────────


class TestAllocation:
    def test_cache_shapes(self, cache: NaiveKVCache):
        """K and V caches should have correct 5D shapes."""
        expected = (4, 1, 4, 32, 8)  # (layers, batch, kv_heads, max_seq, head_dim)
        assert cache.k_cache.shape == expected
        assert cache.v_cache.shape == expected

    def test_initial_seq_len_zero(self, cache: NaiveKVCache):
        assert cache.get_seq_len() == 0

    def test_device_and_dtype(self, cache: NaiveKVCache):
        assert cache.k_cache.dtype == torch.float32
        assert cache.v_cache.dtype == torch.float32
        assert str(cache.k_cache.device) == "cpu"

    def test_factory_from_model_config(self, gpt2_arch_config):
        """from_model_config should create cache with correct dimensions."""
        cache = NaiveKVCache.from_model_config(
            gpt2_arch_config, max_seq_len=512, dtype=torch.float16, device="cpu"
        )
        assert cache.num_layers == 12
        assert cache.num_kv_heads == 12
        assert cache.head_dim == 64  # 768 / 12
        assert cache.max_seq_len == 512


# ─── Append & Get ────────────────────────────────────────────────────────────


class TestAppendAndGet:
    def test_append_single_token(self, cache: NaiveKVCache):
        """Appending 1 token should be retrievable."""
        k = torch.randn(1, 4, 1, 8)  # (batch, kv_heads, 1 token, head_dim)
        v = torch.randn(1, 4, 1, 8)

        # Append to all layers
        for layer in range(4):
            cache.append(layer, k, v)

        assert cache.get_seq_len() == 1

        # Retrieve from layer 0
        k_out, v_out = cache.get(0)
        assert k_out.shape == (1, 4, 1, 8)
        assert torch.allclose(k_out, k)
        assert torch.allclose(v_out, v)

    def test_append_multiple_tokens(self, cache: NaiveKVCache):
        """Appending a prompt of 5 tokens then 1 more decode token."""
        # Prefill: 5 tokens
        k_prefill = torch.randn(1, 4, 5, 8)
        v_prefill = torch.randn(1, 4, 5, 8)
        for layer in range(4):
            cache.append(layer, k_prefill, v_prefill)
        assert cache.get_seq_len() == 5

        # Decode: 1 token
        k_decode = torch.randn(1, 4, 1, 8)
        v_decode = torch.randn(1, 4, 1, 8)
        for layer in range(4):
            cache.append(layer, k_decode, v_decode)
        assert cache.get_seq_len() == 6

        # Full cache should contain concatenation
        k_out, v_out = cache.get(0)
        assert k_out.shape == (1, 4, 6, 8)

    def test_append_respects_layer_order(self, cache: NaiveKVCache):
        """Different layers should store different data."""
        k0 = torch.ones(1, 4, 1, 8) * 1.0
        k1 = torch.ones(1, 4, 1, 8) * 2.0
        v = torch.zeros(1, 4, 1, 8)

        for layer in range(4):
            k = k0 if layer == 0 else k1
            cache.append(layer, k, v)

        k_layer0, _ = cache.get(0)
        k_layer1, _ = cache.get(1)
        assert k_layer0[0, 0, 0, 0].item() == 1.0
        assert k_layer1[0, 0, 0, 0].item() == 2.0

    def test_overflow_raises(self, cache: NaiveKVCache):
        """Exceeding max_seq_len should raise RuntimeError."""
        k_big = torch.randn(1, 4, 33, 8)  # max_seq_len=32
        v_big = torch.randn(1, 4, 33, 8)
        with pytest.raises(RuntimeError, match="overflow"):
            cache.append(0, k_big, v_big)


# ─── Clear ───────────────────────────────────────────────────────────────────


class TestClear:
    def test_clear_resets_seq_len(self, cache: NaiveKVCache):
        k = torch.randn(1, 4, 5, 8)
        v = torch.randn(1, 4, 5, 8)
        for layer in range(4):
            cache.append(layer, k, v)

        assert cache.get_seq_len() == 5
        cache.clear()
        assert cache.get_seq_len() == 0

    def test_get_after_clear_returns_empty(self, cache: NaiveKVCache):
        k = torch.randn(1, 4, 3, 8)
        v = torch.randn(1, 4, 3, 8)
        for layer in range(4):
            cache.append(layer, k, v)
        cache.clear()

        k_out, v_out = cache.get(0)
        assert k_out.shape == (1, 4, 0, 8)


# ─── Memory Reporting ────────────────────────────────────────────────────────


class TestMemoryReporting:
    def test_memory_has_all_keys(self, cache: NaiveKVCache):
        mem = cache.memory_usage_bytes()
        expected_keys = {
            "kv_cache_allocated_bytes",
            "kv_cache_allocated_mb",
            "kv_cache_used_bytes",
            "kv_cache_used_mb",
            "activation_estimate_bytes",
            "activation_estimate_mb",
            "total_estimate_bytes",
            "total_estimate_mb",
        }
        assert expected_keys == set(mem.keys())

    def test_allocated_memory_positive(self, cache: NaiveKVCache):
        mem = cache.memory_usage_bytes()
        assert mem["kv_cache_allocated_bytes"] > 0

    def test_used_memory_grows_with_tokens(self, cache: NaiveKVCache):
        mem0 = cache.memory_usage_bytes()
        assert mem0["kv_cache_used_bytes"] == 0

        k = torch.randn(1, 4, 5, 8)
        v = torch.randn(1, 4, 5, 8)
        for layer in range(4):
            cache.append(layer, k, v)

        mem5 = cache.memory_usage_bytes()
        assert mem5["kv_cache_used_bytes"] > 0


# ─── HF Conversion ──────────────────────────────────────────────────────────


class TestHFConversion:
    def test_empty_cache_returns_none(self, cache: NaiveKVCache):
        assert cache.to_hf_past_key_values() is None

    def test_hf_format_structure(self, cache: NaiveKVCache):
        """HF format is tuple of (k, v) per layer."""
        k = torch.randn(1, 4, 3, 8)
        v = torch.randn(1, 4, 3, 8)
        for layer in range(4):
            cache.append(layer, k, v)

        hf = cache.to_hf_past_key_values()
        assert isinstance(hf, tuple)
        assert len(hf) == 4  # num_layers

        for layer_kv in hf:
            assert len(layer_kv) == 2  # (k, v)
            assert layer_kv[0].shape == (1, 4, 3, 8)
            assert layer_kv[1].shape == (1, 4, 3, 8)

    def test_update_from_hf_past_key_values(self, cache: NaiveKVCache):
        """update_from_hf should extract new tokens and store them."""
        # Simulate HF returning full cache with 5 total tokens
        hf_past = tuple(
            (torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8))
            for _ in range(4)
        )
        # Tell cache these are all new (prefill of 5 tokens)
        cache.update_from_hf_past_key_values(hf_past, num_new_tokens=5)
        assert cache.get_seq_len() == 5

        # Simulate decode step: HF returns 6 total tokens
        hf_past_decode = tuple(
            (torch.randn(1, 4, 6, 8), torch.randn(1, 4, 6, 8))
            for _ in range(4)
        )
        cache.update_from_hf_past_key_values(hf_past_decode, num_new_tokens=1)
        assert cache.get_seq_len() == 6


# ─── Repr ────────────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_format(self, cache: NaiveKVCache):
        r = repr(cache)
        assert "NaiveKVCache" in r
        assert "layers=4" in r
        assert "seq_len=0/32" in r


# ─── Equivalence Test (slow — requires model download) ──────────────────────


@pytest.mark.slow
class TestEquivalence:
    """Verify NaiveKVCache produces identical output to HF's built-in cache."""

    def test_greedy_output_matches_hf_cache(self):
        """Same prompt + greedy sampling → identical tokens with both cache modes."""
        from pravaha.config import CacheConfig, EngineConfig, ModelConfig
        from pravaha.engine import PravahaEngine

        prompt = "The capital of France is"

        # Mode 1: HF cache (Phase 1)
        cfg_hf = EngineConfig(
            model=ModelConfig(device="cpu", dtype="float32"),
            cache=CacheConfig(use_naive_cache=False),
        )
        engine_hf = PravahaEngine(config=cfg_hf)
        text_hf = engine_hf.generate_text(prompt, max_new_tokens=20, temperature=0.0)

        # Mode 2: NaiveKVCache (Phase 2)
        cfg_naive = EngineConfig(
            model=ModelConfig(device="cpu", dtype="float32"),
            cache=CacheConfig(use_naive_cache=True),
        )
        engine_naive = PravahaEngine(config=cfg_naive)
        text_naive = engine_naive.generate_text(
            prompt, max_new_tokens=20, temperature=0.0
        )

        assert text_hf == text_naive, (
            f"Output mismatch!\nHF cache: {text_hf!r}\nNaive cache: {text_naive!r}"
        )
