"""
Tests for Flash Decoding.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pravaha.kernels.flash_decode import flash_decode_attention, flash_decode_fallback


def test_fallback_correctness():
    """Test PyTorch fallback attention correctness against scaled_dot_product_attention."""
    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    
    out_fallback = flash_decode_fallback(q, k, v)
    out_sdpa = F.scaled_dot_product_attention(q, k, v)
    
    torch.testing.assert_close(out_fallback, out_sdpa)


def test_output_shape_validation():
    """Test output shape validation and error handling."""
    B, H, S, D = 2, 4, 128, 64
    
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    out = flash_decode_attention(q, k, v)
    assert out.shape == (B, H, 1, D)
    
    q_bad = torch.randn(B, H, 2, D)
    with pytest.raises(ValueError, match="expects query sequence length 1"):
        flash_decode_attention(q_bad, k, v)
        
    k_bad = torch.randn(B + 1, H, S, D)
    with pytest.raises(ValueError, match="Shape mismatch"):
        flash_decode_attention(q, k_bad, v)


def test_numerical_stability():
    """Test numerical stability with large sequence lengths."""
    B, H, S, D = 1, 1, 8192, 64
    q = torch.randn(B, H, 1, D) * 10
    k = torch.randn(B, H, S, D) * 10
    v = torch.randn(B, H, S, D)
    
    out = flash_decode_fallback(q, k, v)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_head_dimensions():
    """Test that different head dimensions work."""
    B, H, S = 2, 4, 128
    
    for D in [32, 64, 128]:
        q = torch.randn(B, H, 1, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        
        out = flash_decode_fallback(q, k, v)
        assert out.shape == (B, H, 1, D)


def test_batch_processing():
    """Test batch processing (batch_size > 1)."""
    B, H, S, D = 8, 4, 128, 64
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    
    out = flash_decode_fallback(q, k, v)
    assert out.shape == (B, H, 1, D)


def test_causal_masking_behavior():
    """
    Test causal masking behavior.
    For decoding phase, causal masking is implicitly handled by the KV cache.
    """
    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    
    out_fallback = flash_decode_fallback(q, k, v)
    assert torch.all(out_fallback <= v.max() + 1e-5)
    assert torch.all(out_fallback >= v.min() - 1e-5)
