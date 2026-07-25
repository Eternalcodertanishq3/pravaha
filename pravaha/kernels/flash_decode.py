"""
Flash Decoding implementation in Triton.
"""
from __future__ import annotations

import logging
import math

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

logger = logging.getLogger(__name__)

if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SEQ": 64}, num_stages=2, num_warps=2),
            triton.Config({"BLOCK_SEQ": 128}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_SEQ": 256}, num_stages=3, num_warps=8),
        ],
        key=["seq_len", "head_dim"],
    )
    @triton.jit
    def _flash_decode_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,  # noqa: N803
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_od,
        seq_len, head_dim: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,  # noqa: N803
    ):
        """
        Triton kernel for Flash Decoding.
        Computes attention for a single query token per sequence in the batch.
        """
        batch_pid = tl.program_id(0)
        head_pid = tl.program_id(1)

        # Offsets
        q_offset = batch_pid * stride_qb + head_pid * stride_qh
        k_offset = batch_pid * stride_kb + head_pid * stride_kh
        v_offset = batch_pid * stride_vb + head_pid * stride_vh
        o_offset = batch_pid * stride_ob + head_pid * stride_oh

        # Pointers
        q_ptrs = Q_ptr + q_offset + tl.arange(0, head_dim)

        # Load Q
        q = tl.load(q_ptrs)

        # Initialize running stats
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([head_dim], dtype=tl.float32)

        # Loop over sequence
        for start_k in range(0, seq_len, BLOCK_SEQ):
            offs_k = start_k + tl.arange(0, BLOCK_SEQ)
            k_ptrs = K_ptr + k_offset + offs_k[:, None] * stride_ks + tl.arange(0, head_dim)[None, :] * stride_kd
            v_ptrs = V_ptr + v_offset + offs_k[:, None] * stride_vs + tl.arange(0, head_dim)[None, :] * stride_vd

            mask = offs_k[:, None] < seq_len
            k = tl.load(k_ptrs, mask=mask, other=0.0)
            v = tl.load(v_ptrs, mask=mask, other=0.0)

            # Compute Q @ K^T
            qk = tl.sum(q[None, :] * k, axis=1) / math.sqrt(head_dim)

            # Apply mask to qk
            qk = tl.where(offs_k < seq_len, qk, -float("inf"))

            # Update m_i and l_i
            m_ij = tl.max(qk, axis=0)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(qk - m_i_new)

            # Update acc
            acc_scale = alpha
            # cast beta to v dtype to avoid mixed precision upcasting explosion
            acc = acc * acc_scale + tl.sum(beta[:, None].to(v.dtype) * v, axis=0)

            l_i = l_i * alpha + tl.sum(beta, axis=0)
            m_i = m_i_new

        # Write output (cast back to original Q dtype)
        out = acc / l_i
        out = out.to(q.dtype)
        out_ptrs = Out_ptr + o_offset + tl.arange(0, head_dim)
        tl.store(out_ptrs, out)
else:
    # Dummy kernel to satisfy linters if Triton is missing.
    def _flash_decode_kernel(*args, **kwargs):
        pass


def flash_decode_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    PyTorch fallback for Flash Decoding.
    """
    import torch.nn.functional as F

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)

    return out


def flash_decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes Flash Decoding attention.

    Args:
        q: [batch, heads, 1, head_dim]
        k: [batch, heads, seq_len, head_dim]
        v: [batch, heads, seq_len, head_dim]

    Returns:
        output: [batch, heads, 1, head_dim]
    """
    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError("Inputs must be PyTorch tensors")

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("Inputs must be 4D tensors")

    b, h, sq, d = q.shape
    bk, hk, sk, dk = k.shape
    bv, hv, sv, dv = v.shape

    if sq != 1:
        raise ValueError(f"Flash Decoding expects query sequence length 1, got {sq}")

    if b != bk or b != bv or h != hk or h != hv or sk != sv or d != dk or d != dv:
        raise ValueError("Shape mismatch between q, k, v")

    if not HAS_TRITON or not q.is_cuda:
        if HAS_TRITON and not q.is_cuda:
            logger.debug("Tensors not on CUDA, falling back to PyTorch implementation.")
        return flash_decode_fallback(q, k, v)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)

    grid = (b, h)

    _flash_decode_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(3),
        sk, d
    )

    return out


def benchmark_flash_decode(batch_size=8, num_heads=32, seq_len=4096, head_dim=128, iters=100) -> None:
    """
    Benchmarks Flash Decoding kernel vs PyTorch fallback.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Cannot benchmark Triton kernel.")
        return

    q = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

    for _ in range(10):
        _ = flash_decode_attention(q, k, v)
        _ = flash_decode_fallback(q, k, v)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _ = flash_decode_attention(q, k, v)
    end_event.record()
    torch.cuda.synchronize()
    triton_time = start_event.elapsed_time(end_event) / iters

    start_event.record()
    for _ in range(iters):
        _ = flash_decode_fallback(q, k, v)
    end_event.record()
    torch.cuda.synchronize()
    pytorch_time = start_event.elapsed_time(end_event) / iters

    print(f"Triton time: {triton_time:.3f} ms")
    print(f"PyTorch time: {pytorch_time:.3f} ms")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")
