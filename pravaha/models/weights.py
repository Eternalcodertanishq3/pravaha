"""Weight Loading — Safetensors, PyTorch bin, and GGUF formats.

Handles loading model weights from multiple formats with automatic
detection and shard merging. Provides memory profiling utilities.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def load_weights(
    model_path: str,
    dtype: Optional[torch.dtype] = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load model weights from safetensors or PyTorch bin files.

    Auto-detects format and handles sharded models.

    Args:
        model_path: Path to model directory or single weight file.
        dtype: Target dtype to cast weights to. None = keep original.
        device: Device to load weights to.

    Returns:
        State dict mapping parameter names to tensors.
    """
    path = Path(model_path)
    state_dict: dict[str, torch.Tensor] = {}

    if path.is_file():
        if path.suffix == ".safetensors":
            state_dict = _load_safetensors([path], dtype, device)
        elif path.suffix in (".bin", ".pt", ".pth"):
            state_dict = _load_pytorch_bin([path], dtype, device)
        else:
            logger.warning(f"Unknown weight format: {path.suffix}, "
                           "weights will be loaded by transformers.")
    elif path.is_dir():
        safetensor_files = sorted(path.glob("*.safetensors"))
        if safetensor_files:
            state_dict = _load_safetensors(safetensor_files, dtype, device)
        else:
            bin_files = sorted(path.glob("pytorch_model*.bin"))
            if bin_files:
                state_dict = _load_pytorch_bin(bin_files, dtype, device)
            else:
                logger.warning("No weight files found in directory, "
                               "weights will be loaded by transformers.")

    logger.info(f"Loaded {len(state_dict)} tensors from {model_path}")
    return state_dict


def _load_safetensors(
    filepaths: list[Path],
    dtype: Optional[torch.dtype],
    device: str,
) -> dict[str, torch.Tensor]:
    """Load weights from safetensors format."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError(
            "safetensors is required for .safetensors loading. "
            "Install with: pip install safetensors"
        )

    state_dict: dict[str, torch.Tensor] = {}
    for filepath in filepaths:
        shard = load_file(str(filepath), device=device)
        if dtype is not None:
            shard = {k: v.to(dtype) for k, v in shard.items()}
        state_dict.update(shard)
        logger.debug(f"Loaded shard: {filepath.name} ({len(shard)} tensors)")
    return state_dict


def _load_pytorch_bin(
    filepaths: list[Path],
    dtype: Optional[torch.dtype],
    device: str,
) -> dict[str, torch.Tensor]:
    """Load weights from PyTorch .bin format."""
    state_dict: dict[str, torch.Tensor] = {}
    for filepath in filepaths:
        shard = torch.load(filepath, map_location=device, weights_only=True)
        if dtype is not None:
            shard = {k: v.to(dtype) for k, v in shard.items()}
        state_dict.update(shard)
        logger.debug(f"Loaded shard: {filepath.name} ({len(shard)} tensors)")
    return state_dict


def get_weight_memory_profile(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Calculate memory breakdown of loaded weights.

    Returns:
        Dict with total bytes, per-dtype breakdown, and layer stats.
    """
    total_bytes = 0
    dtype_sizes: dict[str, int] = {}
    layer_sizes: dict[str, int] = {}

    for name, tensor in state_dict.items():
        nbytes = tensor.nelement() * tensor.element_size()
        total_bytes += nbytes

        dtype_key = str(tensor.dtype)
        dtype_sizes[dtype_key] = dtype_sizes.get(dtype_key, 0) + nbytes

        # Group by layer prefix
        parts = name.split(".")
        layer_key = ".".join(parts[:3]) if len(parts) > 3 else parts[0]
        layer_sizes[layer_key] = layer_sizes.get(layer_key, 0) + nbytes

    return {
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / (1024 ** 3), 3),
        "total_params": sum(t.nelement() for t in state_dict.values()),
        "num_tensors": len(state_dict),
        "by_dtype": {k: round(v / (1024 ** 2), 1) for k, v in dtype_sizes.items()},
        "top_layers": dict(sorted(layer_sizes.items(), key=lambda x: -x[1])[:10]),
    }


def shard_weights(
    state_dict: dict[str, torch.Tensor],
    num_shards: int = 2,
    strategy: str = "layer",
) -> list[dict[str, torch.Tensor]]:
    """(Stub) Shard weights for tensor parallelism across GPUs.

    Args:
        state_dict: Full model state dict.
        num_shards: Number of GPU shards.
        strategy: Sharding strategy ('layer' or 'tensor').

    Returns:
        List of state dicts, one per shard.
    """
    logger.warning("Tensor parallelism sharding is not yet implemented. "
                   "Returning single shard.")
    return [state_dict]
