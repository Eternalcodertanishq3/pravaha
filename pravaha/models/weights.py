"""Weight loading utilities.

Handles loading model weights from safetensors and PyTorch bin formats,
dtype conversion, and provides stubs for future tensor parallelism.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def load_weights(
    model_path: str | Path,
    dtype: Optional[torch.dtype] = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load model weights from safetensors or PyTorch bin files.

    Priority: safetensors > pytorch_model.bin > model.safetensors
    Handles sharded models (multiple weight files).

    Args:
        model_path: Directory containing weight files.
        dtype: Target dtype to cast weights to. None = keep original.
        device: Device to load weights to.

    Returns:
        Dictionary mapping parameter names to tensors.
    """
    path = Path(model_path)

    if not path.is_dir():
        logger.info(f"Model path '{model_path}' is not a local directory; "
                     "weights will be loaded by transformers.")
        return {}

    # Try safetensors first (faster, safer)
    safetensor_files = sorted(glob.glob(str(path / "*.safetensors")))
    if safetensor_files:
        return _load_safetensors(safetensor_files, dtype, device)

    # Fall back to PyTorch bin files
    bin_files = sorted(glob.glob(str(path / "*.bin")))
    if bin_files:
        return _load_pytorch_bin(bin_files, dtype, device)

    logger.warning(f"No weight files found in {path}. "
                   "Model will be loaded via transformers.")
    return {}


def _load_safetensors(
    files: list[str],
    dtype: Optional[torch.dtype],
    device: str,
) -> dict[str, torch.Tensor]:
    """Load weights from safetensors format."""
    from safetensors.torch import load_file

    state_dict: dict[str, torch.Tensor] = {}
    total_bytes = 0

    for filepath in files:
        logger.info(f"Loading safetensors: {Path(filepath).name}")
        shard = load_file(filepath, device=device)
        for name, tensor in shard.items():
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            state_dict[name] = tensor
            total_bytes += tensor.nelement() * tensor.element_size()

    logger.info(
        f"Loaded {len(state_dict)} parameters from {len(files)} file(s) "
        f"({total_bytes / 1e9:.2f} GB)"
    )
    return state_dict


def _load_pytorch_bin(
    files: list[str],
    dtype: Optional[torch.dtype],
    device: str,
) -> dict[str, torch.Tensor]:
    """Load weights from PyTorch .bin format."""
    state_dict: dict[str, torch.Tensor] = {}
    total_bytes = 0

    for filepath in files:
        logger.info(f"Loading PyTorch bin: {Path(filepath).name}")
        shard = torch.load(filepath, map_location=device, weights_only=True)
        for name, tensor in shard.items():
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            state_dict[name] = tensor
            total_bytes += tensor.nelement() * tensor.element_size()

    logger.info(
        f"Loaded {len(state_dict)} parameters from {len(files)} file(s) "
        f"({total_bytes / 1e9:.2f} GB)"
    )
    return state_dict


def convert_dtype(
    state_dict: dict[str, torch.Tensor],
    target_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Convert all tensors in state_dict to target dtype.

    Skips integer tensors (e.g., position IDs, token type IDs).
    """
    converted = {}
    for name, tensor in state_dict.items():
        if tensor.is_floating_point():
            converted[name] = tensor.to(target_dtype)
        else:
            converted[name] = tensor
    return converted


def estimate_weight_memory(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Calculate memory breakdown of loaded weights.

    Returns:
        Dict with 'total_gb', 'total_params', and per-component breakdowns.
    """
    total_bytes = 0
    total_params = 0
    component_bytes: dict[str, int] = {}

    for name, tensor in state_dict.items():
        nbytes = tensor.nelement() * tensor.element_size()
        total_bytes += nbytes
        total_params += tensor.nelement()

        # Group by component (first dotted segment)
        component = name.split(".")[0]
        component_bytes[component] = component_bytes.get(component, 0) + nbytes

    return {
        "total_gb": total_bytes / (1024**3),
        "total_params": total_params,
        "total_params_billions": total_params / 1e9,
        "components": {k: v / (1024**3) for k, v in component_bytes.items()},
    }


# ─── Future Extension Stubs ──────────────────────────────────────────────────

def shard_weights(
    state_dict: dict[str, torch.Tensor],
    num_shards: int,
    shard_id: int,
) -> dict[str, torch.Tensor]:
    """(Stub) Shard weights for tensor parallelism across GPUs.

    Will be implemented in multi-GPU phase. Currently returns full state_dict.
    """
    if num_shards > 1:
        logger.warning(
            "Tensor parallelism not yet implemented. "
            "Returning full state_dict for shard %d/%d.",
            shard_id, num_shards,
        )
    return state_dict
