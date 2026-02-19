"""Model loader.

Loads HuggingFace transformer models with configurable dtype and device placement.
Supports GPT-2, Llama, Mistral architectures. Designed for inference-only use
(eval mode, no gradients).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from pravaha.models.model_config import ModelArchConfig, parse_model_config

logger = logging.getLogger(__name__)


def _has_accelerate() -> bool:
    """Check if the `accelerate` package is installed."""
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


class ModelLoader:
    """Loads and prepares transformer models for inference.

    Handles weight loading, dtype casting, device placement, and provides
    memory usage reporting. Future-proofed for quantized model loading.
    """

    def load_model(
        self,
        model_path: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        max_memory: Optional[dict] = None,
    ) -> tuple[nn.Module, ModelArchConfig]:
        """Load a HuggingFace causal LM for inference.

        Args:
            model_path: HuggingFace model name or local directory.
            dtype: Target dtype (torch.float16, torch.bfloat16, torch.float32).
            device: Target device ("cuda", "cpu", "cuda:0", etc.).
            max_memory: Optional per-device memory limits for device_map="auto".

        Returns:
            Tuple of (model in eval mode, architecture config).
        """
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Target dtype: {dtype}, device: {device}")

        t0 = time.perf_counter()

        # 1. Parse architecture config
        arch_config = parse_model_config(model_path)
        logger.info(
            f"Architecture: {arch_config.arch_name} | "
            f"{arch_config.num_layers}L, {arch_config.hidden_size}H, "
            f"{arch_config.num_heads}A"
        )

        # 2. Estimate memory requirements
        est_mem = arch_config.estimate_model_memory_bytes(
            dtype_bytes=self._dtype_to_bytes(dtype)
        )
        logger.info(f"Estimated model memory: {est_mem / 1e9:.2f} GB")

        # 3. Check if model fits in GPU memory
        if device.startswith("cuda") and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.device(device))
            gpu_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            free_mem = gpu_mem - torch.cuda.memory_allocated(torch.device(device))
            logger.info(
                f"GPU memory: {gpu_mem / 1e9:.2f} GB total, "
                f"{free_mem / 1e9:.2f} GB free"
            )
            if est_mem > free_mem * 0.9:
                logger.warning(
                    f"Model ({est_mem / 1e9:.2f} GB) may not fit in "
                    f"free GPU memory ({free_mem / 1e9:.2f} GB). "
                    f"Consider using a smaller dtype or quantization."
                )

        # 4. Load model with HuggingFace
        #    `low_cpu_mem_usage` and `device_map` require the `accelerate` package.
        #    We try with those optimizations first, falling back to basic loading.
        has_accelerate = _has_accelerate()

        load_kwargs: dict = {
            "dtype": dtype,  # transformers v5+; older versions use "torch_dtype"
            "trust_remote_code": False,
        }

        if has_accelerate:
            load_kwargs["low_cpu_mem_usage"] = True
            if device != "cpu":
                load_kwargs["device_map"] = device
        else:
            logger.info(
                "'accelerate' not installed — using basic model loading. "
                "Install with: python -m pip install accelerate"
            )

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        # 5. Move to device if not already placed by device_map
        if device == "cpu" or not hasattr(model, "hf_device_map"):
            model = model.to(device)

        # 6. Switch to eval mode and disable gradients
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        t1 = time.perf_counter()
        logger.info(f"Model loaded in {t1 - t0:.2f}s")

        # 7. Report actual memory usage
        if device.startswith("cuda"):
            actual_mem = torch.cuda.memory_allocated(torch.device(device))
            logger.info(f"Actual GPU memory used: {actual_mem / 1e9:.2f} GB")

        return model, arch_config

    def estimate_memory(
        self,
        model_path: str,
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = 1024,
        batch_size: int = 1,
    ) -> dict:
        """Estimate total GPU memory requirements.

        Includes model weights, KV-cache, and activation memory.

        Args:
            model_path: HuggingFace model name or local directory.
            dtype: Target dtype for weights.
            max_seq_len: Maximum sequence length for KV-cache estimation.
            batch_size: Maximum batch size for activation estimation.

        Returns:
            Dict with memory breakdown in GB.
        """
        arch = parse_model_config(model_path)
        dtype_bytes = self._dtype_to_bytes(dtype)

        model_mem = arch.estimate_model_memory_bytes(dtype_bytes)
        kv_cache_mem = (
            arch.total_kv_cache_per_token * max_seq_len * batch_size
        )

        # Rough activation memory estimate:
        # batch_size × seq_len × hidden_size × dtype_bytes × ~4 (intermediates)
        activation_mem = batch_size * max_seq_len * arch.hidden_size * dtype_bytes * 4

        total = model_mem + kv_cache_mem + activation_mem

        return {
            "model_weights_gb": model_mem / (1024**3),
            "kv_cache_gb": kv_cache_mem / (1024**3),
            "activations_gb": activation_mem / (1024**3),
            "total_gb": total / (1024**3),
            "architecture": arch.arch_name,
            "num_params_billions": model_mem / dtype_bytes / 1e9,
        }

    @staticmethod
    def _dtype_to_bytes(dtype: torch.dtype) -> int:
        """Get bytes per element for a torch dtype."""
        return {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
            torch.uint8: 1,
        }.get(dtype, 2)
