"""Model Loader — Load HuggingFace and GGUF models with quantization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArchConfig:
    """Extracted architecture config from the loaded model."""

    num_layers: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    hidden_size: int = 4096
    vocab_size: int = 32000


class ModelLoader:
    """Load transformer models from HuggingFace Hub or local paths."""

    def load(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: Any = "float16",
        quantization: str | None = None,
        trust_remote_code: bool = False,
        use_torch_compile: bool = False,
    ) -> tuple[Any, ArchConfig]:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        from pravaha.models.hf_compat import HFCompatLayer

        # Use HFCompatLayer for universal model support
        compat = HFCompatLayer()

        if isinstance(dtype, str):
            dtype_str = dtype
        else:
            dtype_str = "float16"

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        arch = ArchConfig(
            num_layers=getattr(config, "num_hidden_layers", 32),
            num_kv_heads=getattr(
                config, "num_key_value_heads", getattr(config, "num_attention_heads", 32)
            ),
            head_dim=getattr(
                config,
                "head_dim",
                getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32),
            ),
            hidden_size=getattr(config, "hidden_size", 4096),
            vocab_size=getattr(config, "vocab_size", 32000),
        )

        # Build kwargs via compatibility layer (handles Flash Attn, RoPE, etc.)
        kwargs = compat.get_model_kwargs(
            model_path=model_path,
            device=device,
            dtype_str=dtype_str,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
        )

        logger.info(
            f"Loading model: {model_path} (device={device}, dtype={dtype}, "
            f"quant={quantization or 'none'}, "
            f"flash_attn={'yes' if compat._flash_attn_available else 'no'})"
        )
        model: Any = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        model.eval()

        if use_torch_compile:
            try:
                logger.info("Applying torch.compile() for accelerated inference...")
                # Reduce overhead for generation, using max-autotune if possible in prod
                model = torch.compile(model)
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile: {e}")

        logger.info(f"Model loaded: layers={arch.num_layers}, kv_heads={arch.num_kv_heads}")
        return model, arch
