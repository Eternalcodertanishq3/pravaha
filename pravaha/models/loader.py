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
        dtype: str = "float16",
        quantization: str | None = None,
        trust_remote_code: bool = False,
    ) -> tuple[Any, ArchConfig]:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        torch_dtype = getattr(torch, dtype, torch.float16)
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
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "device_map": "auto" if device == "cuda" else device,
        }
        if quantization == "4bit":
            from pravaha.quantization.bitsandbytes import BitsAndBytesQuantizer

            kwargs["quantization_config"] = BitsAndBytesQuantizer(4).get_config()
        elif quantization == "8bit":
            from pravaha.quantization.bitsandbytes import BitsAndBytesQuantizer

            kwargs["quantization_config"] = BitsAndBytesQuantizer(8).get_config()
        logger.info(
            f"Loading model: {model_path} (device={device}, dtype={dtype}, quant={quantization or 'none'})"
        )
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        model.eval()
        logger.info(f"Model loaded: layers={arch.num_layers}, kv_heads={arch.num_kv_heads}")
        return model, arch
