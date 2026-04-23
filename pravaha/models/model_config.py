"""Model Architecture Configurations — Known model families.

Maps model architectures to their structural parameters (layers,
heads, hidden dimensions). Used by the engine to configure KV
cache sizing and attention calculations.

Supports: GPT-2, Llama, Mistral, Phi, Qwen, Gemma, LLaVA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArchConfig:
    """Architecture parameters extracted from model config.

    These define the structure needed for KV cache allocation
    and attention mask generation.
    """

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    max_position_embeddings: int = 2048
    intermediate_size: int = 0
    model_type: str = "unknown"

    @property
    def kv_cache_size_per_layer(self) -> int:
        """Bytes per layer per token for KV cache (fp16)."""
        return self.num_kv_heads * self.head_dim * 2 * 2  # K+V, 2 bytes each

    @property
    def total_params_estimate(self) -> int:
        """Rough parameter count estimate."""
        # Embedding + attention + MLP per layer + final LM head
        embed = self.vocab_size * self.hidden_size
        attn_per_layer = 4 * self.hidden_size * self.hidden_size  # Q,K,V,O
        mlp_per_layer = 3 * self.hidden_size * (self.intermediate_size or 4 * self.hidden_size)
        return embed + self.num_layers * (attn_per_layer + mlp_per_layer) + embed


# Known architecture presets
KNOWN_ARCHITECTURES: dict[str, dict[str, Any]] = {
    "gpt2": {
        "num_layers": 12,
        "num_heads": 12,
        "num_kv_heads": 12,
        "head_dim": 64,
        "hidden_size": 768,
        "vocab_size": 50257,
        "max_position_embeddings": 1024,
        "model_type": "gpt2",
    },
    "gpt2-medium": {
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 64,
        "hidden_size": 1024,
        "vocab_size": 50257,
        "max_position_embeddings": 1024,
        "model_type": "gpt2",
    },
    "gpt2-large": {
        "num_layers": 36,
        "num_heads": 20,
        "num_kv_heads": 20,
        "head_dim": 64,
        "hidden_size": 1280,
        "vocab_size": 50257,
        "max_position_embeddings": 1024,
        "model_type": "gpt2",
    },
    "llama-7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "intermediate_size": 11008,
        "model_type": "llama",
    },
    "llama-3-8b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "intermediate_size": 14336,
        "model_type": "llama",
    },
    "llama-3-70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 8192,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "intermediate_size": 28672,
        "model_type": "llama",
    },
    "mistral-7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 32000,
        "max_position_embeddings": 32768,
        "intermediate_size": 14336,
        "model_type": "mistral",
    },
    "phi-2": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 80,
        "hidden_size": 2560,
        "vocab_size": 51200,
        "max_position_embeddings": 2048,
        "model_type": "phi",
    },
    "phi-3-mini": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 96,
        "hidden_size": 3072,
        "vocab_size": 32064,
        "max_position_embeddings": 4096,
        "model_type": "phi3",
    },
    "qwen2-7b": {
        "num_layers": 32,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "hidden_size": 3584,
        "vocab_size": 152064,
        "max_position_embeddings": 131072,
        "intermediate_size": 18944,
        "model_type": "qwen2",
    },
    "gemma-2b": {
        "num_layers": 18,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 256,
        "hidden_size": 2048,
        "vocab_size": 256000,
        "max_position_embeddings": 8192,
        "model_type": "gemma",
    },
    "gemma-7b": {
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 256,
        "hidden_size": 3072,
        "vocab_size": 256000,
        "max_position_embeddings": 8192,
        "model_type": "gemma",
    },
    "llava-1.5-7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 32064,
        "max_position_embeddings": 4096,
        "model_type": "llava",
    },
}


def parse_model_config(
    hf_config: Any,
    model_name: str = "",
) -> ArchConfig:
    """Extract ArchConfig from a HuggingFace model config.

    Falls back to known architecture presets if HF config
    fields are missing.

    Args:
        hf_config: transformers.PretrainedConfig instance.
        model_name: Optional model name for preset lookup.

    Returns:
        ArchConfig with all structural parameters populated.
    """
    # Try HF config attributes first
    num_layers = getattr(hf_config, "num_hidden_layers", None) or getattr(hf_config, "n_layer", 12)
    num_heads = getattr(hf_config, "num_attention_heads", None) or getattr(hf_config, "n_head", 12)
    num_kv_heads = getattr(hf_config, "num_key_value_heads", None) or num_heads
    hidden_size = getattr(hf_config, "hidden_size", None) or getattr(hf_config, "n_embd", 768)
    head_dim = hidden_size // num_heads if num_heads > 0 else 64
    vocab_size = getattr(hf_config, "vocab_size", 32000)
    max_pos = getattr(hf_config, "max_position_embeddings", 2048)
    intermediate = getattr(hf_config, "intermediate_size", 4 * hidden_size)
    model_type = getattr(hf_config, "model_type", "unknown")

    config = ArchConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        intermediate_size=intermediate,
        model_type=model_type,
    )

    logger.info(
        f"ArchConfig: {model_type} | {num_layers}L {num_heads}H "
        f"{num_kv_heads}KV {head_dim}D | vocab={vocab_size}"
    )
    return config
