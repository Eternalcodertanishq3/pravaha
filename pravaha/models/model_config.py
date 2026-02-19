"""Model architecture configuration.

Auto-detects model architecture from HuggingFace config.json and extracts
key parameters needed for inference (layer count, hidden sizes, attention heads, etc.).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported architectures and their config key mappings
_ARCH_KEY_MAPS: dict[str, dict[str, str]] = {
    "gpt2": {
        "num_layers": "n_layer",
        "hidden_size": "n_embd",
        "num_heads": "n_head",
        "num_kv_heads": "n_head",  # GPT-2 uses MHA (no GQA)
        "intermediate_size": "n_inner",  # May be None → 4 * hidden_size
        "vocab_size": "vocab_size",
        "max_position_embeddings": "n_positions",
    },
    "llama": {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "num_kv_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
        "max_position_embeddings": "max_position_embeddings",
    },
    "mistral": {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "num_kv_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
        "max_position_embeddings": "max_position_embeddings",
    },
}


@dataclass
class ModelArchConfig:
    """Extracted model architecture parameters.

    These are the essential dimensions needed by the inference engine
    to allocate KV-cache, construct attention masks, and manage memory.
    """

    arch_name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    head_dim: int = field(init=False)
    # Extras from the raw config (e.g., rope_theta, rms_norm_eps)
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_heads

    @property
    def kv_cache_size_per_token(self) -> int:
        """Bytes of KV-cache per token per layer (for fp16).

        KV shape per layer: 2 (K+V) × num_kv_heads × head_dim × 2 bytes (fp16)
        """
        return 2 * self.num_kv_heads * self.head_dim * 2  # 2 bytes for fp16

    @property
    def total_kv_cache_per_token(self) -> int:
        """Total KV-cache bytes per token across all layers (fp16)."""
        return self.kv_cache_size_per_token * self.num_layers

    def estimate_model_memory_bytes(self, dtype_bytes: int = 2) -> int:
        """Rough estimate of model weight memory in bytes.

        This is an approximation based on parameter count.
        """
        # Embedding: vocab_size × hidden_size
        embed_params = self.vocab_size * self.hidden_size

        # Per transformer layer (approximate):
        #   QKV projections: 3 × hidden × hidden  (or adjusted for GQA)
        #   Output projection: hidden × hidden
        #   MLP: 2 × hidden × intermediate + intermediate (gate for Llama)
        #   Layer norms: 2 × hidden
        qkv = self.hidden_size * (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        out_proj = self.hidden_size * self.hidden_size
        mlp = 2 * self.hidden_size * self.intermediate_size
        ln = 2 * self.hidden_size
        layer_params = qkv + out_proj + mlp + ln

        total_params = embed_params + (layer_params * self.num_layers) + self.hidden_size
        return total_params * dtype_bytes


def _detect_architecture(config_dict: dict) -> str:
    """Detect model architecture from HuggingFace config."""
    # Check model_type field first
    model_type = config_dict.get("model_type", "").lower()
    if model_type in _ARCH_KEY_MAPS:
        return model_type

    # Check architectures list
    archs = config_dict.get("architectures", [])
    for arch in archs:
        arch_lower = arch.lower()
        for known_arch in _ARCH_KEY_MAPS:
            if known_arch in arch_lower:
                return known_arch

    raise ValueError(
        f"Unsupported model architecture. "
        f"model_type='{model_type}', architectures={archs}. "
        f"Supported: {list(_ARCH_KEY_MAPS.keys())}"
    )


def parse_model_config(
    model_path: str,
    config_override: Optional[dict] = None,
) -> ModelArchConfig:
    """Parse a HuggingFace model config and extract architecture parameters.

    Args:
        model_path: HuggingFace model name or local directory path.
        config_override: Optional dict to override parsed config values.

    Returns:
        ModelArchConfig with extracted architecture dimensions.
    """
    path = Path(model_path)

    if path.is_dir():
        config_file = path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"No config.json found in {model_path}")
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    else:
        # Remote HuggingFace model — use transformers to fetch config
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path)
        config_dict = hf_config.to_dict()

    arch_name = _detect_architecture(config_dict)
    key_map = _ARCH_KEY_MAPS[arch_name]

    def _get(key: str, default=None):
        hf_key = key_map.get(key, key)
        return config_dict.get(hf_key, default)

    num_layers = _get("num_layers")
    hidden_size = _get("hidden_size")
    num_heads = _get("num_heads")
    num_kv_heads = _get("num_kv_heads", num_heads)  # Default to MHA
    intermediate_size = _get("intermediate_size") or (4 * hidden_size)
    vocab_size = _get("vocab_size")
    max_pos = _get("max_position_embeddings", 2048)

    if config_override:
        num_layers = config_override.get("num_layers", num_layers)
        hidden_size = config_override.get("hidden_size", hidden_size)
        num_heads = config_override.get("num_heads", num_heads)
        num_kv_heads = config_override.get("num_kv_heads", num_kv_heads)

    logger.info(
        f"Detected architecture: {arch_name} | "
        f"layers={num_layers}, hidden={hidden_size}, "
        f"heads={num_heads}, kv_heads={num_kv_heads}, "
        f"vocab={vocab_size}"
    )

    return ModelArchConfig(
        arch_name=arch_name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        extras={
            k: v
            for k, v in config_dict.items()
            if k not in {"model_type", "architectures", "torch_dtype"}
        },
    )
