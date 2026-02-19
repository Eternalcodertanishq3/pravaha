"""Pravāha configuration system.

Pydantic-based configuration dataclasses with YAML loading support.
Each subsystem has its own config, composed into a top-level EngineConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model loading and architecture configuration."""

    model_config = {"protected_namespaces": ()}

    model_path: str = "gpt2"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device: str = "auto"  # "auto" detects CUDA availability; or "cuda", "cpu"
    max_seq_len: int = 1024

    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to the best available device."""
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def torch_dtype(self):
        """Convert string dtype to torch.dtype."""
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map[self.dtype]


class CacheConfig(BaseModel):
    """KV-cache memory management configuration."""

    block_size: int = 16
    num_gpu_blocks: int = 0  # 0 = auto-calculate based on free memory
    num_cpu_blocks: int = 256
    swap_space_gb: float = 4.0
    use_naive_cache: bool = True  # Phase 2: use our managed KV-cache


class SchedulerConfig(BaseModel):
    """Continuous batching scheduler configuration."""

    max_batch_size: int = 32
    max_waiting_requests: int = 256
    policy: Literal["fcfs"] = "fcfs"


class SamplingConfig(BaseModel):
    """Default sampling parameters for token generation."""

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    max_new_tokens: int = 256


class ServerConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent: int = 64


class EngineConfig(BaseModel):
    """Top-level engine configuration, composed of sub-configs."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls.model_validate(raw or {})

    @classmethod
    def default(cls) -> EngineConfig:
        """Return default configuration."""
        return cls()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
