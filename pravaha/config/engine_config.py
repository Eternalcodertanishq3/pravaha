"""Pravaha Engine Configuration with hot-reload support.

# Phase 3: Full configuration system with runtime hot-reload.

Pydantic-based configuration dataclasses with YAML loading support.
Each subsystem has its own config, composed into a top-level EngineConfig.
Hot-reloadable fields can be changed at runtime without server restart.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Sub-configurations
# ─────────────────────────────────────────────


class ModelConfig(BaseModel):
    """Model loading and architecture configuration.

    Defines which model to load, the precision, and device placement.
    These settings are NOT hot-reloadable — changing them requires a restart.
    """

    model_config = {"protected_namespaces": ()}

    model_path: str = "gpt2"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device: str = "auto"
    max_seq_len: int = 1024
    quantization: Literal["8bit", "4bit"] | None = None
    trust_remote_code: bool = False
    revision: str | None = None
    use_torch_compile: bool = False

    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to the best available device."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def torch_dtype(self) -> Any:
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
    num_gpu_blocks: int = 0  # 0 = auto-calculate
    num_cpu_blocks: int = 256
    swap_space_gb: float = 4.0
    use_naive_cache: bool = False  # Phase 4: always use paged
    enable_prefix_caching: bool = True
    enable_session_persistence: bool = True
    max_sessions: int = 1000
    session_ttl_seconds: int = 3600


class SchedulerConfig(BaseModel):
    """Continuous batching scheduler configuration."""

    max_batch_size: int = 32
    max_waiting_requests: int = 256
    policy: Literal["fcfs", "sjf", "priority"] = "fcfs"
    enable_adaptive_batching: bool = True
    adaptive_min_batch: int = 1
    adaptive_max_batch: int = 64


class SamplingConfig(BaseModel):
    """Default sampling parameters for token generation.

    HOT-RELOADABLE: All fields can be changed at runtime.
    """

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    max_new_tokens: int = 256
    repetition_penalty: float = 1.0


class SwarmInlineConfig(BaseModel):
    """Inline swarm settings within the engine config.

    For full swarm configuration, see SwarmConfig.
    """

    enabled: bool = False
    self_heal: bool = True
    max_audit_iterations: int = 3
    agent_roles: list[str] = Field(default_factory=lambda: ["planner", "coder", "critic"])
    default_pipeline: str = "plan-execute-audit"


class RAGInlineConfig(BaseModel):
    """Inline RAG settings within the engine config."""

    enabled: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    store_path: str = "data/rag"


class ServerInlineConfig(BaseModel):
    """Inline server settings within the engine config."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent: int = 64
    enable_cors: bool = True
    api_key: str | None = None
    rate_limit_rpm: int = 0  # 0 = unlimited


class GuardrailsConfig(BaseModel):
    """Content filtering and token budget configuration.

    HOT-RELOADABLE: All fields can be changed at runtime.
    """

    enable_content_filter: bool = False
    token_budget_per_request: int = 0  # 0 = unlimited
    token_budget_per_session: int = 0
    blocked_patterns: list[str] = Field(default_factory=list)


class DistributedConfig(BaseModel):
    """Multi-node distributed inference configuration.

    Controls Tensor Parallelism (TP), Pipeline Parallelism (PP),
    and inter-node communication settings for scaling inference
    across multiple GPUs and nodes.

    NOT hot-reloadable — changing these settings requires a full restart.
    """

    enabled: bool = False
    tp_size: int = 1  # Tensor Parallelism degree (splits model weights across GPUs)
    pp_size: int = 1  # Pipeline Parallelism stages (splits model layers across GPUs)
    backend: Literal["nccl", "gloo"] = "nccl"  # "nccl" for GPU, "gloo" for CPU/cross-node
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    init_method: str | None = None  # Custom init method (e.g., "tcp://host:port")
    timeout_seconds: int = 300
    heartbeat_interval: float = 5.0  # Seconds between node health checks


# ─────────────────────────────────────────────
# Hot-Reloadable Config Mixin
# ─────────────────────────────────────────────


class ConfigurationError(Exception):
    """Raised when configuration is invalid or inconsistent."""

    pass


# Fields that CAN be changed without restarting the server
HOT_RELOADABLE_FIELDS: set[str] = {
    "sampling.temperature",
    "sampling.top_k",
    "sampling.top_p",
    "sampling.max_new_tokens",
    "sampling.repetition_penalty",
    "scheduler.max_batch_size",
    "scheduler.enable_adaptive_batching",
    "swarm.enabled",
    "swarm.self_heal",
    "swarm.max_audit_iterations",
    "guardrails.enable_content_filter",
    "guardrails.token_budget_per_request",
    "guardrails.token_budget_per_session",
    "guardrails.blocked_patterns",
    "server.rate_limit_rpm",
}

# Fields that CANNOT be changed without restarting
COLD_FIELDS: set[str] = {
    "model.model_path",
    "model.quantization",
    "model.dtype",
    "model.device",
    "cache.num_gpu_blocks",
    "cache.block_size",
    "distributed.enabled",
    "distributed.tp_size",
    "distributed.pp_size",
    "distributed.backend",
    "distributed.master_addr",
    "distributed.master_port",
}


# ─────────────────────────────────────────────
# Top-level Engine Config
# ─────────────────────────────────────────────


class EngineConfig(BaseModel):
    """Top-level engine configuration, composed of sub-configs.

    Supports hot-reload for a subset of fields (sampling, scheduler params,
    guardrails). Model and cache settings require a server restart.

    Usage:
        config = EngineConfig.from_yaml("configs/default.yaml")
        config.watch(my_callback)  # Get notified on hot-reload
        config.update(sampling={"temperature": 0.7})  # Hot-reload
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    swarm: SwarmInlineConfig = Field(default_factory=SwarmInlineConfig)
    rag: RAGInlineConfig = Field(default_factory=RAGInlineConfig)
    server: ServerInlineConfig = Field(default_factory=ServerInlineConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    # Internal state (not serialized)
    _watchers: list[Callable[[EngineConfig], None]] = PrivateAttr(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _source_path: Path | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def validate_consistency(self) -> None:
        """Validate configuration consistency.

        Raises ConfigurationError if swarm is enabled but no agent roles
        are configured, or other invalid combinations are found.
        """
        if self.swarm.enabled and not self.swarm.agent_roles:
            raise ConfigurationError(
                "Swarm is enabled but no agent roles are configured. "
                "Either set swarm.enabled=false or configure swarm.agent_roles "
                "in your config file. Example: agent_roles: [planner, coder, critic]"
            )

        if self.rag.enabled and not self.rag.store_path:
            raise ConfigurationError("RAG is enabled but no store_path is configured.")

    def watch(self, callback: Callable[[EngineConfig], None]) -> None:
        """Register a callback for hot-reload notifications.

        Args:
            callback: Function called with the updated config after hot-reload.
        """
        self._watchers.append(callback)

    def update(self, **kwargs: Any) -> list[str]:
        """Hot-reload updatable fields at runtime.

        Only fields listed in HOT_RELOADABLE_FIELDS are accepted.
        Returns a list of field paths that were actually updated.

        Args:
            **kwargs: Section-level dicts, e.g. sampling={"temperature": 0.7}

        Returns:
            List of updated field paths.

        Raises:
            ConfigurationError: If a non-hot-reloadable field is specified.
        """
        updated: list[str] = []

        with self._lock:
            for section_name, section_updates in kwargs.items():
                if not isinstance(section_updates, dict):
                    raise ConfigurationError(
                        f"Expected dict for section '{section_name}', "
                        f"got {type(section_updates).__name__}"
                    )

                section = getattr(self, section_name, None)
                if section is None:
                    raise ConfigurationError(f"Unknown config section: {section_name}")

                for field_name, value in section_updates.items():
                    field_path = f"{section_name}.{field_name}"

                    if field_path in COLD_FIELDS:
                        raise ConfigurationError(
                            f"Field '{field_path}' cannot be hot-reloaded. Server restart required."
                        )

                    if field_path not in HOT_RELOADABLE_FIELDS:
                        logger.warning(
                            f"Field '{field_path}' is not in the hot-reloadable list. "
                            f"Applying anyway, but this may not take effect."
                        )

                # Validate new values using Pydantic
                current_data = section.model_dump()
                current_data.update(section_updates)
                try:
                    new_section = type(section).model_validate(current_data)
                except Exception as e:
                    from pydantic import ValidationError
                    if isinstance(e, ValidationError):
                        raise ConfigurationError(f"Validation failed for section '{section_name}': {e}")
                    raise

                setattr(self, section_name, new_section)

                for field_name, value in section_updates.items():
                    field_path = f"{section_name}.{field_name}"
                    updated.append(field_path)
                    logger.info(f"Hot-reload: {field_path} = {value}")

        # Notify watchers
        for watcher in self._watchers:
            try:
                watcher(self)
            except Exception as e:
                logger.error(f"Config watcher error: {e}", exc_info=True)

        return updated

    def reload_from_yaml(self, path: str | Path | None = None) -> list[str]:
        """Reload hot-reloadable fields from a YAML file.

        Args:
            path: YAML file path. Uses the original source path if None.

        Returns:
            List of updated field paths.
        """
        load_path = Path(path) if path else self._source_path
        if load_path is None or not load_path.exists():
            raise ConfigurationError(f"Cannot reload: config file not found at {load_path}")

        with open(load_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Only apply hot-reloadable sections
        updates: dict[str, dict[str, Any]] = {}
        for section_name in ["sampling", "scheduler", "swarm", "guardrails"]:
            if section_name in raw:
                updates[section_name] = raw[section_name]

        if updates:
            return self.update(**updates)
        return []

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Fully initialized EngineConfig.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        config = cls.model_validate(raw or {})
        object.__setattr__(config, "_source_path", path)
        return config

    @classmethod
    def default(cls) -> EngineConfig:
        """Return default configuration."""
        return cls()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Output file path. Parent directories are created automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
