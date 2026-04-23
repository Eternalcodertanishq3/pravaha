"""Engine Factory — Build engine from config with hardware detection.

Detects available hardware (GPU type, VRAM, CPU cores) and constructs
the appropriate engine configuration automatically.
"""

from __future__ import annotations

import logging

from pravaha.config.engine_config import EngineConfig

logger = logging.getLogger(__name__)


def detect_hardware() -> dict[str, object]:
    """Detect available hardware and return a summary.

    Returns:
        Dictionary with GPU/CPU details.
    """
    info: dict[str, object] = {
        "gpu_available": False,
        "gpu_name": "N/A",
        "gpu_count": 0,
        "vram_gb": 0.0,
        "cpu_cores": 1,
        "ram_gb": 0.0,
    }

    try:
        import psutil

        info["cpu_cores"] = psutil.cpu_count(logical=False) or 1
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_mem = getattr(props, "total_memory", 0)
            info["vram_gb"] = round(total_mem / (1024**3), 1)
    except ImportError:
        pass

    return info


def auto_configure_blocks(vram_gb: float, model_size_gb: float = 2.0) -> int:
    """Auto-calculate optimal number of KV cache blocks based on free VRAM.

    Reserves ~60% of VRAM for the model and activations, uses ~30% for
    KV cache, and keeps ~10% as headroom.

    Args:
        vram_gb: Total VRAM in GB.
        model_size_gb: Estimated model size in GB.

    Returns:
        Recommended number of cache blocks.
    """
    available_for_cache = max(0.5, (vram_gb - model_size_gb) * 0.5)
    # Each block ≈ 2 * num_layers * block_size * num_kv_heads * head_dim * 2 bytes
    # Conservative estimate: ~0.5 MB per block for a 7B model
    blocks = int(available_for_cache * 1024 / 0.5)
    return max(32, min(blocks, 8192))


def build_engine(
    config: EngineConfig | None = None,
    config_path: str | None = None,
    model: str | None = None,
    quantize: str | None = None,
    device: str | None = None,
) -> object:
    """Build an AsyncPravahaEngine from configuration.

    Convenience factory that handles config loading, hardware detection,
    auto-configuration, and validation before engine construction.

    Args:
        config: Pre-built config. If None, loaded from config_path or defaults.
        config_path: Path to YAML config file.
        model: Model path/name override.
        quantize: Quantization override ('4bit', '8bit', None).
        device: Device override.

    Returns:
        Initialized AsyncPravahaEngine.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    # 1. Load or create config
    if config is None:
        if config_path:
            config = EngineConfig.from_yaml(config_path)
        else:
            config = EngineConfig.default()

    # 2. Apply overrides
    if model:
        config.model.model_path = model
    if quantize:
        config.model.quantization = quantize  # type: ignore[assignment]
    if device:
        config.model.device = device

    # 3. Detect hardware
    hw = detect_hardware()
    logger.info(f"Hardware detected: {hw}")

    # 4. Auto-configure if needed
    if config.cache.num_gpu_blocks == 0 and hw["gpu_available"]:
        vram = float(hw["vram_gb"])  # type: ignore[arg-type]
        config.cache.num_gpu_blocks = auto_configure_blocks(vram)
        logger.info(f"Auto-configured {config.cache.num_gpu_blocks} GPU cache blocks")
    elif config.cache.num_gpu_blocks == 0:
        config.cache.num_gpu_blocks = 64  # CPU fallback
        logger.info("Running on CPU — using minimal cache blocks")

    # 5. Validate configuration consistency
    config.validate_consistency()

    # 6. Build engine
    from pravaha.engine.async_engine import AsyncPravahaEngine

    engine = AsyncPravahaEngine(config=config)

    logger.info(
        f"Engine built: model={config.model.model_path}, "
        f"device={config.model.resolved_device}, "
        f"quant={config.model.quantization or 'none'}"
    )

    return engine
