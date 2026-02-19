"""Pravāha top-level inference engine.

Orchestrates model loading, tokenizer initialization, and the decoder loop
into a single clean API. This is the main entry point for using Pravāha.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, Optional

import torch

from pravaha.config import EngineConfig
from pravaha.decoder.decoder import DecoderEngine
from pravaha.decoder.sampling import Sampler, SamplingParams
from pravaha.models.loader import ModelLoader
from pravaha.tokenizer.tokenizer import PravahaTokenizer

logger = logging.getLogger(__name__)

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


class PravahaEngine:
    """Top-level inference engine — ties model, tokenizer, and decoder together.

    Usage:
        engine = PravahaEngine("configs/default.yaml")

        # Streaming generation
        for token in engine.generate("Hello, world"):
            print(token, end="", flush=True)

        # Non-streaming generation
        text = engine.generate_text("Explain quantum computing")
        print(text)
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        config: Optional[EngineConfig] = None,
    ):
        """Initialize the Pravāha engine.

        Args:
            config_path: Path to YAML configuration file.
            config: Direct configuration object (takes priority over config_path).

        Raises:
            FileNotFoundError: If config_path doesn't exist.
            RuntimeError: If model loading fails.
        """
        t0 = time.perf_counter()

        # 1. Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = EngineConfig.from_yaml(config_path)
        else:
            logger.info("No config provided, using defaults")
            self.config = EngineConfig.default()

        # Resolve "auto" device before using it
        self._device = self.config.model.resolved_device

        logger.info(f"Model: {self.config.model.model_path}")
        logger.info(f"Device: {self._device}")
        logger.info(f"Dtype: {self.config.model.dtype}")

        # 2. Load model
        loader = ModelLoader()
        self.model, self.arch_config = loader.load_model(
            model_path=self.config.model.model_path,
            dtype=self.config.model.torch_dtype,
            device=self._device,
        )

        # 3. Initialize tokenizer
        self.tokenizer = PravahaTokenizer(self.config.model.model_path)

        # 4. Create KV-cache (Phase 2: NaiveKVCache)
        self.kv_cache = None
        if self.config.cache.use_naive_cache:
            from pravaha.kv_cache.naive_cache import NaiveKVCache

            self.kv_cache = NaiveKVCache.from_model_config(
                arch_config=self.arch_config,
                max_seq_len=self.config.model.max_seq_len,
                dtype=self.config.model.torch_dtype,
                device=self._device,
            )
            logger.info(f"KV-cache: NaiveKVCache ({self.kv_cache})")
        else:
            logger.info("KV-cache: HuggingFace built-in")

        # 5. Create sampler and decoder
        self.sampler = Sampler()
        self.decoder = DecoderEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            sampler=self.sampler,
            device=self._device,
            kv_cache=self.kv_cache,
        )

        t1 = time.perf_counter()
        logger.info(f"Pravāha engine ready in {t1 - t0:.2f}s")

    def generate(
        self,
        prompt: str,
        **sampling_kwargs,
    ) -> Generator[str, None, None]:
        """Generate tokens from a prompt, yielding each token for streaming.

        Args:
            prompt: Input text prompt.
            **sampling_kwargs: Override default sampling params
                (temperature, top_k, top_p, max_new_tokens, etc.)

        Yields:
            Decoded token strings, one at a time.

        Example:
            for token in engine.generate("Once upon a time", temperature=0.8):
                print(token, end="", flush=True)
        """
        params = self._build_sampling_params(**sampling_kwargs)
        yield from self.decoder.generate(prompt, params)

    def generate_text(
        self,
        prompt: str,
        **sampling_kwargs,
    ) -> str:
        """Generate and return the complete response as a single string.

        Args:
            prompt: Input text prompt.
            **sampling_kwargs: Override default sampling params.

        Returns:
            Complete generated text.
        """
        params = self._build_sampling_params(**sampling_kwargs)
        return self.decoder.generate_text(prompt, params)

    def get_memory_stats(self) -> dict:
        """Return current GPU memory usage breakdown.

        Returns:
            Dict with allocated, reserved, and free GPU memory in GB.
        """
        if not torch.cuda.is_available():
            return {"device": "cpu", "note": "No GPU available"}

        device = torch.device(self._device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        props = torch.cuda.get_device_properties(device)
        total = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)

        return {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
            "total_gb": total / (1024**3),
            "free_gb": (total - allocated) / (1024**3),
            "utilization_pct": (allocated / total) * 100,
        }

    def _build_sampling_params(self, **kwargs) -> SamplingParams:
        """Merge user kwargs with default config to build SamplingParams."""
        defaults = self.config.sampling.model_dump()
        defaults.update({k: v for k, v in kwargs.items() if v is not None})
        return SamplingParams(**defaults)

    def __repr__(self) -> str:
        return (
            f"PravahaEngine("
            f"model={self.config.model.model_path}, "
            f"arch={self.arch_config.arch_name}, "
            f"device={self._device})"
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pravāha Inference Engine")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Prompt to generate")
    args = parser.parse_args()

    # Initialize Engine
    engine = PravahaEngine(config_path=args.config)

    print(f"\nExample 1: Streaming Generation (Prompt: '{args.prompt}')")
    print("-" * 50)
    
    t0 = time.perf_counter()
    # Streaming loop
    for token in engine.generate(args.prompt):
        print(token, end="", flush=True)
    t1 = time.perf_counter()
    
    print(f"\n\n" + "-" * 50)
    print(f"Total time: {t1 - t0:.2f}s")
    
    # Optional: Memory Stats
    print("\nMemory Stats:")
    print(engine.get_memory_stats())
