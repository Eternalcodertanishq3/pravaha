"""Engine Factory — Factory class for building Pravaha engine components.

Decouples subsystem construction and hardware device resolution from the main AsyncPravahaEngine.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pravaha.config.engine_config import EngineConfig
from pravaha.decoder.decoder import DecoderEngine
from pravaha.decoder.sampling import Sampler
from pravaha.memory.block_manager import BlockManager
from pravaha.memory.paged_cache import PagedKVCache
from pravaha.memory.session_cache import SessionKVCache
from pravaha.models.loader import ModelLoader
from pravaha.scheduler.continuous_scheduler import ContinuousScheduler
from pravaha.tokenizer.tokenizer import PravahaTokenizer

logger = logging.getLogger(__name__)


class EngineFactory:
    """Factory for constructing and assembling Pravaha engine subsystems."""

    @staticmethod
    def build_subsystems(config: EngineConfig, device: str) -> dict[str, Any]:
        """Build all engine subsystems.

        Returns:
            Dict containing initialized subsystems: tokenizer, model, arch_config,
            block_manager, kv_cache, decoder, scheduler, session_cache.
        """
        t0 = time.time()

        # 1. Tokenizer
        logger.info(f"EngineFactory: Loading tokenizer from {config.model.model_path}")
        tokenizer = PravahaTokenizer(config.model.model_path)

        # 2. Model & loader
        logger.info(f"EngineFactory: Loading model from {config.model.model_path}")
        loader = ModelLoader()
        model, arch_config = loader.load(
            model_path=config.model.model_path,
            device=device,
            dtype=config.model.torch_dtype,
            quantization=config.model.quantization,
            trust_remote_code=config.model.trust_remote_code,
            use_torch_compile=config.model.use_torch_compile,
        )

        # 3. Block manager
        num_blocks = config.cache.num_gpu_blocks or 256
        block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=config.cache.block_size,
        )

        # 4. Paged KV cache
        kv_cache = PagedKVCache(
            num_layers=arch_config.num_layers,
            num_kv_heads=arch_config.num_kv_heads,
            head_dim=arch_config.head_dim,
            block_size=config.cache.block_size,
            num_blocks=num_blocks,
            dtype=config.model.torch_dtype,
            device=device,
        )

        # 5. Decoder engine
        decoder = DecoderEngine(
            model=model,
            tokenizer=tokenizer,
            sampler=Sampler(),
            device=device,
            kv_cache=kv_cache,
        )

        # 6. Scheduler
        scheduler = ContinuousScheduler(
            num_blocks=num_blocks,
            block_size=config.cache.block_size,
            max_batch_size=config.scheduler.max_batch_size,
            max_seq_len=config.model.max_seq_len,
            max_waiting_requests=config.scheduler.max_waiting_requests,
        )

        # 7. Session cache
        session_cache = SessionKVCache(
            max_sessions=config.cache.max_sessions,
            ttl_seconds=config.cache.session_ttl_seconds,
        )

        elapsed = time.time() - t0
        logger.info(f"EngineFactory: Subsystems constructed in {elapsed:.2f}s.")

        return {
            "tokenizer": tokenizer,
            "model": model,
            "arch_config": arch_config,
            "block_manager": block_manager,
            "kv_cache": kv_cache,
            "decoder": decoder,
            "scheduler": scheduler,
            "session_cache": session_cache,
            "num_blocks": num_blocks,
        }
