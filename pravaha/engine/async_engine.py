"""Async Pravaha Engine — The main inference entry point.

# Phase 3+: Production async engine with continuous batching.
# Fix 1: Race condition — added threading.Event for loop synchronization.

This is the primary interface for inference. It manages:
- Model loading and tokenizer initialization
- Background scheduler loop (in a dedicated thread)
- Request submission via asyncio Futures
- Token streaming via async generators

Fix 1 Applied: The original implementation had a race condition where
the background loop could start processing before the model was fully
loaded. We now use a threading.Event (_ready_event) that gates the
scheduler loop until initialization completes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncGenerator

from pravaha.config.engine_config import EngineConfig
from pravaha.decoder.decoder import DecoderEngine
from pravaha.decoder.sampling import Sampler, SamplingParams
from pravaha.engine.events import EngineEvent, EventBus, EventType
from pravaha.memory.block_manager import BlockManager
from pravaha.memory.paged_cache import PagedKVCache
from pravaha.memory.session_cache import SessionKVCache
from pravaha.models.loader import ModelLoader
from pravaha.scheduler.continuous_scheduler import ContinuousScheduler
from pravaha.scheduler.request import FinishReason, InferenceRequest
from pravaha.tokenizer.tokenizer import PravahaTokenizer

logger = logging.getLogger(__name__)


class AsyncPravahaEngine:
    """Production async inference engine with continuous batching.

    This is the top-level engine that orchestrates all subsystems:
    model loading, tokenization, scheduling, KV-cache management,
    and token generation.

    Fix 1: Race condition prevention.
    The background scheduler loop now waits on self._ready_event before
    processing any requests. This prevents crashes when requests arrive
    during model loading.

    Usage:
        engine = AsyncPravahaEngine(config=EngineConfig.from_yaml("config.yaml"))
        async for token in engine.generate("Hello", SamplingParams()):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        config_path: str | None = None,
    ) -> None:
        """Initialize the engine and start the background loop.

        Args:
            config: Pre-built engine configuration.
            config_path: Path to YAML config (used if config is None).
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = EngineConfig.from_yaml(config_path)
        else:
            self.config = EngineConfig.default()

        # Device resolution
        self._device = self.config.model.resolved_device

        # Event bus for telemetry
        self.event_bus = EventBus()

        # ── Fix 1: Threading synchronization ──
        # This Event gates the background loop until initialization completes.
        # Without this, the loop would try to access self._decoder before
        # the model is loaded, causing an AttributeError race condition.
        self._ready_event = threading.Event()
        self._shutdown_event = threading.Event()

        # Request tracking
        self._request_futures: dict[str, asyncio.Future[list[int]]] = {}
        self._request_queues: dict[str, asyncio.Queue[str]] = {}
        self._active_requests: dict[str, InferenceRequest] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

        # Load model, tokenizer, and build subsystems
        self._initialize_subsystems()

        # ── Fix 5: Swarm config validation at startup ──
        # If swarm is enabled but no agent roles are configured, fail fast
        # with a clear error instead of crashing at request time.
        if hasattr(self.config, "swarm") and self.config.swarm.enabled:
            if not self.config.swarm.agent_roles:
                raise ValueError(
                    "ConfigurationError: Swarm is enabled but no agent roles "
                    "are configured. Set 'swarm.agent_roles' in your config "
                    "or disable swarm with 'swarm.enabled: false'."
                )

        # Start background scheduler thread
        self._bg_thread = threading.Thread(
            target=self._run_scheduler_loop,
            name="pravaha-scheduler",
            daemon=True,
        )
        self._bg_thread.start()

        # Signal that initialization is complete
        # The background loop is now safe to start processing
        self._ready_event.set()

        logger.info("AsyncPravahaEngine initialized and ready.")

    def _initialize_subsystems(self) -> None:
        """Load model and initialize all engine subsystems."""
        t0 = time.time()

        # 1. Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model.model_path}")
        self.tokenizer = PravahaTokenizer(self.config.model.model_path)

        # 2. Load model
        logger.info(f"Loading model: {self.config.model.model_path}")
        loader = ModelLoader()
        self.model, self.arch_config = loader.load(
            model_path=self.config.model.model_path,
            device=self._device,
            dtype=self.config.model.torch_dtype,
            quantization=self.config.model.quantization,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        # 3. Block manager
        num_blocks = self.config.cache.num_gpu_blocks or 256
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=self.config.cache.block_size,
        )

        # 4. Paged KV cache
        self.kv_cache = PagedKVCache(
            num_layers=self.arch_config.num_layers,
            num_kv_heads=self.arch_config.num_kv_heads,
            head_dim=self.arch_config.head_dim,
            block_size=self.config.cache.block_size,
            num_blocks=num_blocks,
            dtype=self.config.model.torch_dtype,
            device=self._device,
        )

        # 5. Decoder engine
        self._decoder = DecoderEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            sampler=Sampler(),
            device=self._device,
            kv_cache=self.kv_cache,
        )

        # 6. Scheduler
        self._scheduler = ContinuousScheduler(
            num_blocks=num_blocks,
            block_size=self.config.cache.block_size,
            max_batch_size=self.config.scheduler.max_batch_size,
            max_seq_len=self.config.model.max_seq_len,
        )

        # 7. Session cache (for multi-turn conversations)
        self.session_cache = SessionKVCache(
            max_sessions=self.config.cache.max_sessions,
            ttl_seconds=self.config.cache.session_ttl_seconds,
        )

        # Metrics
        self._total_requests = 0
        self._total_tokens_generated = 0

        elapsed = time.time() - t0
        self.event_bus.publish(
            EngineEvent(
                event_type=EventType.MODEL_LOADED,
                data={"model": self.config.model.model_path, "load_time_s": round(elapsed, 2)},
            )
        )
        logger.info(f"All subsystems initialized in {elapsed:.1f}s")

    async def generate(
        self,
        prompt: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate tokens for a prompt, yielding one token at a time.

        This is the primary API for inference. Submits the request to the
        scheduler and yields decoded tokens as they're produced.

        Args:
            prompt: Input text prompt.
            params: Sampling parameters. Uses defaults if None.
            session_id: Optional session ID for multi-turn caching.

        Yields:
            Decoded token text, one token per yield.
        """
        if params is None:
            params = SamplingParams(
                temperature=self.config.sampling.temperature,
                top_k=self.config.sampling.top_k,
                top_p=self.config.sampling.top_p,
                max_new_tokens=self.config.sampling.max_new_tokens,
                repetition_penalty=self.config.sampling.repetition_penalty,
            )

        request_id = str(uuid.uuid4())
        self._total_requests += 1

        self.event_bus.publish(
            EngineEvent(
                event_type=EventType.REQUEST_RECEIVED,
                request_id=request_id,
                data={"prompt_len": len(prompt), "session_id": session_id},
            )
        )

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)

        # Create inference request
        request = InferenceRequest(
            request_id=request_id,
            prompt_token_ids=input_ids,
            sampling_params=params,
        )

        # Set up token streaming queue
        token_queue: asyncio.Queue[str] = asyncio.Queue()
        self._request_queues[request_id] = token_queue
        self._active_requests[request_id] = request

        # Capture the current event loop for cross-thread communication
        self._loop = asyncio.get_running_loop()

        # Submit to scheduler
        self._scheduler.add_request(request)

        # Yield tokens as they arrive
        t_start = time.time()
        first_token = True
        tokens_generated = 0

        try:
            while True:
                token_text = await token_queue.get()

                if token_text == "<|DONE|>":
                    break
                if token_text == "<|ERROR|>":
                    break

                if first_token:
                    ttft = (time.time() - t_start) * 1000
                    self.event_bus.publish(
                        EngineEvent(
                            event_type=EventType.TOKEN_GENERATED,
                            request_id=request_id,
                            data={"ttft_ms": round(ttft, 1)},
                        )
                    )
                    first_token = False

                tokens_generated += 1
                self._total_tokens_generated += 1
                yield token_text

        finally:
            # Cleanup
            self._request_queues.pop(request_id, None)
            self._active_requests.pop(request_id, None)

            total_ms = (time.time() - t_start) * 1000
            self.event_bus.publish(
                EngineEvent(
                    event_type=EventType.REQUEST_COMPLETE,
                    request_id=request_id,
                    duration_ms=round(total_ms, 1),
                    data={
                        "tokens": tokens_generated,
                        "tps": round(tokens_generated / (total_ms / 1000), 1)
                        if total_ms > 0
                        else 0,
                    },
                )
            )

    def _run_scheduler_loop(self) -> None:
        """Background thread: continuously schedule and execute batches.

        Fix 1: Waits on _ready_event before starting to prevent race
        conditions during model loading.
        """
        # ── Fix 1: Wait for initialization to complete ──
        # Without this gate, the loop would start immediately when the
        # thread is created, but self._decoder etc. might not exist yet.
        self._ready_event.wait()
        logger.info("Scheduler loop started.")

        while not self._shutdown_event.is_set():
            try:
                if not self._scheduler.has_unfinished_requests():
                    # No work to do — sleep briefly to avoid busy-waiting
                    time.sleep(0.001)
                    continue

                # Run one scheduling step
                scheduled = self._scheduler.step()

                # Process prefill batch
                if scheduled["prefill"]:
                    self._process_prefill(scheduled["prefill"])

                # Process decode batch
                if scheduled["decode"]:
                    self._process_decode(scheduled["decode"])

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}", exc_info=True)
                time.sleep(0.01)

        logger.info("Scheduler loop stopped.")

    def _process_prefill(self, requests: list[InferenceRequest]) -> None:
        """Execute prefill for a batch of new requests."""
        try:
            input_ids_list = [r.prompt_token_ids for r in requests]
            request_ids = [r.request_id for r in requests]
            block_tables = [r.block_table for r in requests]

            next_tokens = self._decoder.step_prefill(input_ids_list, request_ids, block_tables)

            for req, token_id in zip(requests, next_tokens):
                req.generated_token_ids.append(token_id)
                token_text = self.tokenizer.decode_token(token_id)
                self._send_token(req.request_id, token_text)

                # Check for EOS
                if token_id == self.tokenizer.eos_token_id:
                    req.mark_finished(FinishReason.EOS)
                    self._send_token(req.request_id, "<|DONE|>")

        except Exception as e:
            logger.error(f"Prefill error: {e}", exc_info=True)
            for req in requests:
                req.mark_finished(FinishReason.ABORTED)
                self._send_token(req.request_id, "<|ERROR|>")

    def _process_decode(self, requests: list[InferenceRequest]) -> None:
        """Execute one decode step for running requests."""
        try:
            token_ids = [r.generated_token_ids[-1] for r in requests]
            request_ids = [r.request_id for r in requests]
            block_tables = [r.block_table for r in requests]
            context_lens = [r.total_tokens - 1 for r in requests]

            next_tokens = self._decoder.step_decode(
                token_ids, request_ids, block_tables, context_lens
            )

            for req, token_id in zip(requests, next_tokens):
                req.generated_token_ids.append(token_id)
                token_text = self.tokenizer.decode_token(token_id)
                self._send_token(req.request_id, token_text)

                # Check completion conditions
                if token_id == self.tokenizer.eos_token_id:
                    req.mark_finished(FinishReason.EOS)
                    self._send_token(req.request_id, "<|DONE|>")
                elif len(req.generated_token_ids) >= req.sampling_params.max_new_tokens:
                    req.mark_finished(FinishReason.MAX_TOKENS)
                    self._send_token(req.request_id, "<|DONE|>")

        except Exception as e:
            logger.error(f"Decode error: {e}", exc_info=True)
            for req in requests:
                req.mark_finished(FinishReason.ABORTED)
                self._send_token(req.request_id, "<|ERROR|>")

    def _send_token(self, request_id: str, text: str) -> None:
        """Thread-safe: push a token to the request's async queue."""
        queue = self._request_queues.get(request_id)
        if queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(queue.put_nowait, text)

    def abort_request(self, request_id: str) -> bool:
        """Abort a running or pending request.

        Args:
            request_id: Request to abort.

        Returns:
            True if the request was found and aborted.
        """
        aborted = self._scheduler.abort_request(request_id)
        if aborted:
            self._send_token(request_id, "<|DONE|>")
        return aborted

    def stop(self) -> None:
        """Gracefully shut down the engine."""
        logger.info("Stopping engine...")
        self._shutdown_event.set()
        if self._bg_thread.is_alive():
            self._bg_thread.join(timeout=5.0)
        logger.info("Engine stopped.")

    def get_stats(self) -> dict[str, object]:
        """Return engine-level statistics."""
        queue_stats = self._scheduler.get_queue_stats()
        return {
            "total_requests": self._total_requests,
            "total_tokens_generated": self._total_tokens_generated,
            "model": self.config.model.model_path,
            "device": self._device,
            "quantization": self.config.model.quantization or "none",
            "scheduler": queue_stats,
            "block_manager": self.block_manager.get_usage_stats(),
            "session_cache": self.session_cache.get_stats(),
        }

    @property
    def is_ready(self) -> bool:
        """Check if the engine is initialized and ready for requests."""
        return self._ready_event.is_set()
