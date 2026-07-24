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
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any

import psutil

from pravaha.config.engine_config import EngineConfig
from pravaha.decoder.decoder import DecoderEngine
from pravaha.decoder.sampling import Sampler, SamplingParams
from pravaha.engine.event_bus import get_event_bus
from pravaha.engine.events import EngineEvent, EventBus, EventType
from pravaha.engine.load_balancer import AdaptiveLoadBalancer
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

        # Event bus for telemetry (legacy + enhanced)
        self.event_bus = EventBus()
        self._enhanced_bus = get_event_bus()

        # ── Fix 1: Threading synchronization ──
        self._ready_event = threading.Event()
        self._shutdown_event = threading.Event()

        # Request tracking
        self._request_futures: dict[str, asyncio.Future[list[int]]] = {}
        self._request_queues: dict[str, asyncio.Queue[str]] = {}
        self._active_requests: dict[str, InferenceRequest] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

        # ── v3.3: Real metrics tracking ──
        self._token_timestamps: deque[float] = deque(maxlen=200)
        self._current_tps: float = 0.0
        self._total_requests_served: int = 0
        self._total_tokens_generated_v33: int = 0
        self._ttft_history: deque[float] = deque(maxlen=50)
        self._tui_callback: Any = None
        self._start_time: float = time.time()

        # ── v3.3: Adaptive load balancer ──
        self._load_balancer = AdaptiveLoadBalancer()
        self._load_balancer.register_callback(self._on_load_change)

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

        # Start load balancer monitoring
        self._load_balancer.start()

        # Signal that initialization is complete
        self._ready_event.set()

        logger.info("AsyncPravahaEngine initialized and ready.")

    def _initialize_subsystems(self) -> None:
        """Load model and initialize all engine subsystems via EngineFactory."""
        t0 = time.time()
        from pravaha.engine.factory import EngineFactory

        subs = EngineFactory.build_subsystems(self.config, self._device)
        self.tokenizer = subs["tokenizer"]
        self.model = subs["model"]
        self.arch_config = subs["arch_config"]
        self.block_manager = subs["block_manager"]
        self.kv_cache = subs["kv_cache"]
        self._decoder = subs["decoder"]
        self._scheduler = subs["scheduler"]
        self.session_cache = subs["session_cache"]

        # GPU Warmup
        self._warmup_gpu()

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
        self._enhanced_bus.publish("model_loaded", {
            "model": self.config.model.model_path,
            "load_time_s": round(elapsed, 2),
            "device": self._device,
        })
        logger.info(f"All subsystems initialized in {elapsed:.1f}s")

    def _warmup_gpu(self) -> None:
        """Execute dummy forward passes to warm up CUDA kernels and memory pools."""
        try:
            logger.info("Executing GPU warmup passes...")
            dummy_prompt = "Pravaha GPU warmup test sequence."
            dummy_ids = self.tokenizer.encode(dummy_prompt)
            if not dummy_ids:
                dummy_ids = [1, 2, 3]

            block_tables = [[0]]
            # Warmup prefill step
            self._decoder.step_prefill([dummy_ids], ["warmup_req"], block_tables)
            # Warmup decode step
            self._decoder.step_decode([dummy_ids[-1]], ["warmup_req"], block_tables, [len(dummy_ids)])
            logger.info("GPU warmup completed successfully.")
        except Exception as e:
            logger.warning(f"GPU warmup skipped or non-fatal: {e}")

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

        # Check backpressure / overload
        if self._scheduler.is_overloaded(threshold_pct=0.95):
            raise RuntimeError("Server overloaded: queue/memory backpressure threshold reached.")

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)

        # Create inference request
        request = InferenceRequest(
            request_id=request_id,
            prompt_token_ids=input_ids,
            sampling_params=params,
        )

        # Set up token streaming queue (bounded to prevent OOM)
        token_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        self._request_queues[request_id] = token_queue
        self._active_requests[request_id] = request

        # Capture the current event loop for cross-thread communication
        self._loop = asyncio.get_running_loop()

        # Submit to scheduler with capacity check
        if not self._scheduler.add_request(request):
            self._request_queues.pop(request_id, None)
            self._active_requests.pop(request_id, None)
            raise RuntimeError("Scheduler waiting queue is full. Request rejected.")

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
            # Cleanup & abort request if client disconnected or generator exited early
            self.abort_request(request_id)
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
                # Don't send <|ERROR|> directly to the user output if we can avoid it.
                # Send an explicit aborted signal instead.
                self._send_token(req.request_id, "<|ERROR: Request Aborted|>")

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
                self._send_token(req.request_id, "<|ERROR: Request Aborted|>")

    def _send_token(self, request_id: str, text: str) -> None:
        """Thread-safe: push a token to the request's async queue."""
        queue = self._request_queues.get(request_id)
        if queue is not None and self._loop is not None:
            def _safe_put():
                try:
                    queue.put_nowait(text)
                except asyncio.QueueFull:
                    logger.warning(f"Token queue full for request {request_id}. Dropping token.")
            self._loop.call_soon_threadsafe(_safe_put)

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
        self._load_balancer.stop()
        if self._bg_thread.is_alive():
            self._bg_thread.join(timeout=5.0)
        logger.info("Engine stopped.")

    # ── v3.3: TUI callback + real-time metrics ─────────────────

    def set_tui_callback(self, cb: Any) -> None:
        """Set callback for TUI event forwarding."""
        self._tui_callback = cb

    def _record_token(self) -> None:
        """Record a token generation timestamp for TPS calculation."""
        now = time.time()
        self._token_timestamps.append(now)
        self._total_tokens_generated_v33 += 1

        # Calculate real TPS from sliding window
        if len(self._token_timestamps) >= 2:
            window = now - self._token_timestamps[0]
            if window > 0:
                self._current_tps = len(self._token_timestamps) / window

    def _on_load_change(self, snapshot: Any) -> None:
        """React to load balancer changes."""
        logger.info(
            f"Load shift: {snapshot.reason} "
            f"CPU={snapshot.cpu_pct:.0f}% "
            f"RAM={snapshot.ram_pct:.0f}% "
            f"GPU={snapshot.gpu_pct:.0f}%"
        )
        self._enhanced_bus.publish("load_change", snapshot.to_dict())
        if self._tui_callback:
            self._tui_callback("load_change", snapshot.to_dict())

    def get_stats(self) -> dict[str, object]:
        """Return REAL engine statistics. No hardcoded values."""
        queue_stats = self._scheduler.get_queue_stats()
        block_stats = self.block_manager.get_usage_stats()

        # Real hardware stats via psutil
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        gpu_info: dict[str, Any] = {"available": False}
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gpu_info = {
                    "available": True,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1e9, 2),
                    "allocated_gb": round(
                        torch.cuda.memory_allocated(0) / 1e9, 2
                    ),
                    "reserved_gb": round(
                        torch.cuda.memory_reserved(0) / 1e9, 2
                    ),
                }
        except Exception:
            pass

        # TTFT p50
        ttft_p50 = 0.0
        if self._ttft_history:
            sorted_h = sorted(self._ttft_history)
            ttft_p50 = sorted_h[len(sorted_h) // 2]

        # Uptime
        uptime_s = time.time() - self._start_time

        return {
            "model": self.config.model.model_path,
            "device": self._device,
            "quantization": self.config.model.quantization or "none",
            "is_ready": self.is_ready,
            "tokens_per_second": round(self._current_tps, 1),
            "ttft_p50_ms": round(ttft_p50, 1),
            "total_requests": self._total_requests,
            "total_tokens_generated": self._total_tokens_generated,
            "uptime_s": round(uptime_s, 1),
            "scheduler": queue_stats,
            "block_manager": block_stats,
            "session_cache": self.session_cache.get_stats(),
            "hardware": {
                "cpu_pct": cpu,
                "ram_used_gb": round(ram.used / 1e9, 2),
                "ram_total_gb": round(ram.total / 1e9, 2),
                "ram_pct": ram.percent,
            },
            "gpu": gpu_info,
            "load_balancer": self._load_balancer.get_stats(),
        }

    @property
    def is_ready(self) -> bool:
        """Check if the engine is initialized and ready for requests."""
        return self._ready_event.is_set()
