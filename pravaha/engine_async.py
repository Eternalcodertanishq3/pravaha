"""Asynchronous Inference Engine for Pravaha.

Wraps the core DecoderEngine and ContinuousScheduler in an asyncio event loop,
allowing multiple concurrent clients to submit requests and stream responses
without blocking.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import AsyncGenerator, Optional

from pravaha.config import EngineConfig
from pravaha.decoder.decoder import DecoderEngine
from pravaha.decoder.sampling import Sampler, SamplingParams
from pravaha.kv_cache.paged_cache import PagedKVCache
from pravaha.models.loader import ModelLoader
from pravaha.scheduler.request import InferenceRequest, FinishReason
from pravaha.scheduler.scheduler import ContinuousScheduler
from pravaha.tokenizer.tokenizer import PravahaTokenizer

logger = logging.getLogger(__name__)


class AsyncPravahaEngine:
    """Async API for continuous batching (Phase 3)."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize the async engine."""
        self.config = EngineConfig.from_yaml(config_path)
        self._device = self.config.model.resolved_device

        logger.info(f"Initializing AsyncPravahaEngine with model {self.config.model.model_path}")

        # 1. Load Model
        loader = ModelLoader()
        self.model, self.arch_config = loader.load_model(
            model_path=self.config.model.model_path,
            dtype=self.config.model.torch_dtype,
            device=self._device,
            quantization=self.config.model.quantization,
        )

        # 2. Tokenizer
        self.tokenizer = PravahaTokenizer(self.config.model.model_path)

        # 3. Cache & Scheduler Setup (Phase 4 requires PagedKVCache)
        if self.config.cache.use_naive_cache:
            logger.info("Switching to PagedKVCache for Phase 4.")
            self.config.cache.use_naive_cache = False

        self.max_batch_size = 4
        self.block_size = 16
        # Balanced allocation for stability
        self.kv_cache = PagedKVCache.from_model_config(
            arch_config=self.arch_config,
            num_blocks=64,
            block_size=self.block_size,
            dtype=self.config.model.torch_dtype,
            device=self._device,
        )
        
        # 4. Scheduler
        self.scheduler = ContinuousScheduler(
            num_blocks=64,
            block_size=self.block_size,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.config.model.max_seq_len,
        )

        # 5. Decoder
        self.sampler = Sampler()
        self.decoder = DecoderEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            sampler=self.sampler,
            device=self._device,
            kv_cache=self.kv_cache,
        )

        # Background Loop State
        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Communication: request_id -> asyncio.Queue for streaming tokens back to client
        self._output_queues: dict[str, asyncio.Queue] = {}
        # Ensure thread-safe access to scheduler queues API
        self._lock = threading.Lock()
        
        # We need the event loop that the generator will run in
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None 

        # Telemetry counters
        self._total_requests_served: int = 0
        self._total_tokens_generated: int = 0
        self._total_ttft_ms: float = 0.0  # Cumulative time-to-first-token
        self._ttft_count: int = 0

    def start_background_loop(self):
        """Start the continuous batching scheduler in a background thread."""
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
            
        try:
            self._async_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            logger.warning("Created new asyncio loop for AsyncPravahaEngine.")

        self._stop_event.clear()
        self._loop_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._loop_thread.start()
        logger.info("Background scheduler thread started.")

    def stop_background_loop(self):
        """Stop the background scheduler."""
        self._stop_event.set()
        if self._loop_thread:
            self._loop_thread.join()
        logger.info("Background scheduler thread stopped.")

    async def generate(self, prompt: str, params: Optional[SamplingParams] = None) -> AsyncGenerator[str, None]:
        """Async generator API for a single request.
        
        Yields tokens as they are produced by the background scheduler.
        """
        if params is None:
            params = SamplingParams()

        request_id = str(uuid.uuid4())
        
        queue = asyncio.Queue()
        self._output_queues[request_id] = queue

        input_ids = self.tokenizer.encode(prompt)
        request = InferenceRequest(request_id, prompt, input_ids, params)

        with self._lock:
            self.scheduler.add_request(request)

        # Ensure the backend loop is running
        self.start_background_loop()

        try:
            while True:
                # Wait for next token from the background thread
                item = await queue.get()
                
                if isinstance(item, FinishReason):
                    break
                
                if isinstance(item, Exception):
                    raise item

                # Yield generated token text
                token_text = self.tokenizer.decode_token(item)
                yield token_text
                
        finally:
            # Cleanup output queue
            self._output_queues.pop(request_id, None)

    def _scheduler_loop(self):
        """The core synchronous loop running in a background thread."""
        while not self._stop_event.is_set():
            with self._lock:
                has_work = self.scheduler.has_unfinished_requests()
                
            if not has_work:
                # Sleep briefly if no work to avoid pegging CPU
                import time
                time.sleep(0.01)
                continue

            try:
                self._engine_step()
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                # Avoid tight crashing loops
                import time
                time.sleep(1)

    def _engine_step(self):
        """Perform one step (prefill OR decode) using the continuous scheduler."""
        with self._lock:
            scheduled = self.scheduler.step()
            
            prefill_reqs = scheduled["prefill"]
            decode_reqs = scheduled["decode"]
            swap_out_reqs = scheduled["swap_out"]
            swap_in_reqs = scheduled["swap_in"]

        # 0. Handle Swapping
        if swap_out_reqs:
            block_ids = []
            for req in swap_out_reqs:
                block_ids.extend(req.block_table)
            self.kv_cache.swap_out(block_ids)
            logger.info(f"Engine: Swapped out {len(swap_out_reqs)} requests ({len(block_ids)} blocks).")

        if swap_in_reqs:
            block_ids = []
            for req in swap_in_reqs:
                block_ids.extend(req.block_table)
            self.kv_cache.swap_in(block_ids)
            logger.info(f"Engine: Swapped in {len(swap_in_reqs)} requests ({len(block_ids)} blocks).")

        # 1. Batched Prefill Phase
        if prefill_reqs:
            input_ids_list = [req.prompt_token_ids for req in prefill_reqs]
            request_ids = [req.request_id for req in prefill_reqs]
            block_tables = [req.block_table for req in prefill_reqs]
            
            # Run model prefill
            try:
                next_tokens = self.decoder.step_prefill(input_ids_list, request_ids, block_tables)
            except Exception as e:
                logger.error(f"Prefill failed: {e}", exc_info=True)
                for req in prefill_reqs:
                    self._abort_request(req, e)
                return

            # Route outputs
            # Note: We use the lock for marking finished but not for process_token unless needed
            for i, req in enumerate(prefill_reqs):
                self._process_token(req, next_tokens[i])
            return # Disjoint phase

        # 2. Batched Decode Phase
        if decode_reqs:
            last_tokens = [req.get_last_token_id() for req in decode_reqs]
            request_ids = [req.request_id for req in decode_reqs]
            block_tables = [req.block_table for req in decode_reqs]
            context_lens = [req.total_tokens for req in decode_reqs]
            
            # Run model decode
            try:
                next_tokens = self.decoder.step_decode(
                    last_tokens, request_ids, block_tables, context_lens
                )
            except Exception as e:
                logger.error(f"Decode failed: {e}", exc_info=True)
                for req in decode_reqs:
                    self._abort_request(req, e)
                return

            # Route outputs
            for i, req in enumerate(decode_reqs):
                self._process_token(req, next_tokens[i])

    def _process_token(self, request: InferenceRequest, token_id: int):
        """Append a newly generated token and check stopping conditions."""
        request.generated_token_ids.append(token_id)
        self._total_tokens_generated += 1
        
        # Track TTFT on first generated token
        if len(request.generated_token_ids) == 1 and request.start_time is not None:
            ttft_ms = (time.time() - request.start_time) * 1000
            self._total_ttft_ms += ttft_ms
            self._ttft_count += 1
        
        # Enqueue token thread-safely
        queue = self._output_queues.get(request.request_id)
        if queue and self._async_loop:
            self._async_loop.call_soon_threadsafe(queue.put_nowait, token_id)

        # Check stopping criteria
        is_stop_token = token_id == self.tokenizer.eos_token_id or (request.sampling_params and token_id in request.sampling_params.stop_token_ids)
        is_max_length = len(request.generated_token_ids) >= (request.sampling_params.max_new_tokens if request.sampling_params else 100)
        
        if is_stop_token or is_max_length:
            if is_max_length:
                reason = FinishReason.MAX_TOKENS
            elif token_id == self.tokenizer.eos_token_id:
                reason = FinishReason.EOS
            else:
                reason = FinishReason.STOP_TOKEN
                
            with self._lock:
                request.mark_finished(reason)
                self._total_requests_served += 1
            
            if queue and self._async_loop:
                self._async_loop.call_soon_threadsafe(queue.put_nowait, reason)

    def _abort_request(self, request: InferenceRequest, error: Exception):
        with self._lock:
            request.mark_finished(FinishReason.ABORTED)
            self.scheduler.abort_request(request.request_id)
        
        queue = self._output_queues.get(request.request_id)
        if queue and self._async_loop:
            self._async_loop.call_soon_threadsafe(queue.put_nowait, error)

    def get_memory_stats(self) -> dict:
        """Return current GPU memory usage breakdown."""
        import torch
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

    def stop(self):
        """Alias for stop_background_loop."""
        self.stop_background_loop()

    def get_telemetry(self) -> dict:
        """Return live throughput and request telemetry."""
        avg_ttft_ms = 0.0
        if self._ttft_count > 0:
            avg_ttft_ms = self._total_ttft_ms / self._ttft_count

        tokens_per_sec = 0.0
        if self._total_tokens_generated > 0 and self._total_requests_served > 0:
            # Rough estimate based on total output
            tokens_per_sec = self._total_tokens_generated / max(1, self._total_requests_served)

        return {
            "active_requests": len(self._output_queues),
            "total_requests_served": self._total_requests_served,
            "total_tokens_generated": self._total_tokens_generated,
            "avg_time_to_first_token_ms": round(avg_ttft_ms, 2),
        }

    def get_model_info(self) -> dict:
        """Return loaded model metadata."""
        return {
            "model_path": self.config.model.model_path,
            "quantization": self.config.model.quantization or "none",
            "dtype": self.config.model.dtype,
            "device": self._device,
            "architecture": {
                "num_layers": self.arch_config.num_layers,
                "hidden_size": self.arch_config.hidden_size,
                "num_heads": self.arch_config.num_heads,
                "num_kv_heads": self.arch_config.num_kv_heads,
                "vocab_size": self.arch_config.vocab_size,
            },
        }


async def main():
    """Integration test for continuous batching."""
    import time
    
    # Default to INFO to keep logs clean. Change to DEBUG for deeper inspection.
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    engine = AsyncPravahaEngine(config_path="configs/default.yaml")
    
    prompts = [
        "The fastest mammal on land is",
        "Explain the theory of relativity in one sentence.",
        "A short poem about a robot:",
        "Write a python function to add two numbers."
    ]
    
    print(f"\nSubmitting {len(prompts)} concurrent requests for Continuous Batching...")
    print("-" * 50)
    
    # We will concurrently gather the outputs, appending them to a list for each slot
    outputs = ["" for _ in prompts]
    
    async def run_prompt(idx: int, prompt: str):
        # We start the requests with slight staggers to test disjoint queues
        await asyncio.sleep(idx * 0.1) 
        
        start_time = time.perf_counter()
        async for token in engine.generate(prompt, SamplingParams(max_new_tokens=15, temperature=0.7)):
            outputs[idx] += token
            # Quick console viz
            print(f"[Slot {idx}]: {token}", end="", flush=True)
            
        end_time = time.perf_counter()
        print(f"\n[Slot {idx}] -> Finished in {end_time - start_time:.2f}s")
        
    t0 = time.perf_counter()
    
    # Run all 4 requests concurrently
    await asyncio.gather(*(run_prompt(i, p) for i, p in enumerate(prompts)))
    
    t1 = time.perf_counter()
    
    engine.stop_background_loop()
    
    print("\n" + "=" * 50)
    print("Final Outputs:")
    for i, out in enumerate(outputs):
        print(f"[{i}]: {prompts[i]}{out}")
    print(f"\nTotal Batching Time: {t1 - t0:.2f}s for 4 concurrent requests!")

if __name__ == "__main__":
    asyncio.run(main())
