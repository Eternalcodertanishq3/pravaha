from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CUDAGraphDecoderWrapper:
    """
    Wraps a DecoderEngine to capture and replay its decode phase using torch.cuda.CUDAGraph.
    This implementation handles automatic bucketing of batch sizes, padding, static buffer
    management, and VRAM accounting.
    """

    def __init__(
        self,
        decoder_engine: Any,
        buckets: list[int] | None = None,
        warmup_steps: int = 3,
        device: torch.device | None = None,
    ) -> None:
        self.decoder_engine = decoder_engine
        self.buckets = sorted(buckets) if buckets else [1, 2, 4, 8, 16, 32, 64]
        self.warmup_steps = warmup_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State for graph capture
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._warmup_counters: dict[int, int] = {b: 0 for b in self.buckets}

        # Static buffers
        self._static_inputs: dict[int, dict[str, torch.Tensor]] = {}
        self._static_outputs: dict[int, torch.Tensor] = {}
        self._memory_usage_bytes: dict[int, int] = {}

        # Tracking if we are currently capturing to prevent recursive captures
        self._is_capturing = False

        # Shared pool for all graphs to save memory
        self._mempool: tuple[int, int] | None = None
        if torch.cuda.is_available():
            try:
                self._mempool = torch.cuda.graph_pool_handle()
                logger.info("Initialized CUDA graph memory pool.")
            except AttributeError:
                logger.warning("torch.cuda.graph_pool_handle not found. Graphs will use independent memory pools.")

    def _get_bucket(self, batch_size: int) -> int | None:
        """
        Find the smallest bucket size >= batch_size.
        Returns None if batch_size exceeds the maximum bucket size.
        """
        for b in self.buckets:
            if batch_size <= b:
                return b
        return None

    def _allocate_static_buffers(self, bucket_size: int, example_inputs: dict[str, torch.Tensor]) -> None:
        """
        Allocate static pinned memory for graph inputs and outputs.
        CUDA graphs require fixed memory addresses. By pre-allocating these
        buffers, we can copy the actual inputs into them before replaying the graph.
        """
        if bucket_size in self._static_inputs:
            logger.debug(f"Static buffers for bucket {bucket_size} already allocated.")
            return

        logger.info(f"Allocating static buffers for bucket size {bucket_size}.")
        self._static_inputs[bucket_size] = {}

        for k, v in example_inputs.items():
            shape = list(v.shape)
            if shape and shape[0] < bucket_size:
                shape[0] = bucket_size

            # Create tensor with same dtype and shape but for the bucket size
            tensor = torch.zeros(shape, dtype=v.dtype, device=self.device)
            self._static_inputs[bucket_size][k] = tensor

    def verify_buffer_shapes(self, bucket_size: int, token_ids: torch.Tensor, block_tables: torch.Tensor, context_lens: torch.Tensor) -> None:
        """
        Sanity check to ensure runtime input shapes match the allocated static buffer shapes
        (ignoring the batch dimension which is padded).
        """
        static_inputs = self._static_inputs[bucket_size]

        if token_ids.shape[1:] != static_inputs["token_ids"].shape[1:]:
            raise ValueError(f"token_ids shape mismatch. Expected suffix {static_inputs['token_ids'].shape[1:]}, got {token_ids.shape[1:]}")
        if block_tables.shape[1:] != static_inputs["block_tables"].shape[1:]:
            raise ValueError(f"block_tables shape mismatch. Expected suffix {static_inputs['block_tables'].shape[1:]}, got {block_tables.shape[1:]}")
        if context_lens.shape[1:] != static_inputs["context_lens"].shape[1:]:
            raise ValueError(f"context_lens shape mismatch. Expected suffix {static_inputs['context_lens'].shape[1:]}, got {context_lens.shape[1:]}")

    def reset_graphs(self) -> None:
        """
        Clear all captured graphs and free their associated memory.
        Useful when the model structure changes or memory needs to be freed.
        """
        logger.info("Resetting all CUDA graphs and freeing memory.")
        self._graphs.clear()
        self._static_inputs.clear()
        self._static_outputs.clear()
        self._memory_usage_bytes.clear()
        self._warmup_counters = {b: 0 for b in self.buckets}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_budget(self) -> dict[int, int]:
        """
        Return the memory usage of each captured graph in bytes.
        """
        return self._memory_usage_bytes.copy()

    def _capture_graph(self, bucket_size: int, static_kwargs: dict[str, torch.Tensor]) -> None:
        """
        Capture the CUDA graph for a given bucket size.
        This records all CUDA kernel launches into a single graph which can be replayed efficiently.
        """
        logger.info(f"Starting CUDA graph capture for bucket {bucket_size}")
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Cannot capture graph.")
            return

        graph = torch.cuda.CUDAGraph()
        self._is_capturing = True

        try:
            # Measure VRAM before capture
            torch.cuda.synchronize(self.device)
            mem_before = torch.cuda.memory_allocated(self.device)

            # We must use torch.cuda.graph context manager
            pool: Any = self._mempool if isinstance(self._mempool, tuple) else None

            with torch.cuda.graph(graph, pool=pool):  # type: ignore[arg-type]
                # Pass static inputs to the model
                # Assuming decoder_engine has a model attribute that executes the forward pass
                model = getattr(self.decoder_engine, "model", self.decoder_engine)
                if callable(model):
                    self._static_outputs[bucket_size] = model(
                        static_kwargs["token_ids"],
                        static_kwargs["block_tables"],
                        static_kwargs["context_lens"],
                    )
                else:
                    raise RuntimeError("decoder_engine.model is not callable")

            # Measure VRAM after capture
            torch.cuda.synchronize(self.device)
            mem_after = torch.cuda.memory_allocated(self.device)

            self._graphs[bucket_size] = graph
            self._memory_usage_bytes[bucket_size] = mem_after - mem_before

            logger.info(
                f"Successfully captured graph for bucket {bucket_size}. "
                f"VRAM delta: {self._memory_usage_bytes[bucket_size]} bytes"
            )

        except Exception as e:
            logger.error(f"Failed to capture CUDA graph for bucket {bucket_size}: {e}")
            raise
        finally:
            self._is_capturing = False

    def step_decode_graphed(
        self,
        token_ids: torch.Tensor,
        request_ids: list[str],
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Drop-in replacement for step_decode in DecoderEngine.

        This method will:
        1. Determine the appropriate bucket size based on batch size.
        2. Fallback to eager execution if batch size > max bucket or CUDA is unavailable.
        3. Execute warmup steps using eager execution.
        4. Capture the CUDA graph on the first post-warmup execution.
        5. Replay the captured graph for subsequent executions.
        """
        batch_size = token_ids.size(0)
        bucket_size = self._get_bucket(batch_size)

        # Fallback to eager if bucket too large, CUDA unavailable, or already capturing
        if bucket_size is None or not torch.cuda.is_available() or self._is_capturing:
            logger.debug(f"Falling back to eager execution. Batch size: {batch_size}, Bucket: {bucket_size}")
            return self.decoder_engine.step_decode(token_ids, request_ids, block_tables, context_lens)

        # Ensure static buffers are allocated for this bucket
        if bucket_size not in self._static_inputs:
            example_inputs = {
                "token_ids": token_ids,
                "block_tables": block_tables,
                "context_lens": context_lens,
            }
            self._allocate_static_buffers(bucket_size, example_inputs)

        # Verify shapes match the static buffer allocations
        self.verify_buffer_shapes(bucket_size, token_ids, block_tables, context_lens)

        # Warmup phase: run eagerly for a few steps to ensure memory allocations are stable
        if self._warmup_counters[bucket_size] < self.warmup_steps:
            logger.debug(f"Warmup step {self._warmup_counters[bucket_size] + 1}/{self.warmup_steps} for bucket {bucket_size}")
            self._warmup_counters[bucket_size] += 1
            return self.decoder_engine.step_decode(token_ids, request_ids, block_tables, context_lens)

        # Graph capture phase: capture the graph if not already captured
        if bucket_size not in self._graphs:
            self._capture_graph(bucket_size, self._static_inputs[bucket_size])

        # Graph replay phase
        static_inputs = self._static_inputs[bucket_size]

        # Copy runtime data into static buffers (pad automatically by slice)
        # Using non_blocking=True can improve performance if inputs are on CPU pinned memory
        static_inputs["token_ids"][:batch_size].copy_(token_ids, non_blocking=True)
        static_inputs["block_tables"][:batch_size].copy_(block_tables, non_blocking=True)
        static_inputs["context_lens"][:batch_size].copy_(context_lens, non_blocking=True)

        # If the batch size is smaller than the bucket, zero out the remainder of the buffer
        # This prevents stale data from affecting the computation (though attention masks usually handle this)
        if batch_size < bucket_size:
            static_inputs["token_ids"][batch_size:].zero_()
            static_inputs["block_tables"][batch_size:].zero_()
            static_inputs["context_lens"][batch_size:].zero_()

        # Replay the captured graph
        logger.debug(f"Replaying CUDA graph for bucket {bucket_size} (batch size: {batch_size})")
        self._graphs[bucket_size].replay()

        # Return actual output size by slicing the static output tensor
        # clone() is required because the static buffer will be overwritten on the next step
        return self._static_outputs[bucket_size][:batch_size].clone()
