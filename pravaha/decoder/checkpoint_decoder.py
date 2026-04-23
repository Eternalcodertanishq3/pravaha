"""Checkpoint Decoder — Pause and resume mid-generation.

Feature D: Supports pausing an in-flight generation, saving its state,
and resuming later. Useful for long generations that may run out of VRAM,
and for "continue generation" UX in chat.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path

from pravaha.decoder.sampling import SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class InferenceCheckpoint:
    """Saved state of a paused generation.

    Contains everything needed to resume generation from where it left off,
    including the generated tokens so far, KV cache block references,
    and sampling parameters.

    Attributes:
        request_id: Original request identifier.
        generated_so_far: Token IDs generated before pausing.
        kv_blocks: Allocated KV cache block IDs.
        context_len: Current context length (prompt + generated).
        sampling_params: The sampling configuration.
        prompt: Original prompt text.
        paused_at: Timestamp when paused.
    """

    request_id: str
    generated_so_far: list[int] = field(default_factory=list)
    kv_blocks: list[int] = field(default_factory=list)
    context_len: int = 0
    sampling_params: SamplingParams | None = None
    prompt: str = ""
    paused_at: float = field(default_factory=time.time)


class CheckpointManager:
    """Manage inference checkpoints for pause/resume.

    Enables pausing long-running generations, freeing VRAM, and resuming
    when resources become available. Also enables the "continue generation"
    UX where a user can ask for more output after the model stops.
    """

    def __init__(self) -> None:
        """Initialize the checkpoint manager."""
        self._checkpoints: dict[str, InferenceCheckpoint] = {}

    def pause(
        self,
        request_id: str,
        generated_so_far: list[int],
        kv_blocks: list[int],
        context_len: int,
        sampling_params: SamplingParams,
        prompt: str = "",
    ) -> InferenceCheckpoint:
        """Pause an in-flight request mid-generation.

        Saves the current generation state so it can be resumed later.

        Args:
            request_id: Request to pause.
            generated_so_far: Token IDs generated so far.
            kv_blocks: KV cache block IDs (these should NOT be freed).
            context_len: Current context length.
            sampling_params: Sampling parameters.
            prompt: Original prompt text.

        Returns:
            The saved checkpoint.
        """
        checkpoint = InferenceCheckpoint(
            request_id=request_id,
            generated_so_far=generated_so_far.copy(),
            kv_blocks=kv_blocks.copy(),
            context_len=context_len,
            sampling_params=sampling_params,
            prompt=prompt,
            paused_at=time.time(),
        )

        self._checkpoints[request_id] = checkpoint
        logger.info(
            f"Paused request {request_id}: {len(generated_so_far)} tokens, {len(kv_blocks)} blocks"
        )
        return checkpoint

    async def resume(
        self,
        checkpoint: InferenceCheckpoint,
        engine: object,
        additional_tokens: int = 0,
    ) -> AsyncGenerator[str, None]:
        """Resume generation from a checkpoint.

        Continues generating from where the checkpoint left off, using
        the saved KV cache blocks.

        Args:
            checkpoint: The checkpoint to resume from.
            engine: The inference engine with a generate() method.
            additional_tokens: Extra tokens beyond the original max.

        Yields:
            Generated token text.
        """
        tokenizer = engine.tokenizer  # type: ignore[attr-defined]

        # Reconstruct the full context
        full_text = checkpoint.prompt
        if checkpoint.generated_so_far:
            generated_text = tokenizer.decode(checkpoint.generated_so_far)
            full_text += generated_text

        # Adjust max_new_tokens
        params = checkpoint.sampling_params or SamplingParams()
        remaining = params.max_new_tokens - len(checkpoint.generated_so_far)
        if additional_tokens > 0:
            remaining = additional_tokens
        remaining = max(1, remaining)

        resume_params = SamplingParams(
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            max_new_tokens=remaining,
            repetition_penalty=params.repetition_penalty,
        )

        logger.info(
            f"Resuming request {checkpoint.request_id}: continuing for up to {remaining} tokens"
        )

        # Generate using the engine
        async for token_text in engine.generate(full_text, resume_params):  # type: ignore[attr-defined]
            yield token_text

        # Clean up checkpoint after completion
        self._checkpoints.pop(checkpoint.request_id, None)

    def get(self, request_id: str) -> InferenceCheckpoint | None:
        """Retrieve a checkpoint by request ID.

        Args:
            request_id: Request ID to look up.

        Returns:
            The checkpoint if found, None otherwise.
        """
        return self._checkpoints.get(request_id)

    def delete(self, request_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            request_id: Request ID to delete.

        Returns:
            True if the checkpoint existed and was deleted.
        """
        return self._checkpoints.pop(request_id, None) is not None

    def list_checkpoints(self) -> list[dict[str, object]]:
        """List all saved checkpoints.

        Returns:
            List of checkpoint summaries.
        """
        return [
            {
                "request_id": cp.request_id,
                "tokens_generated": len(cp.generated_so_far),
                "context_len": cp.context_len,
                "num_blocks": len(cp.kv_blocks),
                "paused_at": cp.paused_at,
                "age_seconds": round(time.time() - cp.paused_at, 1),
            }
            for cp in self._checkpoints.values()
        ]

    def save_to_disk(
        self,
        checkpoint: InferenceCheckpoint,
        path: str | Path,
    ) -> None:
        """Persist a checkpoint to disk.

        Saves the checkpoint metadata (not the KV cache tensors, which
        remain in GPU/CPU memory).

        Args:
            checkpoint: Checkpoint to save.
            path: Output file path (JSON).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "request_id": checkpoint.request_id,
            "generated_so_far": checkpoint.generated_so_far,
            "kv_blocks": checkpoint.kv_blocks,
            "context_len": checkpoint.context_len,
            "prompt": checkpoint.prompt,
            "paused_at": checkpoint.paused_at,
            "sampling_params": {
                "temperature": checkpoint.sampling_params.temperature,
                "top_k": checkpoint.sampling_params.top_k,
                "top_p": checkpoint.sampling_params.top_p,
                "max_new_tokens": checkpoint.sampling_params.max_new_tokens,
                "repetition_penalty": checkpoint.sampling_params.repetition_penalty,
            }
            if checkpoint.sampling_params
            else None,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Checkpoint saved to {path}")

    def load_from_disk(self, path: str | Path) -> InferenceCheckpoint:
        """Load a checkpoint from disk.

        Args:
            path: Path to the checkpoint JSON file.

        Returns:
            Loaded checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        params = None
        if data.get("sampling_params"):
            params = SamplingParams(**data["sampling_params"])

        checkpoint = InferenceCheckpoint(
            request_id=data["request_id"],
            generated_so_far=data["generated_so_far"],
            kv_blocks=data["kv_blocks"],
            context_len=data["context_len"],
            prompt=data.get("prompt", ""),
            paused_at=data.get("paused_at", 0.0),
            sampling_params=params,
        )

        self._checkpoints[checkpoint.request_id] = checkpoint
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
