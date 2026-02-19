"""Inference request and sequence data structures.

Defines the core request types used by the scheduler to track sequences
through their lifecycle: WAITING → RUNNING → FINISHED (or SWAPPED).

Phase 1: Used as data containers. Phase 3+ will integrate with the
continuous batching scheduler.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pravaha.decoder.sampling import SamplingParams


class SequenceStatus(Enum):
    """Lifecycle status of an inference request."""

    WAITING = "waiting"      # Queued, not yet scheduled
    RUNNING = "running"      # Actively generating tokens
    FINISHED = "finished"    # Generation complete (EOS or max tokens)
    SWAPPED = "swapped"      # KV-cache swapped to CPU (preempted)


class FinishReason(Enum):
    """Reason why generation stopped."""

    EOS = "eos"              # Hit end-of-sequence token
    MAX_TOKENS = "max_tokens"  # Reached max_new_tokens limit
    STOP_TOKEN = "stop_token"  # Hit a custom stop token
    ABORTED = "aborted"      # Client disconnected or request cancelled


@dataclass
class InferenceRequest:
    """Represents a single inference request through its entire lifecycle.

    Tracks prompt, generated tokens, scheduling status, timing, and
    KV-cache block assignments (for paged attention in Phase 4+).

    Attributes:
        request_id: Unique identifier for this request.
        prompt: Original text prompt.
        prompt_token_ids: Tokenized prompt as a list of token IDs.
        sampling_params: Sampling configuration for this request.
        status: Current lifecycle status.
        generated_token_ids: Tokens generated so far.
        arrival_time: When the request was received.
        start_time: When generation started (first scheduled).
        finish_time: When generation completed.
        finish_reason: Why generation stopped.
        block_table: List of physical block IDs for KV-cache (Phase 4).
        num_computed_tokens: Number of tokens whose KV-cache has been computed.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    status: SequenceStatus = SequenceStatus.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    finish_reason: Optional[FinishReason] = None
    # Phase 4: Paged KV-cache block assignments
    block_table: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        """Number of tokens in the prompt."""
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        """Number of tokens generated so far."""
        return len(self.generated_token_ids)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens (prompt + generated)."""
        return self.num_prompt_tokens + self.num_generated_tokens

    @property
    def is_finished(self) -> bool:
        """Whether this request has completed generation."""
        return self.status == SequenceStatus.FINISHED

    @property
    def is_running(self) -> bool:
        """Whether this request is actively generating."""
        return self.status == SequenceStatus.RUNNING

    @property
    def latency(self) -> Optional[float]:
        """Total end-to-end latency in seconds, if finished."""
        if self.finish_time is not None:
            return self.finish_time - self.arrival_time
        return None

    @property
    def time_to_first_token(self) -> Optional[float]:
        """Time from arrival to first generated token, if started."""
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return None

    def mark_running(self) -> None:
        """Transition to RUNNING status."""
        self.status = SequenceStatus.RUNNING
        if self.start_time is None:
            self.start_time = time.time()

    def mark_finished(self, reason: FinishReason) -> None:
        """Transition to FINISHED status."""
        self.status = SequenceStatus.FINISHED
        self.finish_time = time.time()
        self.finish_reason = reason

    def mark_swapped(self) -> None:
        """Transition to SWAPPED status (preempted, KV-cache on CPU)."""
        self.status = SequenceStatus.SWAPPED

    def add_token(self, token_id: int) -> None:
        """Append a generated token."""
        self.generated_token_ids.append(token_id)

    def get_last_token_id(self) -> int:
        """Get the most recently generated token, or last prompt token."""
        if self.generated_token_ids:
            return self.generated_token_ids[-1]
        return self.prompt_token_ids[-1]

    def __repr__(self) -> str:
        return (
            f"InferenceRequest(id={self.request_id}, "
            f"status={self.status.value}, "
            f"prompt_tokens={self.num_prompt_tokens}, "
            f"generated={self.num_generated_tokens})"
        )
