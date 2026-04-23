"""Engine Events — Telemetry dataclasses for the inference pipeline.

Structured event types emitted by the engine for observability,
debugging, and audit tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """Types of engine events."""

    REQUEST_RECEIVED = auto()
    PREFILL_START = auto()
    PREFILL_END = auto()
    DECODE_START = auto()
    DECODE_END = auto()
    TOKEN_GENERATED = auto()
    REQUEST_COMPLETE = auto()
    REQUEST_ABORTED = auto()
    SWAP_OUT = auto()
    SWAP_IN = auto()
    CHECKPOINT_SAVED = auto()
    CHECKPOINT_RESUMED = auto()
    MODEL_LOADED = auto()
    LORA_LOADED = auto()
    CONFIG_RELOADED = auto()
    AUDIT_START = auto()
    AUDIT_ISSUE_FOUND = auto()
    AUDIT_PATCH_APPLIED = auto()
    AUDIT_COMPLETE = auto()
    RAG_QUERY = auto()
    RAG_INGEST = auto()
    BRANCH_CREATED = auto()
    ERROR = auto()


@dataclass
class EngineEvent:
    """A single engine telemetry event.

    Attributes:
        event_type: Type of event.
        timestamp: Unix timestamp when the event occurred.
        request_id: Associated request ID (if applicable).
        data: Event-specific payload.
        duration_ms: Duration of the operation in milliseconds.
    """

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    request_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def __str__(self) -> str:
        parts = [f"[{self.event_type.name}]"]
        if self.request_id:
            parts.append(f"req={self.request_id[:8]}")
        if self.duration_ms is not None:
            parts.append(f"{self.duration_ms:.1f}ms")
        if self.data:
            parts.append(str(self.data))
        return " ".join(parts)


@dataclass
class TokenEvent:
    """Per-token generation event for debugging.

    Attributes:
        position: Token position in the sequence.
        token_id: Generated token ID.
        token_text: Decoded token text.
        logprob: Log probability of the chosen token.
        top_logits: Top-k logits at this position.
        sampling_method: How the token was selected.
    """

    position: int
    token_id: int
    token_text: str = ""
    logprob: float = 0.0
    top_logits: list[tuple[int, float]] = field(default_factory=list)
    sampling_method: str = "multinomial"


@dataclass
class RequestMetrics:
    """Aggregated metrics for a completed request.

    Attributes:
        request_id: Request identifier.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of generated tokens.
        total_tokens: Total tokens processed.
        ttft_ms: Time to first token in milliseconds.
        total_duration_ms: Total request duration.
        tokens_per_second: Generation throughput.
        model_name: Model used for this request.
        was_cached: Whether the response came from cache.
        was_audited: Whether the response went through the audit loop.
        audit_iterations: Number of audit iterations performed.
        cost_usd: Estimated cost in USD.
    """

    request_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0.0
    total_duration_ms: float = 0.0
    tokens_per_second: float = 0.0
    model_name: str = ""
    was_cached: bool = False
    was_audited: bool = False
    audit_iterations: int = 0
    cost_usd: float = 0.0


class EventBus:
    """Simple in-process event bus for engine telemetry.

    Allows components to publish events and subscribers to receive them.
    Used by the TUI, Prometheus exporter, and debug tools.
    """

    def __init__(self, max_history: int = 10000) -> None:
        """Initialize the event bus.

        Args:
            max_history: Maximum events to keep in history.
        """
        self._subscribers: list[Any] = []
        self._history: list[EngineEvent] = []
        self._max_history = max_history

    def subscribe(self, callback: Any) -> None:
        """Register an event subscriber.

        Args:
            callback: Callable(EngineEvent) to invoke on each event.
        """
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Any) -> None:
        """Remove an event subscriber."""
        self._subscribers = [s for s in self._subscribers if s != callback]

    def publish(self, event: EngineEvent) -> None:
        """Publish an event to all subscribers.

        Args:
            event: The event to publish.
        """
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:
                pass  # Don't let subscriber errors affect the engine

    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[EngineEvent]:
        """Get recent events, optionally filtered by type.

        Args:
            event_type: Filter to this event type. None = all types.
            limit: Maximum events to return.

        Returns:
            List of matching events (newest first).
        """
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return list(reversed(events[-limit:]))
