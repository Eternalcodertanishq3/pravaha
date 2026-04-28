"""Event Bus — Decoupled publish/subscribe for engine ↔ TUI events.

Enhanced v3.3 event bus that supports BOTH:
- Structured EventType-based events (backward compatible)
- String-based event types (for TUI/swarm flexibility)

All real-time data flowing between components goes through this
bus. Fully decouples inference engine, swarm agents, and TUI.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class BusEvent:
    """A unified event — works with string or enum types."""

    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    request_id: str | None = None

    def __str__(self) -> str:
        parts = [f"[{self.event_type}]"]
        if self.request_id:
            parts.append(f"req={self.request_id[:8]}")
        if self.data:
            parts.append(str(self.data))
        return " ".join(parts)


class EnhancedEventBus:
    """Thread-safe publish/subscribe event bus.

    Producers (engine, agents, scheduler) call publish().
    Consumers (TUI panels, logger, prometheus) subscribe().

    Supports:
    - Per-type subscriptions: subscribe("agent_started", cb)
    - Wildcard subscriptions: subscribe_all(cb)
    - Event history with configurable depth
    - Per-type event counters
    - Events-per-second calculation over sliding window
    """

    def __init__(self, max_history: int = 2000) -> None:
        self._subscribers: dict[str, list[Callable]] = {}
        self._wildcard_subscribers: list[Callable] = []
        self._history: deque[BusEvent] = deque(maxlen=max_history)
        self._total_events: int = 0
        self._event_counts: dict[str, int] = {}

    # ── Subscribe ──────────────────────────────────────────────

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable) -> None:
        """Subscribe to ALL events (wildcard)."""
        self._wildcard_subscribers.append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Remove a specific subscriber."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                s for s in self._subscribers[event_type] if s != callback
            ]

    def unsubscribe_all(self, callback: Callable) -> None:
        """Remove a wildcard subscriber."""
        self._wildcard_subscribers = [
            s for s in self._wildcard_subscribers if s != callback
        ]

    # ── Publish ────────────────────────────────────────────────

    def publish(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        """Publish an event to all matching subscribers."""
        event = BusEvent(
            event_type=event_type,
            data=data or {},
            request_id=request_id,
        )
        self._history.append(event)
        self._total_events += 1
        self._event_counts[event_type] = (
            self._event_counts.get(event_type, 0) + 1
        )

        # Notify type-specific subscribers
        for cb in self._subscribers.get(event_type, []):
            try:
                cb(event)
            except Exception as e:
                logger.warning(f"EventBus subscriber error ({event_type}): {e}")

        # Notify wildcard subscribers
        for cb in self._wildcard_subscribers:
            try:
                cb(event)
            except Exception as e:
                logger.warning(f"EventBus wildcard subscriber error: {e}")

    # ── Backward compatibility: accept EngineEvent objects ─────

    def publish_legacy(self, engine_event: Any) -> None:
        """Publish an old-style EngineEvent from pravaha.engine.events."""
        event_type_enum = getattr(engine_event, "event_type", None)
        name = event_type_enum.name if event_type_enum else "UNKNOWN"
        self.publish(
            event_type=name,
            data=getattr(engine_event, "data", {}),
            request_id=getattr(engine_event, "request_id", None),
        )

    # ── Query ──────────────────────────────────────────────────

    def get_events_per_second(self, window_s: float = 5.0) -> float:
        """Calculate events/second over a sliding window."""
        cutoff = time.time() - window_s
        recent = sum(1 for e in self._history if e.timestamp > cutoff)
        return recent / window_s if window_s > 0 else 0.0

    def get_recent(
        self,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[BusEvent]:
        """Get recent events, optionally filtered by type."""
        events = list(self._history)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return list(reversed(events[-limit:]))

    def get_stats(self) -> dict[str, Any]:
        """Return bus statistics."""
        return {
            "total_events": self._total_events,
            "event_types": dict(self._event_counts),
            "events_per_second": round(self.get_events_per_second(), 1),
            "subscribers": {
                k: len(v) for k, v in self._subscribers.items()
            },
            "wildcard_subscribers": len(self._wildcard_subscribers),
            "history_size": len(self._history),
        }


# ── Global singleton ──────────────────────────────────────────────

_bus: EnhancedEventBus | None = None


def get_event_bus() -> EnhancedEventBus:
    """Get or create the global event bus singleton."""
    global _bus
    if _bus is None:
        _bus = EnhancedEventBus()
    return _bus
