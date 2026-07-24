"""Circuit Breaker — Fault isolation pattern for external dependencies and agent tools.

Prevents cascading failures by tripping open when failure thresholds are exceeded.
States:
- CLOSED: Normal operation. Errors are counted.
- OPEN: Tripped. Rejects calls immediately with CircuitBreakerOpenError.
- HALF_OPEN: Trial period. Allows limited probe calls to test if service has recovered.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when a call is made while the circuit breaker is in OPEN state."""

    pass


class CircuitBreaker:
    """Thread-safe circuit breaker with async and sync support."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        half_open_success_threshold: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.half_open_success_threshold = half_open_success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_state_change = time.time()
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Internal state transition check (must be called with lock held)."""
        now = time.time()
        if self._state == CircuitState.OPEN:
            if now - self._last_state_change >= self.recovery_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._last_state_change = now
                self._success_count = 0
                logger.info(f"CircuitBreaker '{self.name}': OPEN -> HALF_OPEN (probing recovery).")

    def record_success(self) -> None:
        """Record a successful invocation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._last_state_change = time.time()
                    logger.info(f"CircuitBreaker '{self.name}': HALF_OPEN -> CLOSED (service recovered).")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self, error: Exception) -> None:
        """Record a failed invocation."""
        with self._lock:
            self._failure_count += 1
            logger.warning(
                f"CircuitBreaker '{self.name}': recorded failure #{self._failure_count} "
                f"({type(error).__name__}: {error})"
            )
            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._last_state_change = time.time()
                logger.error(f"CircuitBreaker '{self.name}': CLOSED -> OPEN (threshold reached).")
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._last_state_change = time.time()
                logger.error(f"CircuitBreaker '{self.name}': HALF_OPEN -> OPEN (probe failed).")

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Synchronously execute fn wrapped in the circuit breaker."""
        with self._lock:
            self._check_state_transition()
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"CircuitBreaker '{self.name}' is OPEN. Call rejected."
                )

        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise

    async def call_async(self, coro_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Asynchronously execute coro_fn wrapped in the circuit breaker."""
        with self._lock:
            self._check_state_transition()
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"CircuitBreaker '{self.name}' is OPEN. Call rejected."
                )

        try:
            result = await coro_fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise
