"""Prometheus Metrics — Engine observability via Prometheus.

Exports request counts, latencies, token throughput, cache usage,
and queue depths as Prometheus metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metric definitions and helpers for the engine."""

    def __init__(self) -> None:
        try:
            from prometheus_client import Counter, Gauge, Histogram, Summary
            self.requests_total = Counter("pravaha_requests_total", "Total inference requests", ["model", "status"])
            self.tokens_generated = Counter("pravaha_tokens_generated_total", "Total tokens generated")
            self.ttft_seconds = Histogram("pravaha_ttft_seconds", "Time to first token", buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
            self.request_duration = Histogram("pravaha_request_duration_seconds", "Total request duration", buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0])
            self.tokens_per_second = Gauge("pravaha_tokens_per_second", "Current throughput")
            self.cache_usage_pct = Gauge("pravaha_cache_usage_pct", "KV cache usage percentage")
            self.queue_waiting = Gauge("pravaha_queue_waiting", "Requests in waiting queue")
            self.queue_running = Gauge("pravaha_queue_running", "Requests currently running")
            self.active_sessions = Gauge("pravaha_active_sessions", "Active conversation sessions")
            self.audit_iterations = Counter("pravaha_audit_iterations_total", "Total audit loop iterations")
            self._available = True
        except ImportError:
            logger.warning("prometheus_client not installed, metrics disabled")
            self._available = False

    def record_request(self, model: str, status: str, duration_s: float, ttft_s: float, tokens: int) -> None:
        if not self._available:
            return
        self.requests_total.labels(model=model, status=status).inc()
        self.request_duration.observe(duration_s)
        self.ttft_seconds.observe(ttft_s)
        self.tokens_generated.inc(tokens)

    def update_gauges(self, tps: float, cache_pct: float, waiting: int, running: int, sessions: int) -> None:
        if not self._available:
            return
        self.tokens_per_second.set(tps)
        self.cache_usage_pct.set(cache_pct)
        self.queue_waiting.set(waiting)
        self.queue_running.set(running)
        self.active_sessions.set(sessions)
