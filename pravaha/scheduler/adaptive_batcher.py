"""Adaptive Batcher — Dynamic batch size optimization.

Monitors GPU utilization and request arrival rate to dynamically adjust
batch sizes. Grows batches when GPU is underutilized, shrinks when
latency targets are missed.
"""

from __future__ import annotations

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class AdaptiveBatcher:
    """Dynamically adjust batch size based on system load.

    Algorithm:
    - Track recent latencies (sliding window)
    - If p95 latency < target: grow batch size
    - If p95 latency > target: shrink batch size
    - Clamp to [min_batch, max_batch] range
    """

    def __init__(
        self,
        min_batch: int = 1,
        max_batch: int = 64,
        target_latency_ms: float = 100.0,
        window_size: int = 50,
    ) -> None:
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_latency_ms = target_latency_ms
        self.current_batch_size = min_batch
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._last_adjust = time.time()

    def record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)

    def get_batch_size(self) -> int:
        if len(self._latencies) < 5:
            return self.current_batch_size

        now = time.time()
        if now - self._last_adjust < 1.0:
            return self.current_batch_size

        self._last_adjust = now
        sorted_lat = sorted(self._latencies)
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]

        if p95 < self.target_latency_ms * 0.7:
            self.current_batch_size = min(self.current_batch_size + 2, self.max_batch)
        elif p95 > self.target_latency_ms:
            self.current_batch_size = max(self.current_batch_size - 1, self.min_batch)

        return self.current_batch_size

    def get_stats(self) -> dict:
        sorted_lat = sorted(self._latencies) if self._latencies else [0]
        return {
            "current_batch_size": self.current_batch_size,
            "p50_latency_ms": round(sorted_lat[len(sorted_lat) // 2], 1),
            "p95_latency_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 1)
            if len(sorted_lat) > 1
            else 0,
            "samples": len(self._latencies),
        }
