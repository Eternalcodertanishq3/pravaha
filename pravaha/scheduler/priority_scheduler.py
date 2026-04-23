"""Priority Scheduler — Priority-based request scheduling.

Extends the continuous scheduler with priority queues for swarm agents,
VIP users, and latency-sensitive requests.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field

from pravaha.scheduler.request import InferenceRequest

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PriorityRequest:
    """Wrapper for priority-based scheduling.

    Lower priority values = higher priority (processed first).
    """

    priority: int
    timestamp: float
    request: InferenceRequest = field(compare=False)


class PriorityScheduler:
    """Priority-aware request scheduler.

    Replaces the FCFS waiting queue with a min-heap sorted by priority.
    Audit agents get highest priority (0), then orchestrators (1),
    then regular workers (2), then user requests (3).
    """

    PRIORITY_AUDIT = 0
    PRIORITY_ORCHESTRATOR = 1
    PRIORITY_WORKER = 2
    PRIORITY_USER = 3

    def __init__(self) -> None:
        self._heap: list[PriorityRequest] = []
        self._counter = 0

    def push(self, request: InferenceRequest, priority: int = 3) -> None:
        import time

        self._counter += 1
        entry = PriorityRequest(priority=priority, timestamp=time.time(), request=request)
        heapq.heappush(self._heap, entry)

    def pop(self) -> InferenceRequest | None:
        if self._heap:
            return heapq.heappop(self._heap).request
        return None

    def peek(self) -> InferenceRequest | None:
        if self._heap:
            return self._heap[0].request
        return None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0
