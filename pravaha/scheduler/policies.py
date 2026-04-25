"""Scheduling Policies — Pluggable scheduling strategies.

Supports FCFS, SJF (shortest job first), and priority-based policies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque

from pravaha.scheduler.request import InferenceRequest

logger = logging.getLogger(__name__)


class SchedulingPolicy(ABC):
    """Base class for scheduling policies."""

    @abstractmethod
    def select_next(self, waiting: deque[InferenceRequest]) -> InferenceRequest | None:
        """Select the next request to process from the waiting queue."""
        ...


class FCFSPolicy(SchedulingPolicy):
    """First-Come, First-Served (default)."""

    def select_next(self, waiting: deque[InferenceRequest]) -> InferenceRequest | None:
        return waiting[0] if waiting else None


class SJFPolicy(SchedulingPolicy):
    """Shortest Job First — prioritize shorter prompts."""

    def select_next(self, waiting: deque[InferenceRequest]) -> InferenceRequest | None:
        if not waiting:
            return None
        return min(waiting, key=lambda r: r.num_prompt_tokens)


class PriorityPolicy(SchedulingPolicy):
    """Priority-based — use request priority field."""

    def select_next(self, waiting: deque[InferenceRequest]) -> InferenceRequest | None:
        if not waiting:
            return None
        return min(waiting, key=lambda r: getattr(r, "priority", 3))


def get_policy(name: str) -> SchedulingPolicy:
    """Get a scheduling policy by name.

    Args:
        name: Policy name ('fcfs', 'sjf', 'priority').

    Returns:
        The scheduling policy instance.
    """
    name = name.lower()
    if name == "sjf":
        return SJFPolicy()
    elif name == "priority":
        return PriorityPolicy()
    return FCFSPolicy()
