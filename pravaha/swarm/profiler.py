"""Performance Profiler — Per-agent profiling metrics.

Tracks execution times, token usage, tool success rates
across all agent invocations. Provides mean/p95/max stats.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentProfile:
    """Accumulated profile data for a single agent."""

    role: str
    durations_ms: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    total_calls: int = 0
    errors: int = 0

    def record_call(
        self,
        duration_ms: float,
        tokens: int,
        tool_calls: int = 0,
        tool_successes: int = 0,
        error: bool = False,
    ) -> None:
        self.durations_ms.append(duration_ms)
        self.token_counts.append(tokens)
        self.tool_calls += tool_calls
        self.tool_successes += tool_successes
        self.tool_failures += max(0, tool_calls - tool_successes)
        self.total_calls += 1
        if error:
            self.errors += 1

    def mean_duration(self) -> float:
        return statistics.mean(self.durations_ms) if self.durations_ms else 0.0

    def p95_duration(self) -> float:
        if len(self.durations_ms) < 2:
            return self.mean_duration()
        sorted_d = sorted(self.durations_ms)
        idx = int(len(sorted_d) * 0.95)
        return sorted_d[min(idx, len(sorted_d) - 1)]

    def max_duration(self) -> float:
        return max(self.durations_ms) if self.durations_ms else 0.0

    def token_efficiency(self) -> float:
        """Tokens per millisecond of execution."""
        total_tokens = sum(self.token_counts)
        total_ms = sum(self.durations_ms)
        return total_tokens / total_ms if total_ms > 0 else 0.0

    def tool_success_rate(self) -> float:
        if self.tool_calls == 0:
            return 1.0
        return self.tool_successes / self.tool_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "total_calls": self.total_calls,
            "mean_duration_ms": round(self.mean_duration(), 1),
            "p95_duration_ms": round(self.p95_duration(), 1),
            "max_duration_ms": round(self.max_duration(), 1),
            "total_tokens": sum(self.token_counts),
            "token_efficiency": round(self.token_efficiency(), 3),
            "tool_calls": self.tool_calls,
            "tool_success_rate": round(self.tool_success_rate(), 3),
            "errors": self.errors,
        }


class SwarmProfiler:
    """Per-agent performance profiler for the swarm.

    Aggregates stats across all agent invocations.
    Can export summary reports.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, AgentProfile] = {}
        self._start_time = time.time()

    def get_or_create(self, role: str) -> AgentProfile:
        if role not in self._profiles:
            self._profiles[role] = AgentProfile(role=role)
        return self._profiles[role]

    def record(
        self,
        role: str,
        duration_ms: float,
        tokens: int,
        tool_calls: int = 0,
        tool_successes: int = 0,
        error: bool = False,
    ) -> None:
        """Record a single agent invocation."""
        profile = self.get_or_create(role)
        profile.record_call(
            duration_ms=duration_ms,
            tokens=tokens,
            tool_calls=tool_calls,
            tool_successes=tool_successes,
            error=error,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get full profiling summary."""
        profiles = [p.to_dict() for p in self._profiles.values()]
        profiles.sort(key=lambda x: x["total_calls"], reverse=True)

        total_calls = sum(p["total_calls"] for p in profiles)
        total_tokens = sum(p["total_tokens"] for p in profiles)
        total_errors = sum(p["errors"] for p in profiles)

        return {
            "total_agents_profiled": len(profiles),
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_errors": total_errors,
            "uptime_s": round(time.time() - self._start_time, 1),
            "agents": profiles,
        }

    def get_top_agents(self, n: int = 5, by: str = "total_calls") -> list[dict[str, Any]]:
        """Get top N agents by a metric."""
        profiles = [p.to_dict() for p in self._profiles.values()]
        profiles.sort(key=lambda x: x.get(by, 0), reverse=True)
        return profiles[:n]

    def get_slowest(self, n: int = 5) -> list[dict[str, Any]]:
        """Get slowest agents by p95 duration."""
        return self.get_top_agents(n, by="p95_duration_ms")

    def reset(self) -> None:
        """Reset all profiles."""
        self._profiles.clear()
        self._start_time = time.time()
