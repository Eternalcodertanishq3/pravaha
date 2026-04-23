"""Performance Profiler Agent — Computational bottleneck detector.

Identifies O(n²) algorithms, N+1 queries, blocking I/O in async,
repeated computation, and inefficient data structures.

Triggers on: code, sql, algorithm, async
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class PerformanceProfilerAgent(BaseAgent):
    """Identifies performance bottlenecks and optimization opportunities."""

    role = "performance_profiler"
    priority = 1
    max_tokens = 512
    temperature = 0.1
    system_prompt = (
        "You are a performance analyst. Find:\n"
        "- O(n²) or worse complexity in loops\n"
        "- Repeated identical computation (should be cached)\n"
        "- Blocking I/O calls in async context\n"
        "- N+1 database query patterns\n"
        "- Inefficient data structures for the use case\n"
        "- Missing indexes in SQL schemas\n\n"
        "Return JSON:\n"
        '{"issues": [{"type": "<category>", "location": "<where>", '
        '"impact": "high|medium|low", "complexity_before": "<O(?)>", '
        '"complexity_after": "<O(?)>", "fix": "<how to optimize>"}]}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        code = context.code or context.output or task
        prompt = self.build_prompt(f"Profile for performance:\n```\n{code[:2000]}\n```", context)
        result = await self._generate_json(prompt, engine)
        perf_issues = result.get("issues", [])
        if not isinstance(perf_issues, list):
            perf_issues = []
        issues = [
            {
                "type": p.get("type", "performance"),
                "severity": "major" if p.get("impact") == "high" else "minor",
                "description": f"{p.get('type', '')}: {p.get('fix', '')}",
                "location": p.get("location", ""),
            }
            for p in perf_issues
        ]
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=f"Found {len(issues)} performance issue(s)"
            if issues
            else "PASS: No performance issues",
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=1.0 if not issues else 0.7,
            issues=issues,
            metadata={"perf_issues": len(issues), "details": perf_issues},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "function", "algorithm", "sql", "async"}
