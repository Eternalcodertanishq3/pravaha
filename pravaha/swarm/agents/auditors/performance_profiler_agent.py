"""Performance Profiler Agent — Detect performance anti-patterns."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class PerformanceProfilerAgent(BaseAgent):
    role = "performance_profiler"
    priority = 10
    max_tokens = 1024
    temperature = 0.1

    system_prompt = "You are a performance auditor."

    CHECKS = [
        (r"for\s+\w+\s+in\s+.*:\s*\n\s+for\s+\w+\s+in\s+", "nested_loop", "Nested loop — O(n²) risk"),
        (r"\.append\(.*\)\s*$", "list_append_loop", "Consider list comprehension"),
        (r"import\s+time\s*\n.*time\.sleep", "blocking_sleep", "Blocking sleep in async context"),
    ]

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, issue_id, desc in self.CHECKS:
            if re.search(pattern, code, re.MULTILINE):
                issues.append({"id": issue_id, "description": desc, "severity": "warning"})
        return AgentOutput(
            role=self.role, output=f"Found {len(issues)} perf issue(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
