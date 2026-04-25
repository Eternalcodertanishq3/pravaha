"""Logic Flaw Agent — Detect logical errors in code."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class LogicFlawAgent(BaseAgent):
    role = "logic_flaw"
    priority = 10
    max_tokens = 1024
    temperature = 0.1

    system_prompt = "You are a logic flaw detector."

    CHECKS = [
        (r"if\s+\w+\s*=\s+", "assignment_in_if", "Assignment in if condition (= instead of ==)"),
        (r"while\s+True\s*:", "infinite_loop_risk", "Potential infinite loop (while True without break check)"),
        (r"return\s+.*\n\s+\S", "dead_code", "Code after return statement"),
        (r"except\s+Exception.*:\s*\n\s*pass", "swallowed_exception", "Swallowed exception"),
    ]

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, issue_id, desc in self.CHECKS:
            if re.search(pattern, code, re.MULTILINE):
                issues.append({"id": issue_id, "description": desc, "severity": "error"})
        return AgentOutput(
            role=self.role, output=f"Found {len(issues)} logic flaw(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
