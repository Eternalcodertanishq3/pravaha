"""Type Safety Agent — Type annotation and type-error auditing."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class TypeSafetyAgent(BaseAgent):
    role = "type_safety"
    priority = 10
    max_tokens = 512
    temperature = 0.0

    system_prompt = "You are a type safety auditor."

    CHECKS = [
        (r"def\s+\w+\([^)]*\)\s*:", "missing_return_type", "Function missing return type"),
        (r":\s*Any\b", "any_type", "Using Any type — consider narrowing"),
        (r"#\s*type:\s*ignore", "type_ignore", "type: ignore comment"),
    ]

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, issue_id, desc in self.CHECKS:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line):
                    issues.append({"id": issue_id, "line": i, "description": desc, "severity": "info"})
        return AgentOutput(
            role=self.role, output=f"Found {len(issues)} type issue(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
