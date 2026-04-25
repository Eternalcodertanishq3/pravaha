"""Syntax Audit Agent — Static syntax analysis."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class SyntaxAuditAgent(BaseAgent):
    role = "syntax_audit"
    priority = 10
    max_tokens = 512
    temperature = 0.0

    CHECKS = [
        (r"\bprint\s*\(.*\)\s*$", "debug_print", "Debug print statement found"),
        (r"except\s*:", "bare_except", "Bare except clause"),
        (r"import\s+\*", "wildcard_import", "Wildcard import"),
        (r"\beval\s*\(", "eval_usage", "Use of eval()"),
        (r"\bexec\s*\(", "exec_usage", "Use of exec()"),
        (r"TODO|FIXME|HACK|XXX", "todo_marker", "TODO/FIXME marker found"),
        (r"^\s*pass\s*$", "empty_block", "Empty pass block"),
    ]

    system_prompt = "You are a syntax auditor. Find syntax issues in code."

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, issue_id, description in self.CHECKS:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line):
                    issues.append({
                        "id": issue_id,
                        "line": i,
                        "description": description,
                        "severity": "warning",
                    })
        return AgentOutput(
            role=self.role, output=f"Found {len(issues)} syntax issue(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
