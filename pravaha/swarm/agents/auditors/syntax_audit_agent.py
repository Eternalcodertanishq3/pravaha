"""Syntax Audit Agent — Static syntax analysis with reduced false positives."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class SyntaxAuditAgent(BaseAgent):
    """Static syntax auditor with tighter patterns to reduce false positives."""

    role = "syntax_audit"
    priority = 10
    max_tokens = 512
    temperature = 0.0

    CHECKS = [
        (r'(?<!\s)except\s*:', "bare_except",
         "Bare except clause (catches all exceptions including SystemExit)"),
        (r'\bimport\s+\*', "wildcard_import",
         "Wildcard import (pollutes namespace, hides dependencies)"),
        (r'^(?!\s*#).*\beval\s*\(', "eval_usage",
         "Use of eval()"),
        (r'^(?!\s*#).*\bexec\s*\(', "exec_usage",
         "Use of exec()"),
        (r'(?<!\bdef\s)\bTODO\b|(?<!\bdef\s)\bFIXME\b', "todo_marker",
         "TODO/FIXME marker (unfinished code)"),
    ]

    system_prompt = (
        "You are a syntax auditor. Find syntax issues in code.\n"
        "Focus on patterns that indicate real bugs, not style issues.\n"
        "Bare except, wildcard imports, and eval/exec are the priority."
    )

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues: list[dict[str, Any]] = []
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
