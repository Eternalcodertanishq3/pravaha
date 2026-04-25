"""Privilege Audit Agent — Detect privilege escalation risks."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class PrivilegeAuditAgent(BaseAgent):
    role = "privilege_audit"
    priority = 10
    max_tokens = 1024
    temperature = 0.0

    PRIV_PATTERNS = [
        (r"os\.setuid\(0\)|os\.setgid\(0\)", "root_escalation", "Running as root"),
        (r"chmod\s+777", "world_writable", "World-writable permissions"),
        (r"sudo\s+", "sudo_usage", "sudo usage in code"),
        (r"is_admin\s*=\s*True|is_superuser\s*=\s*True", "hardcoded_admin", "Hardcoded admin flag"),
        (r"DEBUG\s*=\s*True", "debug_enabled", "Debug mode in production"),
    ]

    system_prompt = "You are a privilege escalation auditor."

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, issue_id, desc in self.PRIV_PATTERNS:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({"id": issue_id, "line": i, "description": desc, "severity": "HIGH"})
        return AgentOutput(
            role=self.role, output=f"Privilege audit: {len(issues)} finding(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
