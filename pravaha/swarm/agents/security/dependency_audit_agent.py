"""Dependency Audit Agent — Detect risky imports and dependencies."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class DependencyAuditAgent(BaseAgent):
    role = "dependency_audit"
    priority = 10
    max_tokens = 1024
    temperature = 0.0

    RISKY_IMPORTS = [
        (r"import\s+pickle", "pickle_import", "pickle allows arbitrary code execution"),
        (r"import\s+marshal", "marshal_import", "marshal is not secure for untrusted data"),
        (r"import\s+ctypes", "ctypes_import", "ctypes bypasses memory safety"),
        (r"from\s+lxml.*import.*etree", "lxml_xxe", "lxml can be vulnerable to XXE"),
        (r"import\s+telnetlib", "telnet_import", "telnet is unencrypted"),
        (r"import\s+cgi\b", "cgi_deprecated", "cgi module is deprecated"),
    ]

    system_prompt = "You are a dependency security auditor."

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, issue_id, desc in self.RISKY_IMPORTS:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line):
                    issues.append({"id": issue_id, "line": i, "description": desc, "severity": "MEDIUM"})
        return AgentOutput(
            role=self.role, output=f"Dependency audit: {len(issues)} finding(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
