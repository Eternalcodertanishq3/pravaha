"""Compliance Agent — Regulatory compliance checking."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ComplianceAgent(BaseAgent):
    role = "compliance"
    priority = 10
    max_tokens = 1024
    temperature = 0.1

    COMPLIANCE_CHECKS = [
        (r"log.*password|log.*secret|log.*token", "logging_sensitive", "Logging sensitive data (GDPR/PCI risk)"),
        (r"\.encode\(.*utf-8|\.decode\(", "encoding_check", None),
        (r"print\(.*password|print\(.*secret", "printing_sensitive", "Printing secrets to stdout"),
        (r"cookie.*secure\s*=\s*False", "insecure_cookie", "Cookie without Secure flag"),
        (r"cookie.*httponly\s*=\s*False", "no_httponly", "Cookie without HttpOnly flag"),
    ]

    system_prompt = "You are a compliance auditor for OWASP/GDPR/PCI standards."

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, issue_id, desc in self.COMPLIANCE_CHECKS:
            if desc is None:
                continue
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({"id": issue_id, "line": i, "description": desc, "severity": "HIGH"})
        return AgentOutput(
            role=self.role, output=f"Compliance: {len(issues)} finding(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
