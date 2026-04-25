"""Auth Audit Agent — Authentication/authorization pattern analysis."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class AuthAuditAgent(BaseAgent):
    role = "auth_audit"
    priority = 10
    max_tokens = 1024
    temperature = 0.0

    AUTH_PATTERNS = [
        (r"@app\.route.*\ndef\s+\w+\(", "missing_auth_decorator", "Route without auth decorator"),
        (r"jwt\.decode\(.*verify\s*=\s*False", "jwt_unverified", "JWT verification disabled"),
        (r"session\[.+\]\s*=\s*request\.", "session_fixation", "Session fixation risk"),
        (r"password.*=.*md5|sha1", "weak_password_hash", "Weak password hashing"),
        (r"admin.*=\s*True", "hardcoded_admin", "Hardcoded admin privilege"),
    ]

    system_prompt = "You are an authentication and authorization auditor."

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, issue_id, desc in self.AUTH_PATTERNS:
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                issues.append({"id": issue_id, "description": desc, "severity": "HIGH"})
        return AgentOutput(
            role=self.role, output=f"Auth audit: {len(issues)} finding(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
