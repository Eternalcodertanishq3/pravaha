"""API Security Agent — REST API security analysis."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class APISecurityAgent(BaseAgent):
    role = "api_security"
    priority = 10
    max_tokens = 1024
    temperature = 0.0

    API_PATTERNS = [
        (r"@app\.(get|post|put|delete)\(.*\)\s*\nasync\s+def\s+\w+\(\s*\)", "no_auth_endpoint", "Endpoint without auth params"),
        (r"rate_limit|throttle", "has_rate_limit", None),
        (r"\.query\(.*request\.", "unvalidated_query", "Unvalidated query parameter"),
        (r"response\.headers\[.+\]\s*=\s*request\.", "header_injection", "Header injection from user input"),
    ]

    system_prompt = "You are an API security auditor."

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        has_rate_limiting = False
        for pattern, issue_id, desc in self.API_PATTERNS:
            if desc is None:
                has_rate_limiting = bool(re.search(pattern, code))
                continue
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                issues.append({"id": issue_id, "description": desc, "severity": "MEDIUM"})
        if not has_rate_limiting and "@app." in code:
            issues.append({"id": "missing_rate_limit", "description": "No rate limiting detected", "severity": "MEDIUM"})
        return AgentOutput(
            role=self.role, output=f"API security: {len(issues)} finding(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
