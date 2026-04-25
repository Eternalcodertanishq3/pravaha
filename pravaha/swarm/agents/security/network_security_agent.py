"""Network Security Agent — Network security pattern analysis."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class NetworkSecurityAgent(BaseAgent):
    role = "network_security"
    priority = 10
    max_tokens = 1024
    temperature = 0.0

    NET_PATTERNS = [
        (r"http://(?!localhost|127\.0\.0\.1)", "insecure_http", "Non-local HTTP URL (use HTTPS)"),
        (r"verify\s*=\s*False", "ssl_disabled", "SSL verification disabled"),
        (r"0\.0\.0\.0", "bind_all_interfaces", "Binding to all interfaces"),
        (r"socket\.socket\(", "raw_socket", "Raw socket usage — review needed"),
        (r"allow_origins.*\*", "cors_wildcard", "CORS wildcard allows any origin"),
    ]

    system_prompt = "You are a network security auditor."

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, issue_id, desc in self.NET_PATTERNS:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({"id": issue_id, "line": i, "description": desc, "severity": "MEDIUM"})
        return AgentOutput(
            role=self.role, output=f"Network audit: {len(issues)} finding(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
