"""Security Audit Agent — Enhanced with CVSS scoring."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class SecurityAuditAgent(BaseAgent):
    role = "security_audit"
    priority = 10
    max_tokens = 1024
    temperature = 0.0

    VULN_PATTERNS = [
        (r"\beval\s*\(", "code_injection", "CRITICAL", 9.8, "Use of eval()"),
        (r"\bexec\s*\(", "code_injection", "CRITICAL", 9.8, "Use of exec()"),
        (r"subprocess\..*shell\s*=\s*True", "command_injection", "HIGH", 8.5, "Shell injection risk"),
        (r"os\.system\(", "command_injection", "HIGH", 8.5, "os.system command injection"),
        (r"pickle\.loads?\(", "deserialization", "HIGH", 8.0, "Unsafe deserialization"),
        (r"yaml\.load\((?!.*Loader)", "deserialization", "MEDIUM", 6.5, "Unsafe YAML load"),
        (r"password\s*=\s*['\"]", "hardcoded_secret", "HIGH", 7.5, "Hardcoded password"),
        (r"api[_-]?key\s*=\s*['\"]", "hardcoded_secret", "HIGH", 7.5, "Hardcoded API key"),
        (r"SECRET\s*=\s*['\"]", "hardcoded_secret", "HIGH", 7.5, "Hardcoded secret"),
        (r"\bmd5\b|\bsha1\b", "weak_crypto", "MEDIUM", 5.0, "Weak hash algorithm"),
        (r"verify\s*=\s*False", "ssl_bypass", "HIGH", 7.0, "SSL verification disabled"),
        (r"CORS\(.*origins?\s*=\s*\[?\s*['\"]?\*", "cors_wildcard", "MEDIUM", 5.5, "CORS wildcard origin"),
    ]

    system_prompt = "You are a security auditor with CVSS scoring capability."

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, vuln_type, severity, cvss, desc in self.VULN_PATTERNS:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "id": vuln_type,
                        "line": i,
                        "description": desc,
                        "severity": severity,
                        "cvss": cvss,
                        "snippet": line.strip()[:80],
                    })
        max_cvss = max((float(str(i["cvss"])) for i in issues), default=0.0)
        return AgentOutput(
            role=self.role,
            output=f"Security scan: {len(issues)} finding(s), max CVSS={max_cvss}",
            issues=issues,
            metadata={"total_issues": len(issues), "max_cvss": max_cvss},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
