"""Injection Scanner Agent — Detect injection vulnerabilities."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class InjectionScannerAgent(BaseAgent):
    role = "injection_scanner"
    priority = 10
    max_tokens = 1024
    temperature = 0.0

    INJECTION_PATTERNS = [
        (r"f['\"].*\{.*\}.*SELECT|INSERT|UPDATE|DELETE", "sql_injection", "SQL injection via f-string"),
        (r"\.format\(.*\).*SELECT|INSERT|UPDATE|DELETE", "sql_injection", "SQL injection via .format()"),
        (r"execute\([^,]*\+", "sql_injection", "SQL injection via string concatenation"),
        (r"innerHTML\s*=", "xss", "DOM-based XSS via innerHTML"),
        (r"document\.write\(", "xss", "XSS via document.write"),
        (r"\.raw\(|\.unsafeHTML\(", "xss", "Unsafe HTML rendering"),
        (r"redirect\([^)]*request\.", "open_redirect", "Open redirect from user input"),
        (r"os\.path\.join\(.*request\.", "path_traversal", "Path traversal from user input"),
        (r"\.\./\.\.", "path_traversal", "Path traversal pattern"),
        (r"xml\.etree.*parse\(", "xxe", "Potential XXE vulnerability"),
    ]

    system_prompt = "You are an injection vulnerability scanner."

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, vuln_type, desc in self.INJECTION_PATTERNS:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "id": vuln_type, "line": i, "description": desc,
                        "severity": "CRITICAL",
                    })
        return AgentOutput(
            role=self.role,
            output=f"Injection scan: {len(issues)} finding(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
