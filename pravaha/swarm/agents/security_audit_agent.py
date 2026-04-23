"""Security Audit Agent — OWASP vulnerability scanner.

Scans code for SQL/command injection, hardcoded secrets,
unsafe deserialization, path traversal, and weak crypto.

Triggers on: code, api, database, auth, crypto
"""

from __future__ import annotations

import re
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class SecurityAuditAgent(BaseAgent):
    """Scans for OWASP Top 10 security vulnerabilities."""

    role = "security_audit"
    priority = 1
    max_tokens = 512
    temperature = 0.1

    # Static patterns for common vulnerabilities (pre-LLM check)
    STATIC_PATTERNS = [
        (
            r'(?i)(password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']',
            "Hardcoded secret",
            "critical",
        ),
        (r"(?i)eval\s*\(", "Use of eval()", "critical"),
        (r"(?i)exec\s*\(", "Use of exec()", "major"),
        (r"(?i)subprocess\.call\s*\(.*shell\s*=\s*True", "Shell injection risk", "critical"),
        (r"(?i)pickle\.loads?\s*\(", "Unsafe deserialization (pickle)", "critical"),
        (r"(?i)yaml\.load\s*\([^)]*\)(?!.*Loader)", "Unsafe YAML load without Loader", "major"),
        (r"(?i)os\.system\s*\(", "OS command injection risk", "major"),
        (r"(?i)md5|sha1(?!_)", "Weak hash algorithm (MD5/SHA1)", "minor"),
    ]

    system_prompt = (
        "You are a security auditor. Scan for OWASP Top 10 issues:\n"
        "- SQL/command injection risks\n"
        "- Hardcoded secrets, API keys, passwords\n"
        "- Unsafe deserialization\n"
        "- Path traversal vulnerabilities\n"
        "- Unvalidated inputs\n"
        "- Weak cryptography usage\n\n"
        "Return JSON:\n"
        '{"vulnerabilities": [{"cve_type": "<OWASP category>", '
        '"severity": "critical|major|minor", "location": "<where>", '
        '"description": "<what>", "fix": "<how to fix>"}]}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        code = context.code or context.output or task
        issues: list[dict[str, Any]] = []

        # Phase 1: Static regex scanning (fast, zero cost)
        for pattern, desc, severity in self.STATIC_PATTERNS:
            matches = re.finditer(pattern, code)
            for match in matches:
                line_num = code[: match.start()].count("\n") + 1
                issues.append(
                    {
                        "type": "security",
                        "severity": severity,
                        "description": f"{desc}: {match.group()[:50]}",
                        "location": f"line {line_num}",
                        "source": "static_scan",
                    }
                )

        # Phase 2: LLM deep analysis
        prompt = self.build_prompt(f"Security audit:\n```\n{code[:1500]}\n```", context)
        result = await self._generate_json(prompt, engine)
        vulns = result.get("vulnerabilities", [])
        if isinstance(vulns, list):
            for v in vulns:
                issues.append(
                    {
                        "type": v.get("cve_type", "security"),
                        "severity": v.get("severity", "major"),
                        "description": v.get("description", ""),
                        "location": v.get("location", ""),
                        "fix": v.get("fix", ""),
                        "source": "llm_scan",
                    }
                )

        clean = len(issues) == 0
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output="PASS: No vulnerabilities"
            if clean
            else f"FAIL: {len(issues)} vulnerability(ies)",
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=1.0 if clean else 0.3,
            issues=issues,
            metadata={
                "clean": clean,
                "static_findings": sum(1 for i in issues if i.get("source") == "static_scan"),
                "llm_findings": sum(1 for i in issues if i.get("source") == "llm_scan"),
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "api", "function", "class", "script", "auth", "crypto"}
