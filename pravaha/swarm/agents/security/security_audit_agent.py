"""Security Audit Agent — Enhanced with tighter regex patterns and CVSS scoring."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class SecurityAuditAgent(BaseAgent):
    """Static security scanner with CVSS scoring and false-positive-resistant patterns."""

    role = "security_audit"
    priority = 10
    max_tokens = 1024
    temperature = 0.0

    VULN_PATTERNS = [
        # Hardcoded secrets: require actual string literal AFTER the equals
        # NOT: password = get_env(...) or password = os.environ[...]
        # YES: password = "actual_password_here"
        (r'\bpassword\s*=\s*["\'][^"\']{4,}["\']',
         "hardcoded_password", "HIGH", 8.0,
         "Hardcoded password literal"),

        (r'\bapi[_-]?key\s*=\s*["\'][^"\']{8,}["\']',
         "hardcoded_api_key", "HIGH", 8.0,
         "Hardcoded API key literal"),

        (r'\bsecret[_-]?key\s*=\s*["\'][^"\']{8,}["\']',
         "hardcoded_secret", "HIGH", 7.5,
         "Hardcoded secret key literal"),

        (r'\btoken\s*=\s*["\'][a-zA-Z0-9_\-]{20,}["\']',
         "hardcoded_token", "HIGH", 7.5,
         "Hardcoded token literal (long string)"),

        # Code execution: NOT in comments (line must not start with #)
        (r'^(?!\s*#).*\beval\s*\(',
         "code_exec_eval", "CRITICAL", 9.8,
         "Use of eval() — code injection risk"),

        (r'^(?!\s*#).*\bexec\s*\(',
         "code_exec_exec", "CRITICAL", 9.8,
         "Use of exec() — code injection risk"),

        # Subprocess with shell=True (not in comments)
        (r'^(?!\s*#).*subprocess\..*shell\s*=\s*True',
         "shell_injection", "CRITICAL", 9.0,
         "subprocess with shell=True — command injection"),

        # os.system (not in comments)
        (r'^(?!\s*#).*os\.system\s*\(',
         "command_injection", "HIGH", 8.5,
         "os.system — command injection risk"),

        # Unsafe deserialization
        (r'^(?!\s*#).*pickle\.loads?\s*\(',
         "unsafe_deserial", "HIGH", 8.5,
         "pickle.load/loads — arbitrary code execution on untrusted data"),

        (r'^(?!\s*#).*yaml\.load\s*\([^)]*\)(?!\s*#.*safe)',
         "unsafe_yaml", "MEDIUM", 6.5,
         "yaml.load without Loader — use yaml.safe_load instead"),

        # SSL bypass (only when verify= is explicitly set to False)
        (r'verify\s*=\s*False(?!\s*#\s*nosec)',
         "ssl_bypass", "HIGH", 7.5,
         "SSL certificate verification disabled"),

        # Weak crypto: hashlib.md5/sha1 (not in comments or strings)
        (r'^(?!\s*#)(?!\s*["\']).*hashlib\.(md5|sha1)\s*\(',
         "weak_hash", "MEDIUM", 5.0,
         "MD5/SHA1 is cryptographically weak for security purposes"),

        # CORS wildcard
        (r'CORS\(.*origins?\s*=\s*\[?\s*["\']?\*',
         "cors_wildcard", "MEDIUM", 5.5,
         "CORS wildcard origin"),
    ]

    system_prompt = (
        "You are a security auditor with CVSS scoring capability.\n"
        "Scan code for vulnerabilities using static pattern matching.\n"
        "Report each finding with severity, CVSS score, and remediation."
    )

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues: list[dict[str, Any]] = []
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
