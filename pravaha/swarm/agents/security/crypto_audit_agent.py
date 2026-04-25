"""Crypto Audit Agent — Cryptographic practice analysis."""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class CryptoAuditAgent(BaseAgent):
    role = "crypto_audit"
    priority = 10
    max_tokens = 1024
    temperature = 0.0

    CRYPTO_PATTERNS = [
        (r"\bmd5\b", "weak_hash_md5", "MD5 is cryptographically broken"),
        (r"\bsha1\b", "weak_hash_sha1", "SHA1 is deprecated for security"),
        (r"\bDES\b|\b3DES\b", "weak_cipher", "DES/3DES are deprecated"),
        (r"\bRC4\b", "weak_cipher", "RC4 is broken"),
        (r"ECB\b", "ecb_mode", "ECB mode leaks patterns"),
        (r"random\.random\(|random\.randint\(", "insecure_random", "Use secrets module for crypto"),
        (r"key\s*=\s*b?['\"]", "hardcoded_key", "Hardcoded cryptographic key"),
        (r"iv\s*=\s*b?['\"]", "hardcoded_iv", "Hardcoded initialization vector"),
    ]

    system_prompt = "You are a cryptographic security auditor."

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues = []
        for pattern, issue_id, desc in self.CRYPTO_PATTERNS:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({"id": issue_id, "line": i, "description": desc, "severity": "HIGH"})
        return AgentOutput(
            role=self.role, output=f"Crypto audit: {len(issues)} finding(s)",
            issues=issues, metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
