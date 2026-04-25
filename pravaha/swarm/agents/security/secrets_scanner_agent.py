"""Secrets Scanner Agent — Detect leaked secrets in code.

Uses pattern matching for known secret formats AND Shannon entropy
for high-entropy strings that look like secrets even without known patterns.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class SecretsScannerAgent(BaseAgent):
    role = "secrets_scanner"
    priority = 10
    max_tokens = 512
    temperature = 0.0

    SECRET_PATTERNS = [
        (r"AKIA[0-9A-Z]{16}", "aws_key", "AWS Access Key ID"),
        (r"ghp_[a-zA-Z0-9]{36}", "github_pat", "GitHub Personal Access Token"),
        (r"sk-[a-zA-Z0-9]{48}", "openai_key", "OpenAI API Key"),
        (r"xox[bprs]-[0-9a-zA-Z-]+", "slack_token", "Slack Token"),
        (r"-----BEGIN (RSA |EC )?PRIVATE KEY", "private_key", "Private Key (PEM)"),
        (r"token\s*=\s*['\"][a-zA-Z0-9]{20,}", "generic_token", "Generic API Token"),
        (r"password\s*=\s*['\"][^'\"]{8,}", "hardcoded_password", "Hardcoded Password"),
        (r"AIza[0-9A-Za-z_-]{35}", "google_api_key", "Google API Key"),
    ]

    # Entropy threshold: strings with Shannon entropy > 4.5 on 16+ chars
    ENTROPY_THRESHOLD = 4.5
    MIN_SECRET_LENGTH = 16

    system_prompt = (
        "You are a secrets scanner. Detect leaked API keys, passwords, "
        "tokens, private keys, and high-entropy strings that may be secrets."
    )

    @staticmethod
    def _shannon_entropy(s: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not s:
            return 0.0
        freq = Counter(s)
        length = len(s)
        return -sum(
            (count / length) * math.log2(count / length)
            for count in freq.values()
        )

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues: list[dict[str, Any]] = []
        flagged_lines: set[int] = set()

        # Phase 1: Pattern matching (fast, deterministic)
        for pattern, issue_id, desc in self.SECRET_PATTERNS:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line):
                    issues.append({
                        "id": issue_id,
                        "line": i,
                        "description": desc,
                        "severity": "CRITICAL",
                        "remediation": "Move to environment variable or secrets manager",
                    })
                    flagged_lines.add(i)

        # Phase 2: Entropy-based detection (catch unknown patterns)
        string_re = re.compile(r"""['\"]([a-zA-Z0-9_/+=\-]{16,})['\"]""")
        for i, line in enumerate(code.split("\n"), 1):
            if i in flagged_lines:
                continue
            for match in string_re.finditer(line):
                candidate = match.group(1)
                if len(candidate) >= self.MIN_SECRET_LENGTH:
                    entropy = self._shannon_entropy(candidate)
                    if entropy >= self.ENTROPY_THRESHOLD:
                        issues.append({
                            "id": "high_entropy_string",
                            "line": i,
                            "description": (
                                f"High-entropy string (entropy={entropy:.2f}) "
                                f"may be a leaked secret"
                            ),
                            "severity": "WARNING",
                            "entropy": round(entropy, 2),
                            "remediation": "Review: if this is a secret, move to env var",
                        })

        return AgentOutput(
            role=self.role,
            output=f"Secrets scan: {len(issues)} finding(s)",
            issues=issues,
            metadata={"total_issues": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
