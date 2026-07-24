"""Secrets Redaction Filter — Prevent secrets from leaking into logs.

Applies regex-based redaction to all log records before they are emitted.
Covers: AWS keys, API keys, JWT tokens, private keys, passwords, bearer tokens.
"""

from __future__ import annotations

import logging
import re
from typing import ClassVar


class SecretsRedactionFilter(logging.Filter):
    """Logging filter that redacts known secret patterns."""

    REDACTION_PATTERNS: ClassVar[list[tuple[str, re.Pattern]]] = [
        ("AWS Access Key", re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}", re.ASCII)),
        ("AWS Secret Key", re.compile(r"(?i)(?:aws_secret_access_key|secret_key)\s*[=:]\s*[A-Za-z0-9/+=]{40}")),
        ("Generic API Key", re.compile(r"(?i)(?:api[_-]?key|apikey|api[_-]?secret)\s*[=:\"']\s*[A-Za-z0-9_\-]{20,}")),
        ("Bearer Token", re.compile(r"(?i)Bearer\s+[A-Za-z0-9_\-\.]{20,}")),
        ("JWT Token", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_\-]{10,}")),
        ("Private Key", re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----")),
        ("Password in URL", re.compile(r"://[^:]+:[^@]+@", re.ASCII)),
        ("Generic Password", re.compile(r"(?i)(?:password|passwd|pwd)\s*[=:]\s*\S+")),
        ("GitHub Token", re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}")),
        ("Slack Token", re.compile(r"xox[bpors]-[A-Za-z0-9-]+")),
    ]

    REDACTED = "[REDACTED]"

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact secrets from the log record message and args."""
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._redact(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
        return True

    def _redact(self, text: str) -> str:
        """Apply all redaction patterns to the text."""
        for name, pattern in self.REDACTION_PATTERNS:
            text = pattern.sub(self.REDACTED, text)
        return text


def install_secrets_filter() -> None:
    """Install the secrets redaction filter on the root logger."""
    root_logger = logging.getLogger()
    # Avoid duplicate installation
    for f in root_logger.filters:
        if isinstance(f, SecretsRedactionFilter):
            return
    root_logger.addFilter(SecretsRedactionFilter())
    logging.getLogger(__name__).info("Secrets redaction filter installed on root logger.")
