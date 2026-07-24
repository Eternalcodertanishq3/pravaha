"""PII Redaction Filter — Remove personally identifiable information from logs.

Applies regex-based redaction for: email addresses, phone numbers, SSNs,
credit card numbers, and IP addresses in log output.
"""

from __future__ import annotations

import logging
import re
from typing import ClassVar


class PIIRedactionFilter(logging.Filter):
    """Logging filter that redacts PII patterns."""

    PII_PATTERNS: ClassVar[list[tuple[str, re.Pattern, str]]] = [
        ("Email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[EMAIL_REDACTED]"),
        ("US Phone", re.compile(r"(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"), "[PHONE_REDACTED]"),
        ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
        ("Credit Card", re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"), "[CC_REDACTED]"),
        ("IPv4 Address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_REDACTED]"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact PII from log record message and args."""
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
        """Apply all PII patterns to the text."""
        for name, pattern, replacement in self.PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return text


def install_pii_filter() -> None:
    """Install the PII redaction filter on the root logger."""
    root_logger = logging.getLogger()
    for f in root_logger.filters:
        if isinstance(f, PIIRedactionFilter):
            return
    root_logger.addFilter(PIIRedactionFilter())
    logging.getLogger(__name__).info("PII redaction filter installed on root logger.")
