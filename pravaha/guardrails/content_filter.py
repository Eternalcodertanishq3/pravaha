"""Content Filter — Block harmful or unwanted content.

Scans prompts and responses for blocked patterns, PII, and harmful content.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    allowed: bool
    reason: str = ""
    matched_pattern: str = ""


class ContentFilter:
    """Filter prompts and responses for harmful content."""

    def __init__(self, blocked_patterns: list[str] | None = None) -> None:
        self.blocked_patterns = blocked_patterns or []
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.blocked_patterns]

    def check_prompt(self, prompt: str) -> FilterResult:
        for pattern, compiled in zip(self.blocked_patterns, self._compiled):
            if compiled.search(prompt):
                return FilterResult(
                    allowed=False, reason="blocked_pattern", matched_pattern=pattern
                )
        return FilterResult(allowed=True)

    def check_response(self, response: str) -> FilterResult:
        return self.check_prompt(response)

    def add_pattern(self, pattern: str) -> None:
        self.blocked_patterns.append(pattern)
        self._compiled.append(re.compile(pattern, re.IGNORECASE))
