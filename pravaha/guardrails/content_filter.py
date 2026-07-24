"""Content Filter — Block harmful or unwanted content.

Scans prompts and responses for blocked patterns, PII, and harmful content.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

DEFAULT_BLOCKED_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"you are now",
    r"\[INST\]",
    r"<\|im_start\|>",
    r"<\|system\|>",
    r"ADMIN OVERRIDE",
    r"jailbreak",
]

MAX_PROMPT_LENGTH = 100_000

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    allowed: bool
    reason: str = ""
    matched_pattern: str = ""


class ContentFilter:
    """Filter prompts and responses for harmful content."""

    def __init__(self, blocked_patterns: list[str] | None = None) -> None:
        self.blocked_patterns = blocked_patterns if blocked_patterns is not None else DEFAULT_BLOCKED_PATTERNS.copy()
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.blocked_patterns]

    def check_for_injection(self, prompt: str) -> FilterResult:
        if re.search(r"(?:system|assistant|user)\s*:", prompt, re.IGNORECASE):
            return FilterResult(allowed=False, reason="role_override", matched_pattern="role_override")
        if re.search(r"(?i)(?:ignore\s+previous\s+instructions|bypass\s+instructions|override\s+instructions)", prompt):
            return FilterResult(allowed=False, reason="instruction_override", matched_pattern="instruction_override")
        if len(prompt) > 0:
            alnum_count = sum(1 for c in prompt if c.isalnum() or c.isspace())
            if (1.0 - (alnum_count / len(prompt))) > 0.30:
                return FilterResult(allowed=False, reason="high_special_char_density", matched_pattern="special_chars")
        return FilterResult(allowed=True)

    def check_prompt(self, prompt: str) -> FilterResult:
        if "\x00" in prompt:
            return FilterResult(allowed=False, reason="null_bytes_detected")
        if len(prompt) > MAX_PROMPT_LENGTH:
            return FilterResult(allowed=False, reason="prompt_too_long")

        injection_check = self.check_for_injection(prompt)
        if not injection_check.allowed:
            return injection_check

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
