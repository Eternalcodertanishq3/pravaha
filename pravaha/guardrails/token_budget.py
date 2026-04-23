"""Token Budget — Per-request and per-session token limits.

Enforces maximum token usage to prevent runaway costs.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TokenBudget:
    """Enforce token budgets per request and per session."""

    def __init__(self, per_request: int = 0, per_session: int = 0) -> None:
        self.per_request = per_request  # 0 = unlimited
        self.per_session = per_session
        self._session_usage: dict[str, int] = defaultdict(int)

    def check_request(self, tokens: int) -> bool:
        if self.per_request > 0 and tokens > self.per_request:
            return False
        return True

    def check_session(self, session_id: str, tokens: int) -> bool:
        if self.per_session > 0:
            projected = self._session_usage[session_id] + tokens
            if projected > self.per_session:
                return False
        return True

    def record_usage(self, session_id: str, tokens: int) -> None:
        self._session_usage[session_id] += tokens

    def get_session_usage(self, session_id: str) -> int:
        return self._session_usage[session_id]

    def reset_session(self, session_id: str) -> None:
        self._session_usage.pop(session_id, None)
