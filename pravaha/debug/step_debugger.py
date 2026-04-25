"""Step Debugger — Inspect token generation step-by-step."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TokenStep:
    position: int
    token_id: int
    token_text: str
    logprob: float = 0.0
    top_alternatives: list[tuple[str, float]] = field(default_factory=list)


class StepDebugger:
    """Step through token generation one token at a time."""

    def __init__(self) -> None:
        self._steps: dict[str, list[TokenStep]] = {}
        self._breakpoints: set[int] = set()

    def add_breakpoint(self, position: int) -> None:
        self._breakpoints.add(position)

    def record_step(self, request_id: str, step: TokenStep) -> None:
        if request_id not in self._steps:
            self._steps[request_id] = []
        self._steps[request_id].append(step)

    def should_pause(self, position: int) -> bool:
        return position in self._breakpoints

    def get_steps(self, request_id: str) -> list[TokenStep]:
        return self._steps.get(request_id, [])

    def get_step(self, request_id: str, position: int) -> TokenStep | None:
        steps = self._steps.get(request_id, [])
        for s in steps:
            if s.position == position:
                return s
        return None

    def get_step_info(
        self,
        request_id: str,
        position: int,
    ) -> dict | None:
        """Get structured step info for a position (route-compatible)."""
        step = self.get_step(request_id, position)
        if not step:
            return None
        return {
            "position": step.position,
            "token_id": step.token_id,
            "token_text": step.token_text,
            "logprob": step.logprob,
            "top_tokens": [
                {"text": t, "prob": p}
                for t, p in step.top_alternatives
            ],
        }
