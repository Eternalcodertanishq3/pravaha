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
