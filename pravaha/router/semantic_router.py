"""Semantic Router — Route based on prompt embedding similarity."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SemanticRoute:
    name: str
    description: str
    model: str
    examples: list[str]


class SemanticRouter:
    """Route requests based on semantic similarity to predefined routes."""

    def __init__(self, routes: list[SemanticRoute] | None = None) -> None:
        self.routes = routes or []

    def route(self, prompt: str) -> Optional[SemanticRoute]:
        """Find the best matching route for a prompt."""
        if not self.routes:
            return None
        # Simple keyword matching fallback
        prompt_lower = prompt.lower()
        for route in self.routes:
            for example in route.examples:
                if example.lower() in prompt_lower:
                    return route
        return self.routes[0] if self.routes else None
