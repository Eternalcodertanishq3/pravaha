"""Model Router — Route requests to the optimal model.

Feature 7: Automatically select the best model based on prompt complexity,
token budget, and available resources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """Result of a routing decision."""

    model_name: str
    reason: str
    confidence: float = 1.0


class ModelRouter:
    """Route requests to the optimal model based on complexity analysis."""

    def __init__(self, models: list[str] | None = None) -> None:
        self.models = models or []
        self._default_model = models[0] if models else "default"

    def route(self, prompt: str, max_tokens: int = 256) -> RouteDecision:
        """Decide which model should handle this request."""
        complexity = self._estimate_complexity(prompt)
        if complexity < 0.3 and len(self.models) > 1:
            return RouteDecision(self.models[-1], "simple prompt → small model", complexity)
        return RouteDecision(self._default_model, "default routing", complexity)

    def _estimate_complexity(self, prompt: str) -> float:
        words = len(prompt.split())
        has_code = any(kw in prompt.lower() for kw in ["```", "def ", "class ", "function"])
        has_reasoning = any(kw in prompt.lower() for kw in ["explain", "analyze", "compare", "why"])
        score = min(1.0, words / 500 + (0.3 if has_code else 0) + (0.2 if has_reasoning else 0))
        return score
