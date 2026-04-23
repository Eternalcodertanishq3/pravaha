"""Cost Estimator — Track inference cost per request.

Estimates cost based on token counts and model pricing tiers.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PricingTier:
    """Cost per 1K tokens for a model."""
    model_pattern: str
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


class CostEstimator:
    """Estimate and track inference costs."""

    DEFAULT_TIERS = [
        PricingTier("gpt2", 0.0001, 0.0002),
        PricingTier("llama-7b", 0.0005, 0.001),
        PricingTier("llama-13b", 0.001, 0.002),
        PricingTier("llama-70b", 0.005, 0.01),
        PricingTier("default", 0.001, 0.002),
    ]

    def __init__(self, tiers: list[PricingTier] | None = None) -> None:
        self.tiers = tiers or self.DEFAULT_TIERS
        self.total_cost = 0.0
        self.total_requests = 0

    def estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        tier = self._find_tier(model)
        cost = (input_tokens / 1000 * tier.input_cost_per_1k + output_tokens / 1000 * tier.output_cost_per_1k)
        self.total_cost += cost
        self.total_requests += 1
        return cost

    def _find_tier(self, model: str) -> PricingTier:
        model_lower = model.lower()
        for tier in self.tiers:
            if tier.model_pattern in model_lower:
                return tier
        return self.tiers[-1]

    def get_stats(self) -> dict:
        return {"total_cost_usd": round(self.total_cost, 6), "total_requests": self.total_requests, "avg_cost_usd": round(self.total_cost / max(1, self.total_requests), 6)}
