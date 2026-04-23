"""A/B Router — Split traffic between model variants for testing."""

from __future__ import annotations
import logging
import random

logger = logging.getLogger(__name__)


class ABRouter:
    """A/B test between model variants with configurable traffic split."""

    def __init__(self, model_a: str, model_b: str, b_percentage: float = 10.0) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.b_percentage = b_percentage
        self._a_count = 0
        self._b_count = 0

    def route(self) -> str:
        if random.random() * 100 < self.b_percentage:
            self._b_count += 1
            return self.model_b
        self._a_count += 1
        return self.model_a

    def get_stats(self) -> dict:
        total = self._a_count + self._b_count
        return {"model_a": self.model_a, "model_b": self.model_b, "a_count": self._a_count, "b_count": self._b_count, "b_pct_actual": round(self._b_count / max(1, total) * 100, 1)}
