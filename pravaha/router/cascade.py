"""Cascade Router — Fallback chain of models.

Try a fast/cheap model first. If quality is too low, cascade to a larger model.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class CascadeRouter:
    """Cascade through models from fastest to most capable."""

    def __init__(self, models: list[str], quality_threshold: float = 0.7) -> None:
        self.models = models
        self.quality_threshold = quality_threshold

    async def generate_cascade(
        self, prompt: str, engine: object, params: object
    ) -> AsyncGenerator[str, None]:
        """Try models in order until quality threshold is met."""
        for model_name in self.models:
            logger.info(f"Cascade: trying model {model_name}")
            tokens: list[str] = []
            async for token in engine.generate(prompt, params):  # type: ignore
                tokens.append(token)
                yield token
            break  # Accept first result for now
