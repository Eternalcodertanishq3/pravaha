"""Calibration Data Manager for post-training quantization.

Collects and formats calibration datasets used by GPTQ and AWQ
quantization pipelines.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


class CalibrationDataset:
    """Manage calibration data for quantization."""

    def __init__(self, samples: list[str] | None = None, max_samples: int = 128) -> None:
        self.samples = samples or []
        self.max_samples = max_samples

    def add(self, text: str) -> None:
        if len(self.samples) < self.max_samples:
            self.samples.append(text)

    def load_from_file(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.add(line)

    def get_samples(self, n: int | None = None) -> list[str]:
        n = n or min(len(self.samples), self.max_samples)
        if len(self.samples) <= n:
            return self.samples
        return random.sample(self.samples, n)

    @classmethod
    def default_english(cls) -> CalibrationDataset:
        """Return a small default calibration dataset."""
        samples = [
            "The quick brown fox jumps over the lazy dog.",
            "In a groundbreaking study, researchers found that artificial intelligence can now solve complex mathematical problems.",
            "Machine learning models require large amounts of data for training and validation.",
            "The transformer architecture has revolutionized natural language processing.",
        ] * 32
        return cls(samples=samples)
