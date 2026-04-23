"""Self-Benchmark — Automatic performance benchmarking on startup.

Runs a quick inference benchmark on startup to establish baseline
performance metrics (TTFT, throughput, latency).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results of a self-benchmark run."""

    ttft_ms: float = 0.0
    tokens_per_second: float = 0.0
    total_duration_ms: float = 0.0
    tokens_generated: int = 0
    prompt_tokens: int = 0


class SelfBenchmark:
    """Run self-benchmarks to measure engine performance."""

    BENCHMARK_PROMPT = "The quick brown fox jumps over the lazy dog. In the beginning"

    def __init__(self) -> None:
        self.last_result: BenchmarkResult | None = None

    async def run(self, engine: object, num_tokens: int = 50) -> BenchmarkResult:
        """Run a quick benchmark generating num_tokens tokens."""
        from pravaha.decoder.sampling import SamplingParams

        params = SamplingParams(max_new_tokens=num_tokens, temperature=0.0)

        t0 = time.time()
        first_token_time = None
        tokens = 0

        async for _ in engine.generate(self.BENCHMARK_PROMPT, params):  # type: ignore
            if first_token_time is None:
                first_token_time = time.time()
            tokens += 1

        total = time.time() - t0
        ttft = ((first_token_time or t0) - t0) * 1000
        tps = tokens / total if total > 0 else 0

        self.last_result = BenchmarkResult(
            ttft_ms=round(ttft, 1),
            tokens_per_second=round(tps, 1),
            total_duration_ms=round(total * 1000, 1),
            tokens_generated=tokens,
            prompt_tokens=len(self.BENCHMARK_PROMPT.split()),
        )
        logger.info(f"Benchmark: TTFT={ttft:.0f}ms, TPS={tps:.1f}, tokens={tokens}")
        return self.last_result
