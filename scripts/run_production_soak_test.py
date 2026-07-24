"""Production Soak & Telemetry Benchmark Suite — Measures empirical TTFT, TPS, P95/P99 latency, RAM/VRAM drift, and multi-tenant scaling.

Executes real asynchronous inference requests through AsyncPravahaEngine across varying concurrency levels (1, 5, 10, 25, 50).
Computes exact P50, P95, and P99 quantiles for TTFT, ITL, and total duration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import psutil

from pravaha.config.engine_config import EngineConfig
from pravaha.decoder.sampling import SamplingParams
from pravaha.engine.async_engine import AsyncPravahaEngine

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RequestTelemetry:
    request_id: str
    prompt_len: int
    num_tokens: int
    ttft_ms: float
    total_time_ms: float
    inter_token_latencies_ms: list[float]
    status_code: int


def get_memory_info() -> dict[str, float]:
    """Capture current process RAM and PyTorch CUDA VRAM usage."""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_alloc_mb = 0.0
    vram_res_mb = 0.0

    try:
        import torch
        if torch.cuda.is_available():
            vram_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    return {
        "ram_mb": round(ram_mb, 1),
        "vram_alloc_mb": round(vram_alloc_mb, 1),
        "vram_res_mb": round(vram_res_mb, 1),
        "cpu_pct": psutil.cpu_percent(interval=None),
    }


async def run_single_request(
    engine: AsyncPravahaEngine,
    request_id: str,
    prompt: str,
    max_tokens: int = 30,
) -> RequestTelemetry:
    """Execute single streaming request and collect microsecond telemetry."""
    params = SamplingParams(max_new_tokens=max_tokens, temperature=0.0)
    t0 = time.perf_counter()
    first_token_time = None
    last_token_time = None
    inter_token_latencies: list[float] = []
    tokens_count = 0
    status_code = 200

    try:
        async for token in engine.generate(prompt, params):
            t_now = time.perf_counter()
            if first_token_time is None:
                first_token_time = t_now
                last_token_time = t_now
            else:
                inter_token_latencies.append((t_now - last_token_time) * 1000)
                last_token_time = t_now
            tokens_count += 1
    except RuntimeError as e:
        if "overloaded" in str(e).lower() or "full" in str(e).lower():
            status_code = 429
        else:
            status_code = 500
    except Exception:
        status_code = 500

    t_end = time.perf_counter()
    ttft_ms = ((first_token_time or t_end) - t0) * 1000
    total_time_ms = (t_end - t0) * 1000

    return RequestTelemetry(
        request_id=request_id,
        prompt_len=len(prompt.split()),
        num_tokens=tokens_count,
        ttft_ms=round(ttft_ms, 2),
        total_time_ms=round(total_time_ms, 2),
        inter_token_latencies_ms=[round(l, 2) for l in inter_token_latencies],
        status_code=status_code,
    )


async def run_concurrency_benchmark(
    engine: AsyncPravahaEngine,
    concurrency: int,
    tokens_per_request: int = 25,
) -> dict[str, Any]:
    """Run a batch of concurrent requests and compute quantiles."""
    prompt = "The future of artificial intelligence in software engineering"
    logger.info(f"Running benchmark with concurrency={concurrency}...")

    mem_before = get_memory_info()
    t_start = time.perf_counter()

    tasks = [
        run_single_request(engine, f"bench-c{concurrency}-r{i}", prompt, max_tokens=tokens_per_request)
        for i in range(concurrency)
    ]
    results: list[RequestTelemetry] = await asyncio.gather(*tasks)

    t_total = time.perf_counter() - t_start
    mem_after = get_memory_info()

    # Process metrics
    successful = [r for r in results if r.status_code == 200]
    rate_limited = [r for r in results if r.status_code == 429]
    failed = [r for r in results if r.status_code == 500]

    ttfts = [r.ttft_ms for r in successful]
    totals = [r.total_time_ms for r in successful]
    all_itls = [itl for r in successful for itl in r.inter_token_latencies_ms]
    total_tokens = sum(r.num_tokens for r in successful)

    tps = total_tokens / t_total if t_total > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_duration_sec": round(t_total, 3),
        "requests": {
            "total": concurrency,
            "success": len(successful),
            "rate_limited": len(rate_limited),
            "failed": len(failed),
        },
        "throughput": {
            "total_tokens_generated": total_tokens,
            "system_tps": round(tps, 2),
            "per_user_tps": round(tps / max(1, len(successful)), 2),
        },
        "ttft_quantiles_ms": {
            "p50": round(float(np.percentile(ttfts, 50)), 2) if ttfts else 0.0,
            "p95": round(float(np.percentile(ttfts, 95)), 2) if ttfts else 0.0,
            "p99": round(float(np.percentile(ttfts, 99)), 2) if ttfts else 0.0,
        },
        "itl_quantiles_ms": {
            "p50": round(float(np.percentile(all_itls, 50)), 2) if all_itls else 0.0,
            "p95": round(float(np.percentile(all_itls, 95)), 2) if all_itls else 0.0,
            "p99": round(float(np.percentile(all_itls, 99)), 2) if all_itls else 0.0,
        },
        "total_latency_quantiles_ms": {
            "p50": round(float(np.percentile(totals, 50)), 2) if totals else 0.0,
            "p95": round(float(np.percentile(totals, 95)), 2) if totals else 0.0,
            "p99": round(float(np.percentile(totals, 99)), 2) if totals else 0.0,
        },
        "telemetry_delta": {
            "ram_before_mb": mem_before["ram_mb"],
            "ram_after_mb": mem_after["ram_mb"],
            "ram_drift_mb": round(mem_after["ram_mb"] - mem_before["ram_mb"], 1),
            "vram_alloc_mb": mem_after["vram_alloc_mb"],
        },
    }


async def main_soak_suite() -> dict[str, Any]:
    """Execute complete multi-level benchmark suite."""
    logger.info("Initializing AsyncPravahaEngine for Production Benchmark Suite...")
    config = EngineConfig.default()
    engine = AsyncPravahaEngine(config=config)

    # Wait for engine initialization to settle
    await asyncio.sleep(1.0)

    concurrencies = [1, 5, 10, 25, 50]
    bench_results = []

    mem_initial = get_memory_info()

    for c in concurrencies:
        res = await run_concurrency_benchmark(engine, concurrency=c, tokens_per_request=20)
        bench_results.append(res)
        await asyncio.sleep(0.5)

    # Trigger garbage collection check
    import gc
    gc.collect()
    mem_final = get_memory_info()

    engine.stop()

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "initial_memory": mem_initial,
        "final_memory": mem_final,
        "total_ram_drift_mb": round(mem_final["ram_mb"] - mem_initial["ram_mb"], 1),
        "concurrency_benchmarks": bench_results,
    }

    return summary


if __name__ == "__main__":
    result = asyncio.run(main_soak_suite())
    print(json.dumps(result, indent=2))
