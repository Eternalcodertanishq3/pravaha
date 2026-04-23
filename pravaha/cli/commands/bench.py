"""pravaha bench — Throughput and latency benchmarks."""

from __future__ import annotations

import typer

from pravaha.cli.ascii_art import console, status_box


def bench(
    model: str = typer.Option("gpt2", "--model", "-m", help="Model to benchmark"),
    prompt_len: int = typer.Option(128, "--prompt-len", help="Input prompt length in tokens"),
    output_len: int = typer.Option(50, "--output-len", help="Output tokens to generate"),
    runs: int = typer.Option(3, "--runs", help="Number of benchmark runs"),
    concurrent: int = typer.Option(1, "--concurrent", help="Concurrent requests"),
) -> None:
    """Run comprehensive benchmarks and display results table."""
    console.print("[bold green]Pravaha Benchmark[/bold green]\n")

    console.print(f"Model: {model} | Prompt: {prompt_len} tokens | "
                  f"Output: {output_len} tokens | Runs: {runs}\n")

    with console.status("[bold green]Running benchmarks...[/bold green]"):
        # Import and run benchmark
        try:
            import asyncio
            from pravaha.config.engine_config import EngineConfig
            from pravaha.observability.self_benchmark import StartupBenchmark

            config = EngineConfig.default()
            config.model.model_path = model

            from pravaha.engine.async_engine import AsyncPravahaEngine
            engine = AsyncPravahaEngine(config=config)
            benchmark = StartupBenchmark()
            result = asyncio.run(benchmark.run(engine))

            table = status_box({
                "Throughput": f"{result.get('tokens_per_second', 0):.0f} tok/s",
                "TTFT p50": f"{result.get('ttft_p50_ms', 0):.0f}ms",
                "TTFT p99": f"{result.get('ttft_p99_ms', 0):.0f}ms",
                "VRAM": f"{result.get('vram_gb', 0):.1f} GB",
                "Runs": str(runs),
            }, title="Benchmark Results")
            console.print(table)
            engine.stop()
        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")
            console.print("[dim]Make sure the model is available and dependencies are installed.[/dim]")
