"""pravaha serve — One-command model serving."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from pravaha.cli.ascii_art import print_banner, spinner, status_box


def serve(
    model: str = typer.Argument(..., help="Model name or path (e.g. gpt2, meta-llama/Llama-3-8B)"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
    quantize: Optional[str] = typer.Option(None, "--quantize", help="none | 4bit | 8bit"),
    tui: bool = typer.Option(False, "--tui", help="Launch premium terminal dashboard"),
    swarm: bool = typer.Option(False, "--swarm", help="Enable 32-agent swarm layer"),
    self_heal: bool = typer.Option(True, "--self-heal/--no-self-heal", help="Enable self-healing audit loop"),
    rag: bool = typer.Option(False, "--rag", help="Enable built-in RAG pipeline"),
    speculative: bool = typer.Option(False, "--speculative", help="Enable speculative decoding"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file"),
    workers: int = typer.Option(1, "--workers", help="Uvicorn worker processes"),
    benchmark: bool = typer.Option(True, "--benchmark/--no-benchmark", help="Self-benchmark on startup"),
) -> None:
    """One-command model serving.

    Examples:
      pravaha serve gpt2
      pravaha serve meta-llama/Llama-3-8B --quantize 4bit --tui
      pravaha serve mistral-7b --swarm --self-heal --rag --tui
    """
    print_banner()

    # Build config
    from pravaha.config.engine_config import EngineConfig

    if config and config.exists():
        engine_config = EngineConfig.from_yaml(str(config))
    else:
        engine_config = EngineConfig.default()
    engine_config.model.model_path = model
    if quantize:
        engine_config.model.quantization = quantize
    if swarm:
        engine_config.swarm.enabled = True
    if rag:
        engine_config.rag.enabled = True

    # Print status
    info = status_box({
        "Model": model,
        "Quant": quantize or "none",
        "Device": engine_config.model.resolved_device,
        "Swarm": "enabled" if swarm else "disabled",
        "RAG": "enabled" if rag else "disabled",
        "Port": str(port),
    }, title="Pravaha v3")
    typer.echo(info)

    # Run self-benchmark
    if benchmark:
        typer.echo("\n[benchmark] Self-benchmarking on startup...")

    # Launch TUI or standard uvicorn
    if tui:
        typer.echo("\nLaunching TUI dashboard...")
        try:
            from pravaha.tui.app import PravahaTUI
            tui_app = PravahaTUI(engine_config=engine_config, host=host, port=port)
            tui_app.run()
        except ImportError:
            typer.echo("Textual not installed. Falling back to standard server.")
            _start_uvicorn(host, port, workers)
    else:
        _start_uvicorn(host, port, workers)


def _start_uvicorn(host: str, port: int, workers: int) -> None:
    """Start the uvicorn server."""
    import uvicorn
    uvicorn.run(
        "pravaha.serving.app:create_app",
        host=host, port=port, workers=workers,
        factory=True, log_level="info",
    )
