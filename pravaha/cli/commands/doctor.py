"""Pravāha Doctor — Environment Diagnostics Command."""

from __future__ import annotations

import logging
import sys

import typer

logger = logging.getLogger(__name__)

doctor_app = typer.Typer(help="Diagnose environment setup, CUDA, Rust FFI, and dependencies.")


def check_environment() -> dict[str, str | bool]:
    """Inspect system environment and return status dictionary."""
    status: dict[str, str | bool] = {}

    # Python Version
    status["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # PyTorch & CUDA
    try:
        import torch
        status["pytorch_installed"] = True
        status["pytorch_version"] = torch.__version__
        status["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["cuda_version"] = torch.version.cuda or "Unknown"
        else:
            status["gpu_name"] = "N/A (CPU Mode)"
            status["cuda_version"] = "N/A"
    except ImportError:
        status["pytorch_installed"] = False
        status["pytorch_version"] = "Not Installed"
        status["cuda_available"] = False
        status["gpu_name"] = "N/A"
        status["cuda_version"] = "N/A"

    # Rust Core FFI Module
    try:
        from pravaha import pravaha_core
        status["rust_ffi_available"] = True
    except ImportError:
        status["rust_ffi_available"] = False

    # Triton Kernels
    try:
        import triton
        status["triton_available"] = True
        status["triton_version"] = getattr(triton, "__version__", "Available")
    except ImportError:
        status["triton_available"] = False
        status["triton_version"] = "Not Installed"

    return status


@doctor_app.callback(invoke_without_command=True)
def doctor() -> None:
    """Run environment diagnostics and print friendly setup advice."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    status = check_environment()

    table = Table(title="🌊 Pravāha Environment Diagnostic Report", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="bold")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    # Python
    table.add_row("Python Environment", "[green]OK[/green]", f"Python {status['python_version']}")

    # PyTorch
    if status["pytorch_installed"]:
        table.add_row("PyTorch Framework", "[green]OK[/green]", f"v{status['pytorch_version']}")
    else:
        table.add_row("PyTorch Framework", "[red]MISSING[/red]", "Run `pip install torch`")

    # CUDA
    if status["cuda_available"]:
        table.add_row("GPU Acceleration (CUDA)", "[green]OK[/green]", f"{status['gpu_name']} (CUDA {status['cuda_version']})")
    else:
        table.add_row("GPU Acceleration (CUDA)", "[yellow]CPU Only[/yellow]", "No CUDA GPU detected. Running in CPU fallback mode.")

    # Rust FFI
    if status["rust_ffi_available"]:
        table.add_row("Rust Core Engine FFI", "[green]OK[/green]", "Compiled module `pravaha_core` active")
    else:
        table.add_row(
            "Rust Core Engine FFI",
            "[yellow]Fallback[/yellow]",
            "Pure-Python fallback active. (Build with `maturin develop --manifest-path rust/Cargo.toml` for 5x speedup)",
        )

    # Triton
    if status["triton_available"]:
        table.add_row("Triton FlashDecoding Kernels", "[green]OK[/green]", f"v{status['triton_version']}")
    else:
        table.add_row(
            "Triton FlashDecoding Kernels",
            "[yellow]Fallback[/yellow]",
            "PyTorch SDPA fallback active. (Install `triton` on CUDA Linux/WSL for maximum throughput)",
        )

    console.print(table)

    # Summary Panel
    if status["cuda_available"] and status["rust_ffi_available"]:
        console.print(Panel("[bold green]✨ System Status: High-Performance GPU Engine Ready![/bold green]"))
    else:
        console.print(Panel("[bold yellow]⚡ System Status: Functional with Pure-Python / CPU Fallbacks Active.[/bold yellow]"))
