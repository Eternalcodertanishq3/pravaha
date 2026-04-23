"""pravaha models — Model management commands."""

from __future__ import annotations

import typer

from pravaha.cli.ascii_art import console

models_app = typer.Typer()


@models_app.command("list")
def list_models() -> None:
    """List loaded and available models."""
    import httpx
    try:
        resp = httpx.get("http://localhost:8000/v1/models")
        data = resp.json()
        for m in data.get("data", []):
            console.print(f"  [green]●[/green] {m.get('id', '?')}")
    except Exception:
        console.print("[dim]No running server. Use 'pravaha serve <model>' first.[/dim]")


@models_app.command("info")
def model_info(model: str = typer.Argument(..., help="Model name")) -> None:
    """Show detailed model information."""
    console.print(f"[bold]Model:[/bold] {model}")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model)
        console.print(f"  Type: {config.model_type}")
        console.print(f"  Layers: {getattr(config, 'num_hidden_layers', '?')}")
        console.print(f"  Heads: {getattr(config, 'num_attention_heads', '?')}")
        console.print(f"  Vocab: {getattr(config, 'vocab_size', '?')}")
    except Exception as e:
        console.print(f"[red]Could not load config: {e}[/red]")


@models_app.command("pull")
def pull_model(model: str = typer.Argument(..., help="Model to download")) -> None:
    """Download a model from HuggingFace Hub."""
    console.print(f"Pulling {model}...")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model)
        console.print(f"[green]Downloaded to: {path}[/green]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")


@models_app.command("remove")
def remove_model(model: str = typer.Argument(..., help="Model to remove")) -> None:
    """Remove a cached model."""
    console.print(f"[yellow]Removing {model} from cache...[/yellow]")
    console.print("[dim]Manual removal from HuggingFace cache directory required.[/dim]")
