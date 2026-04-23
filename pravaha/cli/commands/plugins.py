"""pravaha plugin — Plugin management commands."""

from __future__ import annotations

import typer

from pravaha.cli.ascii_art import console

plugin_app = typer.Typer()


@plugin_app.command("list")
def list_plugins() -> None:
    """Show all installed plugins."""
    from pravaha.plugins.registry import PluginRegistry
    registry = PluginRegistry()
    plugins = registry.discover()
    if plugins:
        for p in plugins:
            console.print(f"  [green]●[/green] {p.name} v{p.version} — {p.description}")
    else:
        console.print("[dim]No plugins installed.[/dim]")
        console.print("[dim]Install with: pravaha plugin install <path>[/dim]")


@plugin_app.command("install")
def install_plugin(path: str = typer.Argument(..., help="Path to plugin package")) -> None:
    """Install a local plugin."""
    console.print(f"Installing plugin from: {path}")
    import subprocess
    try:
        subprocess.run(["pip", "install", "-e", path], check=True)
        console.print("[green]Plugin installed successfully.[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Installation failed: {e}[/red]")


@plugin_app.command("remove")
def remove_plugin(name: str = typer.Argument(..., help="Plugin name")) -> None:
    """Remove an installed plugin."""
    console.print(f"[yellow]Removing plugin: {name}[/yellow]")
    import subprocess
    try:
        subprocess.run(["pip", "uninstall", "-y", name], check=True)
        console.print("[green]Plugin removed.[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Removal failed: {e}[/red]")


@plugin_app.command("info")
def plugin_info(name: str = typer.Argument(..., help="Plugin name")) -> None:
    """Show plugin details."""
    from pravaha.plugins.registry import PluginRegistry
    registry = PluginRegistry()
    for p in registry.discover():
        if p.name == name:
            console.print(f"[bold]{p.name}[/bold] v{p.version}")
            console.print(f"  {p.description}")
            return
    console.print(f"[red]Plugin '{name}' not found.[/red]")
