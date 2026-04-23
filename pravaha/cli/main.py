"""Pravaha CLI — Main entry point using Typer."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="pravaha",
    help="Pravaha v3 — The self-healing, swarm-ready LLM inference engine.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _register_commands() -> None:
    """Import and register all sub-commands."""
    from pravaha.cli.commands.bench import bench
    from pravaha.cli.commands.chat import chat
    from pravaha.cli.commands.debug import debug_app
    from pravaha.cli.commands.models import models_app
    from pravaha.cli.commands.plugins import plugin_app
    from pravaha.cli.commands.rag import rag_app
    from pravaha.cli.commands.serve import serve
    from pravaha.cli.commands.swarm import swarm_app

    app.command()(serve)
    app.command()(chat)
    app.command()(bench)
    app.add_typer(models_app, name="models", help="Model management commands.")
    app.add_typer(swarm_app, name="swarm", help="Swarm agent pipeline commands.")
    app.add_typer(rag_app, name="rag", help="RAG document management.")
    app.add_typer(debug_app, name="debug", help="Debug and replay tools.")
    app.add_typer(plugin_app, name="plugin", help="Plugin management.")


_register_commands()

if __name__ == "__main__":
    app()
