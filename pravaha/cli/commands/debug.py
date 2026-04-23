"""pravaha debug — Request replay and token-level debugging."""

from __future__ import annotations

import typer

from pravaha.cli.ascii_art import console

debug_app = typer.Typer()


@debug_app.command("replay")
def replay(request_id: str = typer.Argument(..., help="Request ID to replay")) -> None:
    """Replay a recorded request exactly."""
    import httpx

    try:
        resp = httpx.post(
            "http://localhost:8000/v1/debug/replay", json={"request_id": request_id}, timeout=60.0
        )
        console.print(resp.json())
    except Exception as e:
        console.print(f"[red]Replay error: {e}[/red]")


@debug_app.command("step")
def step(
    request_id: str = typer.Argument(..., help="Request ID"),
    position: int = typer.Option(0, "--pos", "-p", help="Token position"),
) -> None:
    """Step through inference token-by-token, inspecting logits."""
    import httpx

    try:
        resp = httpx.get(
            "http://localhost:8000/v1/debug/step",
            params={"request_id": request_id, "pos": position},
        )
        data = resp.json()
        console.print(f"[bold]Position {position}:[/bold] '{data.get('token_text', '?')}'")
        console.print(f"  Token ID: {data.get('token_id', '?')}")
        top = data.get("top_tokens", [])
        if top:
            console.print("  Top candidates:")
            for t in top[:10]:
                console.print(f"    {t.get('text', '?'):15s}  prob={t.get('prob', 0):.4f}")
    except Exception as e:
        console.print(f"[red]Step error: {e}[/red]")


@debug_app.command("trace")
def trace(request_id: str = typer.Argument(..., help="Request ID")) -> None:
    """Export full token-by-token decision trace."""
    import httpx

    try:
        resp = httpx.get("http://localhost:8000/v1/debug/trace", params={"request_id": request_id})
        import json

        output = json.dumps(resp.json(), indent=2)
        console.print(output)
    except Exception as e:
        console.print(f"[red]Trace error: {e}[/red]")


@debug_app.command("logits")
def logits(
    request_id: str = typer.Argument(..., help="Request ID"),
    token_pos: int = typer.Argument(..., help="Token position"),
) -> None:
    """Show top-10 logits at a specific token position."""
    console.print(f"[bold]Logits at position {token_pos}[/bold]")
    console.print("[dim]Connect to a running server with debug enabled.[/dim]")
