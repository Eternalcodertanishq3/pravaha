"""pravaha swarm — Swarm pipeline commands."""

from __future__ import annotations

import typer

from pravaha.cli.ascii_art import console

swarm_app = typer.Typer()


@swarm_app.command("run")
def run_swarm(
    task: str = typer.Argument(..., help="Task to execute"),
    pipeline: str = typer.Option("plan-execute-audit", "--pipeline", "-p"),
    max_iterations: int = typer.Option(3, "--max-iter"),
) -> None:
    """Execute a swarm pipeline on a task."""
    console.print(f"[bold green]Swarm Execute[/bold green] — pipeline: {pipeline}\n")
    import httpx

    try:
        resp = httpx.post(
            "http://localhost:8000/v1/swarm/run",
            json={"prompt": task, "pipeline": pipeline, "max_audit_iterations": max_iterations},
            timeout=120.0,
        )
        result = resp.json()
        console.print(f"Output:\n{result.get('output', '?')}")
        console.print(
            f"\n[dim]Score: {result.get('final_score', '?')} | "
            f"Iterations: {result.get('audit_iterations', '?')}[/dim]"
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@swarm_app.command("list-agents")
def list_agents() -> None:
    """List all available swarm agents."""
    from pravaha.swarm.agents import ALL_AGENTS

    console.print(f"[bold green]{len(ALL_AGENTS)} agents available:[/bold green]\n")
    for name, cls in ALL_AGENTS.items():
        agent = cls()
        console.print(
            f"  {'●' if agent.priority >= 1 else '○'} {name:25s} "
            f"priority={agent.priority} temp={agent.temperature}"
        )


@swarm_app.command("pipeline")
def show_pipeline(name: str = typer.Argument("plan-execute-audit")) -> None:
    """Show pipeline details."""
    from pravaha.swarm.pipeline import BUILTIN_PIPELINES

    p = BUILTIN_PIPELINES.get(name)
    if p:
        console.print(f"[bold]{p.name}[/bold]: {p.description}")
        console.print(f"  Workers: {', '.join(p.worker_steps)}")
        console.print(f"  Auditors: {', '.join(p.audit_steps)}")
    else:
        console.print(f"[red]Unknown pipeline: {name}[/red]")
