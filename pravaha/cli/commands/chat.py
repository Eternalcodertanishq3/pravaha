"""pravaha chat — Interactive REPL with streaming."""

from __future__ import annotations

from typing import Optional

import typer

from pravaha.cli.ascii_art import console


def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    server: Optional[str] = typer.Option(None, "--server", "-s", help="Connect to running server URL"),
    branch: Optional[str] = typer.Option(None, "--branch", help="Continue from branch ID"),
) -> None:
    """Interactive chat REPL with streaming tokens.

    Commands: /help /model /stats /clear /branch /swarm /rag /audit /debug
    """
    console.print("[bold green]Pravaha Chat[/bold green] — type /help for commands\n")
    session_id = None

    import httpx

    base_url = server or "http://localhost:8000"

    while True:
        try:
            user_input = console.input("[bold cyan]you>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            cmd = user_input.strip().lower()
            if cmd == "/help":
                console.print(
                    "[green]/help[/green] — show commands\n"
                    "[green]/model[/green] — show current model\n"
                    "[green]/stats[/green] — show engine stats\n"
                    "[green]/clear[/green] — clear conversation\n"
                    "[green]/branch[/green] — fork conversation\n"
                    "[green]/swarm <task>[/green] — run swarm pipeline\n"
                    "[green]/rag <query>[/green] — query RAG store\n"
                    "[green]/audit on|off[/green] — toggle audit\n"
                    "[green]/debug[/green] — show debug info\n"
                    "[green]/quit[/green] — exit"
                )
            elif cmd == "/quit":
                break
            elif cmd == "/clear":
                session_id = None
                console.print("[dim]Conversation cleared.[/dim]")
            elif cmd == "/stats":
                try:
                    resp = httpx.get(f"{base_url}/health")
                    console.print(resp.json())
                except Exception:
                    console.print("[red]Could not connect to server.[/red]")
            elif cmd.startswith("/rag "):
                query = cmd[5:].strip()
                try:
                    resp = httpx.get(f"{base_url}/v1/rag/query", params={"query": query})
                    console.print(resp.json())
                except Exception as e:
                    console.print(f"[red]RAG error: {e}[/red]")
            elif cmd.startswith("/swarm "):
                task = cmd[7:].strip()
                try:
                    resp = httpx.post(f"{base_url}/v1/swarm/run", json={"prompt": task})
                    console.print(resp.json())
                except Exception as e:
                    console.print(f"[red]Swarm error: {e}[/red]")
            else:
                console.print(f"[dim]Unknown command: {cmd}[/dim]")
            continue

        # Stream chat completion
        console.print("[bold green]pravaha>[/bold green] ", end="")
        try:
            with httpx.stream(
                "POST", f"{base_url}/v1/chat/completions",
                json={"model": model or "default", "messages": [{"role": "user", "content": user_input}],
                       "stream": True, "session_id": session_id},
                timeout=60.0,
            ) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                console.print(content, end="")
                        except json.JSONDecodeError:
                            pass
            console.print()
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
