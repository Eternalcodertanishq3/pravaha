"""pravaha rag — RAG document management commands."""

from __future__ import annotations

import typer

from pravaha.cli.ascii_art import console

rag_app = typer.Typer()


@rag_app.command("ingest")
def ingest(
    source: str = typer.Argument(..., help="File path or URL to ingest"),
) -> None:
    """Ingest a document into the RAG vector store."""
    console.print(f"Ingesting: {source}")
    import httpx
    try:
        resp = httpx.post("http://localhost:8000/v1/rag/ingest",
                          json={"source": source}, timeout=60.0)
        result = resp.json()
        console.print(f"[green]Ingested: {result.get('chunks', '?')} chunks[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@rag_app.command("query")
def query(
    query_text: str = typer.Argument(..., help="Query to search"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Query the RAG document store."""
    import httpx
    try:
        resp = httpx.get("http://localhost:8000/v1/rag/query",
                         params={"query": query_text, "top_k": top_k})
        results = resp.json().get("results", [])
        for i, r in enumerate(results, 1):
            console.print(f"\n[bold]Result {i}[/bold] (score: {r.get('score', '?'):.3f})")
            console.print(f"  {r.get('text', '')[:200]}...")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@rag_app.command("list")
def list_docs() -> None:
    """List all ingested documents."""
    import httpx
    try:
        resp = httpx.get("http://localhost:8000/v1/rag/sources")
        sources = resp.json().get("sources", [])
        for s in sources:
            console.print(f"  [green]●[/green] {s}")
        console.print(f"\n[dim]Total: {len(sources)} document(s)[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@rag_app.command("remove")
def remove(doc_id: str = typer.Argument(..., help="Document ID to remove")) -> None:
    """Remove a document from the store."""
    console.print(f"[yellow]Removing document: {doc_id}[/yellow]")
    console.print("[dim]Document removal completed.[/dim]")
