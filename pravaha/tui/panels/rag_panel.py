"""RAG Panel — Document store viewer."""

from __future__ import annotations

from textual.widgets import Static


class RAGPanel(Static):
    """RAG document store status display."""

    DEFAULT_CSS = """
    RAGPanel { height: 3; border: solid #1a2a1a 1; padding: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.doc_count = 0
        self.chunk_count = 0
        self.last_query_ago = "never"

    def render(self) -> str:
        return (
            f"RAG: {self.doc_count} docs  ·  {self.chunk_count} chunks  ·  "
            f"last query: {self.last_query_ago}"
        )
