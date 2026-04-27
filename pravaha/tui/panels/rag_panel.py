"""RAG Panel — Full-screen document store and semantic cache view.

Shows stats on embedded documents, vector dimensions, semantic hit rates,
and ingestion progress.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static
from rich.text import Text

from pravaha.tui.dashboard import get_connector

CYAN = "bright_cyan"
GREEN = "bright_green"
YELLOW = "yellow"
MAGENTA = "bright_magenta"


class RAGDetailStatic(Static):
    """Self-refreshing RAG detail view."""

    def on_mount(self) -> None:
        self.set_interval(2.0, self.refresh)

    def render(self) -> Text:
        # Note: We simulate RAG stats since the engine doesn't expose it yet via get_stats()
        # but we build the UI for it.
        t = Text()
        t.append("╔══════════════════════════════════════════════════════════════╗\n", style=CYAN)
        t.append("║                  ", style=CYAN)
        t.append("RAG & SEMANTIC CACHE", style=f"bold {CYAN}")
        t.append("                       ║\n", style=CYAN)
        t.append("╚══════════════════════════════════════════════════════════════╝\n\n", style=CYAN)

        # ── Document Store ──
        t.append(" ◆ Document Store\n\n", style=f"bold {GREEN}")
        t.append(f"   Documents Indexed:    ", style="grey70"); t.append("0\n", style=GREEN)
        t.append(f"   Total Chunks:         ", style="grey70"); t.append("0\n", style=GREEN)
        t.append(f"   Embedding Model:      ", style="grey70"); t.append("n/a\n", style=GREEN)
        t.append(f"   Vector Dimensions:    ", style="grey70"); t.append("n/a\n\n", style=GREEN)

        # ── Semantic Cache ──
        t.append(" ◆ Semantic Cache Stats\n\n", style=f"bold {MAGENTA}")
        t.append(f"   Cache Hits:           ", style="grey70"); t.append("0\n", style=MAGENTA)
        t.append(f"   Cache Misses:         ", style="grey70"); t.append("0\n", style=MAGENTA)
        t.append(f"   Hit Rate:             ", style="grey70"); t.append("0.0%\n\n", style=MAGENTA)

        return t


class RAGScreen(Screen):
    """Full-screen RAG detail panel."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll():
            yield RAGDetailStatic()
        yield Footer()
