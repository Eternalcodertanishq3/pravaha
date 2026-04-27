"""Log Panel — Full-screen engine event and system log viewer.

Shows a large, scrollable view of the most recent engine events.
"""

from __future__ import annotations

from datetime import datetime

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static
from rich.text import Text

from pravaha.tui.dashboard import get_connector

CYAN = "bright_cyan"


class LogDetailStatic(Static):
    """Self-refreshing detailed log view."""

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh)

    def render(self) -> Text:
        conn = get_connector()
        # Get more events for the full screen view
        events = conn.recent_events(100)

        t = Text()
        t.append("╔══════════════════════════════════════════════════════════════╗\n", style=CYAN)
        t.append("║                  ", style=CYAN)
        t.append("SYSTEM EVENT LOG", style=f"bold {CYAN}")
        t.append("                           ║\n", style=CYAN)
        t.append("╚══════════════════════════════════════════════════════════════╝\n\n", style=CYAN)

        if not events:
            t.append(" (no events recorded yet)\n", style="grey42")
            return t

        for ts, msg, color in events:
            t.append(f" [{ts}]  ", style="dark_cyan")
            t.append(f"{msg}\n", style=color)

        return t


class LogScreen(Screen):
    """Full-screen event log panel."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll():
            yield LogDetailStatic()
        yield Footer()
