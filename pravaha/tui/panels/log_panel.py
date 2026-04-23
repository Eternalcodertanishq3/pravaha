"""Log Panel — Structured log viewer."""

from __future__ import annotations

from datetime import datetime

from textual.containers import VerticalScroll
from textual.widgets import Static


class LogPanel(VerticalScroll):
    """Structured log viewer with color-coded levels."""

    DEFAULT_CSS = """
    LogPanel { height: 8; border: solid #1a2a1a 1; padding: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._logs: list[str] = []

    def add_log(self, level: str, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        colors = {"INFO": "#4fc3f7", "WARN": "#ffb300", "ERROR": "#ef5350", "AUDIT": "#ff6b35"}
        color = colors.get(level, "#00e676")
        entry = f"[{color}]{ts} {level:5s}[/{color}] {message}"
        self._logs.append(entry)
        self.mount(Static(entry))
        if len(self._logs) > 100:
            self._logs.pop(0)
            if self.children:
                self.children[0].remove()
        self.scroll_end(animate=False)
