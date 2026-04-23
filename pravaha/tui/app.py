"""Pravaha TUI — Full Textual dashboard application."""

from __future__ import annotations

from typing import Any, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Static

from pravaha.tui.panels.header import HeaderPanel
from pravaha.tui.panels.chat_panel import ChatPanel
from pravaha.tui.panels.metrics_panel import MetricsPanel
from pravaha.tui.panels.queue_panel import QueuePanel
from pravaha.tui.panels.swarm_panel import SwarmPanel
from pravaha.tui.panels.audit_panel import AuditPanel
from pravaha.tui.panels.rag_panel import RAGPanel
from pravaha.tui.panels.log_panel import LogPanel


class PravahaTUI(App):
    """Full 8-panel Textual dashboard for Pravaha v3."""

    CSS_PATH = "pravaha.tcss"
    TITLE = "Pravaha v3"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle Dark"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(
        self,
        engine_config: Any = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.engine_config = engine_config
        self.host = host
        self.port = port

    def compose(self) -> ComposeResult:
        yield HeaderPanel()
        with Horizontal(id="main-content"):
            with Vertical(id="chat-area"):
                yield ChatPanel()
                yield Input(placeholder="Type a message...", id="chat-input")
            with Vertical(id="metrics-area"):
                yield MetricsPanel()
                yield QueuePanel()
        yield SwarmPanel()
        yield AuditPanel()
        with Horizontal(id="bottom-panels"):
            yield RAGPanel()
            yield LogPanel()
        yield Footer()

    def action_refresh(self) -> None:
        """Refresh all panels."""
        self.query_one(MetricsPanel).refresh_metrics()
        self.query_one(SwarmPanel).refresh_agents()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle chat input submission."""
        if event.value.strip():
            chat = self.query_one(ChatPanel)
            chat.add_message("user", event.value)
            event.input.value = ""
            chat.add_message("assistant", "[streaming response...]")
