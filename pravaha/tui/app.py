"""Pravaha TUI — Terminal-native Textual application.

Launch with:  python -m pravaha.tui.app
              pravaha tui              (if CLI is wired)

The dashboard opens *in the terminal*, not in a browser.
When an ``AsyncPravahaEngine`` is passed the dashboard pulls live
telemetry; otherwise it starts in demo/standby mode with real
system metrics (CPU, RAM via psutil).
"""

from __future__ import annotations

import sys
from typing import Any

from textual.app import App, ComposeResult

from pravaha.tui.dashboard import PravahaDashboard, get_connector


class PravahaTUI(App):
    """Cyberpunk terminal dashboard for the Pravaha inference framework.

    This runs entirely in the terminal — no browser window.
    """

    CSS_PATH = "pravaha.tcss"
    TITLE = "Pravaha v3.2"
    SUB_TITLE = "AI Agentic Orchestration Framework"

    SCREENS = {} # Registered in on_mount

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Dark/Light"),
        ("1", "set_state('idle')", "Idle"),
        ("2", "set_state('thinking')", "Think"),
        ("3", "set_state('working')", "Work"),
        ("4", "set_state('success')", "Done"),
        ("5", "set_state('error')", "Error"),
        ("f1", "show_screen('chat')", "F1 Chat"),
        ("f2", "show_screen('swarm')", "F2 Swarm"),
        ("f3", "show_screen('audit')", "F3 Audit"),
        ("f4", "show_screen('metrics')", "F4 Metrics"),
        ("f5", "show_screen('queue')", "F5 Queue"),
        ("f6", "show_screen('rag')", "F6 RAG"),
        ("f7", "show_screen('log')", "F7 Log"),
    ]

    def __init__(
        self,
        engine: Any = None,
        orchestrator: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        conn = get_connector()
        if engine is not None:
            conn.attach_engine(engine)
        if orchestrator is not None:
            conn.attach_orchestrator(orchestrator)

    def compose(self) -> ComposeResult:
        yield PravahaDashboard()

    def on_mount(self) -> None:
        """Log startup event and install screens."""
        from pravaha.tui.panels.chat_panel import ChatScreen
        from pravaha.tui.panels.swarm_panel import SwarmScreen
        from pravaha.tui.panels.audit_panel import AuditScreen
        from pravaha.tui.panels.metrics_panel import MetricsScreen
        from pravaha.tui.panels.queue_panel import QueueScreen
        from pravaha.tui.panels.rag_panel import RAGScreen
        from pravaha.tui.panels.log_panel import LogScreen

        self.install_screen(ChatScreen(), name="chat")
        self.install_screen(SwarmScreen(), name="swarm")
        self.install_screen(AuditScreen(), name="audit")
        self.install_screen(MetricsScreen(), name="metrics")
        self.install_screen(QueueScreen(), name="queue")
        self.install_screen(RAGScreen(), name="rag")
        self.install_screen(LogScreen(), name="log")

        conn = get_connector()
        conn.push_event("Dashboard launched (terminal mode)", "bright_green")

    # ── keybindings ─────────────────────────────────────────────
    def action_show_screen(self, screen_name: str) -> None:
        """Push a detail screen."""
        self.push_screen(screen_name)
    def action_set_state(self, state: str) -> None:
        """Cycle the centre avatar through states (keys 1-5)."""
        from pravaha.tui.avatar.pravaha_avatar import PravahaAvatar

        try:
            avatar = self.query_one(PravahaAvatar)
            avatar.set_state(state)
            get_connector().push_event(f"Avatar -> {state}", "bright_cyan")
        except Exception:
            pass

    # ── public helpers for external wiring ──────────────────────
    def attach_engine(self, engine: Any) -> None:
        """Attach a live engine after the app has started."""
        get_connector().attach_engine(engine)

    def attach_orchestrator(self, orchestrator: Any) -> None:
        """Attach a live orchestrator after the app has started."""
        get_connector().attach_orchestrator(orchestrator)


def main() -> None:
    """Launch the Pravaha TUI dashboard in the terminal."""
    # Ensure proper UTF-8 on Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    app = PravahaTUI()
    app.run()


if __name__ == "__main__":
    main()
