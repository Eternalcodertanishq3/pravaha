"""Pravaha v3.3 TUI — Full cyberpunk dashboard.

All panels visible at once in a 3-column layout:
- Left:   SystemStatus, FlowVisualizer
- Center: Avatar, Chat, LiveEvents
- Right:  AgentGrid, Metrics

Bottom: MetricsBar, LiveEvents
Footer: Wave animation + keybindings

F-keys still open detail screens for deep-dive views.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pravaha.config.engine_config import EngineConfig
    from pravaha.engine.async_engine import AsyncPravahaEngine

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical

# Dashboard connector (singleton)
from pravaha.tui.dashboard import (
    CenterFlowPanel,
    FlowVisualizerPanel,
    LiveEventsPanel,
    MetricsBar,
    PravahaDashboard,
    PravahaFooter,
    get_connector,
)

# New v3.3 panels
from pravaha.tui.panels.agent_grid_panel import AgentGridWidget

# Detail screens (F-key overlays)
from pravaha.tui.panels.audit_panel import AuditScreen
from pravaha.tui.panels.chat_panel import ChatScreen
from pravaha.tui.panels.header import PravahaHeader
from pravaha.tui.panels.log_panel import LogScreen
from pravaha.tui.panels.metrics_panel import MetricsScreen
from pravaha.tui.panels.queue_panel import QueueScreen
from pravaha.tui.panels.rag_panel import RAGScreen
from pravaha.tui.panels.swarm_panel import SwarmScreen
from pravaha.tui.panels.system_status_panel import SystemStatusWidget
from pravaha.tui.panels.wave_panel import WaveWidget

# CSS path
CSS_PATH = Path(__file__).parent / "pravaha.tcss"


class PravahaTUI(App):
    """Pravaha v3.3 — Cyberpunk TUI Dashboard.

    Full 3-column layout with all panels visible at once.
    F-keys open detail screens for deep-dive views.
    """

    TITLE = "PRAVAHA v3.3"
    SUB_TITLE = "AI Agentic Orchestration Framework"
    CSS_PATH = CSS_PATH

    def __init__(
        self,
        engine_config: EngineConfig | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        engine: AsyncPravahaEngine | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.engine_config = engine_config
        self.host = host
        self.port = port
        self.engine = engine

    BINDINGS = [
        Binding("f1", "push_screen('chat')", "Chat", show=True),
        Binding("f2", "push_screen('swarm')", "Swarm", show=True),
        Binding("f3", "push_screen('audit')", "Audit", show=True),
        Binding("f4", "push_screen('metrics')", "Metrics", show=True),
        Binding("f5", "push_screen('queue')", "Queue", show=True),
        Binding("f6", "push_screen('rag')", "RAG", show=True),
        Binding("f7", "push_screen('log')", "Log", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    SCREENS = {
        "chat": ChatScreen,
        "swarm": SwarmScreen,
        "audit": AuditScreen,
        "metrics": MetricsScreen,
        "queue": QueueScreen,
        "rag": RAGScreen,
        "log": LogScreen,
    }

    def compose(self) -> ComposeResult:
        # ── Header ──
        yield PravahaHeader(id="dash-header")

        # ── Main 3-column layout ──
        with Horizontal(id="main-row"):
            # Left column: System Status + Flow Visualizer
            with Vertical(id="left-col"):
                yield SystemStatusWidget(id="sys-status")
                yield FlowVisualizerPanel(id="flow-vis")

            # Center column: Avatar + Events
            with Vertical(id="center-col"):
                yield CenterFlowPanel(id="center-avatar")
                yield LiveEventsPanel(id="center-events")

            # Right column: Agent Grid + Metrics
            with Vertical(id="right-col"):
                yield AgentGridWidget(id="agent-grid")
                yield MetricsBar(id="right-metrics")

        # ── Wave animation ──
        yield WaveWidget(id="wave-panel")

        # ── Footer ──
        yield PravahaFooter(id="dash-footer")

    def on_mount(self) -> None:
        """Push boot events when dashboard mounts."""
        import psutil

        conn = get_connector()
        conn.push_event("Pravaha v3.3 TUI initialized", "bright_cyan")
        conn.push_event(
            f"PID {os.getpid()} │ {psutil.cpu_count(logical=True)} cores │ "
            f"{psutil.virtual_memory().total // (1024**3)} GB RAM",
            "grey70",
        )

        # Sample system metrics immediately
        conn.sample_system()


def run_tui(engine: Any = None, orchestrator: Any = None) -> None:
    """Launch the Pravaha TUI, optionally wired to a live engine."""
    conn = get_connector()
    if engine is not None:
        conn.attach_engine(engine)
    if orchestrator is not None:
        conn.attach_orchestrator(orchestrator)

    app = PravahaTUI()
    app.run()


if __name__ == "__main__":
    run_tui()
