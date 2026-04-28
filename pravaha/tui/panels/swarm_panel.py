"""Swarm Panel — Full-screen 51-agent swarm grid detail view.

Shows every loaded agent with real stats: calls, tokens, duration,
priority, tool/memory attachment, plus a colour-coded activity grid.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from pravaha.tui.dashboard import get_connector

CYAN = "bright_cyan"
GREEN = "bright_green"
YELLOW = "yellow"
MAGENTA = "bright_magenta"
DIM = "grey50"


class SwarmDetailStatic(Static):
    """Self-refreshing swarm grid with real agent data."""

    def on_mount(self) -> None:
        self.set_interval(2.0, self.refresh)

    def render(self) -> Text:
        conn = get_connector()
        agents = conn.agent_list()

        t = Text()
        t.append("╔══════════════════════════════════════════════════════════════╗\n", style=CYAN)
        t.append("║                  ", style=CYAN)
        t.append("SWARM AGENT GRID", style=f"bold {CYAN}")
        t.append("                          ║\n", style=CYAN)
        t.append("╚══════════════════════════════════════════════════════════════╝\n\n", style=CYAN)

        if not agents:
            t.append(" No orchestrator attached — showing default agent roster\n\n", style=YELLOW)
            defaults = [
                "planner", "coder", "critic", "validator", "summarizer",
                "expander", "translator", "reasoner", "merger", "router",
                "memory_agent", "tool_agent", "judge", "refiner",
                "classifier", "extractor", "narrator", "ensemble",
                "debugger", "researcher", "syntax_audit", "logic_flaw",
                "hallucination_hunter", "security_audit", "performance_profiler",
                "consistency_guard", "type_safety", "edge_case_hunter",
                "test_writer", "output_verifier", "self_healer", "patch_applier",
            ]
            # 8-column grid
            for row_start in range(0, len(defaults), 8):
                row = defaults[row_start:row_start + 8]
                for name in row:
                    t.append(f" [{name[:5]:5s}○] ", style="grey42")
                t.append("\n")
            return t

        # ── Activity Grid ──
        t.append(" ◆ Activity Grid\n\n", style=f"bold {GREEN}")
        for row_start in range(0, len(agents), 8):
            row = agents[row_start:row_start + 8]
            for a in row:
                name = a.get("name", "?")[:5]
                calls = a.get("total_calls", 0)
                if calls > 10:
                    sym, col = "●", GREEN
                elif calls > 0:
                    sym, col = "◆", YELLOW
                else:
                    sym, col = "○", "grey42"
                t.append(f" [{name:5s}{sym}] ", style=col)
            t.append("\n")

        t.append(f"\n Total agents: {len(agents)}\n\n", style=DIM)

        # ── Detailed Stats Table ──
        t.append(" ◆ Agent Statistics\n\n", style=f"bold {CYAN}")
        t.append(f" {'Name':20s} {'Priority':>8s} {'Calls':>6s} {'Tokens':>8s} {'Duration':>10s} {'Tools':>6s} {'Mem':>4s}\n", style="bold grey70")
        t.append(f" {'─'*20} {'─'*8} {'─'*6} {'─'*8} {'─'*10} {'─'*6} {'─'*4}\n", style="grey37")

        for a in sorted(agents, key=lambda x: x.get("total_calls", 0), reverse=True):
            name = a.get("name", "?")[:20]
            priority = a.get("priority", 0)
            calls = a.get("total_calls", 0)
            tokens = a.get("total_tokens", 0)
            dur = a.get("total_duration_ms", 0)
            tools = "✓" if a.get("has_tools", False) else "·"
            mem = "✓" if a.get("has_memory", False) else "·"

            name_col = GREEN if calls > 0 else "grey50"
            t.append(f" {name:20s}", style=name_col)
            t.append(f" {priority:>8d}", style=DIM)
            t.append(f" {calls:>6d}", style=YELLOW if calls > 0 else DIM)
            t.append(f" {tokens:>8d}", style=CYAN if tokens > 0 else DIM)
            dur_s = f"{dur:.0f}ms" if dur > 0 else "—"
            t.append(f" {dur_s:>10s}", style=DIM)
            t.append(f" {tools:>6s}", style=GREEN if tools == "✓" else DIM)
            t.append(f" {mem:>4s}\n", style=GREEN if mem == "✓" else DIM)

        return t


class SwarmScreen(Screen):
    """Full-screen swarm agent detail panel."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll():
            yield SwarmDetailStatic()
        yield Footer()
