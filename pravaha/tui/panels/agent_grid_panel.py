"""Agent Grid Panel — Live 51-agent grid with status badges.

Shows all agents in an 8-column grid with color-coded activity
indicators that blink for active agents.
"""

from __future__ import annotations

import time
from typing import Any

from rich.text import Text
from textual.widgets import Static

from pravaha.tui.dashboard import get_connector

CYAN = "bright_cyan"
GREEN = "bright_green"
YELLOW = "yellow"
MAGENTA = "bright_magenta"
RED = "red"
DIM = "grey42"

# Full 51-agent roster for display when orchestrator not attached
ALL_AGENTS = [
    # Workers (20)
    "planner", "researcher", "coder", "debugger", "reasoning",
    "critic", "refiner", "summarizer", "narrator", "expander",
    "extractor", "classifier", "router", "translator", "ensemble",
    "merger", "judge", "memory", "tool", "validator",
    # Auditors (12+1)
    "syntax_aud", "type_safe", "security", "logic_flaw",
    "consist_gd", "halluc_hnt", "edge_case", "perf_prof",
    "out_verify", "self_refl", "test_gen", "patch_app", "regr_guard",
    # Security (10)
    "sec_audit", "inject_sc", "auth_aud", "crypto_aud", "dep_audit",
    "secrets_sc", "net_sec", "priv_aud", "api_sec", "compliance",
    # Design (9)
    "ui_design", "comp_build", "layout_ds", "style_ds",
    "a11y_audit", "ux_review", "design_cr", "proto_bld", "ds_system",
]

# Agent category color mapping
CATEGORY_COLORS = {
    "worker": CYAN,
    "auditor": YELLOW,
    "security": RED,
    "design": MAGENTA,
}


def _agent_category(idx: int) -> str:
    if idx < 20:
        return "worker"
    elif idx < 33:
        return "auditor"
    elif idx < 43:
        return "security"
    else:
        return "design"


class AgentGridWidget(Static):
    """Live 51-agent grid with blinking activity badges."""

    DEFAULT_CSS = """
    AgentGridWidget {
        height: 100%;
        padding: 0 1;
    }
    """

    def on_mount(self) -> None:
        self._tick = 0
        self.set_interval(1.5, self._animate)

    def _animate(self) -> None:
        self._tick += 1
        self.refresh()

    def render(self) -> Text:
        conn = get_connector()
        agents = conn.agent_list()
        blink = self._tick % 2 == 0

        t = Text()
        t.append(" ─ AGENT GRID ─\n", style=f"bold {CYAN}")
        t.append(f" {len(ALL_AGENTS)} agents  │  ", style="grey42")
        t.append("● active  ", style=GREEN)
        t.append("◆ ready  ", style=YELLOW)
        t.append("○ idle\n\n", style=DIM)

        # Build lookup from real agent data
        agent_calls: dict[str, int] = {}
        if agents:
            for a in agents:
                name = a.get("name", "")
                agent_calls[name] = a.get("total_calls", 0)

        cols = 6
        for i, name in enumerate(ALL_AGENTS):
            calls = agent_calls.get(name, 0)
            cat = _agent_category(i)
            cat_color = CATEGORY_COLORS[cat]

            if calls > 10:
                sym = "●" if blink else "◉"
                col = GREEN
            elif calls > 0:
                sym = "◆"
                col = YELLOW
            else:
                sym = "○"
                col = DIM

            display = name[:6]
            t.append(f" {sym}", style=col)
            t.append(f"{display:6s}", style=cat_color if calls > 0 else "grey37")

            if (i + 1) % cols == 0:
                t.append("\n")

        # Summary stats
        if agents:
            active = sum(1 for a in agents if a.get("total_calls", 0) > 0)
            total_calls = sum(a.get("total_calls", 0) for a in agents)
            t.append(f"\n active: {active}  calls: {total_calls}", style="grey50")

        return t
