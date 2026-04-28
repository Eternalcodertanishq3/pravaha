"""Audit Panel — Full-screen self-healing audit loop detail view.

Shows live audit state: active auditors, issues found by severity,
patches applied, iteration progress, and the full audit event log.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from pravaha.tui.dashboard import get_connector

CYAN = "bright_cyan"
GREEN = "bright_green"
YELLOW = "yellow"
RED = "red"
MAGENTA = "bright_magenta"
DIM = "grey50"

# Static/LLM auditor names for reference
STATIC_AUDITORS = [
    "syntax_audit", "type_safety", "security_audit",
    "injection_scanner", "crypto_audit", "secrets_scanner",
    "privilege_audit", "network_security", "compliance",
    "auth_audit", "dependency_audit", "api_security",
]
LLM_AUDITORS = [
    "logic_flaw", "hallucination_hunter", "consistency_guard",
    "edge_case_hunter", "performance_profiler", "output_verifier",
]


class AuditDetailStatic(Static):
    """Self-refreshing audit detail widget."""

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh)

    def render(self) -> Text:
        conn = get_connector()
        agents = conn.agent_list()

        t = Text()
        t.append("╔══════════════════════════════════════════════════════════════╗\n", style=CYAN)
        t.append("║            ", style=CYAN)
        t.append("SELF-HEALING AUDIT LOOP", style=f"bold {CYAN}")
        t.append("                           ║\n", style=CYAN)
        t.append("╚══════════════════════════════════════════════════════════════╝\n\n", style=CYAN)

        # ── Phase A: Static Auditors ──
        t.append(" ◆ Phase A — Static Auditors (instant regex scans)\n\n", style=f"bold {GREEN}")
        for i, name in enumerate(STATIC_AUDITORS):
            # check real agent data
            stats = _find_agent(agents, name)
            calls = stats.get("total_calls", 0) if stats else 0
            dot = "●" if calls > 0 else "○"
            col = GREEN if calls > 0 else "grey42"
            t.append(f"   {dot} ", style=col)
            t.append(f"{name:22s}", style="grey70")
            t.append(f"  calls={calls}\n", style=DIM)

        t.append("\n")

        # ── Phase B: LLM Auditors ──
        t.append(" ◆ Phase B — LLM Auditors (deep analysis)\n\n", style=f"bold {YELLOW}")
        for name in LLM_AUDITORS:
            stats = _find_agent(agents, name)
            calls = stats.get("total_calls", 0) if stats else 0
            tokens = stats.get("total_tokens", 0) if stats else 0
            dot = "●" if calls > 0 else "○"
            col = YELLOW if calls > 0 else "grey42"
            t.append(f"   {dot} ", style=col)
            t.append(f"{name:22s}", style="grey70")
            t.append(f"  calls={calls}  tokens={tokens}\n", style=DIM)

        t.append("\n")

        # ── Confidence Thresholds ──
        t.append(" ◆ Progressive Confidence Thresholds\n\n", style=f"bold {MAGENTA}")
        thresholds = {1: 90.0, 2: 80.0, 3: 70.0}
        for iteration, threshold in thresholds.items():
            t.append(f"   Iteration {iteration}: ", style="grey70")
            bar_len = int(threshold / 5)
            t.append("█" * bar_len, style=GREEN)
            t.append("░" * (20 - bar_len), style="grey27")
            t.append(f"  {threshold:.0f}%\n", style=f"bold {GREEN}")

        t.append("\n")

        # ── Recent audit events from event bus ──
        t.append(" ◆ Recent Audit Events\n\n", style=f"bold {CYAN}")
        events = conn.recent_events(12)
        audit_events = [e for e in events if "AUDIT" in e[1].upper() or "PATCH" in e[1].upper()]
        if audit_events:
            for ts, msg, col in audit_events[:8]:
                t.append(f"   {ts}  ", style="dark_cyan")
                t.append(f"{msg}\n", style=col)
        else:
            t.append("   (no audit events yet — run a pipeline to trigger)\n", style="grey42")

        return t


class AuditScreen(Screen):
    """Full-screen audit loop detail panel."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll():
            yield AuditDetailStatic()
        yield Footer()


def _find_agent(agents: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for a in agents:
        if a.get("name", "") == name:
            return a
    return None
