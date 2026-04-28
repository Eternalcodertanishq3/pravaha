"""Queue Panel — Full-screen request queue and scheduler visualization.

Shows active, waiting, and swapped requests inside the continuous batching scheduler.
"""

from __future__ import annotations

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
RED = "red"


class QueueDetailStatic(Static):
    """Self-refreshing queue detail view."""

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh)

    def render(self) -> Text:
        conn = get_connector()
        sched = conn.scheduler_stats()

        t = Text()
        t.append("╔══════════════════════════════════════════════════════════════╗\n", style=CYAN)
        t.append("║                  ", style=CYAN)
        t.append("CONTINUOUS BATCHING QUEUE", style=f"bold {CYAN}")
        t.append("                  ║\n", style=CYAN)
        t.append("╚══════════════════════════════════════════════════════════════╝\n\n", style=CYAN)

        if not sched:
            t.append(" Engine/Scheduler not attached. Showing STANDBY.\n", style=YELLOW)
            return t

        waiting = sched.get("waiting", 0)
        running = sched.get("running", 0)
        swapped = sched.get("swapped", 0)
        finished = sched.get("finished", 0)
        reqs = sched.get("requests", [])

        # ── Queue Summary ──
        t.append(" ◆ Queue Status\n\n", style=f"bold {CYAN}")
        t.append(f"   Running:  {running:3d}\n", style=GREEN)
        t.append(f"   Waiting:  {waiting:3d}\n", style=YELLOW)
        t.append(f"   Swapped:  {swapped:3d}\n", style=MAGENTA)
        t.append(f"   Finished: {finished:3d}\n\n", style="grey50")

        # ── Active Requests ──
        t.append(" ◆ Active Requests In-Flight\n\n", style=f"bold {GREEN}")

        if not reqs:
            t.append("   (No active requests)\n", style="grey50")
        else:
            t.append(f"   {'Request ID':12s} {'Tokens':>8s} {'Progress':>10s}  {'Bar':30s}\n", style="bold grey70")
            t.append(f"   {'─'*12} {'─'*8} {'─'*10}  {'─'*30}\n", style="grey37")

            for r in reqs:
                rid = str(r.get("id", "?"))[:12]
                toks = r.get("tokens", 0)
                prog = r.get("progress", 0.0)

                t.append(f"   {rid:12s} ", style=CYAN)
                t.append(f"{toks:>8d} ", style=YELLOW)
                t.append(f"{prog:>9.1f}%  ", style=GREEN)
                t.append(self._gauge(prog, 100, 30) + "\n", style=GREEN)

        return t

    @staticmethod
    def _gauge(value: float, max_val: float, width: int = 30) -> str:
        ratio = min(value / max_val, 1.0) if max_val > 0 else 0
        filled = int(ratio * width)
        return "█" * filled + "░" * (width - filled)


class QueueScreen(Screen):
    """Full-screen queue detail panel."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll():
            yield QueueDetailStatic()
        yield Footer()
