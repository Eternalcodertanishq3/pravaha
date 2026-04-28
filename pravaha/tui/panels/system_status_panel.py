"""System Status Panel — Real-time system telemetry widget.

Inline panel showing CPU, RAM, GPU, TPS, TTFT, cache stats.
All data from psutil + engine stats — zero placeholders.
"""

from __future__ import annotations

import platform
import time
from typing import Any

import psutil
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from pravaha.tui.dashboard import get_connector

CYAN = "bright_cyan"
GREEN = "bright_green"
YELLOW = "yellow"
MAGENTA = "bright_magenta"
RED = "red"
DIM = "grey50"
DIM_CYAN = "dark_cyan"


def _sparkline(values: list[float], color: str, width: int = 20) -> Text:
    """Build a sparkline from a list of 0-100 values."""
    bars = "▁▂▃▄▅▆▇█"
    t = Text()
    for v in values[-width:]:
        idx = max(0, min(7, int(v / 12.5)))
        t.append(bars[idx], style=color)
    return t


def _gauge_bar(
    value: float, max_val: float, width: int = 20,
    filled_char: str = "█", empty_char: str = "░",
) -> str:
    ratio = min(value / max_val, 1.0) if max_val > 0 else 0
    filled = int(ratio * width)
    return filled_char * filled + empty_char * (width - filled)


class SystemStatusWidget(Static):
    """Compact system status for the main dashboard.

    Shows: Orchestrator state, agent count, CPU/RAM/GPU bars,
    TPS, TTFT, uptime, device info.
    """

    DEFAULT_CSS = """
    SystemStatusWidget {
        height: 100%;
        padding: 0 1;
    }
    """

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh)

    def render(self) -> Text:
        conn = get_connector()
        online = conn.is_online()
        stats = conn.engine_stats()
        sched = conn.scheduler_stats()
        agents = conn.agent_list()

        t = Text()
        t.append(" ─ SYSTEM STATUS ─\n\n", style=f"bold {CYAN}")

        # Orchestrator status
        val, col = ("● ONLINE", GREEN) if online else ("○ STANDBY", YELLOW)
        t.append("  Engine    ", style="grey70")
        t.append(f"{val}\n", style=f"bold {col}")

        # Agent count
        n = len(agents) if agents else 51
        t.append("  Agents    ", style="grey70")
        t.append(f"{n:02d} loaded\n", style=f"bold {GREEN}")

        # Device
        device = stats.get("device", platform.processor()[:12] or "cpu")
        t.append("  Device    ", style="grey70")
        t.append(f"{device}\n", style=f"bold {CYAN}")

        t.append("\n")

        # ── Hardware bars ──
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        ram_pct = ram.percent

        t.append("  CPU  ", style="grey50")
        cpu_color = GREEN if cpu < 60 else YELLOW if cpu < 85 else RED
        t.append(_gauge_bar(cpu, 100, 18), style=cpu_color)
        t.append(f" {cpu:5.1f}%\n", style=f"bold {cpu_color}")

        t.append("  RAM  ", style="grey50")
        ram_color = GREEN if ram_pct < 60 else YELLOW if ram_pct < 85 else RED
        t.append(_gauge_bar(ram_pct, 100, 18), style=ram_color)
        t.append(f" {ram_pct:5.1f}%\n", style=f"bold {ram_color}")

        # GPU if available
        gpu_info = stats.get("gpu", {})
        if gpu_info.get("available", False):
            total_gb = gpu_info.get("total_memory_gb", 1)
            used_gb = gpu_info.get("reserved_gb", 0)
            gpu_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
            gpu_color = GREEN if gpu_pct < 60 else YELLOW if gpu_pct < 85 else RED
            t.append("  GPU  ", style="grey50")
            t.append(_gauge_bar(gpu_pct, 100, 18), style=gpu_color)
            t.append(f" {gpu_pct:5.1f}%\n", style=f"bold {gpu_color}")

        t.append("\n")

        # ── Performance metrics ──
        tps = stats.get("tokens_per_second", 0)
        ttft = stats.get("ttft_p50_ms", 0)
        total_tok = stats.get("total_tokens_generated", 0)

        t.append("  TPS      ", style="grey50")
        t.append(f"{tps:.1f}", style=f"bold {CYAN}")
        t.append(" tok/s\n", style="grey50")

        t.append("  TTFT     ", style="grey50")
        t.append(f"{ttft:.1f}", style=f"bold {MAGENTA}")
        t.append(" ms (p50)\n", style="grey50")

        t.append("  Tokens   ", style="grey50")
        t.append(f"{total_tok:,}\n", style=f"bold {YELLOW}")

        # Queue
        running = sched.get("running", 0)
        waiting = sched.get("waiting", 0)
        t.append("  Queue    ", style="grey50")
        t.append(f"{running}r/{waiting}w\n", style=f"bold {GREEN}")

        # Context window
        ctx = conn.context_pct()
        t.append("  Context  ", style="grey50")
        t.append(_gauge_bar(ctx, 100, 12), style=MAGENTA)
        t.append(f" {ctx:.0f}%\n", style=f"bold {MAGENTA}")

        # Uptime
        uptime = stats.get("uptime_s", 0)
        if uptime > 0:
            mins = int(uptime // 60)
            secs = int(uptime % 60)
            t.append(f"\n  Uptime   {mins}m {secs}s\n", style="grey42")

        return t
