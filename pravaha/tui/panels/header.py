"""Header Panel — Live clock, device detection, uptime, model name.

The crown of the TUI — a cyberpunk header with real-time system info
and animated decorative elements.
"""

from __future__ import annotations

import platform
import time
from datetime import datetime

import psutil
from rich.text import Text
from textual.widgets import Static

from pravaha.tui.dashboard import get_connector

CYAN = "bright_cyan"
DIM_CYAN = "dark_cyan"
GREEN = "bright_green"
DIM = "grey50"


class PravahaHeader(Static):
    """Animated header with real-time clock, model, and hardware info."""

    DEFAULT_CSS = """
    PravahaHeader {
        height: 4;
        content-align: center middle;
        text-align: center;
        padding: 0 1;
    }
    """

    def on_mount(self) -> None:
        self._start_time = time.time()
        self.set_interval(1.0, self.refresh)

    def render(self) -> Text:
        conn = get_connector()
        stats = conn.engine_stats()
        now = datetime.now()
        clock = now.strftime("%H:%M:%S")

        # Animated decorative tick
        tick_chars = ["~", "≈", "∿", "≈"]
        tick = tick_chars[int(time.time()) % len(tick_chars)]

        t = Text()

        # Line 1: Brand
        t.append(f"  {tick}{tick}  ", style=DIM_CYAN)
        t.append("P  R  A  V  A  H  A", style=f"bold {CYAN}")
        t.append(f"  {tick}{tick}   ", style=DIM_CYAN)

        # Clock
        t.append(f"⏱ {clock}", style=f"bold {GREEN}")
        t.append("\n")

        # Line 2: Separator
        t.append("  ~ ── ── ── ── ── ── ── ── ── ── ── ── ── ~", style=DIM_CYAN)
        t.append("\n")

        # Line 3: System info
        t.append("  ", style="")

        # Model
        model = stats.get("model", "no model")
        if len(str(model)) > 25:
            model = str(model).split("/")[-1][:25]
        t.append(f"Model: {model}", style="grey70")
        t.append("  │  ", style="grey37")

        # Device
        device = stats.get("device", "cpu")
        gpu_info = stats.get("gpu", {})
        if gpu_info.get("available"):
            gpu_name = gpu_info.get("name", "GPU")[:15]
            t.append(f"⊕ {gpu_name}", style=f"bold {GREEN}")
        else:
            cpu_name = platform.processor()[:15] or "CPU"
            t.append(f"⊕ {cpu_name}", style="grey70")
        t.append("  │  ", style="grey37")

        # RAM
        ram = psutil.virtual_memory()
        t.append(f"RAM: {ram.used // (1024**3)}/{ram.total // (1024**3)}GB", style="grey70")
        t.append("  │  ", style="grey37")

        # Cores
        cores = psutil.cpu_count(logical=True)
        t.append(f"Cores: {cores}", style="grey70")

        return t
