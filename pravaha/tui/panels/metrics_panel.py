"""Metrics Panel — Full-screen detailed metrics and performance view.

Shows expanded system metrics, memory blocks, tokenizer throughput,
and engine statistics.
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
DIM = "grey50"


class MetricsDetailStatic(Static):
    """Self-refreshing detailed metrics view."""

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh)

    def render(self) -> Text:
        conn = get_connector()
        stats = conn.engine_stats()
        block_stats = conn.block_stats()

        t = Text()
        t.append("╔══════════════════════════════════════════════════════════════╗\n", style=CYAN)
        t.append("║                  ", style=CYAN)
        t.append("SYSTEM & ENGINE METRICS", style=f"bold {CYAN}")
        t.append("                     ║\n", style=CYAN)
        t.append("╚══════════════════════════════════════════════════════════════╝\n\n", style=CYAN)

        # ── Hardware ──
        t.append(" ◆ Hardware Utilization\n\n", style=f"bold {GREEN}")
        cpu = conn.cpu_pct()
        mem = conn.mem_pct()

        t.append("   CPU Usage:    ", style="grey70")
        t.append(f"{self._gauge(cpu, 100, 40)}  {cpu:5.1f}%\n", style=GREEN)

        t.append("   Memory Usage: ", style="grey70")
        t.append(f"{self._gauge(mem, 100, 40)}  {mem:5.1f}%\n\n", style=CYAN)

        # ── Engine Core ──
        t.append(" ◆ Engine Statistics\n\n", style=f"bold {YELLOW}")
        model = stats.get("model", "unknown")
        device = stats.get("device", "unknown")
        quant = stats.get("quantization", "none")
        total_req = stats.get("total_requests", 0)
        total_tok = stats.get("total_tokens_generated", 0)

        t.append("   Model:        ", style="grey70")
        t.append(f"{model}\n", style=YELLOW)
        t.append("   Device:       ", style="grey70")
        t.append(f"{device}\n", style=YELLOW)
        t.append("   Quantization: ", style="grey70")
        t.append(f"{quant}\n", style=YELLOW)
        t.append("   Total Reqs:   ", style="grey70")
        t.append(f"{total_req:,}\n", style=YELLOW)
        t.append("   Total Tokens: ", style="grey70")
        t.append(f"{total_tok:,}\n\n", style=YELLOW)

        # ── KV Cache & Blocks ──
        t.append(" ◆ KV Cache & Memory Blocks\n\n", style=f"bold {MAGENTA}")
        if block_stats:
            total_blocks = block_stats.get("total_blocks", 1)
            used_blocks = block_stats.get("used_blocks", 0)
            free_blocks = block_stats.get("free_blocks", 0)
            usage_pct = block_stats.get("usage_pct", 0.0)

            t.append("   Block Usage:  ", style="grey70")
            t.append(f"{self._gauge(usage_pct, 100, 40)}  {usage_pct:5.1f}%\n", style=MAGENTA)
            t.append("   Total Blocks: ", style="grey70")
            t.append(f"{total_blocks:,}\n", style=MAGENTA)
            t.append("   Used Blocks:  ", style="grey70")
            t.append(f"{used_blocks:,}\n", style=MAGENTA)
            t.append("   Free Blocks:  ", style="grey70")
            t.append(f"{free_blocks:,}\n", style=MAGENTA)
        else:
            t.append("   (Block statistics unavailable)\n", style=DIM)

        return t

    @staticmethod
    def _gauge(value: float, max_val: float, width: int = 40) -> str:
        ratio = min(value / max_val, 1.0) if max_val > 0 else 0
        filled = int(ratio * width)
        return "█" * filled + "░" * (width - filled)


class MetricsScreen(Screen):
    """Full-screen metrics detail panel."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll():
            yield MetricsDetailStatic()
        yield Footer()
