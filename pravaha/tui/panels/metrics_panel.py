"""Metrics Panel — GPU/CPU/throughput ASCII gauges."""

from __future__ import annotations

from textual.widgets import Static


class MetricsPanel(Static):
    """Real-time metrics with ASCII gauge visualizations."""

    DEFAULT_CSS = """
    MetricsPanel { height: auto; border: solid #1a2a1a 1; padding: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.throughput = 0.0
        self.ttft_p50 = 0.0
        self.ttft_p99 = 0.0
        self.vram_used = 0.0
        self.vram_total = 24.0
        self.requests_served = 0
        self.total_tokens = 0

    def render(self) -> str:
        vram_bar = self._gauge(self.vram_used, self.vram_total, 20)
        tput_bar = self._gauge(min(self.throughput, 1000), 1000, 20)
        return (
            f"Throughput: {tput_bar}  {self.throughput:.0f} tok/s\n"
            f"TTFT p50: {self.ttft_p50:.0f}ms  p99: {self.ttft_p99:.0f}ms\n"
            f"VRAM: {vram_bar}  {self.vram_used:.1f}/{self.vram_total:.1f} GB\n"
            f"Served: {self.requests_served:,}  Tokens: {self.total_tokens:,}"
        )

    @staticmethod
    def _gauge(value: float, max_val: float, width: int = 20) -> str:
        ratio = min(value / max_val, 1.0) if max_val > 0 else 0
        filled = int(ratio * width)
        return "█" * filled + "░" * (width - filled)

    def refresh_metrics(self) -> None:
        self.refresh()
