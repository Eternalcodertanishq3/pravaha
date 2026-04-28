"""Wave Panel — Animated ASCII sine wave for dashboard footer.

Runs at ~10fps with gradient coloring. Pure aesthetic element
that makes the TUI feel alive and cyberpunk.
"""

from __future__ import annotations

import math
import time

from rich.text import Text
from textual.widgets import Static

# Gradient colors for the wave (bottom to top)
WAVE_GRADIENT = [
    "grey27", "grey37", "grey42", "dark_cyan",
    "cyan3", "bright_cyan", "bright_cyan",
]


class WaveWidget(Static):
    """Animated ASCII sine wave with gradient coloring."""

    DEFAULT_CSS = """
    WaveWidget {
        height: 5;
        padding: 0;
    }
    """

    def on_mount(self) -> None:
        self._tick = 0
        self.set_interval(0.1, self._animate)

    def _animate(self) -> None:
        self._tick += 1
        self.refresh()

    def render(self) -> Text:
        width = 80
        height = 4
        t = Text()

        phase = self._tick * 0.15
        for row in range(height):
            for col in range(width):
                # Multi-layer sine wave
                x = col / width * math.pi * 4
                y1 = math.sin(x + phase) * 0.4
                y2 = math.sin(x * 1.5 + phase * 0.7) * 0.3
                y3 = math.sin(x * 0.5 + phase * 1.3) * 0.3
                combined = y1 + y2 + y3

                # Map to row
                row_norm = 1.0 - (row / (height - 1))  # 1.0 at top, 0.0 at bottom
                wave_norm = (combined + 1.0) / 2.0     # 0.0 to 1.0

                # Character selection based on wave intensity at this position
                if wave_norm > row_norm + 0.1:
                    char = "█"
                    color_idx = min(len(WAVE_GRADIENT) - 1, int(wave_norm * (len(WAVE_GRADIENT) - 1)))
                elif wave_norm > row_norm - 0.05:
                    char = "▓"
                    color_idx = min(len(WAVE_GRADIENT) - 1, max(0, int(wave_norm * (len(WAVE_GRADIENT) - 1)) - 1))
                elif wave_norm > row_norm - 0.15:
                    char = "░"
                    color_idx = max(0, min(len(WAVE_GRADIENT) - 1, int(wave_norm * (len(WAVE_GRADIENT) - 1)) - 2))
                else:
                    char = " "
                    color_idx = 0

                if char != " ":
                    t.append(char, style=WAVE_GRADIENT[color_idx])
                else:
                    t.append(char)
            t.append("\n")

        return t
