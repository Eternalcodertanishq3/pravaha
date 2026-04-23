"""Queue Panel — Request queue visualization."""

from __future__ import annotations
from textual.widgets import Static


class QueuePanel(Static):
    """Request queue status with ASCII bar visualization."""

    DEFAULT_CSS = """
    QueuePanel { height: 4; border: solid #1a2a1a 1; padding: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_slots = 32

    def render(self) -> str:
        filled = int((self.active / self.max_slots) * 20) if self.max_slots > 0 else 0
        bar = "▓" * filled + "░" * (20 - filled)
        return f"Queue: {bar}  {self.active}/{self.max_slots} slots"
