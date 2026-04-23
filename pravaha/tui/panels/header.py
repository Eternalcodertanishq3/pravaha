"""Header Panel — Banner + clock + model badge + GPU status."""

from __future__ import annotations
from datetime import datetime
from textual.widgets import Static


class HeaderPanel(Static):
    """Top bar showing model info, GPU status, and time."""

    DEFAULT_CSS = """
    HeaderPanel { height: 3; background: #111111; border-bottom: solid #00e676 1; padding: 0 2; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "loading..."
        self.gpu_info = ""
        self.quant = ""

    def render(self) -> str:
        now = datetime.now().strftime("%H:%M:%S")
        return (
            f" PRAVAHA v3  ·  {self.model_name}  ·  "
            f"{self.quant}  ·  {self.gpu_info}  {now}"
        )

    def update_info(self, model: str, quant: str = "", gpu: str = "") -> None:
        self.model_name = model
        self.quant = quant or "fp16"
        self.gpu_info = gpu or "CPU"
        self.refresh()
