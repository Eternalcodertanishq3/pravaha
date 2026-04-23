"""Chat Panel — Streaming chat with token rendering."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static


class ChatPanel(VerticalScroll):
    """Streaming chat display with rich token rendering."""

    DEFAULT_CSS = """
    ChatPanel { height: 100%; border: solid #1a2a1a 1; background: #0a0a0a; padding: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        color = "#00e676" if role == "assistant" else "#4fc3f7"
        label = "pravaha" if role == "assistant" else "you"
        self._messages.append((role, content))
        self.mount(Static(f"[{color}]{label}>[/{color}] {content}"))
        self.scroll_end(animate=False)

    def append_token(self, token: str) -> None:
        if self.children:
            last = self.children[-1]
            if isinstance(last, Static):
                last.update(str(last.renderable) + token)
