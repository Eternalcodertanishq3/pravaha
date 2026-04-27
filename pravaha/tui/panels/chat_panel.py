"""Chat Panel — Full-screen streaming chat interface.

Opens as a detail screen where the user can send prompts to the
engine and watch tokens stream in real-time.
"""

from __future__ import annotations

import asyncio
from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static
from rich.text import Text

from pravaha.tui.dashboard import get_connector

CYAN = "bright_cyan"
GREEN = "bright_green"
DIM = "grey50"


class ChatHistory(VerticalScroll):
    """Scrolling chat history with rich rendering."""

    DEFAULT_CSS = """
    ChatHistory {
        height: 1fr;
        background: #0a0e14;
        border: round #1a3a4a;
        padding: 1 2;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._active_message_text: Text | None = None
        self._active_message_widget: Static | None = None

    def add_message(self, role: str, content: str) -> None:
        label = Text()
        if role == "user":
            label.append(" you > ", style="bold bright_cyan")
            label.append(content, style="grey85")
        else:
            label.append(" pravaha > ", style="bold bright_green")
            label.append(content, style="grey85")
            
        widget = Static(label)
        self.mount(widget)
        self.scroll_end(animate=False)
        
        self._active_message_text = label
        self._active_message_widget = widget

    def append_token(self, token: str) -> None:
        """Append a token to the last message (streaming)."""
        if self._active_message_text is not None and self._active_message_widget is not None:
            self._active_message_text.append(token, style="grey85")
            self._active_message_widget.update(self._active_message_text)

    def add_system(self, msg: str) -> None:
        t = Text()
        t.append(f" system: {msg}", style="grey50")
        self.mount(Static(t))
        self.scroll_end(animate=False)


class ChatBanner(Static):
    """Top banner for the chat screen."""

    def render(self) -> Text:
        conn = get_connector()
        t = Text()
        t.append("╔══════════════════════════════════════════════════════════════╗\n", style=CYAN)
        t.append("║                  ", style=CYAN)
        t.append("PRAVAHA CHAT INTERFACE", style=f"bold {CYAN}")
        t.append("                    ║\n", style=CYAN)
        t.append("╚══════════════════════════════════════════════════════════════╝\n", style=CYAN)
        if conn.is_online():
            stats = conn.engine_stats()
            model = stats.get("model", "unknown")
            quant = stats.get("quantization", "none")
            t.append(f" Model: {model}  ·  Quantization: {quant}", style=DIM)
        else:
            t.append(" Engine: STANDBY — attach an engine to chat", style="yellow")
        return t


class ChatScreen(Screen):
    """Full-screen chat interface with prompt input and streaming output."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ChatBanner()
        yield ChatHistory(id="chat-history")
        yield Input(
            placeholder="Type a prompt and press Enter...",
            id="chat-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        history = self.query_one("#chat-history", ChatHistory)
        history.add_system("Chat session started. Type a prompt below.")
        conn = get_connector()
        if not conn.is_online():
            history.add_system(
                "Engine is in STANDBY mode. Messages will be echoed back."
            )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""

        history = self.query_one("#chat-history", ChatHistory)
        history.add_message("user", prompt)

        conn = get_connector()

        if conn.engine is not None and conn.is_online():
            # Real engine: stream tokens
            history.add_message("assistant", "")
            conn.push_event(f"Chat prompt: {prompt[:40]}...", CYAN)
            try:
                from pravaha.decoder.sampling import SamplingParams
                params = SamplingParams(max_new_tokens=256)
                async for token in conn.engine.generate(prompt, params):
                    history.append_token(token)
                conn.push_event("Chat response complete", GREEN)
            except Exception as exc:
                history.add_system(f"Error: {exc}")
                conn.push_event(f"Chat error: {exc}", "red")
        else:
            # Standby: echo
            history.add_message(
                "assistant",
                f"[standby] Echo: {prompt}",
            )
