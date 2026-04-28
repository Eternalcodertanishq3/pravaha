"""Pravaha Avatar Widget — Animated robotic ASCII face for the TUI.

A Textual widget with a reactive ``state`` property.  When the state
changes the avatar re-renders with a new face expression, and the
frame index cycles on a timer to produce a breathing/blinking effect.
"""

from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

# ── palette shortcuts ────────────────────────────────────────────
C = "bright_cyan"
G = "bright_green"
Y = "yellow"
M = "bright_magenta"
D = "grey50"

# ── frames per state (two frames → animated blink) ──────────────

_IDLE = [
    [
        ("         ╭─╮         ", C),
        ("         │○│         ", C),
        ("      ╭──┴──╮       ", C),
        ("      │ ◉  ◉ │       ", C),
        ("      │  ▽   │       ", C),
        ("      ╰──┬──╯       ", C),
        ("     ╭───┴───╮      ", C),
        ("     │  A I  │      ", M),
        ("     ╰───────╯      ", C),
    ],
    [
        ("         ╭─╮         ", C),
        ("         │◦│         ", C),
        ("      ╭──┴──╮       ", C),
        ("      │ ◉  ◉ │       ", C),
        ("      │  ━   │       ", C),
        ("      ╰──┬──╯       ", C),
        ("     ╭───┴───╮      ", C),
        ("     │  A I  │      ", M),
        ("     ╰───────╯      ", C),
    ],
]

_THINKING = [
    [
        ("     ◌   ╭─╮   ◌   ", Y),
        ("         │◦│         ", Y),
        ("      ╭──┴──╮       ", Y),
        ("      │ ◑  ◐ │       ", Y),
        ("      │  ▿   │       ", Y),
        ("      ╰──┬──╯       ", Y),
        ("     ╭───┴───╮      ", Y),
        ("     │ · · · │      ", Y),
        ("     ╰───────╯      ", Y),
    ],
    [
        ("   ◍     ╭─╮     ◍ ", Y),
        ("         │●│         ", Y),
        ("      ╭──┴──╮       ", Y),
        ("      │ ◐  ◑ │       ", Y),
        ("      │  ▿   │       ", Y),
        ("      ╰──┬──╯       ", Y),
        ("     ╭───┴───╮      ", Y),
        ("     │ · · · │      ", Y),
        ("     ╰───────╯      ", Y),
    ],
]

_WORKING = [
    [
        ("         ╭─╮         ", G),
        ("         │●│         ", G),
        ("      ╭──┴──╮       ", G),
        ("      │ ▪  ▪ │       ", G),
        ("      │  ━   │       ", G),
        ("      ╰──┬──╯       ", G),
        ("     ╭───┴───╮      ", G),
        ("     │ ░▒▓▒░ │      ", G),
        ("     ╰───────╯      ", G),
    ],
    [
        ("         ╭─╮         ", G),
        ("         │○│         ", G),
        ("      ╭──┴──╮       ", G),
        ("      │ ▪  ▪ │       ", G),
        ("      │  ━   │       ", G),
        ("      ╰──┬──╯       ", G),
        ("     ╭───┴───╮      ", G),
        ("     │ ▓▒░▒▓ │      ", G),
        ("     ╰───────╯      ", G),
    ],
]

_SUCCESS = [
    [
        ("    ✦    ╭─╮    ✦   ", G),
        ("         │★│         ", G),
        ("      ╭──┴──╮       ", G),
        ("      │ ★  ★ │       ", G),
        ("      │ ╰━╯  │       ", G),
        ("      ╰──┬──╯       ", G),
        ("     ╭───┴───╮      ", G),
        ("     │  ✓ ✓  │      ", G),
        ("     ╰───────╯      ", G),
    ],
]

_ERROR = [
    [
        ("    ⚠    ╭─╮    ⚠   ", M),
        ("         │!│         ", M),
        ("      ╭──┴──╮       ", M),
        ("      │ ✕  ✕ │       ", M),
        ("      │  △   │       ", M),
        ("      ╰──┬──╯       ", M),
        ("     ╭───┴───╮      ", "red"),
        ("     │ ERR   │      ", "red"),
        ("     ╰───────╯      ", M),
    ],
]

_STATE_FRAMES: dict[str, list[list[tuple[str, str]]]] = {
    "idle": _IDLE,
    "thinking": _THINKING,
    "working": _WORKING,
    "success": _SUCCESS,
    "error": _ERROR,
}

STATE_LABELS: dict[str, str] = {
    "idle": "Standing By",
    "thinking": "Planning...",
    "working": "Generating...",
    "success": "Complete",
    "error": "Error",
}


class PravahaAvatar(Widget):
    """Animated Pravaha robot avatar with reactive state cycling."""

    DEFAULT_CSS = """
    PravahaAvatar {
        width: auto;
        height: auto;
        content-align: center middle;
    }
    """

    state: reactive[str] = reactive("idle")
    _frame_idx: reactive[int] = reactive(0)

    def on_mount(self) -> None:
        """Start the animation timer (toggle frame every 800ms)."""
        self.set_interval(0.8, self._advance_frame)

    def _advance_frame(self) -> None:
        frames = _STATE_FRAMES.get(self.state, _IDLE)
        self._frame_idx = (self._frame_idx + 1) % len(frames)

    def render(self) -> Text:
        frames = _STATE_FRAMES.get(self.state, _IDLE)
        frame = frames[self._frame_idx % len(frames)]
        text = Text()
        for i, (line, color) in enumerate(frame):
            text.append(line, style=color)
            if i < len(frame) - 1:
                text.append("\n")
        return text

    def set_state(self, new_state: str) -> None:
        if new_state in _STATE_FRAMES:
            self.state = new_state
            self._frame_idx = 0

    def watch_state(self) -> None:
        self.refresh()

    def watch__frame_idx(self) -> None:
        self.refresh()
