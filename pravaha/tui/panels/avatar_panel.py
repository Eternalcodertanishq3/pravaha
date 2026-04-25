"""Avatar Panel — Animated pixel avatar in the TUI."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class AvatarPanel(Widget):
    """Animated avatar panel with state-driven pixel art.

    Shows the current agent state with animated frames.
    500ms frame rate for smooth animation.
    """

    DEFAULT_CSS = """
    AvatarPanel {
        width: 20;
        height: 12;
        border: solid green;
        padding: 0 1;
    }
    """

    state: reactive[str] = reactive("idle")
    agent_name: reactive[str] = reactive("")
    _frame_index: int = 0

    def compose(self):
        yield Static(id="avatar-art")
        yield Static(id="avatar-label")

    def on_mount(self) -> None:
        self.set_interval(0.5, self._advance_frame)

    def _advance_frame(self) -> None:
        from pravaha.tui.avatar.pravaha_avatar import AVATAR_STATES

        frames = AVATAR_STATES.get(self.state, AVATAR_STATES["idle"])
        self._frame_index = (self._frame_index + 1) % len(frames)
        self._render_frame()

    def _render_frame(self) -> None:
        from pravaha.tui.avatar.pravaha_avatar import (
            AVATAR_STATES,
            STATE_COLORS,
            STATE_LABELS,
        )

        frames = AVATAR_STATES.get(self.state, AVATAR_STATES["idle"])
        frame = frames[self._frame_index % len(frames)]
        color = STATE_COLORS.get(self.state, "cyan")
        label = STATE_LABELS.get(self.state, "")

        art = self.query_one("#avatar-art", Static)
        art.update("\n".join(frame))

        label_widget = self.query_one("#avatar-label", Static)
        name_str = f" [{color}]{self.agent_name}[/]" if self.agent_name else ""
        label_widget.update(f"[{color}]{label}[/]{name_str}")

    def watch_state(self, _old: str, _new: str) -> None:
        self._frame_index = 0
        self._render_frame()

    def set_state(self, state: str, agent_name: str = "") -> None:
        self.agent_name = agent_name
        self.state = state


# Create __init__.py
__all__ = ["AvatarPanel"]
