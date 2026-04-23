"""ASCII Art — Banners, boxes, progress bars, spinners, agent grids.

All visual components used by the CLI and TUI to create Pravaha's
premium terminal experience.
"""

from __future__ import annotations

from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

BANNER = r"""
██████╗ ██████╗  █████╗ ██╗   ██╗ █████╗ ██╗  ██╗ █████╗
██╔══██╗██╔══██╗██╔══██╗██║   ██║██╔══██╗██║  ██║██╔══██╗
██████╔╝██████╔╝███████║██║   ██║███████║███████║███████║
██╔═══╝ ██╔══██╗██╔══██║╚██╗ ██╔╝██╔══██║██╔══██║██╔══██║
██║     ██║  ██║██║  ██║ ╚████╔╝ ██║  ██║██║  ██║██║  ██║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
  प्रवाह  ·  self-healing · swarm-ready · hackable
"""


def print_banner() -> None:
    """Print the Pravaha banner with colored styling."""
    text = Text(BANNER)
    text.stylize("bold bright_green")
    console.print(text)
    console.print("[dim]v3.0.0  ·  MIT License  ·  github.com/pravaha[/dim]\n")


def status_box(rows: dict[str, str], title: str = "") -> str:
    """Draw a Unicode box with key→value rows."""
    max_key = max(len(k) for k in rows) if rows else 0
    max_val = max(len(v) for v in rows.values()) if rows else 0
    width = max(max_key + max_val + 7, len(title) + 4, 40)

    lines = []
    header = f"─ {title} " if title else "─"
    lines.append(f"┌{header}{'─' * (width - len(header) - 1)}┐")
    for key, val in rows.items():
        padding = width - len(key) - len(val) - 5
        lines.append(f"│  {key}{' ' * (max_key - len(key))}  {val}{' ' * max(padding, 1)} │")
    lines.append(f"└{'─' * (width - 1)}┘")
    return "\n".join(lines)


def ascii_progress(
    value: float, max_val: float, width: int = 40, label: str = ""
) -> str:
    """Render an ASCII progress bar."""
    ratio = min(value / max_val, 1.0) if max_val > 0 else 0
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {value:.1f}/{max_val:.1f} {label}"


def spinner(msg: str) -> Any:
    """Return a Rich spinner context manager."""
    return console.status(f"[bold green]{msg}[/bold green]", spinner="dots")


def agent_grid(agents: list[dict[str, Any]]) -> str:
    """Render 8-column grid of agent badges with status indicators."""
    badges = []
    for agent in agents:
        name = agent.get("name", "?")[:8]
        status = agent.get("status", "idle")
        if status == "active":
            badges.append(f"[{name}●]")
        elif status == "audit":
            badges.append(f"[{name}◆]")
        else:
            badges.append(f"[{name}○]")

    lines = []
    for i in range(0, len(badges), 8):
        lines.append("  ".join(badges[i:i + 8]))
    return "\n".join(lines)
