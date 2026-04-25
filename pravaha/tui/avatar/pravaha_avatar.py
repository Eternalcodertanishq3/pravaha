"""Pravaha Avatar — Pixel-art ASCII avatar with animated states.

5 states × 2+ frames each:
- idle: Peaceful breathing
- thinking: Swirling dots
- working: Active typing
- success: Celebration
- audit: Scanner mode
"""

from __future__ import annotations

# ── COLOR CODES ──────────────────────────────────────────────────
CYAN = "[bold cyan]"
GREEN = "[bold green]"
YELLOW = "[bold yellow]"
RED = "[bold red]"
MAGENTA = "[bold magenta]"
BLUE = "[bold blue]"
WHITE = "[bold white]"
DIM = "[dim]"
RESET = "[/]"

# ── AVATAR FRAMES ────────────────────────────────────────────────

IDLE_FRAMES = [
    [
        f"  {CYAN}╭━━━━━╮{RESET}  ",
        f"  {CYAN}│ ◉  ◉ │{RESET}  ",
        f"  {CYAN}│  ▽   │{RESET}  ",
        f"  {CYAN}╰━━━━━╯{RESET}  ",
        f"  {DIM}  ╱│╲  {RESET}  ",
        f"  {DIM}   │   {RESET}  ",
        f"  {DIM}  ╱ ╲  {RESET}  ",
    ],
    [
        f"  {CYAN}╭━━━━━╮{RESET}  ",
        f"  {CYAN}│ ◉  ◉ │{RESET}  ",
        f"  {CYAN}│  ━   │{RESET}  ",
        f"  {CYAN}╰━━━━━╯{RESET}  ",
        f"  {DIM}  ╱│╲  {RESET}  ",
        f"  {DIM}   │   {RESET}  ",
        f"  {DIM}  ╱ ╲  {RESET}  ",
    ],
]

THINKING_FRAMES = [
    [
        f"  {YELLOW}╭━━━━━╮{RESET} {YELLOW}◌{RESET}",
        f"  {YELLOW}│ ◑  ◑ │{RESET} {YELLOW}◍{RESET}",
        f"  {YELLOW}│  ▿   │{RESET}  ",
        f"  {YELLOW}╰━━━━━╯{RESET}  ",
        f"  {DIM}  ╱│╲  {RESET}  ",
        f"  {DIM}   │   {RESET}  ",
        f"  {DIM}  ╱ ╲  {RESET}  ",
    ],
    [
        f"  {YELLOW}╭━━━━━╮{RESET}  ",
        f" {YELLOW}◍{RESET}{YELLOW}│ ◐  ◐ │{RESET}  ",
        f"  {YELLOW}│  ▿   │{RESET}  ",
        f"  {YELLOW}╰━━━━━╯{RESET} {YELLOW}◌{RESET}",
        f"  {DIM}  ╱│╲  {RESET}  ",
        f"  {DIM}   │   {RESET}  ",
        f"  {DIM}  ╱ ╲  {RESET}  ",
    ],
    [
        f" {YELLOW}◌{RESET}{YELLOW}╭━━━━━╮{RESET}  ",
        f"  {YELLOW}│ ◒  ◒ │{RESET}  ",
        f"  {YELLOW}│  ▿   │{RESET} {YELLOW}◍{RESET}",
        f"  {YELLOW}╰━━━━━╯{RESET}  ",
        f"  {DIM}  ╱│╲  {RESET}  ",
        f"  {DIM}   │   {RESET}  ",
        f"  {DIM}  ╱ ╲  {RESET}  ",
    ],
]

WORKING_FRAMES = [
    [
        f"  {GREEN}╭━━━━━╮{RESET}  ",
        f"  {GREEN}│ ▪  ▪ │{RESET}  ",
        f"  {GREEN}│  ━   │{RESET}  ",
        f"  {GREEN}╰━━━━━╯{RESET}  ",
        f"  {GREEN} ╱│╲▄ {RESET}  ",
        f"  {GREEN}  │ ░▒{RESET}  ",
        f"  {DIM}  ╱ ╲  {RESET}  ",
    ],
    [
        f"  {GREEN}╭━━━━━╮{RESET}  ",
        f"  {GREEN}│ ▪  ▪ │{RESET}  ",
        f"  {GREEN}│  ━   │{RESET}  ",
        f"  {GREEN}╰━━━━━╯{RESET}  ",
        f"  {GREEN}▄╱│╲  {RESET}  ",
        f"  {GREEN}▒░│   {RESET}  ",
        f"  {DIM}  ╱ ╲  {RESET}  ",
    ],
]

SUCCESS_FRAMES = [
    [
        f"  {GREEN}╭━━━━━╮{RESET}  ",
        f"  {GREEN}│ ★  ★ │{RESET}  ",
        f"  {GREEN}│ ╰━╯  │{RESET}  ",
        f"  {GREEN}╰━━━━━╯{RESET}  ",
        f"  {GREEN}╲ │ ╱  {RESET}  ",
        f"  {GREEN}  │    {RESET}  ",
        f"  {GREEN}╱   ╲  {RESET}  ",
    ],
    [
        f" {GREEN}✦{RESET}{GREEN}╭━━━━━╮{RESET}{GREEN}✦{RESET}",
        f"  {GREEN}│ ★  ★ │{RESET}  ",
        f"  {GREEN}│ ╰━╯  │{RESET}  ",
        f"  {GREEN}╰━━━━━╯{RESET}  ",
        f" {GREEN}✦╲ │ ╱✦ {RESET}  ",
        f"  {GREEN}  │    {RESET}  ",
        f"  {GREEN}╱   ╲  {RESET}  ",
    ],
]

AUDIT_FRAMES = [
    [
        f"  {MAGENTA}╭━━━━━╮{RESET}  ",
        f"  {MAGENTA}│ ◎  ◎ │{RESET}  ",
        f"  {MAGENTA}│  ▼   │{RESET}  ",
        f"  {MAGENTA}╰━━━━━╯{RESET}  ",
        f"  {MAGENTA} ╱│╲🔍{RESET} ",
        f"  {DIM}   │   {RESET}  ",
        f"  {DIM}  ╱ ╲  {RESET}  ",
    ],
    [
        f"  {MAGENTA}╭━━━━━╮{RESET}  ",
        f"  {MAGENTA}│ ◉  ◉ │{RESET}  ",
        f"  {MAGENTA}│  ▼   │{RESET}  ",
        f"  {MAGENTA}╰━━━━━╯{RESET}  ",
        f" {MAGENTA}🔍╱│╲  {RESET} ",
        f"  {DIM}   │   {RESET}  ",
        f"  {DIM}  ╱ ╲  {RESET}  ",
    ],
]

# ── STATE MAP ────────────────────────────────────────────────────

AVATAR_STATES: dict[str, list[list[str]]] = {
    "idle": IDLE_FRAMES,
    "thinking": THINKING_FRAMES,
    "working": WORKING_FRAMES,
    "success": SUCCESS_FRAMES,
    "audit": AUDIT_FRAMES,
}

STATE_COLORS: dict[str, str] = {
    "idle": "cyan",
    "thinking": "yellow",
    "working": "green",
    "success": "green",
    "audit": "magenta",
}

STATE_LABELS: dict[str, str] = {
    "idle": "Standing By",
    "thinking": "Planning...",
    "working": "Generating...",
    "success": "✓ Complete",
    "audit": "Auditing...",
}
