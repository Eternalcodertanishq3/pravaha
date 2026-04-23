"""Swarm Panel — 32-agent live status grid."""

from __future__ import annotations
from textual.widgets import Static


class SwarmPanel(Static):
    """32-agent grid with status badges."""

    DEFAULT_CSS = """
    SwarmPanel { height: 5; border: solid #1a2a1a 1; padding: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._agents: list[dict] = []

    def render(self) -> str:
        if not self._agents:
            defaults = [
                "plan", "code", "crit", "valid", "summ", "expd", "tran", "reas",
                "merg", "rout", "memo", "tool", "judg", "refi", "clas", "extr",
                "narr", "ensm", "debug", "resrc",
                "synx", "logc", "halu", "secu", "perf", "cons", "type", "edge",
                "test", "veri", "self", "ptch",
            ]
            badges = [f"[{n}○]" for n in defaults]
        else:
            badges = []
            for a in self._agents:
                name = a.get("name", "?")[:4]
                status = a.get("status", "idle")
                sym = "●" if status == "active" else ("◆" if status == "audit" else "○")
                badges.append(f"[{name}{sym}]")
        lines = []
        for i in range(0, len(badges), 8):
            lines.append(" ".join(badges[i:i + 8]))
        return "Agents: " + "\n        ".join(lines)

    def refresh_agents(self) -> None:
        self.refresh()
