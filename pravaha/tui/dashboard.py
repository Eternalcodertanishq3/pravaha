"""TUI Dashboard — Rich terminal UI for monitoring the engine."""

from __future__ import annotations

import time
from typing import Any


class Dashboard:
    """Rich terminal dashboard showing engine stats in real-time."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._running = False

    def render_once(self) -> str:
        """Render a single frame of the dashboard as text."""
        stats = self.engine.get_stats()
        sched = stats.get("scheduler", {})
        blocks = stats.get("block_manager", {})

        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║        Pravāha v3 — Inference Engine Dashboard          ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Model: {str(stats.get('model', 'N/A'))[:46]:46s}  ║",
            f"║  Device: {str(stats.get('device', 'N/A'))[:45]:45s}  ║",
            f"║  Quant: {str(stats.get('quantization', 'none'))[:46]:46s}  ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Total Requests: {stats.get('total_requests', 0):>6}                               ║",
            f"║  Total Tokens:   {stats.get('total_tokens_generated', 0):>6}                               ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Waiting: {sched.get('waiting', 0):>4}  Running: {sched.get('running', 0):>4}  Swapped: {sched.get('swapped', 0):>4}    ║",
            f"║  Free Blocks: {blocks.get('free_blocks', 0):>5} / {blocks.get('total_blocks', 0):>5}                        ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    def start(self, refresh_rate: float = 1.0) -> None:
        """Start the live dashboard (blocking)."""
        self._running = True
        try:
            from rich.live import Live
            from rich.text import Text

            with Live(auto_refresh=False) as live:
                while self._running:
                    output = self.render_once()
                    live.update(Text(output))
                    live.refresh()
                    time.sleep(refresh_rate)
        except ImportError:
            # Fallback without Rich
            import os

            while self._running:
                os.system("cls" if os.name == "nt" else "clear")
                print(self.render_once())
                time.sleep(refresh_rate)

    def stop(self) -> None:
        self._running = False


class RequestPanel:
    """Panel showing active request details."""

    def render(self, stats: dict) -> str:
        requests = stats.get("scheduler", {}).get("requests", [])
        if not requests:
            return "  No active requests."
        lines = ["  ID         Status    Tokens  Progress"]
        lines.append("  " + "-" * 42)
        for req in requests[:10]:
            rid = str(req.get("id", ""))[:8]
            status = str(req.get("status", ""))[:8]
            tokens = req.get("tokens", 0)
            progress = req.get("progress", 0)
            lines.append(f"  {rid:8s}  {status:8s}  {tokens:>6}  {progress:>5.1f}%")
        return "\n".join(lines)


class BlockMapPanel:
    """Panel showing KV cache block allocation map."""

    def render(self, block_map: list[int], cols: int = 64) -> str:
        symbols = {0: "░", 1: "█", 2: "▓"}
        lines = ["  Block Map (░=Free █=Used ▓=Shared)"]
        row = "  "
        for i, status in enumerate(block_map):
            row += symbols.get(status, "?")
            if (i + 1) % cols == 0:
                lines.append(row)
                row = "  "
        if len(row.strip()) > 0:
            lines.append(row)
        return "\n".join(lines)


class AgentPanel:
    """Panel showing swarm agent status."""

    def render(self, agents: list[dict]) -> str:
        if not agents:
            return "  No agents loaded."
        lines = ["  Agent           Role         Calls  Tokens"]
        lines.append("  " + "-" * 46)
        for agent in agents[:20]:
            name = str(agent.get("name", ""))[:14]
            role = str(agent.get("role", ""))[:10]
            calls = agent.get("total_calls", 0)
            tokens = agent.get("total_tokens", 0)
            lines.append(f"  {name:14s}  {role:10s}  {calls:>5}  {tokens:>6}")
        return "\n".join(lines)
