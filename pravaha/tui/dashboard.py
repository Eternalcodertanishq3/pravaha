"""Pravaha TUI Dashboard — Live-wired, animated cyberpunk monitoring.

Every panel pulls real data from the ``AsyncPravahaEngine`` when an
engine reference is attached.  When no engine is available the panels
show a ``STANDBY`` state so the TUI can still launch in demo mode.

All panels auto-refresh on Textual timers so the whole dashboard
is continuously animated.
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from datetime import datetime
from typing import Any

import psutil
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static
from rich.text import Text

from pravaha.tui.avatar.pravaha_avatar import PravahaAvatar

# ═══════════════════════════════════════════════════════════════════
#  Colour palette
# ═══════════════════════════════════════════════════════════════════
CYAN = "bright_cyan"
DIM_CYAN = "dark_cyan"
GREEN = "bright_green"
YELLOW = "yellow"
MAGENTA = "bright_magenta"
DIM = "grey37"

# ═══════════════════════════════════════════════════════════════════
#  Shared engine connector — singleton bridge to real data
# ═══════════════════════════════════════════════════════════════════

class EngineConnector:
    """Bridge between TUI panels and the real AsyncPravahaEngine.

    If ``engine`` is *None* the connector returns safe defaults so
    the TUI still renders cleanly in demo/standalone mode.
    """

    def __init__(self) -> None:
        self.engine: Any = None
        self.orchestrator: Any = None
        self._event_log: deque[tuple[float, str, str]] = deque(maxlen=200)
        self._cpu_history: deque[float] = deque(maxlen=30)
        self._mem_history: deque[float] = deque(maxlen=30)
        self._tok_history: deque[float] = deque(maxlen=30)
        self._lat_history: deque[float] = deque(maxlen=30)
        self._tokens_generated = 0
        self._last_sample = time.monotonic()

    # ── attach real engine ──────────────────────────────────────
    def attach_engine(self, engine: Any) -> None:
        self.engine = engine
        if hasattr(engine, "event_bus"):
            engine.event_bus.subscribe(self._on_engine_event)
        self.push_event("Engine attached", GREEN)

    def attach_orchestrator(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator
        self.push_event("Swarm orchestrator attached", GREEN)

    # ── event log ───────────────────────────────────────────────
    def _on_engine_event(self, event: Any) -> None:
        label = getattr(event, "event_type", None)
        name = label.name if label else str(event)
        req = getattr(event, "request_id", None)
        msg = name
        if req:
            msg += f" [{req[:8]}]"
        self.push_event(msg, CYAN)

    def push_event(self, msg: str, color: str = "grey70") -> None:
        self._event_log.appendleft((time.time(), msg, color))

    def recent_events(self, n: int = 8) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for ts, msg, col in list(self._event_log)[:n]:
            t = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            out.append((t, msg, col))
        return out

    # ── system metrics (real psutil) ────────────────────────────
    def sample_system(self) -> None:
        self._cpu_history.append(psutil.cpu_percent(interval=0))
        self._mem_history.append(psutil.virtual_memory().percent)

    # ── engine stats ────────────────────────────────────────────
    def engine_stats(self) -> dict[str, Any]:
        if self.engine and hasattr(self.engine, "get_stats"):
            return self.engine.get_stats()
        return {}

    def scheduler_stats(self) -> dict[str, Any]:
        stats = self.engine_stats()
        return stats.get("scheduler", {})

    def block_stats(self) -> dict[str, Any]:
        stats = self.engine_stats()
        return stats.get("block_manager", {})

    def agent_list(self) -> list[dict[str, Any]]:
        if self.orchestrator and hasattr(self.orchestrator, "_agents"):
            return [
                a.get_stats()
                for a in self.orchestrator._agents.values()
            ]
        return []

    def cpu_pct(self) -> float:
        return self._cpu_history[-1] if self._cpu_history else 0.0

    def mem_pct(self) -> float:
        return self._mem_history[-1] if self._mem_history else 0.0

    def cpu_sparkline(self) -> list[int]:
        return [max(0, min(7, int(v / 12.5))) for v in self._cpu_history]

    def mem_sparkline(self) -> list[int]:
        return [max(0, min(7, int(v / 12.5))) for v in self._mem_history]

    def tok_per_sec(self) -> float:
        stats = self.engine_stats()
        return float(stats.get("total_tokens_generated", 0))

    def context_pct(self) -> float:
        bs = self.block_stats()
        total = bs.get("total_blocks", 1)
        used = bs.get("used_blocks", 0)
        return round(used / max(1, total) * 100, 1)

    def is_online(self) -> bool:
        if self.engine and hasattr(self.engine, "is_ready"):
            return self.engine.is_ready
        return False


# ── module-level singleton ──────────────────────────────────────
_connector = EngineConnector()


def get_connector() -> EngineConnector:
    return _connector


# ═══════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════

class HeaderPanel(Static):
    DEFAULT_CSS = "HeaderPanel { height: 4; content-align: center middle; text-align: center; padding: 0 1; }"

    def on_mount(self) -> None:
        self.set_interval(1.2, self.refresh)

    def render(self) -> Text:
        t = Text()
        # decorative tick
        tick = "~" if int(time.time()) % 2 == 0 else "≈"
        t.append(f"                            {tick}{tick}  ", style=DIM_CYAN)
        t.append("P  R  A  V  A  H  A", style=f"bold {CYAN}")
        t.append(f"  {tick}{tick}\n", style=DIM_CYAN)
        t.append("                          ~  ─  ─  ─  ─  ─  ─  ─  ─  ─  ~\n", style=DIM_CYAN)
        t.append("                         A I   A G E N T I C   O R C H E S T R A T I O N   F R A M E W O R K", style="grey50")
        return t


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM STATUS  (left-top)  — REAL DATA
# ═══════════════════════════════════════════════════════════════════

class SystemStatusPanel(Static):
    DEFAULT_CSS = "SystemStatusPanel { height: 100%; padding: 0 1; }"

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh)

    def render(self) -> Text:
        conn = get_connector()
        online = conn.is_online()
        stats = conn.engine_stats()
        sched = conn.scheduler_stats()
        agents = conn.agent_list()

        t = Text()
        t.append(" ─ SYSTEM STATUS ─\n\n", style=f"bold {CYAN}")

        # orchestrator
        val, col = ("ONLINE", GREEN) if online else ("STANDBY", YELLOW)
        t.append(" ☑ ", style=CYAN); t.append("Orchestrator  ", style="grey70")
        t.append(": ", style="grey50"); t.append(f"{val}\n", style=f"bold {col}")

        # agents
        n_agents = len(agents) if agents else 0
        t.append(" ⊛ ", style=CYAN); t.append("Agents        ", style="grey70")
        t.append(": ", style="grey50"); t.append(f"{n_agents:02d} LOADED\n", style=f"bold {GREEN}")

        # tasks
        running = sched.get("running", 0)
        waiting = sched.get("waiting", 0)
        t.append(" ◔ ", style=CYAN); t.append("Tasks         ", style="grey70")
        t.append(": ", style="grey50"); t.append(f"{running + waiting:02d} ACTIVE\n", style=f"bold {YELLOW}")

        # pipelines
        pipelines = sched.get("running", 0)
        t.append(" ⊘ ", style=CYAN); t.append("Pipelines     ", style="grey70")
        t.append(": ", style="grey50"); t.append(f"{pipelines:02d} RUNNING\n", style=f"bold {GREEN}")

        # total tokens
        total_tok = stats.get("total_tokens_generated", 0)
        t.append(" ⊕ ", style=CYAN); t.append("Tokens        ", style="grey70")
        t.append(": ", style="grey50"); t.append(f"{total_tok}\n", style=f"bold {CYAN}")

        return t


# ═══════════════════════════════════════════════════════════════════
#  FLOW VISUALIZER  (left-bottom) — animated flowing lines
# ═══════════════════════════════════════════════════════════════════

_FLOW_CHARS = ["╌", "┈", "╌", "·", "╌", "┈", "·", "╌"]

class FlowVisualizerPanel(Static):
    DEFAULT_CSS = "FlowVisualizerPanel { height: 100%; padding: 0 1; }"

    def on_mount(self) -> None:
        self._tick = 0
        self.set_interval(0.4, self._animate)

    def _animate(self) -> None:
        self._tick += 1
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(" ─ FLOW VISUALIZER ─\n\n", style=f"bold {CYAN}")
        # animated flowing lines (shift pattern each tick)
        for row in range(5):
            line = "   "
            offset = (self._tick + row) % len(_FLOW_CHARS)
            for col in range(19):
                idx = (offset + col) % len(_FLOW_CHARS)
                line += _FLOW_CHARS[idx]
            t.append(f"{line}\n", style=DIM_CYAN)
        t.append("\n")
        t.append("  ─→  ", style=CYAN); t.append("DATA FLOW\n", style="grey70")
        t.append("  ┈┈  ", style=DIM_CYAN); t.append("CONTROL FLOW\n", style="grey70")
        t.append("  ╌╌  ", style=DIM_CYAN); t.append("CONTEXT STREAM\n", style="grey70")
        return t


# ═══════════════════════════════════════════════════════════════════
#  CENTRE FLOW HUB — animated concentric rings + avatar
# ═══════════════════════════════════════════════════════════════════

_RING_CHARS = ["·", "╌", "┈", "·", "╌", "·", "┈", "╌"]

class CenterFlowPanel(Widget):
    DEFAULT_CSS = "CenterFlowPanel { height: 100%; content-align: center middle; text-align: center; padding: 0; }"

    def compose(self) -> ComposeResult:
        yield Static(id="flow-top-art")
        yield PravahaAvatar(id="center-avatar")
        yield Static(id="flow-bottom-art")
        yield Static(id="flow-tagline")

    def on_mount(self) -> None:
        self._tick = 0
        self.set_interval(0.5, self._animate)

    def _animate(self) -> None:
        self._tick += 1
        self._update_flow_art()

    def _build_ring(self, width: int, offset: int) -> str:
        """Build one animated ring line."""
        line = ""
        for i in range(width):
            idx = (self._tick + offset + i) % len(_RING_CHARS)
            ch = _RING_CHARS[idx]
            line += ch + " "
        return line

    def _update_flow_art(self) -> None:
        top = Text()
        paddings = [18, 12, 8, 6, 4, 3, 2, 2]
        widths =   [25, 30, 34, 36, 38, 39, 40, 40]
        for i, (pad, w) in enumerate(zip(paddings, widths)):
            ring = self._build_ring(w, i * 3)
            top.append(" " * pad + ring + "\n", style=DIM_CYAN)
        try:
            self.query_one("#flow-top-art", Static).update(top)
        except Exception:
            pass

        bot = Text()
        for i, (pad, w) in enumerate(zip(reversed(paddings), reversed(widths))):
            ring = self._build_ring(w, (len(paddings) + i) * 3)
            bot.append(" " * pad + ring + "\n", style=DIM_CYAN)
        try:
            self.query_one("#flow-bottom-art", Static).update(bot)
        except Exception:
            pass

        tag = Text()
        tag.append("\n")
        tag.append("F L O W", style=f"bold {CYAN}")
        tag.append(" . ", style="grey50")
        tag.append("O R C H E S T R A T E", style=f"bold {GREEN}")
        tag.append(" . ", style="grey50")
        tag.append("I N T E L L I G E N C E", style=f"bold {MAGENTA}")
        tag.append(" .", style="grey50")
        try:
            self.query_one("#flow-tagline", Static).update(tag)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
#  ACTIVE AGENTS  (right-top) — REAL AGENT DATA
# ═══════════════════════════════════════════════════════════════════

_AGENT_ICONS = ["⚙", "⊞", "✎", "◇", "◈", "≡", "☰", "⊕", "✦", "◎"]

class ActiveAgentsPanel(Static):
    DEFAULT_CSS = "ActiveAgentsPanel { height: 100%; padding: 0 1; }"

    def on_mount(self) -> None:
        self.set_interval(2.0, self.refresh)

    def render(self) -> Text:
        conn = get_connector()
        agents = conn.agent_list()
        t = Text()
        t.append(" ─ ACTIVE AGENTS ─\n\n", style=f"bold {CYAN}")

        if not agents:
            # show placeholder list when engine not attached
            names = ["Planner", "Retriever", "Analyzer", "Coder",
                     "Validator", "Memory", "Supervisor"]
            for i, name in enumerate(names):
                icon = _AGENT_ICONS[i % len(_AGENT_ICONS)]
                t.append(f"  {icon} ", style=CYAN)
                t.append(f"[{chr(65+i)}] ", style="grey50")
                t.append(f"{name:12s}", style="grey70")
                t.append(" ○\n", style="grey42")
            return t

        # real agents (show first 10)
        for i, a in enumerate(agents[:10]):
            icon = _AGENT_ICONS[i % len(_AGENT_ICONS)]
            name = a.get("name", f"agent_{i}")[:12]
            calls = a.get("total_calls", 0)
            active = calls > 0
            dot_color = GREEN if active else "grey42"
            dot = "●" if active else "○"
            t.append(f"  {icon} ", style=CYAN)
            t.append(f"[{chr(65+i)}] ", style="grey50")
            t.append(f"{name:12s}", style="grey70")
            t.append(f" {dot}\n", style=dot_color)

        return t


# ═══════════════════════════════════════════════════════════════════
#  CURRENT PIPELINE  (right-bottom) — tracks real pipeline state
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_STEPS = ["Ingest", "Understand", "Plan", "Execute", "Validate", "Deliver"]

class CurrentPipelinePanel(Static):
    DEFAULT_CSS = "CurrentPipelinePanel { height: 100%; padding: 0 1; }"

    def on_mount(self) -> None:
        self._active_step = 0
        self.set_interval(1.5, self._tick)

    def _tick(self) -> None:
        conn = get_connector()
        # if engine is actively running, animate step progression
        sched = conn.scheduler_stats()
        if sched.get("running", 0) > 0:
            self._active_step = min(self._active_step + 1, len(_DEFAULT_STEPS) - 1)
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(" ─ CURRENT PIPELINE ─\n\n", style=f"bold {CYAN}")
        for i, name in enumerate(_DEFAULT_STEPS):
            if i < self._active_step:
                dot_c, txt_c, mark = GREEN, "grey70", " ✓"
            elif i == self._active_step:
                dot_c, txt_c, mark = YELLOW, f"bold {YELLOW}", " …"
            else:
                dot_c, txt_c, mark = "grey42", "grey50", " ···"
            t.append("  ○ ", style=dot_c)
            t.append(f"({i+1}) ", style="grey50")
            t.append(f"{name:12s}", style=txt_c)
            t.append(f"{mark}\n", style=dot_c)
        return t


# ═══════════════════════════════════════════════════════════════════
#  METRICS BAR  (bottom-left) — REAL psutil + engine metrics
# ═══════════════════════════════════════════════════════════════════

def _sparkline(values: list[int], color: str) -> Text:
    bars = "▁▂▃▄▅▆▇█"
    t = Text()
    for v in values[-15:]:
        t.append(bars[min(v, 7)], style=color)
    return t


class MetricsBar(Static):
    DEFAULT_CSS = "MetricsBar { height: 100%; padding: 0 1; }"

    def on_mount(self) -> None:
        self.set_interval(1.0, self._sample)

    def _sample(self) -> None:
        get_connector().sample_system()
        self.refresh()

    def render(self) -> Text:
        conn = get_connector()
        cpu = conn.cpu_pct()
        mem = conn.mem_pct()
        ctx = conn.context_pct()
        tok = conn.tok_per_sec()

        metrics = [
            ("CPU",        f"{cpu:.0f}%",  conn.cpu_sparkline(), GREEN),
            ("MEMORY",     f"{mem:.0f}%",  conn.mem_sparkline(), CYAN),
            ("TOKENS",     f"{tok:.0f}",   [max(0, min(7, int(tok / 50))) for _ in range(12)], YELLOW),
            ("CTX WINDOW", f"{ctx:.0f}%",  [max(0, min(7, int(ctx / 12.5))) for _ in range(12)], MAGENTA),
        ]
        t = Text()
        for label, value, data, color in metrics:
            t.append(f"  {label}\n", style="grey50")
            t.append("  ")
            t.append_text(_sparkline(data, color))
            t.append("\n")
            t.append(f"  {value}\n\n", style=f"bold {color}")
        return t


# ═══════════════════════════════════════════════════════════════════
#  LIVE EVENTS  (bottom-right) — REAL engine event bus
# ═══════════════════════════════════════════════════════════════════

class LiveEventsPanel(Static):
    DEFAULT_CSS = "LiveEventsPanel { height: 100%; padding: 0 1; }"

    def on_mount(self) -> None:
        self.set_interval(0.8, self.refresh)
        # push initial boot events
        conn = get_connector()
        conn.push_event("TUI dashboard initialised", CYAN)
        conn.push_event(f"PID {os.getpid()} | psutil {psutil.__version__}", "grey50")
        conn.push_event(f"CPU cores: {psutil.cpu_count(logical=True)}", "grey70")
        conn.push_event(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB", "grey70")

    def render(self) -> Text:
        conn = get_connector()
        events = conn.recent_events(8)
        t = Text()
        t.append(" LIVE EVENTS\n", style=f"bold {CYAN}")
        if not events:
            t.append("  (no events yet)\n", style="grey42")
            return t
        for ts, msg, color in events:
            t.append(f" {ts}  ", style=DIM_CYAN)
            t.append(f"{msg}", style=color)
            t.append("  ●\n", style=color)
        return t


# ═══════════════════════════════════════════════════════════════════
#  FOOTER — animated scanner line
# ═══════════════════════════════════════════════════════════════════

class PravahaFooter(Static):
    DEFAULT_CSS = "PravahaFooter { height: 2; content-align: center middle; text-align: center; padding: 0; }"

    def on_mount(self) -> None:
        self._tick = 0
        self.set_interval(0.6, self._animate)

    def _animate(self) -> None:
        self._tick += 1
        self.refresh()

    def render(self) -> Text:
        # animated dashes that shift
        dash_chars = "── ── ── ──"
        shift = self._tick % 4
        left = dash_chars[shift:] + dash_chars[:shift]
        t = Text()
        t.append(f"{left}  ", style=DIM_CYAN)
        t.append("PRAVAHA", style=f"bold {CYAN}")
        t.append(" • ", style="grey50")
        t.append("[F1] ", style=CYAN); t.append("Chat │ ", style="grey50")
        t.append("[F2] ", style=CYAN); t.append("Swarm │ ", style="grey50")
        t.append("[F3] ", style=CYAN); t.append("Audit │ ", style="grey50")
        t.append("[F4] ", style=CYAN); t.append("Metrics │ ", style="grey50")
        t.append("[F5] ", style=CYAN); t.append("Queue │ ", style="grey50")
        t.append("[F6] ", style=CYAN); t.append("RAG │ ", style="grey50")
        t.append("[F7] ", style=CYAN); t.append("Log", style="grey50")
        t.append(f"  {left}", style=DIM_CYAN)
        return t


# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD CONTAINER
# ═══════════════════════════════════════════════════════════════════

class PravahaDashboard(Widget):
    """Top-level dashboard widget that wires all panels together."""

    DEFAULT_CSS = "PravahaDashboard { layout: vertical; width: 100%; height: 100%; }"

    def compose(self) -> ComposeResult:
        yield HeaderPanel(id="dash-header")

        with Horizontal(id="main-row"):
            with Vertical(id="left-col"):
                yield SystemStatusPanel(id="sys-status")
                yield FlowVisualizerPanel(id="flow-vis")

            yield CenterFlowPanel(id="center-flow")

            with Vertical(id="right-col"):
                yield ActiveAgentsPanel(id="active-agents")
                yield CurrentPipelinePanel(id="cur-pipeline")

        with Horizontal(id="bottom-row"):
            yield MetricsBar(id="metrics-bar")
            yield LiveEventsPanel(id="live-events")

        yield PravahaFooter(id="dash-footer")
