"""Swarm Agent Registry — Unified registry for all 51 agents.

Imports from 4 subfolders:
- workers/ (20 agents) — task execution
- auditors/ (12 agents) — quality assurance
- security/ (10 agents) — security analysis
- design/ (9 agents) — UI/UX design
"""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext
from pravaha.swarm.agents.workers import WORKER_AGENTS
from pravaha.swarm.agents.auditors import AUDIT_AGENTS
from pravaha.swarm.agents.security import SECURITY_AGENTS
from pravaha.swarm.agents.design import DESIGN_AGENTS

# ── Unified Registry (51 agents) ─────────────────────────────────

ALL_AGENTS: dict[str, type] = {}
ALL_AGENTS.update(WORKER_AGENTS)     # 20
ALL_AGENTS.update(AUDIT_AGENTS)      # 12
ALL_AGENTS.update(SECURITY_AGENTS)   # 10
ALL_AGENTS.update(DESIGN_AGENTS)     #  9
# Total: 51

# ── Category Maps ────────────────────────────────────────────────

AGENT_CATEGORIES: dict[str, dict[str, type]] = {
    "workers": WORKER_AGENTS,
    "auditors": AUDIT_AGENTS,
    "security": SECURITY_AGENTS,
    "design": DESIGN_AGENTS,
}

__all__ = [
    "ALL_AGENTS",
    "AGENT_CATEGORIES",
    "WORKER_AGENTS",
    "AUDIT_AGENTS",
    "SECURITY_AGENTS",
    "DESIGN_AGENTS",
    "BaseAgent",
    "AgentOutput",
    "SharedContext",
]
