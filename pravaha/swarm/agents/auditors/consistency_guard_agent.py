"""Consistency Guard Agent — Check for contradictions in output."""

from __future__ import annotations

from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ConsistencyGuardAgent(BaseAgent):
    role = "consistency_guard"
    priority = 10
    max_tokens = 1024
    temperature = 0.1

    system_prompt = (
        "You are a consistency auditor. Check output for:\n"
        "- Contradictory statements\n"
        "- Conflicting data points\n"
        "- Inconsistent naming or terminology\n"
        "Report each contradiction with line references."
    )

    def can_handle(self, task_type: str) -> bool:
        return True
