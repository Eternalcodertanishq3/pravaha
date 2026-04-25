"""Self Reflection Agent — Post-task self-analysis."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class SelfReflectionAgent(BaseAgent):
    role = "self_reflection"
    priority = 13
    max_tokens = 512
    temperature = 0.3

    system_prompt = (
        "Reflect on the swarm's performance for this task.\n"
        "What went well? What could improve? Log internally."
    )

    def can_handle(self, task_type: str) -> bool:
        return True
