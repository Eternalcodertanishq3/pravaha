"""Memory Agent — Context memory management."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class MemoryAgent(BaseAgent):
    role = "memory"
    priority = 1
    max_tokens = 512
    temperature = 0.1

    system_prompt = (
        "You are a memory manager agent. Summarize and store key context.\n"
        "Extract: entities, decisions, outcomes, and open questions.\n"
        "Output a structured summary for future agent reference."
    )

    def can_handle(self, task_type: str) -> bool:
        return True
