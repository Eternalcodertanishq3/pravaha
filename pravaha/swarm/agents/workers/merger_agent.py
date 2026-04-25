"""Merger Agent — Multi-output merging."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class MergerAgent(BaseAgent):
    role = "merger"
    priority = 6
    max_tokens = 1024
    temperature = 0.3

    system_prompt = (
        "You are a merger agent. Merge multiple text outputs into one:\n"
        "- Remove duplicates\n"
        "- Resolve contradictions\n"
        "- Create a unified, coherent document"
    )

    def can_handle(self, task_type: str) -> bool:
        return True
