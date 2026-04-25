"""Narrator Agent — Creative narrative writing."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class NarratorAgent(BaseAgent):
    role = "narrator"
    priority = 5
    max_tokens = 2048
    temperature = 0.8

    system_prompt = (
        "You are a creative narrator. Write compelling, vivid prose.\n"
        "- Rich sensory details\n"
        "- Strong voice and tone\n"
        "- Natural dialogue\n"
        "- Show, don't tell"
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"writing", "creative"}
