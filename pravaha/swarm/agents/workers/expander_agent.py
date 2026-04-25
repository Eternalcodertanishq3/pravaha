"""Expander Agent — Content expansion and elaboration."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class ExpanderAgent(BaseAgent):
    role = "expander"
    priority = 5
    max_tokens = 1536
    temperature = 0.6

    system_prompt = (
        "You are a content expander. Elaborate on the given content:\n"
        "- Add supporting details and examples\n"
        "- Develop arguments with evidence\n"
        "- Maintain consistency with original tone\n"
        "- Don't contradict existing content"
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"writing", "creative", "general"}
