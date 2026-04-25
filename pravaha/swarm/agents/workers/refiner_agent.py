"""Refiner Agent — Iterative output refinement."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class RefinerAgent(BaseAgent):
    role = "refiner"
    priority = 4
    max_tokens = 1024
    temperature = 0.3

    system_prompt = (
        "You are an output refiner. Take the current output and improve it:\n"
        "- Fix any remaining issues identified by critics or auditors\n"
        "- Improve clarity and readability\n"
        "- Ensure consistent formatting\n"
        "- Remove redundancy\n"
        "Output the refined version directly."
    )

    def can_handle(self, task_type: str) -> bool:
        return True
