"""Validator Agent — Output validation against requirements."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class ValidatorAgent(BaseAgent):
    role = "validator"
    priority = 5
    max_tokens = 512
    temperature = 0.1

    system_prompt = (
        "You are a validation agent. Check output against requirements:\n"
        "- Does it address all parts of the task?\n"
        "- Is the format correct?\n"
        "- Are there any logical inconsistencies?\n"
        "VERDICT: VALID or INVALID with reasons."
    )

    def can_handle(self, task_type: str) -> bool:
        return True
