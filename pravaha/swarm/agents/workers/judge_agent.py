"""Judge Agent — Final quality judgment."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class JudgeAgent(BaseAgent):
    role = "judge"
    priority = 7
    max_tokens = 512
    temperature = 0.2

    system_prompt = (
        "You are the final judge. Give a pass/fail decision:\n"
        "VERDICT: PASS or FAIL\n"
        "SCORE: 0-100\n"
        "REASON: Why"
    )

    def can_handle(self, task_type: str) -> bool:
        return True
