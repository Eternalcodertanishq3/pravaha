"""Output Verifier Agent — Final quality scoring."""

from __future__ import annotations

from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class OutputVerifierAgent(BaseAgent):
    role = "output_verifier"
    priority = 11
    max_tokens = 512
    temperature = 0.1

    system_prompt = (
        "You are the final output verifier. Score the output 0-100.\n"
        "Criteria: correctness, completeness, safety, readability.\n"
        "Output: SCORE: XX\nISSUES: list\nVERDICT: PASS/FAIL"
    )

    def can_handle(self, task_type: str) -> bool:
        return True
