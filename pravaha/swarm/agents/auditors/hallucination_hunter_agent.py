"""Hallucination Hunter Agent — Detect fabricated claims."""

from __future__ import annotations

from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class HallucinationHunterAgent(BaseAgent):
    role = "hallucination_hunter"
    priority = 10
    max_tokens = 1024
    temperature = 0.1

    system_prompt = (
        "You are a hallucination detector. Check for:\n"
        "- Fabricated statistics or citations\n"
        "- Invented API methods or libraries\n"
        "- False claims stated as facts\n"
        "- Made-up code that doesn't exist\n"
        "Tag each finding: [HALLUCINATION] description"
    )

    def can_handle(self, task_type: str) -> bool:
        return True
