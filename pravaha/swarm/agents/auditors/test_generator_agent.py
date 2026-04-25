"""Test Generator Agent — Auto-generate test cases."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class TestGeneratorAgent(BaseAgent):
    role = "test_generator"
    priority = 10
    max_tokens = 2048
    temperature = 0.2

    system_prompt = (
        "Generate comprehensive pytest test cases for the given code.\n"
        "Cover: happy path, edge cases, error cases.\n"
        "Use descriptive test names and docstrings."
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
