"""Edge Case Hunter Agent — Find unhandled edge cases."""

from __future__ import annotations

from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class EdgeCaseHunterAgent(BaseAgent):
    role = "edge_case_hunter"
    priority = 10
    max_tokens = 1024
    temperature = 0.2

    system_prompt = (
        "You are an edge case hunter. Find unhandled scenarios:\n"
        "- Empty inputs, None values, boundary conditions\n"
        "- Race conditions, timeouts\n"
        "- Unusual but valid inputs\n"
        "List each edge case and how to handle it."
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "analysis"}
