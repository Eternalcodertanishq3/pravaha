"""Ensemble Agent — Multi-response aggregation."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class EnsembleAgent(BaseAgent):
    role = "ensemble"
    priority = 6
    max_tokens = 1024
    temperature = 0.3

    system_prompt = (
        "You are an ensemble aggregator. Combine multiple agent outputs\n"
        "into a single coherent response:\n"
        "- Merge complementary information\n"
        "- Resolve contradictions (favor higher-confidence sources)\n"
        "- Maintain coherent structure"
    )

    def can_handle(self, task_type: str) -> bool:
        return True
