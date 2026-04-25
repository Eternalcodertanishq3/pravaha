"""Classifier Agent — Content classification."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class ClassifierAgent(BaseAgent):
    role = "classifier"
    priority = 3
    max_tokens = 512
    temperature = 0.1

    system_prompt = (
        "You are a classification agent. Classify content into categories:\n"
        "- Determine the primary category\n"
        "- List applicable secondary categories\n"
        "- Provide confidence level for each\n"
        "- Output in JSON format"
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"classification", "analysis", "general"}
