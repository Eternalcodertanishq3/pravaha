"""Extractor Agent — Structured data extraction."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class ExtractorAgent(BaseAgent):
    role = "extractor"
    priority = 3
    max_tokens = 1024
    temperature = 0.1

    system_prompt = (
        "You are a data extraction agent. Extract structured data:\n"
        "- Identify entities, relationships, and attributes\n"
        "- Output in JSON format\n"
        "- Be precise — no hallucinated fields\n"
        "- Handle missing data with null, not guesses"
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"extraction", "analysis", "general"}
