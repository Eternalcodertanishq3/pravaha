"""Summarizer Agent — Concise summary generation."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class SummarizerAgent(BaseAgent):
    role = "summarizer"
    priority = 5
    max_tokens = 512
    temperature = 0.3

    system_prompt = (
        "You are a summarization agent. Create concise, accurate summaries.\n"
        "- Preserve all key facts and conclusions\n"
        "- Maintain logical flow\n"
        "- Use bullet points for multiple items\n"
        "- Target 20% of original length"
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"research", "analysis", "writing", "general"}
