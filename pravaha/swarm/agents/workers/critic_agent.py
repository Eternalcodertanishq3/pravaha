"""Critic Agent — Evaluates quality and finds weaknesses."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class CriticAgent(BaseAgent):
    role = "critic"
    priority = 3
    max_tokens = 1024
    temperature = 0.4

    system_prompt = (
        "You are a quality critic. Evaluate the provided output for:\n"
        "- Correctness: Are there factual or logical errors?\n"
        "- Completeness: Does it address all requirements?\n"
        "- Clarity: Is it well-organized and understandable?\n"
        "- Edge cases: Are they handled?\n\n"
        "Give a score 1-10 with detailed justification.\n"
        "Format: SCORE: N/10 followed by STRENGTHS: and WEAKNESSES:"
    )

    def can_handle(self, task_type: str) -> bool:
        return True
