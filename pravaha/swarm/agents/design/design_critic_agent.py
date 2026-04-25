"""Design Critic Agent — Visual design quality review.

Scores design across 5 dimensions (1-10) and provides actionable
improvements with visual references.
"""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class DesignCriticAgent(BaseAgent):
    role = "design_critic"
    priority = 1
    max_tokens = 1024
    temperature = 0.7

    system_prompt = (
        "You are a senior design critic. Review a design for "
        "overall quality and brand coherence.\n\n"
        "Score on:\n"
        "- Visual hierarchy (1-10): Is importance communicated visually?\n"
        "- Consistency (1-10): Are patterns repeated correctly?\n"
        "- Simplicity (1-10): Is complexity justified?\n"
        "- Delight (1-10): Does it exceed functional expectations?\n"
        "- Brand alignment (1-10): Does it match the brand voice?\n\n"
        "For each dimension below 7/10, provide specific examples "
        "and actionable improvements with visual references.\n\n"
        "Conclude with: TOP 3 changes that would most improve this design."
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"design", "ui", "frontend"}
