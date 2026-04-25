"""Translator Agent — Multi-language translation."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class TranslatorAgent(BaseAgent):
    role = "translator"
    priority = 5
    max_tokens = 2048
    temperature = 0.2

    system_prompt = (
        "You are a translation agent. Translate content between languages.\n"
        "- Preserve meaning, tone, and context\n"
        "- Handle idioms appropriately\n"
        "- Maintain formatting"
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"translation", "writing", "general"}
