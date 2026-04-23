"""Translator Agent — Professional multilingual translation."""

from __future__ import annotations
import time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class TranslatorAgent(BaseAgent):
    """Translates content between languages with cultural sensitivity."""

    role = "translator"
    priority = 0
    max_tokens = 2048
    temperature = 0.2
    system_prompt = (
        "You are a professional translator. Translate with cultural sensitivity.\n\n"
        "Rules:\n"
        "1. Preserve tone and register of the original\n"
        "2. For technical terms, use target-language conventions\n"
        "3. Keep proper nouns, brand names, and code unchanged\n"
        "4. Add translator notes [TN: ...] for ambiguous phrases\n"
        "5. Maintain original formatting (lists, headings, etc.)\n"
        "6. If no target language specified, translate to English"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        prompt = self.build_prompt(task, context)
        output = await self._generate(prompt, engine)
        notes = output.count("[TN:")
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(role=self.role, output=output, tokens_used=self._total_tokens,
                           duration_ms=duration, confidence=0.85,
                           metadata={"translator_notes": notes})

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"translation", "general"}
