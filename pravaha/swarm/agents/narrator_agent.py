"""Narrator Agent — Technical-to-readable prose conversion."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class NarratorAgent(BaseAgent):
    """Converts technical output into clear, engaging prose."""

    role = "narrator"
    priority = 0
    max_tokens = 1024
    temperature = 0.7
    system_prompt = (
        "You are a technical writer. Convert technical output into clear, "
        "readable prose for a non-expert audience.\n\n"
        "Rules:\n"
        "1. Use analogies to explain complex concepts\n"
        "2. Avoid jargon — if you must use a technical term, define it\n"
        "3. Be engaging: use active voice and varied sentence structure\n"
        "4. Break complex ideas into digestible paragraphs\n"
        "5. Use examples to illustrate abstract points\n"
        "6. Maintain accuracy while improving accessibility"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        content = context.output or context.code or task
        prompt = self.build_prompt(f"Rewrite for a general audience:\n\n{content[:2000]}", context)
        output = await self._generate(prompt, engine)
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.8,
            metadata={"original_type": "technical", "target_audience": "general"},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"writing", "general", "research"}
