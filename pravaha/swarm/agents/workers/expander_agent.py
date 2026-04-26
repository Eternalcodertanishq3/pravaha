"""Expander Agent — Content expansion with ratio tracking."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ExpanderAgent(BaseAgent):
    """Expand brief input with depth, detail, and examples."""

    role = "expander"
    priority = 5
    max_tokens = 2048
    temperature = 0.6

    system_prompt = """You are a content expansion specialist.

    Take brief input and expand it with depth and detail.

    Expansion rules:
    1. Every abstract claim needs a concrete example
    2. Every complex concept needs an analogy
    3. Every step needs a "why this matters" explanation
    4. Technical content needs a real-world application
    5. Target 3-5x word count of the input
    6. Maintain the original tone and voice exactly
    7. Add section headers if output exceeds 300 words
    8. Never contradict or weaken the original content

    What NOT to do:
    - Don't add padding ("In conclusion...", "As we can see...")
    - Don't add unrelated tangents
    - Don't change the core argument
    - Don't over-explain obvious things

    Track expansion: report [EXPANSION_RATIO: X.Xx] at the end.
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        content = context.output or task
        input_words = len(content.split())

        prompt = self.build_prompt(
            f"Expand this content to 3-5x its current length "
            f"({input_words} words → target {input_words * 3}-{input_words * 5} words):\n\n"
            f"{content}",
            context,
        )
        output = await self._generate(prompt, engine)
        output_words = len(output.split())
        ratio = output_words / max(1, input_words)

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.85 if 2.5 <= ratio <= 6.0 else 0.6,
            metadata={
                "input_words": input_words,
                "output_words": output_words,
                "expansion_ratio": round(ratio, 2),
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"writing", "creative", "general", "research"}
