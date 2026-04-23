"""Summarizer Agent — Intelligent content condensation.

Reduces input to 20% of original length while preserving all
key facts and structure. No information is truly lost — only
padding is removed.

Priority: 0 (worker).
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class SummarizerAgent(BaseAgent):
    """Condenses long content while preserving essential information."""

    role = "summarizer"
    priority = 0
    max_tokens = 512
    temperature = 0.3
    system_prompt = (
        "You are a summarizer. Condense the input to its essential points.\n\n"
        "Rules:\n"
        "1. Preserve ALL key facts, decisions, and data points\n"
        "2. Remove padding, filler words, and redundant explanations\n"
        "3. Output should be ≤20% of input length\n"
        "4. Maintain the original structure (bullets, numbered lists)\n"
        "5. Keep technical terms exact — do not paraphrase jargon\n"
        "6. If the input contains code, keep the code but remove comments\n"
        "7. Prioritize actionable information over background\n\n"
        "Format: Use bullet points for maximum information density."
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()

        content = task if len(task) > 100 else (context.output or context.research or "")
        prompt = self.build_prompt(f"Summarize the following:\n\n{content}", context)
        output = await self._generate(prompt, engine)

        # Compute compression ratio
        input_len = len(content.split())
        output_len = len(output.split())
        ratio = output_len / input_len if input_len > 0 else 1.0

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.9 if ratio <= 0.3 else 0.7,
            metadata={
                "input_words": input_len,
                "output_words": output_len,
                "compression_ratio": round(ratio, 3),
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return True  # Can summarize anything
