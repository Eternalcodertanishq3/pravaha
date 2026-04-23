"""Expander Agent — Content expansion with depth and detail."""

from __future__ import annotations
import time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ExpanderAgent(BaseAgent):
    """Expands brief content into detailed, thorough output."""

    role = "expander"
    priority = 0
    max_tokens = 2048
    temperature = 0.7
    system_prompt = (
        "You are a content expander. Take the given outline or brief and "
        "expand it into full, detailed content.\n\n"
        "Rules:\n"
        "1. Maintain consistent voice and depth throughout\n"
        "2. Add concrete examples for every abstract concept\n"
        "3. Include relevant analogies for complex ideas\n"
        "4. Target 3-5x expansion of the original length\n"
        "5. Do NOT add information that contradicts the original\n"
        "6. Add section headings if the output is long"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        content = task if len(task) > 50 else (context.output or "")
        prompt = self.build_prompt(f"Expand the following:\n\n{content}", context)
        output = await self._generate(prompt, engine)
        input_len = len(content.split())
        output_len = len(output.split())
        expansion = output_len / input_len if input_len > 0 else 1.0
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(role=self.role, output=output, tokens_used=self._total_tokens,
                           duration_ms=duration, confidence=0.8,
                           metadata={"expansion_ratio": round(expansion, 2)})

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"writing", "general", "research"}
