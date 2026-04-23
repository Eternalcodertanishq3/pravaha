"""Refiner Agent — Iterative quality improvement."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class RefinerAgent(BaseAgent):
    """Iteratively improves output quality based on feedback."""

    role = "refiner"
    priority = 1
    max_tokens = 2048
    temperature = 0.4
    system_prompt = (
        "You are an iterative improver. Take the output and make it 30%% better.\n\n"
        "Rules:\n"
        "1. Improve clarity, precision, and structure\n"
        "2. Do NOT change parts that are already correct\n"
        "3. If feedback is provided, address every point\n"
        "4. Mark significant changes with # REFINED: <what changed>\n"
        "5. Show ONLY the improved version — no explanations"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        content = context.output or context.code or ""
        feedback = context.feedback or "Improve overall quality"
        enriched = f"Output to refine:\n{content[:2000]}\n\nFeedback:\n{feedback[:500]}"
        prompt = self.build_prompt(enriched, context)
        output = await self._generate(prompt, engine)
        refinements = output.count("# REFINED:")
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.85,
            metadata={"refinements_made": refinements},
        )

    def can_handle(self, task_type: str) -> bool:
        return True
