"""Refiner Agent — Addresses critic feedback point by point."""

from __future__ import annotations

import re
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class RefinerAgent(BaseAgent):
    """Read context.feedback from CriticAgent and address each specific point."""

    role = "refiner"
    priority = 4
    max_tokens = 2048
    temperature = 0.3

    system_prompt = """You are a precision refiner.

    You receive feedback from a critic and the original output.
    Your job is to address EACH feedback point specifically.

    Rules:
    1. Read every feedback point carefully
    2. For each issue: fix it in the output
    3. Mark every change with: # REFINED: <reason from feedback>
    4. Do NOT introduce new content unrelated to the feedback
    5. Do NOT remove working code/content that wasn't criticized
    6. Preserve the structure and style of the original
    7. If feedback contradicts itself, follow the higher-scoring dimension
    8. If a feedback point is unclear, make a conservative fix
    9. After all fixes, do a final coherence check
    10. Report how many feedback points were addressed

    Quality gate: every marked REFINED comment must trace to a
    specific feedback point. No phantom improvements.
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        content = context.code or context.output or ""
        feedback = context.feedback or ""

        if not feedback:
            return AgentOutput(
                role=self.role, output=content,
                duration_ms=0.0, confidence=0.7,
                metadata={"feedback_points_addressed": 0, "improvements_made": 0},
            )

        # Count feedback points (lines with scores or bullets)
        feedback_lines = [
            line.strip() for line in feedback.split("\n")
            if line.strip() and (
                re.search(r"\d+/10", line)
                or line.strip().startswith(("-", "•", "*", "→"))
            )
        ]

        prompt = self.build_prompt(
            f"Feedback from critic:\n{feedback[:1500]}\n\n"
            f"Original output to refine:\n{content[:2500]}\n\n"
            f"Address each feedback point. Mark changes with # REFINED: <reason>",
            context,
        )
        output = await self._generate(prompt, engine)

        # Track improvements
        refined_count = output.count("# REFINED:")
        context.output = output

        if self._memory:
            self._memory.store(
                f"Refined: {refined_count} changes from {len(feedback_lines)} feedback points",
                importance=0.5,
            )

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=min(0.95, 0.5 + 0.1 * refined_count),
            metadata={
                "feedback_points_addressed": len(feedback_lines),
                "improvements_made": refined_count,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return True
