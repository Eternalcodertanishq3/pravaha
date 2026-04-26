"""Critic Agent — Rigorous 4-dimension quality evaluation."""

from __future__ import annotations

import re
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class CriticAgent(BaseAgent):
    """Evaluate output across correctness, completeness, clarity, efficiency."""

    role = "critic"
    priority = 3
    max_tokens = 1024
    temperature = 0.4

    system_prompt = """You are a rigorous quality critic.

    Evaluate output across FOUR dimensions. Score each 1-10:

    1. CORRECTNESS (1-10)
       - Are facts accurate?
       - Is logic sound with no contradictions?
       - Does code actually work (not just look plausible)?
       Score 10 only if you are certain everything is correct.

    2. COMPLETENESS (1-10)
       - Does output address ALL parts of the original task?
       - Are edge cases handled?
       - Are examples/tests included where needed?
       Score 10 only if nothing is missing.

    3. CLARITY (1-10)
       - Is it immediately understandable to the intended audience?
       - Is structure logical with good flow?
       - Are variable/function names descriptive?

    4. EFFICIENCY (1-10)
       - Is the solution concise without being sparse?
       - No unnecessary complexity, no over-engineering?
       - Optimal approach for the problem size?

    For each dimension below 7:
    - Give ONE specific example of the problem
    - Give ONE concrete fix (actionable, not vague)

    Format EXACTLY as:
    CORRECTNESS: N/10 — [example] → [fix]
    COMPLETENESS: N/10 — [example] → [fix]
    CLARITY: N/10 — [example] → [fix]
    EFFICIENCY: N/10 — [example] → [fix]
    OVERALL: N/10
    TOP FIX: [single most impactful improvement]
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        content = context.code or context.output or ""
        if not content:
            return AgentOutput(
                role=self.role,
                output="No content to critique.",
                duration_ms=0.0,
                confidence=0.5,
            )

        enriched = (
            f"Original task: {task}\n\n"
            f"Output to critique:\n{content[:2500]}"
        )
        prompt = self.build_prompt(enriched, context)
        output = await self._generate(prompt, engine)

        # Store critique as feedback for RefinerAgent
        context.feedback = output

        # Parse scores
        scores: dict[str, int] = {}
        for dim in ["correctness", "completeness", "clarity", "efficiency", "overall"]:
            m = re.search(
                rf"{dim}:\s*(\d+)/10",
                output, re.IGNORECASE,
            )
            if m:
                scores[dim] = int(m.group(1))

        overall = scores.get("overall", sum(scores.values()) // max(len(scores), 1))

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=overall / 10.0,
            metadata={"scores": scores, "overall": overall},
        )

    def can_handle(self, task_type: str) -> bool:
        return True
