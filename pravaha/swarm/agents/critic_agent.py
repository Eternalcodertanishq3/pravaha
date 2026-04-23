"""Critic Agent — Multi-dimensional quality evaluation.

Evaluates output quality across clarity, correctness, completeness,
and efficiency. Provides specific scores and actionable feedback
that RefinerAgent can consume.

Priority: 1 (senior worker).
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class CriticAgent(BaseAgent):
    """Evaluates output quality with dimensional scoring."""

    role = "critic"
    priority = 1
    max_tokens = 512
    temperature = 0.6
    system_prompt = (
        "You are a quality critic. Evaluate the output for:\n\n"
        "1. **Clarity** (1-10): Is the output easy to understand?\n"
        "2. **Correctness** (1-10): Is the content factually/logically correct?\n"
        "3. **Completeness** (1-10): Does it fully address the task?\n"
        "4. **Efficiency** (1-10): Is it concise without being sparse?\n\n"
        "For each dimension:\n"
        "- Give a numerical score\n"
        "- Provide ONE specific example supporting your score\n"
        "- Suggest ONE concrete improvement\n\n"
        "End with:\n"
        "**Overall**: X/10\n"
        "**Top Priority Fix**: The single most impactful improvement\n\n"
        "Be specific and actionable. 'Could be better' is not useful feedback. "
        "'The error handling on line 15 silently swallows exceptions — add "
        "logging' is useful feedback."
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()

        content = context.code or context.output or ""
        enriched_task = f"Original task: {task}\n\nOutput to evaluate:\n{content[:2000]}"

        prompt = self.build_prompt(enriched_task, context)
        output = await self._generate(prompt, engine)

        # Store feedback for RefinerAgent
        context.feedback = output

        # Extract scores (heuristic)
        scores = {}
        for dimension in ["clarity", "correctness", "completeness", "efficiency", "overall"]:
            for line in output.lower().split("\n"):
                if dimension in line and "/10" in line:
                    try:
                        import re
                        nums = re.findall(r'(\d+)/10', line)
                        if nums:
                            scores[dimension] = int(nums[0])
                    except (ValueError, IndexError):
                        pass

        overall = scores.get("overall", 5)

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
        return True  # Critics evaluate everything
