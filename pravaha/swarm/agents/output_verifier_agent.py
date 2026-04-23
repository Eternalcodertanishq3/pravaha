"""Output Verifier Agent — Task satisfaction scorer."""

from __future__ import annotations

import re
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class OutputVerifierAgent(BaseAgent):
    """Scores task satisfaction 0-100 with requirement tracking."""

    role = "output_verifier"
    priority = 2
    max_tokens = 512
    temperature = 0.1
    system_prompt = (
        "You are a task verifier. Given the original task and output:\n"
        "1. Score task satisfaction 0-100\n"
        "2. List requirements MET\n"
        "3. List requirements NOT MET\n"
        "4. List requirements PARTIALLY met\n"
        "5. Give a one-line verdict\n\n"
        "Return JSON:\n"
        '{"score": <0-100>, "met": ["..."], "not_met": ["..."], '
        '"partial": ["..."], "verdict": "...", "should_retry": true|false}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        content = context.output or context.code or context.merged_output or ""
        prompt = self.build_prompt(
            f"Original task: {task[:500]}\n\nOutput to verify:\n{content[:1500]}", context
        )
        result = await self._generate_json(prompt, engine)
        score = result.get("score", 50)
        if isinstance(score, str):
            nums = re.findall(r"\d+", score)
            score = int(nums[0]) if nums else 50
        verdict = result.get("verdict", "")
        should_retry = result.get("should_retry", score < 70)
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=verdict,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=score / 100.0,
            metadata={"score": score, "should_retry": should_retry, "details": result},
        )

    def can_handle(self, task_type: str) -> bool:
        return True  # Always runs
