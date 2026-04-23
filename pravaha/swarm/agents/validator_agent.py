"""Validator Agent — Factual correctness verification.

Verifies each factual claim in the output, marking them as
verified [V], unverifiable [?], or incorrect [X]. Provides
corrections for incorrect claims.

Priority: 1 (senior worker).
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ValidatorAgent(BaseAgent):
    """Checks factual correctness of all claims in output."""

    role = "validator"
    priority = 1
    max_tokens = 512
    temperature = 0.1
    system_prompt = (
        "You are a fact validator. For EVERY factual claim in the output:\n\n"
        "1. Mark verified claims with [V] — you are confident this is correct\n"
        "2. Mark unverifiable claims with [?] — cannot confirm or deny\n"
        "3. Mark incorrect claims with [X] — this is factually wrong\n\n"
        "For each [X] claim, provide the correct information.\n"
        "For each [?] claim, explain what would be needed to verify.\n\n"
        "Format:\n"
        "[V] 'Python was created by Guido van Rossum' — Correct\n"
        "[X] 'Python 3.12 was released in 2024' — Incorrect: released Oct 2023\n"
        "[?] 'This library supports 100k concurrent connections' — Need benchmarks\n\n"
        "End with a summary:\n"
        "VERIFIED: N claims  |  UNCERTAIN: N claims  |  INCORRECT: N claims\n"
        "VERDICT: PASS / FAIL / NEEDS_REVIEW"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()

        content = context.output or context.code or context.research or ""
        enriched_task = f"Validate the factual claims in this output:\n\n{content[:2000]}"

        prompt = self.build_prompt(enriched_task, context)
        output = await self._generate(prompt, engine)

        # Count verdicts
        verified = output.count("[V]")
        uncertain = output.count("[?]")
        incorrect = output.count("[X]")
        total = verified + uncertain + incorrect

        passed = "PASS" in output.upper() and "FAIL" not in output.upper()
        confidence = verified / total if total > 0 else 0.5

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=confidence,
            metadata={
                "verified": verified,
                "uncertain": uncertain,
                "incorrect": incorrect,
                "passed": passed,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"research", "analysis", "writing", "general", "facts"}
