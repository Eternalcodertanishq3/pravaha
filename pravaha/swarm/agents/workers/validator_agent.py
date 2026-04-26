"""Validator Agent — Output validation with verification tags."""

from __future__ import annotations

import re
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ValidatorAgent(BaseAgent):
    """Validate output with [VERIFIED]/[?]/[X] tags and compute verdict."""

    role = "validator"
    priority = 5
    max_tokens = 1024
    temperature = 0.2

    PASS_THRESHOLD = 0.1  # Max 10% incorrect rate to PASS

    system_prompt = """You are an output validator.

    Review the output and mark EVERY factual claim or code statement:
    - [VERIFIED] — you are confident this is correct
    - [?] — uncertain, cannot confirm without more context
    - [X] — this is incorrect or contains an error

    Rules:
    1. Mark EVERY significant claim (aim for 5+ tags minimum)
    2. For code: check syntax, logic, edge cases, return types
    3. For text: check facts, dates, names, logical consistency
    4. [VERIFIED] requires you to have HIGH confidence
    5. Default to [?] when unsure — never inflate confidence
    6. For [X] tags: briefly explain what's wrong after the tag
    7. Compute a final ratio: verified / total claims

    End with:
    VERIFIED: N
    UNCERTAIN: N
    INCORRECT: N
    VERDICT: PASS or FAIL
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        content = context.patched_output or context.code or context.output or ""
        if not content:
            return AgentOutput(
                role=self.role, output="No content to validate.",
                confidence=0.0, metadata={"verdict": "FAIL"},
            )

        # Check if content already has validation tags
        existing_verified = len(re.findall(r"\[VERIFIED\]", content))
        existing_uncertain = len(re.findall(r"\[\?\]", content))
        existing_incorrect = len(re.findall(r"\[X\]", content))
        has_tags = (existing_verified + existing_uncertain + existing_incorrect) > 0

        if has_tags:
            # Content already tagged — compute verdict directly
            output = content
            verified = existing_verified
            uncertain = existing_uncertain
            incorrect = existing_incorrect
        else:
            # Ask LLM to add validation tags
            prompt = self.build_prompt(
                f"Original task: {task}\n\n"
                f"Output to validate:\n{content[:2500]}",
                context,
            )
            output = await self._generate(prompt, engine)

            # Count tags in LLM output
            verified = len(re.findall(r"\[VERIFIED\]", output))
            uncertain = len(re.findall(r"\[\?\]", output))
            incorrect = len(re.findall(r"\[X\]", output))

        total = verified + uncertain + incorrect
        incorrect_rate = incorrect / max(total, 1)
        verdict = "PASS" if incorrect_rate < self.PASS_THRESHOLD else "FAIL"

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=(verified / max(total, 1)),
            metadata={
                "verified": verified,
                "uncertain": uncertain,
                "incorrect": incorrect,
                "total_claims": total,
                "incorrect_rate": round(incorrect_rate, 3),
                "verdict": verdict,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return True
