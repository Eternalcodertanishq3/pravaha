"""Judge Agent — Final quality gate with weighted scoring."""

from __future__ import annotations

import re
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class JudgeAgent(BaseAgent):
    """Final quality gate: PASS/FAIL with weighted multi-criterion scoring."""

    role = "judge"
    priority = 7
    max_tokens = 512
    temperature = 0.2

    # Scoring weights (must sum to 1.0)
    WEIGHTS = {
        "correctness": 0.35,
        "completeness": 0.25,
        "relevance": 0.20,
        "clarity": 0.20,
    }
    PASS_THRESHOLD = 65.0

    system_prompt = """You are the final quality judge.

    Score the output on these weighted dimensions (0-100 each):
    1. CORRECTNESS (35%): Are facts accurate? Does code work?
    2. COMPLETENESS (25%): All requirements addressed? Edge cases?
    3. RELEVANCE (20%): Does output match the original task?
    4. CLARITY (20%): Well-organized, understandable, concise?

    Format EXACTLY as:
    CORRECTNESS: N/100
    COMPLETENESS: N/100
    RELEVANCE: N/100
    CLARITY: N/100
    WEIGHTED_SCORE: N/100
    VERDICT: PASS or FAIL
    TOP_ISSUE: [single biggest problem, or "None" if PASS]
    IMPROVEMENT: [one actionable fix, or "None" if PASS]
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        content = context.patched_output or context.code or context.output or ""
        if not content:
            return AgentOutput(
                role=self.role, output="VERDICT: FAIL\nNo content to judge.",
                confidence=0.0, metadata={"verdict": "FAIL", "score": 0},
            )

        prompt = self.build_prompt(
            f"Original task: {task}\n\nOutput to judge:\n{content[:2500]}", context,
        )
        output = await self._generate(prompt, engine)

        # Parse scores
        scores: dict[str, float] = {}
        for dim in self.WEIGHTS:
            m = re.search(rf"{dim}:\s*(\d+)/100", output, re.IGNORECASE)
            if m:
                scores[dim] = float(m.group(1))

        # Compute weighted score
        weighted = sum(
            scores.get(dim, 50) * w for dim, w in self.WEIGHTS.items()
        )
        verdict = "PASS" if weighted >= self.PASS_THRESHOLD else "FAIL"

        context.extra["judge_score"] = weighted
        context.extra["judge_verdict"] = verdict

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=weighted / 100.0,
            metadata={
                "scores": scores,
                "weighted_score": round(weighted, 1),
                "verdict": verdict,
                "pass_threshold": self.PASS_THRESHOLD,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return True
