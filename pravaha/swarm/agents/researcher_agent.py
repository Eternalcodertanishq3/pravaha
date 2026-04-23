"""Researcher Agent — Information gathering and synthesis.

Gathers relevant context, cites sources with confidence levels,
and flags uncertain claims. Feeds structured research to downstream
agents like Coder or Narrator.

Priority: 1 (senior worker) — runs early to provide context.
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ResearcherAgent(BaseAgent):
    """Gathers and synthesizes information with confidence tracking."""

    role = "researcher"
    priority = 1
    max_tokens = 1024
    temperature = 0.5
    system_prompt = (
        "You are a research synthesizer. Your job is to gather and organize "
        "all relevant information about the given topic.\n\n"
        "Rules:\n"
        "1. Cite what you know with confidence levels:\n"
        "   [HIGH] — well-established facts you are certain about\n"
        "   [MEDIUM] — likely correct but verify before critical use\n"
        "   [LOW] — uncertain, may be outdated or incomplete\n"
        "2. Flag any claim you cannot verify with [UNCERTAIN]\n"
        "3. Organize findings by relevance to the task\n"
        "4. Include both supporting and contradicting evidence\n"
        "5. Summarize key takeaways at the end\n"
        "6. If referencing APIs, libraries, or tools, specify exact versions\n\n"
        "Format your output as:\n"
        "## Key Findings\n"
        "- [HIGH] Finding 1...\n"
        "- [MEDIUM] Finding 2...\n\n"
        "## Uncertainties\n"
        "- [UNCERTAIN] Claim that needs verification...\n\n"
        "## Summary\n"
        "Concise summary of research relevant to the task."
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()

        # Enrich prompt with plan if available
        prompt = self.build_prompt(task, context)
        output = await self._generate(prompt, engine)

        # Store research in shared context
        context.research = output

        # Count confidence-tagged claims
        high = output.count("[HIGH]")
        medium = output.count("[MEDIUM]")
        low = output.count("[LOW]")
        uncertain = output.count("[UNCERTAIN]")

        # Self-assessed confidence based on claim distribution
        total_claims = high + medium + low + uncertain
        confidence = (
            (high * 1.0 + medium * 0.7 + low * 0.3) / total_claims
            if total_claims > 0
            else 0.5
        )

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=min(confidence, 1.0),
            metadata={
                "claims_high": high,
                "claims_medium": medium,
                "claims_low": low,
                "claims_uncertain": uncertain,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"research", "analysis", "general", "writing"}
