"""Merger Agent — Multi-output merging with conflict detection."""

from __future__ import annotations

import re
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class MergerAgent(BaseAgent):
    """Merge multiple outputs with real conflict detection and resolution."""

    role = "merger"
    priority = 6
    max_tokens = 1024
    temperature = 0.3

    # Contradictory modal patterns
    CONFLICT_PATTERNS = [
        (r"\bshould\b.*\bnot\b", r"\bshould\b(?!.*\bnot\b)"),
        (r"\bmust\b.*\bnot\b", r"\bmust\b(?!.*\bnot\b)"),
        (r"\bdo\b.*\bnot\b", r"\bdo\b(?!.*\bnot\b)"),
        (r"\bis\b.*\bnot\b", r"\bis\b(?!.*\bnot\b)"),
        (r"\bTrue\b", r"\bFalse\b"),
        (r"\benable\b", r"\bdisable\b"),
        (r"\byes\b", r"\bno\b"),
    ]

    system_prompt = """You are a merger agent that combines multiple outputs.

    Process:
    1. Read all input sources carefully
    2. Detect CONFLICTS: contradictory statements about the same topic
       Mark each: [CONFLICT: source A says X, source B says Y]
    3. RESOLVE conflicts by choosing the better-evidenced version
       Mark: [RESOLVED: chose X because Z]
    4. MERGE complementary information without duplication
    5. Maintain a coherent narrative structure
    6. Preserve technical accuracy over stylistic preferences
    7. If sources agree, note: [AGREED: both confirm X]
    8. Output the unified document with all markers

    Never silently drop conflicting information. Always mark and resolve.
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()

        # Gather all available outputs to merge
        sources: list[str] = []
        for name, ao in context.agent_outputs.items():
            if ao.output and name != self.role:
                sources.append(f"[Source: {name}]\n{ao.output[:500]}")

        if not sources:
            content = context.output or task
            sources = [content]

        # Detect conflicts via regex before LLM
        conflicts_detected = 0
        all_text = " ".join(sources)
        for pos_pattern, neg_pattern in self.CONFLICT_PATTERNS:
            has_pos = bool(re.search(pos_pattern, all_text, re.IGNORECASE))
            has_neg = bool(re.search(neg_pattern, all_text, re.IGNORECASE))
            if has_pos and has_neg:
                conflicts_detected += 1

        combined = "\n\n---\n\n".join(sources)
        prompt = self.build_prompt(
            f"Merge these {len(sources)} sources "
            f"(detected {conflicts_detected} potential conflicts):\n\n{combined}",
            context,
        )
        output = await self._generate(prompt, engine)
        context.merged_output = output

        resolved_count = output.count("[RESOLVED")
        agreed_count = output.count("[AGREED")

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=min(0.95, 0.6 + 0.1 * agreed_count),
            metadata={
                "sources_merged": len(sources),
                "conflicts_detected": conflicts_detected,
                "conflicts_resolved": resolved_count,
                "agreements": agreed_count,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return True
