"""Narrator Agent — Jargon replacement and readability improvement."""

from __future__ import annotations

import re
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class NarratorAgent(BaseAgent):
    """Detect jargon and rewrite for clarity, targeting grade-8 readability."""

    role = "narrator"
    priority = 5
    max_tokens = 1536
    temperature = 0.4

    # Common technical jargon → plain language mapping
    JARGON_MAP: dict[str, str] = {
        "utilize": "use",
        "implement": "build",
        "instantiate": "create",
        "propagate": "spread",
        "leverage": "use",
        "paradigm": "approach",
        "refactor": "restructure",
        "deprecated": "outdated",
        "enumerate": "list",
        "concatenate": "join",
        "serialize": "convert to text",
        "deserialize": "convert from text",
        "idempotent": "safely repeatable",
        "orthogonal": "independent",
        "agnostic": "independent of",
        "boilerplate": "template code",
        "syntactic sugar": "shorthand",
    }

    system_prompt = """You are a narrative clarity specialist.

    Rewrite technical content for maximum readability:

    1. Target Flesch-Kincaid Grade Level 8 (understandable by a 14-year-old)
    2. Replace jargon with plain language equivalents
    3. Use ACTIVE voice, never passive
    4. Break long sentences (>25 words) into shorter ones
    5. Add analogies for complex concepts
    6. Preserve ALL technical accuracy — simplify language, not meaning
    7. Keep code blocks, variable names, and API names UNCHANGED
    8. Add transition phrases between sections for flow
    9. Use "you" to address the reader directly
    10. End with a one-sentence summary

    Mark changes: [SIMPLIFIED: original → rewrite]
    Mark analogies: [ANALOGY: complex concept ≈ simple comparison]
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        content = context.output or task

        # Phase 1: Static jargon detection
        jargon_found: list[str] = []
        for jargon in self.JARGON_MAP:
            if re.search(rf"\b{jargon}\b", content, re.IGNORECASE):
                jargon_found.append(jargon)

        # Phase 2: LLM rewrite
        jargon_hint = ""
        if jargon_found:
            replacements = ", ".join(
                f'"{j}" → "{self.JARGON_MAP[j]}"' for j in jargon_found[:10]
            )
            jargon_hint = f"\n\nDetected jargon to replace: {replacements}"

        prompt = self.build_prompt(
            f"Rewrite this for grade-8 readability:{jargon_hint}\n\n{content[:3000]}",
            context,
        )
        output = await self._generate(prompt, engine)

        # Count improvements
        analogies_added = output.count("[ANALOGY:")
        simplified = output.count("[SIMPLIFIED:")

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.8,
            metadata={
                "jargon_replaced": len(jargon_found),
                "analogies_added": analogies_added,
                "simplifications": simplified,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"writing", "creative", "general", "research"}
