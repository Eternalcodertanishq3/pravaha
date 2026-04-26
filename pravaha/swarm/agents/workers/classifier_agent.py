"""Classifier Agent — Precision content classification."""

from __future__ import annotations

import json
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ClassifierAgent(BaseAgent):
    """Classify tasks along multiple dimensions for intelligent routing."""

    role = "classifier"
    priority = 3
    max_tokens = 256
    temperature = 0.1
    available_tools = ["memory"]

    system_prompt = """You are a precision content classifier.

    Classify the input along these EXACT dimensions:
    1. task_type: code | research | writing | analysis | math |
                  translation | design | security | general
    2. complexity: 1 (trivial) to 5 (requires deep expertise)
    3. domain: software | science | business | creative | legal |
               medical | financial | general
    4. requires_tools: true | false
    5. recommended_pipeline: name of the best pipeline for this task

    Output ONLY this JSON:
    {
      "task_type": "...",
      "complexity": N,
      "domain": "...",
      "requires_tools": true|false,
      "recommended_pipeline": "...",
      "confidence": 0.0-1.0,
      "reasoning": "one sentence justification"
    }

    Never add text outside the JSON. Never guess randomly.
    Base complexity on: specialized knowledge needed, steps required,
    ambiguity in requirements, domain expertise needed.
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        prompt = self.build_prompt(task, context)
        result = await self._generate_json(prompt, engine, max_tokens=150)

        # Write classification into shared context
        context.task_type = result.get("task_type", "general")
        context.extra["complexity"] = result.get("complexity", 3)
        context.extra["domain"] = result.get("domain", "general")
        context.extra["classification"] = result

        # Store in memory for future similar tasks
        if self._memory:
            self._memory.store(
                f"Classified '{task[:40]}' as "
                f"{context.task_type}/{result.get('domain')} "
                f"complexity={result.get('complexity')}",
                importance=0.4,
            )

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        confidence = float(str(result.get("confidence", 0.8)))

        return AgentOutput(
            role=self.role,
            output=json.dumps(result),
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=confidence,
            metadata=result,
        )

    def can_handle(self, task_type: str) -> bool:
        return True  # Classifier handles any input
