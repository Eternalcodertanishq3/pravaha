"""Classifier Agent — Domain/intent classification."""

from __future__ import annotations
import time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ClassifierAgent(BaseAgent):
    """Classifies input by domain, intent, and complexity."""

    role = "classifier"
    priority = 0
    max_tokens = 256
    temperature = 0.1
    system_prompt = (
        "You are an input classifier. Determine the characteristics of the input.\n\n"
        "Return ONLY valid JSON:\n"
        '{"domain": "<area>", "intent": "<what user wants>", '
        '"complexity": <1-5>, "urgency": "low|medium|high", '
        '"agents": ["<recommended_agent_1>", "<recommended_agent_2>"]}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        prompt = self.build_prompt(task, context)
        result = await self._generate_json(prompt, engine, max_tokens=128)
        context.task_type = result.get("domain", "general")
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(role=self.role, output=str(result), tokens_used=self._total_tokens,
                           duration_ms=duration, confidence=0.9, metadata=result)

    def can_handle(self, task_type: str) -> bool:
        return True
