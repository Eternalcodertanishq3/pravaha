"""Judge Agent — Quality arbiter with structured scoring."""

from __future__ import annotations
import re, time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class JudgeAgent(BaseAgent):
    """Scores outputs and picks the best among candidates."""

    role = "judge"
    priority = 2
    max_tokens = 256
    temperature = 0.2
    system_prompt = (
        "You are a quality judge. Given one or multiple outputs, "
        "evaluate and select the best.\n\n"
        "Return ONLY valid JSON:\n"
        '{"winner": <index 0-based or -1 if single>, '
        '"score": <0-100>, "reason": "<one sentence justification>", '
        '"dimensions": {"accuracy": <0-100>, "completeness": <0-100>, '
        '"clarity": <0-100>, "efficiency": <0-100>}}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        content = context.merged_output or context.output or context.code or ""
        prompt = self.build_prompt(f"Judge this output:\n\n{content[:2000]}", context)
        result = await self._generate_json(prompt, engine)
        score = result.get("score", 50)
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(role=self.role, output=result.get("reason", str(result)),
                           tokens_used=self._total_tokens, duration_ms=duration,
                           confidence=score / 100.0, metadata=result)

    def can_handle(self, task_type: str) -> bool:
        return True
