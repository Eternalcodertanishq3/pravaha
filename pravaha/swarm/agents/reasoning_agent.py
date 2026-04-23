"""Reasoning Agent — Step-by-step chain-of-thought reasoning."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ReasoningAgent(BaseAgent):
    """Chain-of-thought reasoning for complex problems."""

    role = "reasoning"
    priority = 1
    max_tokens = 2048
    temperature = 0.1
    system_prompt = (
        "You are a logical reasoner. Think step by step.\n\n"
        "Rules:\n"
        "1. Show your work — label each step: Step 1, Step 2, ...\n"
        "2. Check each step before proceeding to the next\n"
        "3. Conclude ONLY from stated premises — no assumptions\n"
        "4. If a step requires an assumption, state it explicitly\n"
        "5. If you find a contradiction, stop and explain it\n"
        "6. End with: CONCLUSION: <your final answer>\n"
        "7. Rate your confidence: HIGH / MEDIUM / LOW"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        prompt = self.build_prompt(task, context)
        output = await self._generate(prompt, engine)
        context.reasoning = output
        steps = sum(
            1
            for line in output.split("\n")
            if "step" in line.lower() and any(c.isdigit() for c in line)
        )
        has_conclusion = "CONCLUSION:" in output.upper()
        if "CONFIDENCE" in output.upper():
            parts = output.upper().split("CONFIDENCE")[-1][:20]
            confidence_str = "HIGH" if "HIGH" in parts else "MEDIUM"
        else:
            confidence_str = "MEDIUM"
        conf_map = {"HIGH": 0.95, "MEDIUM": 0.7, "LOW": 0.4}
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=conf_map.get(confidence_str, 0.7),
            metadata={"steps": steps, "has_conclusion": has_conclusion},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"analysis", "math", "reasoning", "code", "general"}
