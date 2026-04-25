"""Reasoning Agent — Chain-of-thought with real verification."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ReasoningAgent(BaseAgent):
    """Autonomous reasoning agent with math verification via code execution."""

    role = "reasoning"
    priority = 1
    max_tokens = 2048
    temperature = 0.1
    max_react_steps = 4
    available_tools = ["execute_python"]

    system_prompt = (
        "You are an autonomous reasoning agent.\n\n"
        "You think rigorously and verify your conclusions.\n\n"
        "For mathematical claims: use execute_python to verify.\n"
        "For logical claims: check each step against premises.\n"
        "For causal claims: distinguish correlation from causation.\n\n"
        "Never state a conclusion you haven't checked.\n"
        "Show all work. Label each step. Identify assumptions.\n\n"
        "Format:\n"
        "Step 1: <reasoning>\n"
        "Step 2: <reasoning>\n"
        "...\n"
        "CONCLUSION: <your final answer>\n"
        "CONFIDENCE: HIGH | MEDIUM | LOW"
    )

    async def run(
        self, task: str, context: SharedContext, engine: Any
    ) -> AgentOutput:
        if self._tool_registry and self.available_tools:
            result = await self.run_react(task, context, engine)
            context.reasoning = result.output
            return result

        t0 = time.time()
        prompt = self.build_prompt(task, context)
        output = await self._generate(prompt, engine)
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        context.reasoning = output
        steps = sum(
            1
            for line in output.split("\n")
            if "step" in line.lower() and any(c.isdigit() for c in line)
        )
        has_conclusion = "CONCLUSION:" in output.upper()

        # Parse confidence
        if "CONFIDENCE" in output.upper():
            parts = output.upper().split("CONFIDENCE")[-1][:20]
            if "HIGH" in parts:
                confidence_str = "HIGH"
            elif "LOW" in parts:
                confidence_str = "LOW"
            else:
                confidence_str = "MEDIUM"
        else:
            confidence_str = "MEDIUM"

        conf_map = {"HIGH": 0.95, "MEDIUM": 0.7, "LOW": 0.4}

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
