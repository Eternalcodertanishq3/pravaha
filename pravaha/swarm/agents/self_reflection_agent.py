"""Self Reflection Agent — Meta-cognitive engine auditor."""

from __future__ import annotations
import time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class SelfReflectionAgent(BaseAgent):
    """Audits the engine's own decisions for improvement."""

    role = "self_reflection"
    priority = 1
    max_tokens = 512
    temperature = 0.2
    system_prompt = (
        "You are the engine's internal auditor. Review the inference "
        "decisions made for this request:\n"
        "- Was the right pipeline chosen for this task type?\n"
        "- Were agent token budgets appropriate?\n"
        "- Did any agent exceed its scope?\n"
        "- Were there unnecessary steps?\n"
        "- What would you do differently?\n\n"
        "This output is logged for system improvement, not shown to user.\n\n"
        "Return JSON:\n"
        '{"pipeline_quality": "good|ok|poor", '
        '"improvements": ["..."], "inefficiencies": ["..."], '
        '"token_waste_estimate": <0-100>}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        agents_used = list(context.agent_outputs.keys())
        prompt = self.build_prompt(
            f"Reflect on pipeline:\nTask: {task[:300]}\nAgents used: {agents_used}\n"
            f"Outputs generated: {len(context.agent_outputs)}", context)
        result = await self._generate_json(prompt, engine)
        quality = result.get("pipeline_quality", "ok")
        improvements = result.get("improvements", [])
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role, output=f"Pipeline: {quality} | {len(improvements)} improvement(s)",
            tokens_used=self._total_tokens, duration_ms=duration,
            confidence={"good": 0.9, "ok": 0.7, "poor": 0.4}.get(quality, 0.5),
            metadata=result,
        )

    def can_handle(self, task_type: str) -> bool:
        return True  # Always runs (logged only)
