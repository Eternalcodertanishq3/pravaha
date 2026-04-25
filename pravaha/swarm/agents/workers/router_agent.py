"""Router Agent — Task routing and classification."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class RouterAgent(BaseAgent):
    role = "router"
    priority = 0
    max_tokens = 256
    temperature = 0.1

    system_prompt = (
        "You are a task router. Classify the task type and recommend\n"
        "the optimal agent pipeline. Output JSON:\n"
        '{"task_type": "code|research|writing|analysis|creative",\n'
        ' "pipeline": ["planner","coder","debugger"],\n'
        ' "complexity": "low|medium|high"}'
    )

    def can_handle(self, task_type: str) -> bool:
        return True
