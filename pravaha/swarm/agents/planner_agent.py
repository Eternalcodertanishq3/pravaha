"""Planner Agent — Task decomposition specialist.

Breaks complex user requests into numbered, independently
executable sub-tasks. Outputs structured plans that downstream
agents (Researcher, Coder, etc.) consume directly.

Priority: 2 (orchestrator) — runs first in most pipelines.
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class PlannerAgent(BaseAgent):
    """Decomposes tasks into clear, numbered sub-tasks."""

    role = "planner"
    priority = 2
    max_tokens = 512
    temperature = 0.3
    system_prompt = (
        "You are a task decomposition expert. Break the given task into "
        "clear, numbered subtasks. Each subtask must be independently "
        "executable by a specialist agent.\n\n"
        "Rules:\n"
        "1. Each subtask should be specific and actionable\n"
        "2. Order subtasks by dependency (independent first)\n"
        "3. Estimate complexity per subtask: LOW / MEDIUM / HIGH\n"
        "4. Tag each subtask with the best agent type: "
        "[code] [research] [analysis] [writing] [translation]\n"
        "5. Output ONLY a numbered list with tags and complexity\n\n"
        "Format:\n"
        "1. [code][HIGH] Implement the data processing pipeline\n"
        "2. [research][LOW] Gather API documentation for the service\n"
        "3. [writing][MEDIUM] Write user-facing documentation"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        prompt = self.build_prompt(task, context)
        output = await self._generate(prompt, engine)

        # Store plan in shared context for downstream agents
        context.plan = output

        # Parse subtask count for metadata
        subtask_count = sum(
            1 for line in output.split("\n") if line.strip() and line.strip()[0].isdigit()
        )

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.9,
            metadata={"subtask_count": subtask_count},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "research", "writing", "analysis", "math", "general"}
