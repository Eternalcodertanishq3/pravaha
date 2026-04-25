"""Planner Agent — Truly autonomous task decomposition with memory."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class PlannerAgent(BaseAgent):
    """Autonomous planning agent with persistent memory.

    Uses memory to recall similar past tasks and adapts plans
    based on prior successes and failures.
    """

    role = "planner"
    priority = 2
    max_tokens = 1536
    temperature = 0.3
    max_react_steps = 3
    available_tools = ["memory"]

    system_prompt = (
        "You are an autonomous planning agent.\n\n"
        "You decompose tasks and adapt your plan based on constraints.\n\n"
        "Workflow:\n"
        "1. Recall similar past tasks from memory\n"
        "2. Identify what makes this task unique\n"
        "3. Decompose into subtasks ordered by dependency\n"
        "4. Assign estimated complexity and best agent per subtask\n"
        "5. Flag risks and contingencies\n\n"
        "Output format:\n"
        "PLAN:\n"
        "1. [agent_type][complexity:low|medium|high] Description — why this order\n"
        "2. ...\n"
        "RISKS: What could go wrong\n"
        "CONTINGENCY: Fallback if step N fails"
    )

    async def run(
        self, task: str, context: SharedContext, engine: Any
    ) -> AgentOutput:
        # Inject memory context if available
        if self._memory:
            past = self._memory.get_recent(limit=3)
            if past:
                context.extra["planner_memory"] = past

        if self._tool_registry and self.available_tools:
            result = await self.run_react(task, context, engine)
        else:
            t0 = time.time()
            prompt = self.build_prompt(task, context)
            output = await self._generate(prompt, engine)
            duration = (time.time() - t0) * 1000
            self._total_duration_ms += duration

            # Parse plan structure
            steps = [
                line.strip()
                for line in output.split("\n")
                if line.strip() and line.strip()[0].isdigit()
            ]
            result = AgentOutput(
                role=self.role,
                output=output,
                tokens_used=self._total_tokens,
                duration_ms=duration,
                metadata={
                    "steps": len(steps),
                    "has_risks": "RISK" in output.upper(),
                    "has_contingency": "CONTINGENCY" in output.upper(),
                },
            )

        context.plan = result.output
        if self._memory:
            self._memory.store(
                f"Plan for: {task[:60]} → {len(result.output.split(chr(10)))} steps"
            )
        return result

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "research", "writing", "analysis", "general"}
