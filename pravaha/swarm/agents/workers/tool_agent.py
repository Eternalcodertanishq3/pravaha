"""Tool Agent — General-purpose tool orchestrator."""

from __future__ import annotations

import json
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ToolAgent(BaseAgent):
    """Execute the most appropriate tool for the task and report results."""

    role = "tool"
    priority = 2
    max_tokens = 1024
    temperature = 0.2
    max_react_steps = 4
    available_tools = [
        "execute_python", "read_file", "web_search",
        "fetch_url", "run_shell", "memory",
    ]

    system_prompt = """You are a tool orchestration agent.

    You have access to ALL available tools. Your job:
    1. Analyze the task to determine which tool(s) are needed
    2. Execute the tool with correct arguments
    3. Interpret the results
    4. Report findings clearly

    Tool selection priority:
    - Code execution/testing → execute_python
    - File analysis → read_file
    - Information lookup → web_search or fetch_url
    - System commands → run_shell (use sparingly)
    - Context recall → memory

    Rules:
    - ALWAYS use a tool — you exist to execute, not to guess
    - If a tool fails, try an alternative approach
    - Report tool name, args used, and raw result
    - Summarize findings in 2-3 sentences after tool output
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()

        if self._tool_registry and self.available_tools:
            result = await self.run_react(task, context, engine)
        else:
            # Fallback: no tools available, just analyze
            prompt = self.build_prompt(task, context)
            output = await self._generate(prompt, engine)
            duration = (time.time() - t0) * 1000
            self._total_duration_ms += duration
            result = AgentOutput(
                role=self.role,
                output=output,
                tokens_used=self._total_tokens,
                duration_ms=duration,
                confidence=0.5,
                metadata={"tool_used": None, "fallback": True},
            )

        # Track tool usage in metadata
        tools_used = []
        for step in result.trajectory:
            if step.action:
                tools_used.append(step.action.tool_name)

        result.metadata["tools_executed"] = tools_used
        result.metadata["tool_count"] = len(tools_used)

        if self._memory and tools_used:
            self._memory.store(
                f"Used tools {tools_used} for '{task[:40]}'",
                importance=0.4,
            )

        return result

    def can_handle(self, task_type: str) -> bool:
        return True
