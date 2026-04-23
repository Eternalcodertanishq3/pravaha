"""Tool Agent — Function call formatting and execution."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ToolAgent(BaseAgent):
    """Formats and routes function/tool calls."""

    role = "tool"
    priority = 1
    max_tokens = 512
    temperature = 0.1
    system_prompt = (
        "You are a tool executor. Given a function definition and arguments, "
        "format the correct function call.\n\n"
        "Return ONLY valid JSON:\n"
        '{"name": "<function_name>", "arguments": {<key>: <value>, ...}}\n\n'
        "Rules:\n"
        "1. Match argument types to the function signature exactly\n"
        "2. Include all required arguments\n"
        "3. Use default values for optional arguments when appropriate\n"
        "4. Validate argument types before outputting\n"
        "5. If the task doesn't require a tool call, return: "
        '{"name": null, "reason": "<why no tool is needed>"}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        prompt = self.build_prompt(task, context)
        result = await self._generate_json(prompt, engine)
        tool_name = result.get("name")
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=str(result),
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.9 if tool_name else 0.5,
            metadata={"tool_call": result, "tool_needed": tool_name is not None},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "function", "api", "general"}
