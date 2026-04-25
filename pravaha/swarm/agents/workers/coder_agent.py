"""Coder Agent — Autonomous code generation with real verification."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class CoderAgent(BaseAgent):
    """Autonomous coding agent that VERIFIES code actually runs.

    Uses execute_python to test generated code. Does not report
    'it should work' — it proves it does work.
    """

    role = "coder"
    priority = 1
    max_tokens = 2048
    temperature = 0.2
    max_react_steps = 6
    available_tools = ["execute_python", "read_file", "web_search"]

    system_prompt = (
        "You are an autonomous coding agent.\n\n"
        "You write code, then VERIFY it actually runs correctly.\n\n"
        "Workflow:\n"
        "1. Analyze the requirements\n"
        "2. Write the code\n"
        "3. ALWAYS call execute_python to verify it runs\n"
        "4. If it fails, debug and fix (up to 3 attempts)\n"
        "5. Only report success if execution is actually clean\n\n"
        "You do NOT report 'it should work' — you verify it does work.\n"
        "The execute_python tool runs code and returns actual stdout/stderr.\n\n"
        "If you need to look up an API or library, use web_search.\n"
        "If you need to read existing code, use read_file."
    )

    async def run(
        self, task: str, context: SharedContext, engine: Any
    ) -> AgentOutput:
        # Check memory for similar past tasks
        if self._memory:
            past = self._memory.get_recent(limit=3)
            if past:
                context.extra["coder_memory"] = past

        if self._tool_registry and self.available_tools:
            result = await self.run_react(task, context, engine)
        else:
            t0 = time.time()
            prompt = self.build_prompt(task, context)
            output = await self._generate(prompt, engine)
            duration = (time.time() - t0) * 1000
            self._total_duration_ms += duration

            # Extract code blocks
            code_blocks = []
            if "```" in output:
                parts = output.split("```")
                for i in range(1, len(parts), 2):
                    block = parts[i]
                    if block.startswith("python"):
                        block = block[6:]
                    code_blocks.append(block.strip())

            result = AgentOutput(
                role=self.role,
                output=output,
                tokens_used=self._total_tokens,
                duration_ms=duration,
                metadata={
                    "code_blocks": len(code_blocks),
                    "lines": output.count("\n") + 1,
                },
            )

        context.code = result.output

        # Record this episode in memory
        if self._memory:
            success = result.confidence > 0.7
            self._memory.store(
                f"Task: {task[:50]} → Success: {success}",
                importance=0.7 if success else 0.4,
            )

        return result

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "debug", "general"}
