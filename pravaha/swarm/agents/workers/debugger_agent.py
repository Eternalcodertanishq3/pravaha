"""Debugger Agent — Autonomous debugging with real execution verification."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class DebuggerAgent(BaseAgent):
    """Autonomous debugging agent that FIX and VERIFY.

    Doesn't just identify bugs — fixes them and proves the fix
    works by executing the code.
    """

    role = "debugger"
    priority = 1
    max_tokens = 2048
    temperature = 0.1
    max_react_steps = 5
    available_tools = ["execute_python", "read_file"]

    system_prompt = (
        "You are an autonomous debugging agent.\n\n"
        "You don't just identify bugs — you FIX them and VERIFY the fix.\n\n"
        "Workflow:\n"
        "1. Read the code and identify the bug\n"
        "2. Understand the root cause (not just the symptom)\n"
        "3. Apply the minimal fix\n"
        "4. Execute the fixed code to verify it works\n"
        "5. Run edge cases to confirm robustness\n"
        "6. Report the exact fix applied and proof it works\n\n"
        "You MUST call execute_python to verify your fix.\n"
        "'It looks correct' is not acceptable.\n"
        "'I ran it and got exit_code=0' is acceptable.\n\n"
        "Output format:\n"
        "BUG: <description of the bug>\n"
        "ROOT CAUSE: <why it happens>\n"
        "FIX: <minimal code change>\n"
        "VERIFICATION: <output of running the fixed code>"
    )

    async def run(
        self, task: str, context: SharedContext, engine: Any
    ) -> AgentOutput:
        if self._tool_registry and self.available_tools:
            return await self.run_react(task, context, engine)

        t0 = time.time()
        prompt = self.build_prompt(task, context)
        output = await self._generate(prompt, engine)
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        has_bug = "BUG:" in output.upper()
        has_fix = "FIX:" in output.upper()
        has_verify = "VERIFICATION:" in output.upper()

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            metadata={
                "identified_bug": has_bug,
                "proposed_fix": has_fix,
                "verified": has_verify,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "debug", "general"}
