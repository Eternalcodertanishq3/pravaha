"""Debugger Agent — Root cause analysis and fix generation.

Analyzes code or errors, identifies the root cause with clear
explanation, and provides minimal correct fixes. Designed to
work with output from CoderAgent.

Priority: 1 (senior worker).
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class DebuggerAgent(BaseAgent):
    """Finds bugs, explains root causes, provides minimal fixes."""

    role = "debugger"
    priority = 1
    max_tokens = 1024
    temperature = 0.1
    system_prompt = (
        "You are a debugging expert. Analyze the code or error provided "
        "and identify the root cause.\n\n"
        "Your analysis MUST follow this structure:\n"
        "1. **Bug Location**: Exact file/line/function where the bug lives\n"
        "2. **Root Cause**: WHY the bug occurs (not just what happens)\n"
        "3. **Impact**: What breaks because of this bug\n"
        "4. **Fix**: The MINIMAL correct fix — change as few lines as possible\n"
        "5. **Prevention**: How to prevent this class of bug in the future\n\n"
        "Rules:\n"
        "- Be specific: 'line 42 passes str where int expected' not 'type error'\n"
        "- Always explain the root cause, not just the symptom\n"
        "- Provide the fix as a diff or corrected code block\n"
        "- If multiple bugs exist, list each separately\n"
        "- Consider edge cases the original code might miss"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()

        # Prioritize code from context for analysis
        if context.code and "debug" not in task.lower():
            enriched_task = f"{task}\n\nCode to analyze:\n```\n{context.code}\n```"
        else:
            enriched_task = task

        prompt = self.build_prompt(enriched_task, context)
        output = await self._generate(prompt, engine)

        # Count bugs found (heuristic: look for numbered findings)
        bugs_found = sum(
            1
            for line in output.split("\n")
            if line.strip().startswith(("**Bug", "Bug ", "Issue ", "- Bug"))
        )

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.8,
            metadata={"bugs_found": max(bugs_found, 1)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "function", "class", "script", "debug"}
