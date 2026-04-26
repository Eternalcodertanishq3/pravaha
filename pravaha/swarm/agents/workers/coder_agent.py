"""Coder Agent — Autonomous code generation with real verification."""

from __future__ import annotations

import re
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

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract the first code block from markdown output.

        Handles: ```python, ```py, ```python3, ```typescript,
        ```javascript, ```rust, ```go, ```java, ``` (blank)
        Falls back to the raw text if no fence is found.
        """
        # Match fenced code blocks with any language tag (or none)
        pattern = re.compile(
            r"```(?:[a-zA-Z0-9+#_\-]*)?\s*\n(.*?)\n```",
            re.DOTALL,
        )
        matches = pattern.findall(text)

        if matches:
            # Return the LONGEST code block (most complete implementation)
            return max(matches, key=len).strip()

        # No fences — check if entire text looks like code
        lines = text.strip().split("\n")
        code_indicators = sum(
            1 for line in lines
            if line.strip().startswith((
                "def ", "class ", "import ", "from ", "async def ",
                "function ", "const ", "let ", "var ", "fn ", "pub fn ",
                "func ", "package ",
            ))
        )
        if code_indicators >= 2 or (
            len(lines) > 5 and lines[0].strip().startswith("#")
        ):
            return text.strip()

        return text.strip()

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        # Check memory for similar past tasks
        if self._memory:
            past = self._memory.get_recent(limit=3)
            if past:
                context.extra["coder_memory"] = past

        # Recall past failures via episodic memory (Bonus 5)
        if self._memory and hasattr(self._memory, "_store"):
            try:
                from pravaha.swarm.memory.episodic_memory import EpisodicMemory

                episodes = EpisodicMemory()
                failures = episodes.get_failures("coder", limit=3)
                if failures:
                    context.extra["past_failures"] = [
                        f.get("task", "")[:50] for f in failures
                    ]
            except Exception:
                pass

        if self._tool_registry and self.available_tools:
            result = await self.run_react(task, context, engine)
        else:
            t0 = time.time()
            prompt = self.build_prompt(task, context)
            output = await self._generate(prompt, engine)
            duration = (time.time() - t0) * 1000
            self._total_duration_ms += duration

            # Extract code using robust extractor
            code = self._extract_code(output)
            code_blocks = 1 if code != output else 0

            result = AgentOutput(
                role=self.role,
                output=output,
                tokens_used=self._total_tokens,
                duration_ms=duration,
                metadata={
                    "code_blocks": code_blocks,
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
