"""Test Generator Agent — Automated test suite creation."""

from __future__ import annotations
import time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class TestGeneratorAgent(BaseAgent):
    """Auto-generates comprehensive test suites for code output."""

    role = "test_generator"
    priority = 1
    max_tokens = 2048
    temperature = 0.2
    system_prompt = (
        "You are a test engineer. Write a complete pytest test suite:\n"
        "- Happy path tests for all public functions\n"
        "- Error case tests (invalid input, exceptions)\n"
        "- Edge case tests (empty, null, overflow, unicode)\n"
        "- Mock external dependencies with unittest.mock\n"
        "- Assert all function contracts and return types\n"
        "- Use descriptive test names: test_<function>_<scenario>\n\n"
        "Return ONLY the test code, ready to run. No explanations."
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        code = context.code or task
        prompt = self.build_prompt(f"Write tests for:\n```python\n{code[:2000]}\n```", context)
        output = await self._generate(prompt, engine)
        context.tests = output
        test_count = output.count("def test_")
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(role=self.role, output=output, tokens_used=self._total_tokens,
                           duration_ms=duration, confidence=0.85,
                           metadata={"test_count": test_count})

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "function", "class", "module"}
