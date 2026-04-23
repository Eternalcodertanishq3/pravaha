"""Type Safety Agent — Type annotation and mismatch checker.

Finds missing type annotations, type mismatches, None-dereference
risks, and unsafe casts in generated code.

Triggers on: code, python, typescript, java
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class TypeSafetyAgent(BaseAgent):
    """Checks code for type safety issues and missing annotations."""

    role = "type_safety"
    priority = 1
    max_tokens = 512
    temperature = 0.1
    system_prompt = (
        "You are a type safety checker. For Python/TypeScript code:\n"
        "- Find missing type annotations on functions\n"
        "- Find type mismatches (passing str where int expected)\n"
        "- Find None-dereference risks (accessing .attr on Optional)\n"
        "- Find unsafe casts or type: ignore comments\n"
        "- Find incorrect generic type usage\n\n"
        "Return JSON:\n"
        '{"type_errors": [{"location": "<where>", "error": "<what>", '
        '"expected_type": "<type>", "actual_type": "<type>", '
        '"fix": "<correction>"}], "clean": true|false}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        code = context.code or context.output or task
        prompt = self.build_prompt(f"Type-check this code:\n```\n{code[:2000]}\n```", context)
        result = await self._generate_json(prompt, engine)
        type_errors = result.get("type_errors", [])
        if not isinstance(type_errors, list):
            type_errors = []
        issues = [
            {
                "type": "type_safety",
                "severity": "major",
                "description": f"{e.get('error', '')}: expected {e.get('expected_type', '?')}, got {e.get('actual_type', '?')}",
                "location": e.get("location", ""),
                "fix": e.get("fix", ""),
            }
            for e in type_errors
        ]
        clean = len(issues) == 0
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output="PASS: Type-safe" if clean else f"FAIL: {len(issues)} type issue(s)",
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=1.0 if clean else 0.5,
            issues=issues,
            metadata={"clean": clean, "type_errors": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "function", "class", "module", "script"}
