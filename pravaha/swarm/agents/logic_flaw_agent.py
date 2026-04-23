"""Logic Flaw Agent — Reasoning contradiction detector.

Finds logical contradictions, infinite loops, off-by-one errors,
incorrect conditionals, and invalid algorithm assumptions.

Triggers on: code, reasoning, analysis, math
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class LogicFlawAgent(BaseAgent):
    """Detects logical errors and reasoning contradictions."""

    role = "logic_flaw"
    priority = 1
    max_tokens = 512
    temperature = 0.1
    system_prompt = (
        "You are a logic flaw detector. Examine this reasoning/code for:\n"
        "- Logical contradictions (says X then contradicts X)\n"
        "- Infinite loops or unreachable code paths\n"
        "- Off-by-one errors in loops and ranges\n"
        "- Incorrect conditional logic (wrong operator, missing case)\n"
        "- Invalid algorithm assumptions\n\n"
        "Return JSON:\n"
        '{"flaws": [{"type": "<category>", "location": "<where>", '
        '"description": "<what is wrong>", "severity": "critical|major|minor"}], '
        '"clean": true|false}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        content = context.code or context.output or context.reasoning or task
        prompt = self.build_prompt(f"Analyze for logic flaws:\n\n{content[:2000]}", context)
        result = await self._generate_json(prompt, engine)
        flaws = result.get("flaws", [])
        if not isinstance(flaws, list):
            flaws = []
        issues = [
            {
                "type": f.get("type", "logic"),
                "severity": f.get("severity", "major"),
                "description": f.get("description", str(f)),
                "location": f.get("location", ""),
            }
            for f in flaws
        ]
        clean = len(issues) == 0
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output="PASS: No logic flaws" if clean else f"FAIL: {len(issues)} flaw(s)",
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=1.0 if clean else 0.4,
            issues=issues,
            metadata={"clean": clean, "flaw_count": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "reasoning", "analysis", "math", "function"}
