"""Edge Case Hunter Agent — Missing boundary condition detector."""

from __future__ import annotations
import time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class EdgeCaseHunterAgent(BaseAgent):
    """Identifies unhandled edge cases and boundary conditions."""

    role = "edge_case_hunter"
    priority = 1
    max_tokens = 512
    temperature = 0.1
    system_prompt = (
        "You are an edge case analyst. Find unhandled cases:\n"
        "- Empty inputs ([], '', None, 0)\n"
        "- Overflow/underflow (max int, very long strings)\n"
        "- Concurrent access / race conditions\n"
        "- Network failure / timeout scenarios\n"
        "- Malformed input data\n"
        "- Unicode / encoding edge cases\n\n"
        "Return JSON:\n"
        '{"edge_cases": [{"case": "...", "fix_required": true|false}]}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        code = context.code or context.output or task
        prompt = self.build_prompt(f"Find edge cases:\n```\n{code[:2000]}\n```", context)
        result = await self._generate_json(prompt, engine)
        edge_cases = result.get("edge_cases", [])
        if not isinstance(edge_cases, list):
            edge_cases = []
        critical = [e for e in edge_cases if e.get("fix_required", True)]
        issues = [{"type": "edge_case", "severity": "major", "description": e.get("case", str(e))}
                   for e in critical]
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(role=self.role, output=f"{len(critical)} edge case(s) need fixes",
                           tokens_used=self._total_tokens, duration_ms=duration,
                           confidence=1.0 if not critical else 0.5, issues=issues,
                           metadata={"total": len(edge_cases), "fix_required": len(critical)})

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "function", "api", "algorithm", "class"}
