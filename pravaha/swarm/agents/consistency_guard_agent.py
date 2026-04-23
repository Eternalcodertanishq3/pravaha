"""Consistency Guard Agent — Cross-agent contradiction checker.

Compares all agent outputs in the shared context and flags
contradicting facts, inconsistent names, and incompatible interfaces.

Triggers on: multi_agent_output, always_in_swarm
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ConsistencyGuardAgent(BaseAgent):
    """Checks cross-agent output consistency and flags contradictions."""

    role = "consistency_guard"
    priority = 1
    max_tokens = 512
    temperature = 0.1
    system_prompt = (
        "You are a consistency checker. Compare all agent outputs in context.\n\n"
        "Find:\n"
        "- Contradicting facts between outputs\n"
        "- Inconsistent variable/function names across code snippets\n"
        "- Incompatible API interfaces between modules\n"
        "- Conflicting recommendations from different agents\n\n"
        "Return JSON:\n"
        '{"contradictions": [{"source_a": "<agent>", "source_b": "<agent>", '
        '"conflict": "<what contradicts>", "resolution": "<suggested fix>"}], '
        '"consistent": true|false}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        outputs = {name: ao.output[:300] for name, ao in context.agent_outputs.items()}
        if len(outputs) < 2:
            duration = (time.time() - t0) * 1000
            return AgentOutput(
                role=self.role,
                output="PASS: Insufficient outputs to compare",
                duration_ms=duration,
                confidence=1.0,
            )
        combined = "\n\n".join(f"[{name}]: {text}" for name, text in outputs.items())
        prompt = self.build_prompt(f"Check consistency:\n\n{combined}", context)
        result = await self._generate_json(prompt, engine)
        contradictions = result.get("contradictions", [])
        if not isinstance(contradictions, list):
            contradictions = []
        issues = [
            {
                "type": "contradiction",
                "severity": "major",
                "description": c.get("conflict", str(c)),
                "location": f"{c.get('source_a', '?')} vs {c.get('source_b', '?')}",
            }
            for c in contradictions
        ]
        consistent = len(issues) == 0
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output="PASS: Outputs consistent"
            if consistent
            else f"FAIL: {len(issues)} contradiction(s)",
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=1.0 if consistent else 0.4,
            issues=issues,
            metadata={"consistent": consistent, "contradictions": len(issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return True  # Always runs in swarm context
