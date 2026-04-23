"""Merger Agent — Multi-output synthesis expert."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class MergerAgent(BaseAgent):
    """Merges multiple agent outputs into the best unified result."""

    role = "merger"
    priority = 2
    max_tokens = 2048
    temperature = 0.3
    system_prompt = (
        "You are a synthesis expert. Merge the provided outputs into "
        "the best unified result.\n\n"
        "Rules:\n"
        "1. Eliminate redundancy across outputs\n"
        "2. Resolve conflicts by choosing the most well-supported claim\n"
        "3. Preserve unique insights from each source\n"
        "4. Maintain consistent formatting and voice\n"
        "5. Note any unresolvable conflicts with [CONFLICT: ...]\n"
        "6. The merged output should be BETTER than any individual input"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        outputs = [
            f"--- Output from {name} ---\n{ao.output[:500]}"
            for name, ao in context.agent_outputs.items()
            if ao.output
        ]
        if not outputs:
            outputs = [context.output or task]
        combined = "\n\n".join(outputs)
        prompt = self.build_prompt(f"Merge these outputs:\n\n{combined}", context)
        output = await self._generate(prompt, engine)
        context.merged_output = output
        conflicts = output.count("[CONFLICT")
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.85 if conflicts == 0 else 0.6,
            metadata={"sources_merged": len(outputs), "conflicts": conflicts},
        )

    def can_handle(self, task_type: str) -> bool:
        return True
