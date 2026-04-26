"""Ensemble Agent — Multi-candidate synthesis and aggregation."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class EnsembleAgent(BaseAgent):
    """Synthesize multiple agent outputs into a single superior output."""

    role = "ensemble"
    priority = 6
    max_tokens = 1536
    temperature = 0.2

    system_prompt = """You are an ensemble aggregator.

    You receive multiple outputs from different agents or runs.
    Your job is to produce a single superior output that is better
    than any individual input.

    Process:
    1. Read all candidate outputs carefully
    2. Identify AGREEMENTS — facts/decisions that appear in 2+ outputs
       (these are HIGH CONFIDENCE — keep them)
    3. Identify CONTRADICTIONS — different answers to the same question
       (resolve by choosing the most evidence-backed version)
    4. Identify UNIQUE INSIGHTS — good ideas appearing in only one output
       (evaluate each: is it actually good? If yes, include it)
    5. Synthesize into a coherent, unified output

    Mark your synthesis sections:
    [CONSENSUS] — backed by multiple sources
    [RESOLVED: chose X over Y because Z] — resolved contradiction
    [UNIQUE: from source N] — valuable unique insight included

    The final output must be more complete and accurate than any
    individual input. Never just pick the longest output.
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        candidates = [
            ao.output[:600]
            for ao in context.agent_outputs.values()
            if ao.output and ao.confidence > 0.4
        ]

        if len(candidates) <= 1:
            output = candidates[0] if candidates else context.output or task
            duration = (time.time() - t0) * 1000
            return AgentOutput(
                role=self.role, output=output,
                duration_ms=duration, confidence=0.7,
            )

        combined = "\n\n---\n\n".join(
            f"[Candidate {i + 1}]\n{c}" for i, c in enumerate(candidates)
        )
        prompt = self.build_prompt(
            f"Synthesize these {len(candidates)} candidates:\n\n{combined}",
            context,
        )
        output = await self._generate(prompt, engine)
        context.merged_output = output

        consensus_count = output.count("[CONSENSUS]")
        resolved_count = output.count("[RESOLVED")

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=min(0.95, 0.6 + 0.05 * consensus_count),
            metadata={
                "candidates": len(candidates),
                "consensus_points": consensus_count,
                "resolved_contradictions": resolved_count,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return True
