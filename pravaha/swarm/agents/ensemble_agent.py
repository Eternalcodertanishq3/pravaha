"""Ensemble Agent — Multi-model voting and synthesis."""

from __future__ import annotations
import time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class EnsembleAgent(BaseAgent):
    """Coordinates ensemble of outputs using agreement analysis."""

    role = "ensemble"
    priority = 2
    max_tokens = 512
    temperature = 0.5
    system_prompt = (
        "You are an ensemble coordinator. Given outputs from multiple "
        "models or runs:\n\n"
        "1. Identify points of AGREEMENT (high confidence)\n"
        "2. Identify points of DISAGREEMENT (flag or use majority)\n"
        "3. Synthesize the best final answer\n\n"
        "Return JSON:\n"
        '{"agreed": ["<point1>", ...], "disagreed": ["<point1>", ...], '
        '"majority_vote": "<the consensus answer>", '
        '"confidence": <0.0-1.0>, "final_answer": "<synthesized output>"}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        candidates = [ao.output[:300] for ao in context.agent_outputs.values()]
        if len(candidates) <= 1:
            output = candidates[0] if candidates else context.output or task
            duration = (time.time() - t0) * 1000
            return AgentOutput(role=self.role, output=output, duration_ms=duration, confidence=0.7)
        combined = "\n---\n".join(f"Candidate {i+1}:\n{c}" for i, c in enumerate(candidates))
        prompt = self.build_prompt(f"Ensemble these candidates:\n\n{combined}", context)
        result = await self._generate_json(prompt, engine)
        final = result.get("final_answer", str(result))
        conf = result.get("confidence", 0.7)
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(role=self.role, output=final, tokens_used=self._total_tokens,
                           duration_ms=duration, confidence=conf,
                           metadata={"candidates": len(candidates), "ensemble_result": result})

    def can_handle(self, task_type: str) -> bool:
        return True
