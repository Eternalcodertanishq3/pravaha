"""Router Agent — Task classification and routing."""

from __future__ import annotations
import time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class RouterAgent(BaseAgent):
    """Classifies input and routes to the appropriate pipeline."""

    role = "router"
    priority = 2
    max_tokens = 128
    temperature = 0.1
    system_prompt = (
        "You are a task classifier. Classify the input as ONE of:\n"
        "code | research | writing | analysis | math | translation | general\n\n"
        "Return ONLY a single JSON object:\n"
        '{"category": "<type>", "complexity": <1-5>, '
        '"agents": ["<agent1>", "<agent2>", ...]}\n\n'
        "Agent choices: planner, researcher, coder, debugger, critic, "
        "validator, summarizer, expander, translator, reasoning, "
        "merger, narrator, extractor, classifier"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        prompt = self.build_prompt(task, context)
        result = await self._generate_json(prompt, engine, max_tokens=64)
        category = result.get("category", "general")
        context.task_type = category
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(role=self.role, output=category, tokens_used=self._total_tokens,
                           duration_ms=duration, confidence=0.9, metadata=result)

    def can_handle(self, task_type: str) -> bool:
        return True  # Router handles everything
