"""Memory Agent — Long-term context compression and management."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class MemoryAgent(BaseAgent):
    """Compresses conversation history while preserving key facts."""

    role = "memory"
    priority = 1
    max_tokens = 512
    temperature = 0.2
    system_prompt = (
        "You are a context manager. Compress the conversation history "
        "to fit in 512 tokens while preserving:\n\n"
        "1. Key decisions made so far\n"
        "2. Facts established as true\n"
        "3. Current task state and progress\n"
        "4. Open questions that need answers\n"
        "5. User preferences expressed\n\n"
        "Format as bullet points. Most recent and important first."
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        history = str(context.conversation_history[-20:]) if context.conversation_history else task
        prompt = self.build_prompt(f"Compress this context:\n\n{history[:2000]}", context)
        output = await self._generate(prompt, engine)
        context.context_summary = output
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.85,
            metadata={"original_length": len(history), "compressed_length": len(output)},
        )

    def can_handle(self, task_type: str) -> bool:
        return True
