"""Memory Agent — Context memory management with store and retrieve."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class MemoryAgent(BaseAgent):
    """Manage persistent memory: store key decisions, retrieve past context."""

    role = "memory"
    priority = 1
    max_tokens = 512
    temperature = 0.1
    available_tools = ["memory"]

    system_prompt = """You are a memory manager agent.

    Your job is to compress and manage context across sessions.

    Store operations — extract and persist:
    1. Key DECISIONS made (and why)
    2. ENTITIES mentioned (people, files, APIs, models)
    3. OUTCOMES of actions (success/failure + reason)
    4. OPEN QUESTIONS that need follow-up
    5. ERROR PATTERNS that should be avoided

    Retrieve operations — find relevant past context:
    1. Search for similar tasks from history
    2. Recall past failures to avoid repeating them
    3. Find relevant decisions for consistency

    Compress aggressively: each memory item should be ONE sentence.
    Use the format: [TYPE] content
    Example: [DECISION] Using SQLite for persistence due to zero-config.
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        items_stored = 0
        items_retrieved = 0

        # Phase 1: Retrieve relevant past context
        retrieved: list[str] = []
        if self._memory:
            recent = self._memory.get_recent(limit=5)
            important = self._memory.get_important(min_importance=0.6)
            retrieved = list(set(recent + important))
            items_retrieved = len(retrieved)

        # Phase 2: Generate compressed summary from current context
        content = context.output or context.code or task
        input_words = len(content.split())

        prompt = self.build_prompt(
            f"Compress this context into key memory items "
            f"(one sentence each, prefixed with [TYPE]):\n\n"
            f"{content[:2000]}",
            context,
        )
        output = await self._generate(prompt, engine)

        # Phase 3: Store each extracted memory item
        if self._memory:
            for line in output.split("\n"):
                line = line.strip()
                if line.startswith("[") and "]" in line:
                    importance = 0.8 if "[DECISION]" in line or "[ERROR]" in line else 0.5
                    self._memory.store(line, importance=importance)
                    items_stored += 1

        # Update context summary
        summary_parts = []
        if retrieved:
            summary_parts.append("Past context:\n" + "\n".join(f"  {r}" for r in retrieved[:3]))
        summary_parts.append(f"Current: {output[:300]}")
        context.context_summary = "\n".join(summary_parts)

        output_words = len(output.split())
        compression = output_words / max(1, input_words)

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.85,
            metadata={
                "items_stored": items_stored,
                "items_retrieved": items_retrieved,
                "compression_ratio": round(compression, 2),
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return True
