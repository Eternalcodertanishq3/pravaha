"""Extractor Agent — Structured data extraction from text."""

from __future__ import annotations
import time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ExtractorAgent(BaseAgent):
    """Extracts structured data from unstructured text."""

    role = "extractor"
    priority = 0
    max_tokens = 1024
    temperature = 0.1
    system_prompt = (
        "You are a data extractor. Extract structured data from "
        "unstructured text.\n\n"
        "Rules:\n"
        "1. Output ONLY valid JSON matching the requested schema\n"
        "2. If a field cannot be found, use null instead of guessing\n"
        "3. Normalize dates to ISO 8601 format\n"
        "4. Normalize currency to USD with 2 decimal places\n"
        "5. Extract ALL instances, not just the first\n"
        "6. No text outside the JSON output"
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        content = context.output or task
        prompt = self.build_prompt(f"Extract structured data:\n\n{content[:2000]}", context)
        result = await self._generate_json(prompt, engine)
        is_valid = not result.get("parse_error", False)
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(role=self.role, output=str(result), tokens_used=self._total_tokens,
                           duration_ms=duration, confidence=0.9 if is_valid else 0.4,
                           metadata={"extracted_data": result, "valid_json": is_valid})

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"analysis", "research", "general"}
