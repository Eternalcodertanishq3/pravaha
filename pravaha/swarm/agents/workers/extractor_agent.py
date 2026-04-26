"""Extractor Agent — Structured data extraction with JSON output."""

from __future__ import annotations

import json
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ExtractorAgent(BaseAgent):
    """Extract structured data from unstructured text into JSON."""

    role = "extractor"
    priority = 4
    max_tokens = 1024
    temperature = 0.1
    available_tools = ["read_file"]

    system_prompt = """You are a precision data extractor.

    Extract structured data from unstructured text. Rules:

    1. Output ONLY valid JSON — no markdown, no explanation
    2. Use ISO 8601 for dates (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
    3. Use null for missing fields — NEVER guess or fabricate
    4. Normalize strings: trim whitespace, consistent casing
    5. Numbers must be actual numbers, not strings
    6. Arrays for multi-value fields, objects for nested data
    7. Include a "_confidence" field (0.0-1.0) for each extracted field
    8. Include a "_source" field noting which part of input it came from
    9. If input is ambiguous, extract ALL possible interpretations
    10. Preserve original language for proper nouns

    Schema enforcement:
    - If a schema is provided, match it exactly
    - Extra fields beyond schema go in "_extra" object
    - Missing required fields get null with "_confidence": 0.0
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        content = context.output or task

        prompt = self.build_prompt(
            f"Extract structured data from:\n\n{content[:3000]}", context,
        )
        result = await self._generate_json(prompt, engine)

        # Track extraction quality
        fields_extracted = len([
            k for k in result if not k.startswith("_") and k != "parse_error"
        ])
        null_count = sum(1 for v in result.values() if v is None)

        context.extra["extracted_data"] = result

        if self._memory:
            self._memory.store(
                f"Extracted {fields_extracted} fields from '{task[:40]}'",
                importance=0.5,
            )

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        confidence = max(0.3, 1.0 - (null_count / max(fields_extracted, 1)) * 0.5)

        return AgentOutput(
            role=self.role,
            output=json.dumps(result, indent=2),
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=confidence,
            metadata={
                "fields_extracted": fields_extracted,
                "null_count": null_count,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"extraction", "analysis", "general"}
