"""Researcher Agent — Autonomous research with real web search."""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class ResearcherAgent(BaseAgent):
    """Autonomous research agent that actually searches and verifies."""

    role = "researcher"
    priority = 1
    max_tokens = 2048
    temperature = 0.3
    max_react_steps = 5
    available_tools = ["web_search", "fetch_url"]

    system_prompt = (
        "You are an autonomous research agent.\n\n"
        "You don't just recall training knowledge — you actually search\n"
        "and verify current information.\n\n"
        "Workflow:\n"
        "1. Search for current, accurate information\n"
        "2. Fetch the most relevant pages for details\n"
        "3. Cross-reference at least 2 sources\n"
        "4. Distinguish facts (verified) from claims (unverified)\n"
        "5. Always report the source URL alongside each fact\n\n"
        "Use web_search then fetch_url to get full content.\n"
        "Never state facts without attempting to verify them.\n\n"
        "Tag each claim:\n"
        "[VERIFIED] — confirmed by 2+ sources\n"
        "[LIKELY] — found in 1 source\n"
        "[UNVERIFIED] — from training knowledge only"
    )

    async def run(
        self, task: str, context: SharedContext, engine: Any
    ) -> AgentOutput:
        if self._tool_registry and self.available_tools:
            result = await self.run_react(task, context, engine)
        else:
            t0 = time.time()
            prompt = self.build_prompt(task, context)
            output = await self._generate(prompt, engine)
            duration = (time.time() - t0) * 1000
            self._total_duration_ms += duration

            claims = [
                l.strip()
                for l in output.split("\n")
                if any(t in l for t in ["[VERIFIED]", "[LIKELY]", "[UNVERIFIED]"])
            ]
            result = AgentOutput(
                role=self.role,
                output=output,
                tokens_used=self._total_tokens,
                duration_ms=duration,
                metadata={"claims_tagged": len(claims)},
            )

        context.research = result.output
        return result

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"research", "analysis", "general"}
