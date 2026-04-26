"""Router Agent — Task routing with pipeline recommendation."""

from __future__ import annotations

import json
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext

# Built-in pipelines that the router can recommend
BUILTIN_PIPELINES = [
    "plan-execute-audit",
    "research-summarize",
    "code-review",
    "secure-code-review",
    "design-component",
    "full-secure-design",
]


class RouterAgent(BaseAgent):
    """Route tasks to the optimal pipeline based on content analysis."""

    role = "router"
    priority = 2
    max_tokens = 256
    temperature = 0.1

    system_prompt = """You are a task router for an AI agent swarm.

    Analyze the task and decide:
    1. task_type: code | research | writing | analysis | math |
                  translation | design | security | general
    2. recommended_pipeline: one of these exact names:
       - plan-execute-audit (default for code tasks)
       - research-summarize (for information gathering)
       - code-review (for existing code analysis)
       - secure-code-review (for security-sensitive code)
       - design-component (for UI/UX work)
       - full-secure-design (for secure design + build)
    3. complexity_estimate: 1-5 (affects iteration count)
    4. confidence: 0.0-1.0

    Output ONLY JSON:
    {
      "task_type": "...",
      "recommended_pipeline": "...",
      "complexity_estimate": N,
      "confidence": 0.0-1.0,
      "reasoning": "one sentence"
    }

    Routing heuristics:
    - Code with "security" or "audit" keywords → secure-code-review
    - Code with "review" or "fix" → code-review
    - "Design", "UI", "component" → design-component
    - "Research", "find", "what is" → research-summarize
    - Default for code → plan-execute-audit
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()

        # Quick heuristic pre-check (avoid LLM call for obvious cases)
        task_lower = task.lower()
        heuristic_type = "general"
        heuristic_pipeline = "plan-execute-audit"

        if any(kw in task_lower for kw in ["security", "audit", "vulnerability", "cve"]):
            heuristic_type = "security"
            heuristic_pipeline = "secure-code-review"
        elif any(kw in task_lower for kw in ["design", "ui", "layout", "component"]):
            heuristic_type = "design"
            heuristic_pipeline = "design-component"
        elif any(kw in task_lower for kw in ["research", "find", "search", "what is"]):
            heuristic_type = "research"
            heuristic_pipeline = "research-summarize"
        elif any(kw in task_lower for kw in ["code", "write", "implement", "function", "class"]):
            heuristic_type = "code"
        elif any(kw in task_lower for kw in ["review", "fix", "debug", "bug"]):
            heuristic_type = "code"
            heuristic_pipeline = "code-review"

        # Use LLM for refined routing
        prompt = self.build_prompt(task, context)
        result = await self._generate_json(prompt, engine, max_tokens=150)

        if result.get("parse_error"):
            # Fallback to heuristic
            result = {
                "task_type": heuristic_type,
                "recommended_pipeline": heuristic_pipeline,
                "complexity_estimate": 3,
                "confidence": 0.6,
                "reasoning": "Heuristic fallback (LLM parse failed)",
            }

        # Write routing decision into context
        routed_type = result.get("task_type", heuristic_type)
        context.task_type = routed_type
        context.extra["recommended_pipeline"] = result.get(
            "recommended_pipeline", heuristic_pipeline,
        )
        context.extra["complexity_estimate"] = result.get("complexity_estimate", 3)

        if self._memory:
            self._memory.store(
                f"Routed '{task[:40]}' → {routed_type}/{result.get('recommended_pipeline')}",
                importance=0.3,
            )

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        confidence = float(str(result.get("confidence", 0.7)))

        return AgentOutput(
            role=self.role,
            output=json.dumps(result),
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=confidence,
            metadata=result,
        )

    def can_handle(self, task_type: str) -> bool:
        return True
