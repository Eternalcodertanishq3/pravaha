"""Hallucination Hunter Agent — Fabricated fact detector.

Flags potential hallucinations by assessing confidence per claim
and checking for impossible or unverifiable statements.

Triggers on: text, research, facts, code
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class HallucinationHunterAgent(BaseAgent):
    """Detects fabricated facts and unsupported claims."""

    role = "hallucination_hunter"
    priority = 1
    max_tokens = 512
    temperature = 0.1
    system_prompt = (
        "You are a hallucination detector. For each factual claim:\n"
        "1. Assess confidence: HIGH / MEDIUM / LOW\n"
        "2. Flag claims you cannot verify as [UNCERTAIN]\n"
        "3. Flag impossible claims as [HALLUCINATION]\n"
        "4. For code: flag calls to non-existent APIs or libraries\n\n"
        "Return JSON:\n"
        '{"claims": [{"text": "...", "confidence": "HIGH|MEDIUM|LOW", '
        '"status": "verified|uncertain|hallucination", "reason": "..."}], '
        '"hallucination_risk": "low|medium|high"}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        content = context.output or context.research or context.code or task
        ref_context = context.research[:500] if context.research else ""
        prompt = self.build_prompt(
            f"Check for hallucinations:\n\nContent:\n{content[:1500]}\n\n"
            f"Reference context:\n{ref_context}",
            context,
        )
        result = await self._generate_json(prompt, engine)
        claims = result.get("claims", [])
        if not isinstance(claims, list):
            claims = []
        hallucinations = [c for c in claims if c.get("status") == "hallucination"]
        uncertain = [c for c in claims if c.get("status") == "uncertain"]
        risk = result.get("hallucination_risk", "medium")
        issues = [
            {
                "type": "hallucination",
                "description": h.get("text", ""),
                "severity": "critical",
                "reason": h.get("reason", ""),
            }
            for h in hallucinations
        ]
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role,
            output=f"Risk: {risk} | Hallucinations: {len(hallucinations)} | Uncertain: {len(uncertain)}",
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence={"low": 0.95, "medium": 0.6, "high": 0.2}.get(risk, 0.5),
            issues=issues,
            metadata={
                "hallucination_risk": risk,
                "hallucinations": len(hallucinations),
                "uncertain": len(uncertain),
                "total_claims": len(claims),
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"research", "writing", "analysis", "general", "code", "facts"}
