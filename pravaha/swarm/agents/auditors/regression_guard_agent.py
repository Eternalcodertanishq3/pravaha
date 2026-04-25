"""Regression Guard Agent — Detect regressions from patches."""

from __future__ import annotations

from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class RegressionGuardAgent(BaseAgent):
    role = "regression_guard"
    priority = 11
    max_tokens = 1024
    temperature = 0.1

    system_prompt = (
        "You are a regression guard. Compare patched code against\n"
        "the original and verify:\n"
        "- No functionality was removed\n"
        "- No new bugs introduced by patches\n"
        "- All original test cases still pass\n"
        "Report any regressions found."
    )

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ""
        patched = ctx.patched_output or ""
        if not code or not patched or code == patched:
            return AgentOutput(
                role=self.role,
                output="No regression check needed (no patches applied)",
                metadata={"skipped": True},
            )
        return await super().run(task, ctx, engine)

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
