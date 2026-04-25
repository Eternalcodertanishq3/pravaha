"""Patch Applier Agent — Apply fixes from audit findings."""

from __future__ import annotations

from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class PatchApplierAgent(BaseAgent):
    role = "patch_applier"
    priority = 12
    max_tokens = 2048
    temperature = 0.1

    system_prompt = (
        "You are the patch applier. Given audit findings and code,\n"
        "apply the minimal set of fixes. Output the patched code.\n"
        "Mark each patch: [PATCH] description of change"
    )

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        result = await super().run(task, ctx, engine)
        ctx.patched_output = result.output
        return result

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
