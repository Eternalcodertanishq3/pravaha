"""Patch Applier Agent — Minimal fix application from audit reports.

Takes original output + audit issues, applies MINIMAL fixes to
resolve each issue. Preserves original structure and style.
Adds inline comments for every change made.

Triggers on: on_issues_found (runs after auditors find problems)
"""

from __future__ import annotations
import time
from typing import Any
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class PatchApplierAgent(BaseAgent):
    """Applies minimal fixes from audit reports to output."""

    role = "patch_applier"
    priority = 1
    max_tokens = 2048
    temperature = 0.1
    system_prompt = (
        "You are a code/text patcher. Given the original output and "
        "a list of audit issues:\n\n"
        "Rules:\n"
        "1. Apply MINIMAL fixes to resolve each issue\n"
        "2. Do NOT change working parts of the code\n"
        "3. Preserve the original structure and style\n"
        "4. For each fix, add an inline comment: # PATCHED: <reason>\n"
        "5. If an issue cannot be fixed without major rewrite, "
        "   add a comment: # TODO: <issue description>\n"
        "6. Return the complete patched output. Nothing else.\n\n"
        "The goal is surgical precision — change the minimum to fix the maximum."
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        original = context.code or context.output or ""
        all_issues = []
        for report in context.audit_reports:
            if isinstance(report, dict):
                all_issues.extend(report.get("issues", []))

        if not all_issues:
            duration = (time.time() - t0) * 1000
            return AgentOutput(role=self.role, output=original,
                               confidence=1.0, duration_ms=duration)

        issue_text = "\n".join(f"- [{i.get('severity', 'major')}] {i.get('description', str(i))}"
                               for i in all_issues[:15])
        prompt = self.build_prompt(
            f"Apply these fixes:\n{issue_text}\n\nOriginal:\n```\n{original[:2000]}\n```",
            context)
        output = await self._generate(prompt, engine)
        context.patched_output = output

        patches_applied = output.count("# PATCHED:")
        todos = output.count("# TODO:")

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        return AgentOutput(
            role=self.role, output=output, tokens_used=self._total_tokens,
            duration_ms=duration, confidence=0.8 if patches_applied > 0 else 0.5,
            patches=[i.get("description", str(i)) for i in all_issues[:15]],
            metadata={"patches_applied": patches_applied, "todos_left": todos,
                       "issues_received": len(all_issues)},
        )

    def can_handle(self, task_type: str) -> bool:
        return True  # Triggered by audit loop, not task type
