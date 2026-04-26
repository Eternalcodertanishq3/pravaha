"""Patch Applier Agent — Apply fixes from audit findings with execution verification."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


def _extract_runnable_code(text: str) -> str:
    """Extract importable Python code from agent output."""
    # Try fenced code blocks first
    fences = re.findall(
        r"```(?:python|py)?\s*\n(.*?)\n```",
        text, re.DOTALL,
    )
    if fences:
        return max(fences, key=len).strip()

    # Heuristic: if majority of lines look like Python, use entire output
    lines = text.strip().split("\n")
    python_lines = sum(
        1 for line in lines
        if line.strip().startswith((
            "def ", "class ", "import ", "from ", "#", "@", "if ", "for ",
            "while ", "try:", "with ", "return ", "    ",
        ))
    )
    if python_lines / max(len(lines), 1) > 0.4:
        return text.strip()
    return ""


class PatchApplierAgent(BaseAgent):
    """Apply minimal fixes from audit findings, then verify patches execute."""

    role = "patch_applier"
    priority = 12
    max_tokens = 2048
    temperature = 0.1
    available_tools = ["execute_python"]

    system_prompt = """You are the patch applier for a self-healing code pipeline.

    Given audit findings and the current code, apply the MINIMAL set of
    fixes needed to resolve each issue.

    Rules:
    1. Fix only what the auditors flagged — no drive-by refactoring
    2. Mark each change: # PATCHED: <description of fix>
    3. Preserve all existing functionality
    4. If an issue is a false positive, add: # PATCHED: false positive (reason)
    5. Output the COMPLETE patched code, not just the diff
    6. Prioritize CRITICAL and HIGH severity fixes first
    7. For security fixes: prefer removal over workaround
    8. Test mentally: would this patch introduce new issues?
    """

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        code = context.code or context.output or ""
        all_issues = []

        # Gather issues from audit reports
        for report in context.audit_reports:
            all_issues.extend(report.get("issues", []))

        if not all_issues:
            return AgentOutput(
                role=self.role, output=code, duration_ms=0.0,
                confidence=0.9, metadata={"patches_applied": 0},
            )

        # Build targeted patch prompt
        issues_desc = "\n".join(
            f"- [{i.get('severity', 'unknown')}] Line {i.get('line', '?')}: "
            f"{i.get('description', i.get('id', 'unknown'))}"
            for i in all_issues[:15]
        )
        prompt = self.build_prompt(
            f"Audit findings ({len(all_issues)} issues):\n{issues_desc}\n\n"
            f"Code to patch:\n{code[:3000]}",
            context,
        )
        output = await self._generate(prompt, engine)
        context.patched_output = output
        patches_applied = output.count("# PATCHED:")

        # Verify patched code actually executes
        execution_verified = False
        execution_result: dict[str, Any] | None = None

        if self._tool_registry and "execute_python" in self.available_tools:
            code_to_test = _extract_runnable_code(output)
            if code_to_test:
                try:
                    raw = await self._tool_registry.execute(
                        "execute_python",
                        {"code": code_to_test, "timeout_s": 5},
                    )
                    execution_result = json.loads(raw)
                    execution_verified = execution_result.get("success", False)

                    if not execution_verified:
                        stderr = execution_result.get("stderr", "")[:200]
                        context.extra["patch_execution_error"] = stderr
                        output += f"\n# PATCH_EXECUTION_FAILED: {stderr}"
                except (json.JSONDecodeError, Exception):
                    pass

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.9 if execution_verified else
                       (0.7 if patches_applied > 0 else 0.4),
            patches=[
                i.get("description", str(i)) for i in all_issues[:15]
            ],
            metadata={
                "patches_applied": patches_applied,
                "execution_verified": execution_verified,
                "execution_result": execution_result,
                "total_issues": len(all_issues),
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "code"
