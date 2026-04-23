"""Syntax Audit Agent — Structural code validity scanner.

Scans generated code for syntax errors using AST parsing first,
then LLM analysis for complex issues. Returns structured JSON
with issue locations and fix hints.

Triggers on: code, function, class, script
"""

from __future__ import annotations

import ast
import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class SyntaxAuditAgent(BaseAgent):
    """Scans code for syntax errors with AST + LLM analysis."""

    role = "syntax_audit"
    priority = 1
    max_tokens = 512
    temperature = 0.1
    system_prompt = (
        "You are a syntax auditor. Given code output, scan it for:\n"
        "- Syntax errors (invalid Python/JS/etc syntax)\n"
        "- Unmatched brackets, braces, parentheses\n"
        "- Unterminated strings or comments\n"
        "- Invalid indentation\n\n"
        "Report each issue as JSON:\n"
        '{"issues": [{"line": N, "severity": "error|warning", '
        '"description": "...", "fix_hint": "..."}], "clean": true|false}'
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        code = context.code or context.output or task
        issues: list[dict[str, Any]] = []

        # Phase 1: Static AST analysis (instant, no LLM cost)
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(
                {
                    "line": e.lineno or 0,
                    "severity": "error",
                    "description": f"SyntaxError: {e.msg}",
                    "fix_hint": f"Check syntax near line {e.lineno}: {e.text.strip() if e.text else ''}",
                }
            )

        # Phase 2: LLM analysis for semantic syntax issues AST can't catch
        if not issues:
            prompt = self.build_prompt(
                f"Audit this code for syntax issues:\n```\n{code}\n```", context
            )
            result = await self._generate_json(prompt, engine)
            llm_issues = result.get("issues", [])
            if isinstance(llm_issues, list):
                issues.extend(llm_issues)

        clean = len(issues) == 0
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output="PASS: No syntax errors" if clean else f"FAIL: {len(issues)} issue(s) found",
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=1.0 if clean else 0.3,
            issues=issues,
            metadata={"clean": clean, "issue_count": len(issues), "used_ast": True},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "function", "class", "script", "module"}
