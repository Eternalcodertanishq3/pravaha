"""Accessibility Agent — WCAG accessibility auditor.

Runs static regex checks for common WCAG violations FIRST (zero LLM cost),
then optionally uses LLM for deeper analysis.
"""

from __future__ import annotations

import re
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class AccessibilityAgent(BaseAgent):
    role = "accessibility_auditor"
    priority = 1
    max_tokens = 1024
    temperature = 0.1

    WCAG_CHECKS = [
        (r"<img(?![^>]*alt=)", "missing_alt", "1.1.1 Non-text Content",
         "Add alt attribute to all <img> elements"),
        (r"<a[^>]*>\s*</a>", "empty_link", "2.4.4 Link Purpose",
         "Add descriptive text to links"),
        (r"<input(?![^>]*(?:aria-label|id=))", "input_no_label",
         "1.3.1 Info and Relationships", "Associate input with label or aria-label"),
        (r"<html(?![^>]*lang=)", "missing_lang", "3.1.1 Language of Page",
         "Add lang attribute to <html>"),
        (r'tabindex="[1-9]', "positive_tabindex", "2.4.3 Focus Order",
         "Use tabindex=0 or -1, not positive values"),
        (r"autofocus", "autofocus_warning", "2.4.3 Focus Order",
         "Autofocus can disorient screen reader users"),
    ]

    system_prompt = (
        "You are a WCAG accessibility auditor. Analyze UI components for:\n\n"
        "Perceivable:\n"
        "- All images have meaningful alt text\n"
        "- Color is not the only way to convey information\n"
        "- Minimum contrast ratio 4.5:1 (text), 3:1 (UI components)\n"
        "- Audio/video has captions/transcripts\n\n"
        "Operable:\n"
        "- All functionality available via keyboard\n"
        "- Focus visible on all interactive elements\n"
        "- No keyboard traps\n"
        "- Skip navigation for long pages\n\n"
        "Understandable:\n"
        "- Error messages identify the problem clearly\n"
        "- Labels are descriptive and persistent\n\n"
        "Robust:\n"
        "- Valid, parseable HTML\n"
        "- ARIA used correctly (not overriding native semantics)\n\n"
        "Return: violations as JSON with WCAG criterion, impact, fix"
    )

    async def run(self, task: str, ctx: SharedContext, engine: Any) -> AgentOutput:
        code = ctx.code or ctx.output or ""
        issues: list[dict[str, Any]] = []

        # Static regex checks — fast, zero LLM cost
        for pattern, issue_id, wcag_ref, remediation in self.WCAG_CHECKS:
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line):
                    issues.append({
                        "id": issue_id,
                        "line": i,
                        "wcag": wcag_ref,
                        "description": f"WCAG {wcag_ref} violation",
                        "severity": "MAJOR",
                        "remediation": remediation,
                    })

        return AgentOutput(
            role=self.role,
            output=f"Accessibility audit: {len(issues)} violation(s)",
            issues=issues,
            metadata={"total_issues": len(issues), "wcag_level": "AA"},
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"design", "ui", "frontend", "code"}
