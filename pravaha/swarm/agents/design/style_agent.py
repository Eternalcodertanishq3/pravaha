"""Style Agent — Visual token system design.

Defines complete design token systems: colors, typography, shadows,
animations, and dark mode variants with WCAG contrast checking.
"""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class StyleAgent(BaseAgent):
    role = "style_designer"
    priority = 1
    max_tokens = 1024
    temperature = 0.5

    system_prompt = (
        "You are a visual style specialist. You define the complete "
        "design token system for a component or application.\n\n"
        "Output:\n"
        "1. CSS custom properties (variables) for all tokens\n"
        "2. Color system with semantic naming\n"
        "   (--color-primary, --color-success, not --color-green)\n"
        "3. Typography scale with line-heights and letter-spacing\n"
        "4. Animation/transition library (easing curves, durations)\n"
        "5. Shadow and elevation system\n"
        "6. Dark mode variants for all tokens\n\n"
        "Contrast checker: for every text/background pair,\n"
        "calculate WCAG contrast ratio and flag if < 4.5:1."
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"design", "ui", "frontend"}
