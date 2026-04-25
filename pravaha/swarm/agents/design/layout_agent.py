"""Layout Agent — Spatial organization specialist.

Determines optimal CSS Grid / Flexbox layouts with responsive
breakpoints and z-axis layering strategy.
"""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class LayoutAgent(BaseAgent):
    role = "layout_designer"
    priority = 1
    max_tokens = 1024
    temperature = 0.3

    system_prompt = (
        "You are a layout specialist. You determine optimal spatial "
        "organization for UI components.\n\n"
        "Output specifications for:\n"
        "1. CSS Grid layouts (grid-template-areas)\n"
        "2. Flexbox arrangements (direction, wrap, alignment)\n"
        "3. Responsive grid breakpoints\n"
        "4. Whitespace and rhythm analysis\n"
        "5. Visual hierarchy through size and spacing\n"
        "6. Z-axis layering (z-index strategy)\n\n"
        "For each layout decision, explain the reasoning:\n"
        '"This 3-column grid at 1024px+ because..."'
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"design", "ui", "frontend"}
