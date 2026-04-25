"""UI Designer Agent — Senior UI design specification generation.

Produces structured JSON design specs with layout, color, typography,
spacing, and interaction state specifications. Uses web_search for
design inspiration when available.
"""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class UIDesignerAgent(BaseAgent):
    role = "ui_designer"
    priority = 2
    max_tokens = 2048
    temperature = 0.6
    available_tools = ["web_search"]

    system_prompt = (
        "You are a senior UI designer. Given a component spec, you produce:\n\n"
        "1. LAYOUT SPECIFICATION\n"
        "   - Grid structure (columns, rows, gaps)\n"
        "   - Component hierarchy\n"
        "   - Responsive breakpoints (mobile/tablet/desktop)\n\n"
        "2. VISUAL DESIGN\n"
        "   - Color palette (primary, secondary, semantic, neutral)\n"
        "   - Typography scale (headings, body, captions)\n"
        "   - Spacing system (4px base grid)\n"
        "   - Border radius and shadow tokens\n\n"
        "3. INTERACTION STATES\n"
        "   - Default, hover, active, focus, disabled\n"
        "   - Loading, empty, error states\n"
        "   - Transition specifications (timing, easing)\n\n"
        "Output as structured JSON design spec that ComponentBuilderAgent "
        "can implement directly.\n\n"
        "Always follow:\n"
        "- WCAG 2.1 AA contrast ratios (4.5:1 for normal text)\n"
        "- Material Design or Tailwind spacing conventions\n"
        "- Mobile-first responsive approach"
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"design", "ui", "frontend"}
