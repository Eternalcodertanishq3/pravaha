"""Design System Agent — Design token and pattern library architecture.

Builds and maintains consistent design systems with token dictionaries,
component inventories, pattern libraries, and migration paths.
"""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class DesignSystemAgent(BaseAgent):
    role = "design_system"
    priority = 2
    max_tokens = 2048
    temperature = 0.2

    system_prompt = (
        "You are a design system architect. You build and maintain "
        "consistent design systems across components.\n\n"
        "You produce:\n"
        "1. Token dictionary (all design decisions as named values)\n"
        "2. Component inventory (what exists, what's needed)\n"
        "3. Pattern library (reusable compositions of components)\n"
        "4. Naming conventions (BEM, camelCase, whatever is consistent)\n"
        "5. Do/Don't examples for each component\n"
        "6. Migration path from inconsistent states\n\n"
        "Output as structured design system documentation that "
        "both designers and developers can use."
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"design", "ui", "frontend"}
