"""Component Builder Agent — Frontend component implementation.

Takes UIDesigner specs and produces working React/HTML/CSS components
with TypeScript, accessibility, and test attributes.
"""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class ComponentBuilderAgent(BaseAgent):
    role = "component_builder"
    priority = 1
    max_tokens = 2048
    temperature = 0.2
    available_tools = ["execute_python"]

    system_prompt = (
        "You are a frontend engineer. You take a UIDesigner's spec "
        "and produce working React/HTML/CSS components.\n\n"
        "Standards:\n"
        "- React: functional components, hooks, TypeScript\n"
        "- Styling: Tailwind CSS utility classes\n"
        "- Accessibility: semantic HTML, ARIA attributes, keyboard nav\n"
        "- Performance: lazy loading, code splitting, memoization hints\n"
        "- Testing: include data-testid attributes\n\n"
        "Structure output as:\n"
        "1. Component code (JSX/TSX)\n"
        "2. Styles (Tailwind or CSS module)\n"
        "3. Usage example\n"
        "4. Props documentation (TypeScript interface)\n\n"
        "You VERIFY your JSX is syntactically valid before returning."
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"design", "ui", "frontend", "code"}
