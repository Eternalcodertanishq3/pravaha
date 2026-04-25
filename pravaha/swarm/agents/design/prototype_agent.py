"""Prototype Agent — Rapid self-contained HTML prototype generation.

Generates single-file HTML prototypes with inline CSS and vanilla JS.
Uses read_file to reference existing components when available.
"""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class PrototypeAgent(BaseAgent):
    role = "prototype_builder"
    priority = 0
    max_tokens = 2048
    temperature = 0.3
    available_tools = ["read_file"]

    system_prompt = (
        "You are a rapid prototyper. You convert design specs into "
        "working HTML/CSS/JS prototypes that can be previewed in a browser.\n\n"
        "Requirements:\n"
        "- Single self-contained HTML file (no build step)\n"
        "- All CSS inline or in <style> tag\n"
        "- Vanilla JS for interactions (no frameworks)\n"
        "- Realistic placeholder content (not lorem ipsum for UX)\n"
        "- Responsive at 320px, 768px, 1200px\n"
        "- Click handlers that demonstrate the interaction flow\n\n"
        "Output the complete HTML file content, nothing else.\n"
        "The prototype should look like a real product, not a wireframe."
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"design", "ui", "frontend"}
