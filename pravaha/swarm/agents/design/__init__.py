"""Design Agents — 9 design-focused agents.

Registry keys match the spec:
- ui_designer, component_builder, layout_designer, style_designer,
  accessibility_auditor, ux_reviewer, design_critic, prototype_builder,
  design_system
"""

from pravaha.swarm.agents.design.accessibility_agent import AccessibilityAgent
from pravaha.swarm.agents.design.component_builder_agent import ComponentBuilderAgent
from pravaha.swarm.agents.design.design_critic_agent import DesignCriticAgent
from pravaha.swarm.agents.design.design_system_agent import DesignSystemAgent
from pravaha.swarm.agents.design.layout_agent import LayoutAgent
from pravaha.swarm.agents.design.prototype_agent import PrototypeAgent
from pravaha.swarm.agents.design.style_agent import StyleAgent
from pravaha.swarm.agents.design.ui_designer_agent import UIDesignerAgent
from pravaha.swarm.agents.design.ux_reviewer_agent import UXReviewerAgent

DESIGN_AGENTS: dict[str, type] = {
    "ui_designer": UIDesignerAgent,
    "component_builder": ComponentBuilderAgent,
    "layout_designer": LayoutAgent,
    "style_designer": StyleAgent,
    "accessibility_auditor": AccessibilityAgent,
    "ux_reviewer": UXReviewerAgent,
    "design_critic": DesignCriticAgent,
    "prototype_builder": PrototypeAgent,
    "design_system": DesignSystemAgent,
}

__all__ = ["DESIGN_AGENTS"]
