"""Tests for design agents — accessibility, UI designer, design critic."""

from __future__ import annotations

import pytest

from pravaha.swarm.agents.base_agent import SharedContext


class TestAccessibilityAgent:
    @pytest.mark.asyncio
    async def test_detects_missing_alt(self) -> None:
        from pravaha.swarm.agents.design.accessibility_agent import AccessibilityAgent
        agent = AccessibilityAgent()
        ctx = SharedContext()
        ctx.code = '<img src="photo.jpg">\n'
        result = await agent.run("audit", ctx, None)
        assert any(i["id"] == "missing_alt" for i in result.issues)

    @pytest.mark.asyncio
    async def test_clean_html(self) -> None:
        from pravaha.swarm.agents.design.accessibility_agent import AccessibilityAgent
        agent = AccessibilityAgent()
        ctx = SharedContext()
        ctx.code = '<img src="photo.jpg" alt="A cat">\n'
        result = await agent.run("audit", ctx, None)
        assert not any(i["id"] == "missing_alt" for i in result.issues)

    @pytest.mark.asyncio
    async def test_detects_positive_tabindex(self) -> None:
        from pravaha.swarm.agents.design.accessibility_agent import AccessibilityAgent
        agent = AccessibilityAgent()
        ctx = SharedContext()
        ctx.code = '<button tabindex="5">Click</button>\n'
        result = await agent.run("audit", ctx, None)
        assert any(i["id"] == "positive_tabindex" for i in result.issues)


class TestUIDesigner:
    def test_can_handle(self) -> None:
        from pravaha.swarm.agents.design.ui_designer_agent import UIDesignerAgent
        agent = UIDesignerAgent()
        assert agent.can_handle("design")
        assert agent.can_handle("ui")
        assert not agent.can_handle("code")

    def test_has_tools(self) -> None:
        from pravaha.swarm.agents.design.ui_designer_agent import UIDesignerAgent
        agent = UIDesignerAgent()
        assert "web_search" in agent.available_tools


class TestDesignCritic:
    def test_can_handle(self) -> None:
        from pravaha.swarm.agents.design.design_critic_agent import DesignCriticAgent
        agent = DesignCriticAgent()
        assert agent.can_handle("design")

    def test_system_prompt_has_scoring(self) -> None:
        from pravaha.swarm.agents.design.design_critic_agent import DesignCriticAgent
        agent = DesignCriticAgent()
        assert "1-10" in agent.system_prompt
        assert "Visual hierarchy" in agent.system_prompt


class TestPrototypeBuilder:
    def test_has_read_file_tool(self) -> None:
        from pravaha.swarm.agents.design.prototype_agent import PrototypeAgent
        agent = PrototypeAgent()
        assert agent.role == "prototype_builder"
        assert "read_file" in agent.available_tools
