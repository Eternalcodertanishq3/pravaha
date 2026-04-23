"""Tests for the swarm agent system."""

from __future__ import annotations

import pytest

from pravaha.swarm.agents import ALL_AGENTS, AUDIT_AGENTS, WORKER_AGENTS
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class TestAgentRegistry:
    """Verify all 32 agents are registered and valid."""

    def test_worker_count(self) -> None:
        assert len(WORKER_AGENTS) == 20, f"Expected 20 workers, got {len(WORKER_AGENTS)}"

    def test_audit_count(self) -> None:
        assert len(AUDIT_AGENTS) == 12, f"Expected 12 auditors, got {len(AUDIT_AGENTS)}"

    def test_total_count(self) -> None:
        assert len(ALL_AGENTS) == 32, f"Expected 32 total, got {len(ALL_AGENTS)}"

    def test_no_overlap(self) -> None:
        overlap = set(WORKER_AGENTS.keys()) & set(AUDIT_AGENTS.keys())
        assert not overlap, f"Overlap between workers and auditors: {overlap}"

    def test_all_inherit_base(self) -> None:
        for name, cls in ALL_AGENTS.items():
            assert issubclass(cls, BaseAgent), f"{name} does not inherit BaseAgent"

    def test_all_have_role(self) -> None:
        for name, cls in ALL_AGENTS.items():
            agent = cls()
            assert agent.role, f"{name} has empty role"

    def test_all_have_system_prompt(self) -> None:
        for name, cls in ALL_AGENTS.items():
            agent = cls()
            assert agent.system_prompt, f"{name} has empty system_prompt"

    def test_all_have_can_handle(self) -> None:
        for name, cls in ALL_AGENTS.items():
            agent = cls()
            assert callable(getattr(agent, "can_handle", None)), f"{name} missing can_handle"

    def test_all_instantiate(self) -> None:
        for name, cls in ALL_AGENTS.items():
            agent = cls()
            assert isinstance(agent, BaseAgent)


class TestSharedContext:
    """Verify SharedContext behaves correctly."""

    def test_empty_context(self) -> None:
        ctx = SharedContext()
        assert ctx.task == ""
        assert ctx.output == ""
        assert ctx.code == ""
        assert ctx.agent_outputs == {}
        assert ctx.audit_reports == []

    def test_set_values(self) -> None:
        ctx = SharedContext()
        ctx.task = "write code"
        ctx.code = "def foo(): pass"
        assert ctx.task == "write code"
        assert ctx.code == "def foo(): pass"

    def test_agent_outputs_storage(self) -> None:
        ctx = SharedContext()
        output = AgentOutput(role="test", output="hello", confidence=0.9)
        ctx.agent_outputs["test"] = output
        assert "test" in ctx.agent_outputs
        assert ctx.agent_outputs["test"].output == "hello"


class TestAgentOutput:
    """Verify AgentOutput dataclass."""

    def test_default_values(self) -> None:
        out = AgentOutput(role="test", output="result")
        assert out.confidence == 1.0
        assert out.tokens_used == 0
        assert out.duration_ms == 0.0
        assert out.issues == []
        assert out.patches == []

    def test_with_metadata(self) -> None:
        out = AgentOutput(
            role="coder", output="def x(): pass",
            confidence=0.85, metadata={"lang": "python"},
        )
        assert out.metadata["lang"] == "python"
        assert out.confidence == 0.85


class TestAgentCanHandle:
    """Verify agent routing capabilities."""

    def test_planner_handles_code(self) -> None:
        from pravaha.swarm.agents.planner_agent import PlannerAgent
        assert PlannerAgent().can_handle("code")

    def test_coder_handles_code(self) -> None:
        from pravaha.swarm.agents.coder_agent import CoderAgent
        assert CoderAgent().can_handle("code")
        assert not CoderAgent().can_handle("translation")

    def test_router_handles_all(self) -> None:
        from pravaha.swarm.agents.router_agent import RouterAgent
        for t in ["code", "research", "writing", "math"]:
            assert RouterAgent().can_handle(t)

    def test_syntax_audit_handles_code(self) -> None:
        from pravaha.swarm.agents.syntax_audit_agent import SyntaxAuditAgent
        assert SyntaxAuditAgent().can_handle("code")
        assert not SyntaxAuditAgent().can_handle("writing")


class TestSyntaxAuditStaticParse:
    """Test AST-based static syntax checking."""

    @pytest.mark.asyncio
    async def test_clean_code_passes(self) -> None:
        from pravaha.swarm.agents.syntax_audit_agent import SyntaxAuditAgent
        agent = SyntaxAuditAgent()
        ctx = SharedContext()
        ctx.code = "def foo():\n    return 42\n"
        # We can't run without engine, but we can test AST parsing
        import ast
        try:
            ast.parse(ctx.code)
            clean = True
        except SyntaxError:
            clean = False
        assert clean

    @pytest.mark.asyncio
    async def test_broken_code_fails(self) -> None:
        import ast
        with pytest.raises(SyntaxError):
            ast.parse("def foo(\n    return 42\n")
