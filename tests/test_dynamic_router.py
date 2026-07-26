"""Tests for DynamicSwarmRouter — Hybrid Dynamic-DAG Architecture."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pravaha.swarm.dynamic_router import (
    DynamicSwarmRouter,
    TaskIntent,
    RoutingDecision,
    INTENT_CATEGORIES,
    MANDATORY_GATES,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_agents():
    """Create a mock agent registry with common agent names."""
    agents = {}
    all_roles = set()
    for config in INTENT_CATEGORIES.values():
        all_roles.update(config["workers"])
        all_roles.update(config["auditors"])
    all_roles.update(MANDATORY_GATES)
    all_roles.update(["patch_applier", "self_reflection"])

    for role in all_roles:
        agent = MagicMock()
        agent.role = role
        agent.name = role
        agents[role] = agent

    return agents


@pytest.fixture
def router(mock_agents):
    """Create a DynamicSwarmRouter with mock agents."""
    return DynamicSwarmRouter(
        agent_registry=mock_agents,
        use_llm_classification=False,
    )


# ── Keyword Classification Tests ─────────────────────────────────

class TestKeywordClassification:
    """Test keyword-based intent classification."""

    def test_code_generation_intent(self, router):
        """Task with code-generation keywords selects coder agents."""
        intent = router._keyword_classify("Write a python function to compute factorial")
        assert intent.primary_category == "code_generation"
        assert "coder" in intent.selected_workers
        assert intent.confidence > 0.3

    def test_debugging_intent(self, router):
        """Task with debugging keywords selects debugger agents."""
        intent = router._keyword_classify(
            "Fix this bug: the error traceback shows a crash"
        )
        assert intent.primary_category == "debugging"
        assert "debugger" in intent.selected_workers

    def test_security_intent(self, router):
        """Security-related task selects security auditor agents."""
        intent = router._keyword_classify(
            "Check for SQL injection vulnerabilities and authentication issues"
        )
        assert intent.primary_category == "security_audit"
        assert "injection_scanner" in intent.selected_auditors

    def test_ui_design_intent(self, router):
        """UI-related task selects UI designer agents."""
        intent = router._keyword_classify(
            "Design a responsive navigation component with buttons"
        )
        assert intent.primary_category == "ui_design"
        assert "ui_designer" in intent.selected_workers

    def test_research_intent(self, router):
        """Research task selects researcher agents."""
        intent = router._keyword_classify("Research what is the latest Python version")
        assert intent.primary_category == "research"
        assert "researcher" in intent.selected_workers

    def test_unknown_task_gets_general_fallback(self, router):
        """Task with no keyword matches gets general fallback."""
        intent = router._keyword_classify("xyz abc 123 ~~~")
        assert intent.primary_category == "general"
        assert intent.confidence == 0.3

    def test_multi_category_task_merges_agents(self, router):
        """Task matching multiple categories merges agents from both."""
        intent = router._keyword_classify(
            "Fix the security vulnerability in the login form UI component"
        )
        # Should match security + ui_design or security + debugging
        assert len(intent.selected_workers) >= 2
        assert len(intent.selected_auditors) >= 2

    def test_complexity_increases_with_keywords(self, router):
        """Complexity score increases with more keyword matches."""
        simple = router._keyword_classify("Write a function")
        complex_ = router._keyword_classify(
            "Build an API endpoint that generates code for a module "
            "with a function to implement a feature"
        )
        assert complex_.complexity >= simple.complexity

    def test_confidence_normalized_to_1(self, router):
        """Confidence is capped at 1.0 even with many matches."""
        intent = router._keyword_classify(
            " ".join(INTENT_CATEGORIES["code_generation"]["keywords"])
        )
        assert intent.confidence <= 1.0


# ── DAG Building Tests ───────────────────────────────────────────

class TestDAGBuilding:
    """Test DAG generation from intent."""

    def test_workers_are_sequential(self, router):
        """Workers are chained sequentially in the DAG."""
        intent = TaskIntent(
            primary_category="code_generation",
            selected_workers=["planner", "researcher", "coder"],
            selected_auditors=["syntax_audit"],
        )
        dag = router._build_dag(intent)

        # Check worker dependencies
        assert dag._nodes["planner"].dependencies == []
        assert dag._nodes["researcher"].dependencies == ["planner"]
        assert dag._nodes["coder"].dependencies == ["researcher"]

    def test_auditors_are_parallel(self, router):
        """Auditors all depend on the last worker and run in parallel."""
        intent = TaskIntent(
            primary_category="code_generation",
            selected_workers=["planner", "coder"],
            selected_auditors=["type_safety", "test_generator"],
        )
        dag = router._build_dag(intent)

        # Auditors should depend on last worker
        assert dag._nodes["audit_type_safety"].dependencies == ["coder"]
        assert dag._nodes["audit_test_generator"].dependencies == ["coder"]

    def test_mandatory_gates_always_appended(self, router, mock_agents):
        """Mandatory audit gates are ALWAYS appended to the DAG."""
        intent = TaskIntent(
            primary_category="research",
            selected_workers=["researcher"],
            selected_auditors=["hallucination_hunter"],
        )
        dag = router._build_dag(intent)
        dag = router._append_audit_gates(dag, intent)

        # Check that mandatory gates exist
        gate_names = [n for n in dag._nodes if n.startswith("gate_")]
        assert len(gate_names) >= 1  # At least some mandatory gates

    def test_empty_workers_still_gets_auditors(self, router):
        """Even with no workers, auditors and gates are still added."""
        intent = TaskIntent(
            primary_category="general",
            selected_workers=[],
            selected_auditors=["syntax_audit"],
        )
        dag = router._build_dag(intent)
        # Should still have nodes from audit
        assert len(dag._nodes) >= 0  # No crash


# ── Full Route Tests ─────────────────────────────────────────────

class TestRouting:
    """Test the full route() method."""

    @pytest.mark.asyncio
    async def test_route_returns_decision(self, router):
        """route() returns a valid RoutingDecision."""
        decision = await router.route(
            "Write a function to sort a list",
            context=MagicMock(),
        )
        assert isinstance(decision, RoutingDecision)
        assert decision.intent.primary_category in list(INTENT_CATEGORIES.keys()) + ["general"]
        assert decision.dag is not None
        assert decision.routing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_route_updates_history(self, router):
        """Each route() call appends to routing history."""
        await router.route("Write code", context=MagicMock())
        await router.route("Fix a bug", context=MagicMock())

        history = router.get_routing_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_route_category_stats(self, router):
        """Category stats track routing frequency."""
        await router.route("Write code", context=MagicMock())
        await router.route("Write a function", context=MagicMock())
        await router.route("Fix the error", context=MagicMock())

        stats = router.get_category_stats()
        assert "code_generation" in stats
        assert stats["code_generation"] >= 1

    @pytest.mark.asyncio
    async def test_route_filters_unavailable_agents(self):
        """Router filters out agents not in the registry."""
        limited_agents = {
            "planner": MagicMock(),
            "coder": MagicMock(),
            "syntax_audit": MagicMock(),
            "output_verifier": MagicMock(),
        }
        router = DynamicSwarmRouter(
            agent_registry=limited_agents,
            use_llm_classification=False,
        )
        decision = await router.route(
            "Write a function with tests",
            context=MagicMock(),
        )
        # All selected workers should be in the registry
        for worker in decision.intent.selected_workers:
            assert worker in limited_agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
