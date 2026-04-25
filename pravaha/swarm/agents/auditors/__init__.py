"""Auditors — All 12 audit agents + regression guard."""

from pravaha.swarm.agents.auditors.syntax_audit_agent import SyntaxAuditAgent
from pravaha.swarm.agents.auditors.type_safety_agent import TypeSafetyAgent
from pravaha.swarm.agents.auditors.logic_flaw_agent import LogicFlawAgent
from pravaha.swarm.agents.auditors.consistency_guard_agent import ConsistencyGuardAgent
from pravaha.swarm.agents.auditors.hallucination_hunter_agent import HallucinationHunterAgent
from pravaha.swarm.agents.auditors.edge_case_hunter_agent import EdgeCaseHunterAgent
from pravaha.swarm.agents.auditors.performance_profiler_agent import PerformanceProfilerAgent
from pravaha.swarm.agents.auditors.output_verifier_agent import OutputVerifierAgent
from pravaha.swarm.agents.auditors.patch_applier_agent import PatchApplierAgent
from pravaha.swarm.agents.auditors.self_reflection_agent import SelfReflectionAgent
from pravaha.swarm.agents.auditors.test_generator_agent import TestGeneratorAgent
from pravaha.swarm.agents.auditors.regression_guard_agent import RegressionGuardAgent

AUDIT_AGENTS: dict[str, type] = {
    "syntax_audit": SyntaxAuditAgent,
    "type_safety": TypeSafetyAgent,
    "logic_flaw": LogicFlawAgent,
    "consistency_guard": ConsistencyGuardAgent,
    "hallucination_hunter": HallucinationHunterAgent,
    "edge_case_hunter": EdgeCaseHunterAgent,
    "performance_profiler": PerformanceProfilerAgent,
    "output_verifier": OutputVerifierAgent,
    "patch_applier": PatchApplierAgent,
    "self_reflection": SelfReflectionAgent,
    "test_generator": TestGeneratorAgent,
    "regression_guard": RegressionGuardAgent,
}

__all__ = ["AUDIT_AGENTS"]
