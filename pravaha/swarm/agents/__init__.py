"""Pravaha Swarm Agents — 20 Workers + 12 Auditors.

All 32 agents are individually implemented in their own files.
This __init__.py provides the master registries for discovery.
"""

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext
from pravaha.swarm.agents.classifier_agent import ClassifierAgent
from pravaha.swarm.agents.coder_agent import CoderAgent
from pravaha.swarm.agents.consistency_guard_agent import ConsistencyGuardAgent
from pravaha.swarm.agents.critic_agent import CriticAgent
from pravaha.swarm.agents.debugger_agent import DebuggerAgent
from pravaha.swarm.agents.edge_case_hunter_agent import EdgeCaseHunterAgent
from pravaha.swarm.agents.ensemble_agent import EnsembleAgent
from pravaha.swarm.agents.expander_agent import ExpanderAgent
from pravaha.swarm.agents.extractor_agent import ExtractorAgent
from pravaha.swarm.agents.hallucination_hunter_agent import HallucinationHunterAgent
from pravaha.swarm.agents.judge_agent import JudgeAgent
from pravaha.swarm.agents.logic_flaw_agent import LogicFlawAgent
from pravaha.swarm.agents.memory_agent import MemoryAgent
from pravaha.swarm.agents.merger_agent import MergerAgent
from pravaha.swarm.agents.narrator_agent import NarratorAgent
from pravaha.swarm.agents.output_verifier_agent import OutputVerifierAgent
from pravaha.swarm.agents.patch_applier_agent import PatchApplierAgent
from pravaha.swarm.agents.performance_profiler_agent import PerformanceProfilerAgent

# ── 20 Worker Agents ──
from pravaha.swarm.agents.planner_agent import PlannerAgent
from pravaha.swarm.agents.reasoning_agent import ReasoningAgent
from pravaha.swarm.agents.refiner_agent import RefinerAgent
from pravaha.swarm.agents.researcher_agent import ResearcherAgent
from pravaha.swarm.agents.router_agent import RouterAgent
from pravaha.swarm.agents.security_audit_agent import SecurityAuditAgent
from pravaha.swarm.agents.self_reflection_agent import SelfReflectionAgent
from pravaha.swarm.agents.summarizer_agent import SummarizerAgent

# ── 12 Self-Healing Audit Agents ──
from pravaha.swarm.agents.syntax_audit_agent import SyntaxAuditAgent
from pravaha.swarm.agents.test_generator_agent import TestGeneratorAgent
from pravaha.swarm.agents.tool_agent import ToolAgent
from pravaha.swarm.agents.translator_agent import TranslatorAgent
from pravaha.swarm.agents.type_safety_agent import TypeSafetyAgent
from pravaha.swarm.agents.validator_agent import ValidatorAgent

# Master registries
WORKER_AGENTS: dict[str, type[BaseAgent]] = {
    "planner": PlannerAgent,
    "researcher": ResearcherAgent,
    "coder": CoderAgent,
    "debugger": DebuggerAgent,
    "critic": CriticAgent,
    "validator": ValidatorAgent,
    "summarizer": SummarizerAgent,
    "expander": ExpanderAgent,
    "translator": TranslatorAgent,
    "reasoning": ReasoningAgent,
    "merger": MergerAgent,
    "router": RouterAgent,
    "memory": MemoryAgent,
    "tool": ToolAgent,
    "judge": JudgeAgent,
    "refiner": RefinerAgent,
    "classifier": ClassifierAgent,
    "extractor": ExtractorAgent,
    "narrator": NarratorAgent,
    "ensemble": EnsembleAgent,
}

AUDIT_AGENTS: dict[str, type[BaseAgent]] = {
    "syntax_audit": SyntaxAuditAgent,
    "logic_flaw": LogicFlawAgent,
    "hallucination_hunter": HallucinationHunterAgent,
    "security_audit": SecurityAuditAgent,
    "performance_profiler": PerformanceProfilerAgent,
    "consistency_guard": ConsistencyGuardAgent,
    "type_safety": TypeSafetyAgent,
    "edge_case_hunter": EdgeCaseHunterAgent,
    "test_generator": TestGeneratorAgent,
    "output_verifier": OutputVerifierAgent,
    "self_reflection": SelfReflectionAgent,
    "patch_applier": PatchApplierAgent,
}

ALL_AGENTS: dict[str, type[BaseAgent]] = {**WORKER_AGENTS, **AUDIT_AGENTS}

__all__ = [
    "BaseAgent",
    "AgentOutput",
    "SharedContext",
    "WORKER_AGENTS",
    "AUDIT_AGENTS",
    "ALL_AGENTS",
]
