"""Swarm configuration — agent roles, budgets, audit settings.

Defines the full swarm topology: which agents to activate, their token budgets,
audit pipeline ordering, and pipeline DAG definitions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentRoleConfig(BaseModel):
    """Configuration for a single agent role in the swarm.

    Attributes:
        name: Unique role identifier (e.g., 'planner', 'coder').
        enabled: Whether this agent is active.
        priority: Scheduling priority (0=worker, 1=senior, 2=orchestrator).
        max_tokens: Maximum tokens this agent may generate per invocation.
        temperature: Sampling temperature override for this role.
        model_override: Use a different model for this agent (e.g., a larger one).
    """

    name: str
    enabled: bool = True
    priority: int = 0
    max_tokens: int = 1024
    temperature: float = 0.5
    model_override: Optional[str] = None


class AuditConfig(BaseModel):
    """Configuration for the self-healing audit loop.

    Attributes:
        enabled: Master switch for the audit system.
        max_iterations: Maximum number of audit→patch→re-audit cycles.
        min_satisfaction_score: Minimum OutputVerifier score to pass without retry.
        pipeline: Ordered list of audit agent names to run.
        skip_for_types: Output types that skip certain auditors.
    """

    enabled: bool = True
    max_iterations: int = 3
    min_satisfaction_score: float = 70.0
    pipeline: list[str] = Field(default_factory=lambda: [
        "syntax_audit",
        "type_safety",
        "security_audit",
        "logic_flaw",
        "consistency_guard",
        "hallucination_hunter",
        "edge_case_hunter",
        "performance_profiler",
        "output_verifier",
    ])
    skip_for_types: dict[str, list[str]] = Field(default_factory=lambda: {
        "text": ["syntax_audit", "type_safety"],
        "analysis": ["syntax_audit"],
    })


class PipelineStepConfig(BaseModel):
    """A single step in a pipeline DAG.

    Attributes:
        agent: Agent role name to execute.
        depends_on: List of step names that must complete before this step.
        condition: Optional condition expression (e.g., 'output.confidence < 0.8').
    """

    agent: str
    depends_on: list[str] = Field(default_factory=list)
    condition: Optional[str] = None


class PipelineConfig(BaseModel):
    """A named pipeline defining a DAG of agent execution.

    Attributes:
        name: Pipeline identifier.
        description: Human-readable description.
        steps: Ordered list of pipeline steps.
    """

    name: str
    description: str = ""
    steps: list[PipelineStepConfig] = Field(default_factory=list)


class SwarmConfig(BaseModel):
    """Full swarm configuration: agents, audit, and pipelines.

    Loaded from a dedicated YAML file (e.g., configs/swarm_default.yaml).
    Defines which agents participate, their budgets, and how the audit
    loop operates.
    """

    enabled: bool = False
    agent_roles: list[AgentRoleConfig] = Field(default_factory=list)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    pipelines: list[PipelineConfig] = Field(default_factory=list)
    default_pipeline: str = "plan-execute-audit"
    shared_prefix_caching: bool = True
    max_concurrent_agents: int = 8
    total_token_budget: int = 50000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SwarmConfig":
        """Load swarm configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Swarm config not found: {path}, using defaults.")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls.model_validate(raw or {})

    @classmethod
    def default_with_all_agents(cls) -> "SwarmConfig":
        """Create a default config with all 32 agents enabled."""
        worker_roles = [
            AgentRoleConfig(name="planner", priority=2, max_tokens=512, temperature=0.3),
            AgentRoleConfig(name="researcher", priority=1, max_tokens=1024, temperature=0.5),
            AgentRoleConfig(name="coder", priority=1, max_tokens=2048, temperature=0.2),
            AgentRoleConfig(name="debugger", priority=1, max_tokens=1024, temperature=0.1),
            AgentRoleConfig(name="critic", priority=1, max_tokens=512, temperature=0.6),
            AgentRoleConfig(name="validator", priority=1, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="summarizer", priority=0, max_tokens=512, temperature=0.3),
            AgentRoleConfig(name="expander", priority=0, max_tokens=2048, temperature=0.7),
            AgentRoleConfig(name="translator", priority=0, max_tokens=2048, temperature=0.2),
            AgentRoleConfig(name="reasoning", priority=1, max_tokens=2048, temperature=0.1),
            AgentRoleConfig(name="merger", priority=2, max_tokens=2048, temperature=0.3),
            AgentRoleConfig(name="router", priority=2, max_tokens=128, temperature=0.1),
            AgentRoleConfig(name="memory", priority=1, max_tokens=512, temperature=0.2),
            AgentRoleConfig(name="tool", priority=1, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="judge", priority=2, max_tokens=256, temperature=0.2),
            AgentRoleConfig(name="refiner", priority=1, max_tokens=2048, temperature=0.4),
            AgentRoleConfig(name="classifier", priority=0, max_tokens=256, temperature=0.1),
            AgentRoleConfig(name="extractor", priority=0, max_tokens=1024, temperature=0.1),
            AgentRoleConfig(name="narrator", priority=0, max_tokens=1024, temperature=0.7),
            AgentRoleConfig(name="ensemble", priority=2, max_tokens=512, temperature=0.5),
        ]

        audit_roles = [
            AgentRoleConfig(name="syntax_audit", priority=1, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="logic_flaw", priority=1, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="hallucination_hunter", priority=1, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="security_audit", priority=1, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="performance_profiler", priority=1, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="consistency_guard", priority=1, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="type_safety", priority=1, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="edge_case_hunter", priority=1, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="test_generator", priority=1, max_tokens=2048, temperature=0.2),
            AgentRoleConfig(name="output_verifier", priority=2, max_tokens=512, temperature=0.1),
            AgentRoleConfig(name="self_reflection", priority=1, max_tokens=512, temperature=0.2),
            AgentRoleConfig(name="patch_applier", priority=1, max_tokens=2048, temperature=0.1),
        ]

        return cls(
            enabled=True,
            agent_roles=worker_roles + audit_roles,
        )
