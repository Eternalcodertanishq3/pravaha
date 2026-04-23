"""Swarm Pipeline — Named execution pipelines for common workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pipeline:
    """A named sequence of agent executions."""

    name: str
    description: str = ""
    worker_steps: list[str] = field(default_factory=list)
    audit_steps: list[str] = field(default_factory=list)


# Built-in pipelines
BUILTIN_PIPELINES: dict[str, Pipeline] = {
    "plan-execute-audit": Pipeline(
        name="plan-execute-audit",
        description="Plan task, execute with coder, audit and patch.",
        worker_steps=["planner", "researcher", "coder"],
        audit_steps=[
            "syntax_audit",
            "security_audit",
            "logic_flaw",
            "edge_case_hunter",
            "output_verifier",
            "patch_applier",
        ],
    ),
    "research-summarize": Pipeline(
        name="research-summarize",
        description="Research a topic and summarize findings.",
        worker_steps=["researcher", "reasoning", "summarizer"],
        audit_steps=["hallucination_hunter", "consistency_guard", "output_verifier"],
    ),
    "code-review": Pipeline(
        name="code-review",
        description="Generate code, debug, test, and review.",
        worker_steps=["coder", "debugger", "critic", "refiner"],
        audit_steps=[
            "syntax_audit",
            "type_safety",
            "security_audit",
            "performance_profiler",
            "test_generator",
            "output_verifier",
            "patch_applier",
        ],
    ),
    "creative-write": Pipeline(
        name="creative-write",
        description="Creative writing with narrative quality.",
        worker_steps=["narrator", "expander", "refiner"],
        audit_steps=["consistency_guard", "output_verifier"],
    ),
    "extract-classify": Pipeline(
        name="extract-classify",
        description="Extract data and classify it.",
        worker_steps=["extractor", "classifier", "validator"],
        audit_steps=["output_verifier"],
    ),
}
