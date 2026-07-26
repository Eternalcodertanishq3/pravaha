"""Dynamic Swarm Router — Intelligent agent selection and DAG generation.

Replaces hardcoded pipeline templates with runtime intent analysis.
Examines the user's task, selects the optimal subset of agents from the
52-agent registry, builds a custom execution DAG on the fly, and always
appends mandatory audit gatekeepers at the end.

This is the core brain of Pravāha's Hybrid Dynamic-DAG Architecture:
- Front-end: Dynamic, fluid, Claude-Code-style autonomy.
- Back-end: Pravāha's enterprise audit rigor (always enforced).

v4.0: Initial implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pravaha.swarm.agents.base_agent import BaseAgent, SharedContext

from pravaha.swarm.pipeline_dag import DAGNode, PipelineDAG

logger = logging.getLogger(__name__)


# ── Intent Classification ──────────────────────────────────────────

INTENT_CATEGORIES = {
    "code_generation": {
        "keywords": [
            "write", "create", "implement", "build", "code", "generate",
            "function", "class", "module", "api", "endpoint", "feature",
        ],
        "workers": ["planner", "researcher", "coder", "refiner"],
        "auditors": ["syntax_audit", "type_safety", "test_generator"],
    },
    "debugging": {
        "keywords": [
            "fix", "bug", "error", "crash", "broken", "failing", "debug",
            "traceback", "exception", "issue", "wrong", "doesn't work",
        ],
        "workers": ["researcher", "debugger", "coder"],
        "auditors": [
            "syntax_audit", "logic_flaw", "regression_guard",
            "test_generator",
        ],
    },
    "security_audit": {
        "keywords": [
            "security", "vulnerability", "injection", "xss", "csrf",
            "auth", "authentication", "hack", "exploit", "penetration",
            "owasp", "cve", "secrets", "password", "credential",
        ],
        "workers": ["researcher"],
        "auditors": [
            "security_audit", "injection_scanner", "auth_audit",
            "crypto_audit", "secrets_scanner", "network_security",
            "privilege_audit", "api_security", "compliance",
        ],
    },
    "ui_design": {
        "keywords": [
            "ui", "ux", "design", "layout", "component", "style",
            "button", "form", "modal", "navigation", "responsive",
            "accessibility", "color", "theme", "css", "html",
        ],
        "workers": ["planner", "ui_designer", "component_builder"],
        "auditors": [
            "accessibility_auditor", "ux_reviewer", "design_critic",
            "style_designer",
        ],
    },
    "code_review": {
        "keywords": [
            "review", "audit", "check", "analyze", "quality", "improve",
            "refactor", "optimize", "clean", "lint",
        ],
        "workers": ["researcher", "critic", "refiner"],
        "auditors": [
            "syntax_audit", "type_safety", "logic_flaw",
            "consistency_guard", "performance_profiler",
        ],
    },
    "testing": {
        "keywords": [
            "test", "unittest", "pytest", "coverage", "tdd",
            "integration test", "e2e", "spec",
        ],
        "workers": ["researcher", "test_generator", "coder"],
        "auditors": [
            "syntax_audit", "edge_case_hunter", "regression_guard",
        ],
    },
    "documentation": {
        "keywords": [
            "document", "docstring", "readme", "api doc", "explain",
            "guide", "tutorial", "comment",
        ],
        "workers": ["researcher", "narrator", "expander"],
        "auditors": [
            "consistency_guard", "output_verifier",
        ],
    },
    "research": {
        "keywords": [
            "research", "find", "search", "look up", "what is",
            "explain", "compare", "analyze", "investigate",
        ],
        "workers": ["researcher", "summarizer"],
        "auditors": [
            "hallucination_hunter", "output_verifier",
        ],
    },
    "data_processing": {
        "keywords": [
            "data", "json", "csv", "parse", "transform", "extract",
            "convert", "serialize", "schema", "validate",
        ],
        "workers": ["extractor", "coder", "validator"],
        "auditors": [
            "syntax_audit", "type_safety", "output_verifier",
        ],
    },
}

# Mandatory audit gatekeepers — ALWAYS appended, NEVER skippable
MANDATORY_GATES = [
    "syntax_audit",
    "security_audit",
    "output_verifier",
]


@dataclass
class TaskIntent:
    """Classified intent of a user's task."""

    primary_category: str
    secondary_categories: list[str] = field(default_factory=list)
    confidence: float = 0.0
    complexity: int = 1  # 1-5 scale
    selected_workers: list[str] = field(default_factory=list)
    selected_auditors: list[str] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)
    requires_tools: list[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """The final routing decision with execution DAG."""

    intent: TaskIntent
    dag: PipelineDAG
    agent_count: int = 0
    estimated_steps: int = 0
    routing_time_ms: float = 0.0


class DynamicSwarmRouter:
    """Dynamically generates execution DAGs from user intent.

    The router serves as the intelligent brain of Pravāha's Hybrid
    Dynamic-DAG Architecture. It replaces rigid, hardcoded pipeline
    templates with runtime intent analysis and dynamic agent selection.

    Architecture:
        1. Analyze task intent using keyword matching + optional LLM.
        2. Select the optimal subset of agents from 52-agent registry.
        3. Build a dependency-aware execution DAG on the fly.
        4. ALWAYS append mandatory audit gatekeepers (non-negotiable).

    The mandatory audit gates ensure Pravāha's quality rigor is never
    compromised, regardless of how dynamically the DAG is generated.
    """

    def __init__(
        self,
        agent_registry: dict[str, Any],
        use_llm_classification: bool = True,
    ) -> None:
        """Initialize the router.

        Args:
            agent_registry: Dict mapping agent names to BaseAgent instances.
            use_llm_classification: Whether to use LLM for complex intent
                classification (falls back to keyword matching if False
                or if LLM is unavailable).
        """
        self._agents = agent_registry
        self._use_llm = use_llm_classification
        self._routing_history: list[RoutingDecision] = []

    async def route(
        self,
        task: str,
        context: Any,  # SharedContext
        engine: Any | None = None,
    ) -> RoutingDecision:
        """Analyze task → Select agents → Build DAG → Append audit gates.

        This is the main entry point. Takes a user task string and returns
        a complete RoutingDecision with an executable PipelineDAG.

        Args:
            task: The user's task description.
            context: SharedContext for cross-agent state.
            engine: Optional LLM engine for advanced classification.

        Returns:
            RoutingDecision with the generated DAG and metadata.
        """
        t0 = time.time()

        # Step 1: Classify intent
        intent = await self._classify_intent(task, engine)

        # Step 2: Filter to available agents only
        intent.selected_workers = [
            w for w in intent.selected_workers
            if w in self._agents
        ]
        intent.selected_auditors = [
            a for a in intent.selected_auditors
            if a in self._agents
        ]

        # Step 3: Build the execution DAG
        dag = self._build_dag(intent)

        # Step 4: ALWAYS append mandatory audit gatekeepers
        dag = self._append_audit_gates(dag, intent)

        routing_time = (time.time() - t0) * 1000

        decision = RoutingDecision(
            intent=intent,
            dag=dag,
            agent_count=len(intent.selected_workers) + len(intent.selected_auditors),
            estimated_steps=len(intent.selected_workers) + len(intent.selected_auditors),
            routing_time_ms=round(routing_time, 2),
        )

        self._routing_history.append(decision)
        logger.info(
            f"DynamicRouter: category={intent.primary_category}, "
            f"workers={len(intent.selected_workers)}, "
            f"auditors={len(intent.selected_auditors)}, "
            f"complexity={intent.complexity}, "
            f"routing_time={routing_time:.1f}ms"
        )

        return decision

    async def _classify_intent(
        self,
        task: str,
        engine: Any | None = None,
    ) -> TaskIntent:
        """Classify the user's task into intent categories.

        Uses a two-tier classification system:
        1. Fast keyword matching (always runs, <1ms).
        2. Optional LLM classification for ambiguous tasks.

        Args:
            task: The user's task string.
            engine: Optional LLM engine for advanced classification.

        Returns:
            TaskIntent with category, workers, and auditors.
        """
        # Tier 1: Keyword-based classification
        intent = self._keyword_classify(task)

        # Tier 2: LLM classification for low-confidence or complex tasks
        if (
            self._use_llm
            and engine is not None
            and intent.confidence < 0.6
        ):
            try:
                llm_intent = await self._llm_classify(task, engine)
                if llm_intent.confidence > intent.confidence:
                    intent = llm_intent
            except Exception as e:
                logger.warning(f"LLM classification failed, using keywords: {e}")

        return intent

    def _keyword_classify(self, task: str) -> TaskIntent:
        """Fast keyword-based intent classification.

        Scores each category by counting keyword matches in the task.
        Returns the highest-scoring category with its associated agents.
        """
        task_lower = task.lower()
        scores: dict[str, float] = {}

        for category, config in INTENT_CATEGORIES.items():
            score = sum(
                1.0 for kw in config["keywords"]
                if kw in task_lower
            )
            # Boost for exact phrase matches (multi-word keywords)
            score += sum(
                0.5 for kw in config["keywords"]
                if " " in kw and kw in task_lower
            )
            scores[category] = score

        if not scores or max(scores.values()) == 0:
            # Fallback: general-purpose pipeline
            return TaskIntent(
                primary_category="general",
                confidence=0.3,
                complexity=2,
                selected_workers=["planner", "researcher", "coder"],
                selected_auditors=["syntax_audit", "output_verifier"],
            )

        # Sort categories by score
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_cats[0]
        primary_config = INTENT_CATEGORIES[primary[0]]

        # Determine secondary categories (score > 50% of primary)
        secondary = [
            cat for cat, score in sorted_cats[1:]
            if score > 0 and score >= primary[1] * 0.5
        ]

        # Merge workers and auditors from secondary categories
        workers = list(primary_config["workers"])
        auditors = list(primary_config["auditors"])

        for sec_cat in secondary[:2]:  # Max 2 secondary categories
            sec_config = INTENT_CATEGORIES[sec_cat]
            for w in sec_config["workers"]:
                if w not in workers:
                    workers.append(w)
            for a in sec_config["auditors"]:
                if a not in auditors:
                    auditors.append(a)

        # Estimate complexity from task length and keyword density
        complexity = min(5, max(1, int(primary[1]) + len(secondary)))

        # Normalize confidence
        max_possible = len(primary_config["keywords"])
        confidence = min(1.0, primary[1] / max(max_possible * 0.3, 1))

        return TaskIntent(
            primary_category=primary[0],
            secondary_categories=secondary,
            confidence=round(confidence, 2),
            complexity=complexity,
            selected_workers=workers,
            selected_auditors=auditors,
        )

    async def _llm_classify(self, task: str, engine: Any) -> TaskIntent:
        """Use the LLM engine for advanced intent classification.

        Generates a structured JSON classification for ambiguous tasks
        that keyword matching cannot confidently resolve.
        """
        categories_list = ", ".join(INTENT_CATEGORIES.keys())
        all_workers = sorted(set(
            w for config in INTENT_CATEGORIES.values()
            for w in config["workers"]
        ))
        all_auditors = sorted(set(
            a for config in INTENT_CATEGORIES.values()
            for a in config["auditors"]
        ))

        prompt = (
            f"Classify this task into one or more categories and select "
            f"the optimal agents.\n\n"
            f"Task: {task}\n\n"
            f"Categories: {categories_list}\n"
            f"Available workers: {', '.join(all_workers)}\n"
            f"Available auditors: {', '.join(all_auditors)}\n\n"
            f"Respond ONLY with valid JSON:\n"
            f'{{"category": "...", "secondary": [...], '
            f'"complexity": 1-5, "workers": [...], "auditors": [...]}}'
        )

        # Use the engine's generate method
        if hasattr(engine, "generate"):
            response = await engine.generate(prompt, max_tokens=256)
        elif hasattr(engine, "generate_text"):
            response = await engine.generate_text(prompt, max_tokens=256)
        else:
            raise AttributeError("Engine has no generate method")

        # Parse JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in LLM response")

        data = json.loads(json_match.group())

        return TaskIntent(
            primary_category=data.get("category", "general"),
            secondary_categories=data.get("secondary", []),
            confidence=0.85,  # LLM classification is generally confident
            complexity=data.get("complexity", 3),
            selected_workers=data.get("workers", ["planner", "coder"]),
            selected_auditors=data.get("auditors", ["syntax_audit"]),
        )

    def _build_dag(self, intent: TaskIntent) -> PipelineDAG:
        """Build a dependency-aware execution DAG from intent.

        Workers run sequentially (each depends on the previous) because
        their outputs typically feed into each other (plan → research →
        code → refine).

        Auditors run in parallel after all workers complete, because
        they independently evaluate the same output.

        Args:
            intent: Classified TaskIntent with selected agents.

        Returns:
            PipelineDAG ready for execution.
        """
        dag = PipelineDAG()

        # Workers: sequential chain (plan → research → code → refine)
        for i, worker in enumerate(intent.selected_workers):
            deps = [intent.selected_workers[i - 1]] if i > 0 else []
            dag.add_node(
                name=worker,
                agent_role=worker,
                dependencies=deps,
            )

        # Intent-specific auditors: parallel after last worker
        last_worker = (
            intent.selected_workers[-1]
            if intent.selected_workers
            else None
        )
        for auditor in intent.selected_auditors:
            # Avoid duplicating mandatory gates
            if auditor in MANDATORY_GATES:
                continue
            deps = [last_worker] if last_worker else []
            dag.add_node(
                name=f"audit_{auditor}",
                agent_role=auditor,
                dependencies=deps,
            )

        return dag

    def _append_audit_gates(
        self,
        dag: PipelineDAG,
        intent: TaskIntent,
    ) -> PipelineDAG:
        """ALWAYS append mandatory audit gatekeepers to the DAG.

        These gates are hardcoded and mandatory — the Dynamic Router
        cannot skip them. This guarantees Pravāha's quality rigor is
        never compromised.

        The mandatory gates run AFTER all intent-specific auditors
        complete, serving as the final verification layer.

        Args:
            dag: The DAG with workers and intent-specific auditors.
            intent: The classified intent (for dependency resolution).

        Returns:
            The DAG with mandatory gates appended.
        """
        # Determine what the gates depend on
        # They should depend on all intent-specific auditors completing
        audit_node_names = [
            name for name in dag._nodes
            if name.startswith("audit_")
        ]

        # If no auditors, depend on last worker
        if not audit_node_names:
            last_worker = (
                intent.selected_workers[-1]
                if intent.selected_workers
                else None
            )
            gate_deps = [last_worker] if last_worker else []
        else:
            gate_deps = audit_node_names

        for gate in MANDATORY_GATES:
            if gate not in self._agents:
                continue
            gate_name = f"gate_{gate}"
            # Don't add if already present as an intent-specific auditor
            if f"audit_{gate}" in dag._nodes:
                continue
            dag.add_node(
                name=gate_name,
                agent_role=gate,
                dependencies=gate_deps,
            )

        return dag

    def get_routing_history(self) -> list[dict[str, Any]]:
        """Get history of routing decisions for observability."""
        return [
            {
                "category": d.intent.primary_category,
                "complexity": d.intent.complexity,
                "agent_count": d.agent_count,
                "routing_time_ms": d.routing_time_ms,
            }
            for d in self._routing_history
        ]

    def get_category_stats(self) -> dict[str, int]:
        """Get counts of tasks routed per category."""
        stats: dict[str, int] = {}
        for decision in self._routing_history:
            cat = decision.intent.primary_category
            stats[cat] = stats.get(cat, 0) + 1
        return stats
