"""Swarm Orchestrator — Coordinate agent execution and pipeline management.

Manages the 51-agent swarm: task decomposition, worker pipeline
execution, and the self-healing audit loop. Uses SharedContext
for cross-agent communication.

v3.3: EventBus integration, SwarmProfiler, memory warm-up,
audit score progression tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pravaha.engine.event_bus import get_event_bus
from pravaha.swarm.agents import ALL_AGENTS
from pravaha.swarm.agents.base_agent import AgentMemory, AgentOutput, BaseAgent, SharedContext
from pravaha.swarm.memory.memory_store import MemoryStore
from pravaha.swarm.profiler import SwarmProfiler
from pravaha.swarm.tools import ToolRegistry

logger = logging.getLogger(__name__)

# Phase A: Static-only auditors (no LLM, instant regex scans)
STATIC_AUDITORS = [
    "syntax_audit", "type_safety", "security_audit",
    "injection_scanner", "crypto_audit", "secrets_scanner",
    "privilege_audit", "network_security", "compliance",
    "auth_audit", "dependency_audit", "api_security",
]

# Phase B: LLM auditors (require generation, expensive)
LLM_AUDITORS = [
    "logic_flaw", "hallucination_hunter", "consistency_guard",
    "edge_case_hunter", "performance_profiler", "output_verifier",
]

# Progressive confidence threshold: exit early if score is high enough
MIN_SCORES_BY_ITERATION = {
    1: 90.0,  # After first iteration, need 90+ to exit early
    2: 80.0,  # After second, 80+ is good enough
    3: 70.0,  # After third, 70+ is the minimum
}


class SwarmOrchestrator:
    """Orchestrate the 51-agent swarm execution with self-healing audit.

    v3.2 additions:
    - Parallel static auditor execution via asyncio.gather
    - Progressive confidence threshold for early exit
    - Issue severity triage (only CRITICAL/HIGH go to PatchApplier)
    - Execution-verified patching with result logging
    """

    def __init__(self, enabled_agents: list[str] | None = None) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._avatar_callback: Any = None

        # Initialize tool registry and memory store
        self._tools = ToolRegistry.default()
        self._memory = MemoryStore()

        # v3.3: Event bus and profiler
        self._bus = get_event_bus()
        self._profiler = SwarmProfiler()
        self._score_progression: list[float] = []

        for name, cls in ALL_AGENTS.items():
            if enabled_agents is None or name in enabled_agents:
                agent = cls()
                agent.attach_tools(self._tools)
                agent.attach_memory(
                    AgentMemory(self._memory, agent.role)
                )
                self._agents[name] = agent

        self.context = SharedContext()
        logger.info(f"SwarmOrchestrator: {len(self._agents)} agents loaded (v3.3)")
        self._bus.publish("swarm_init", {"agents_loaded": len(self._agents)})

    def set_avatar_callback(self, callback: Any) -> None:
        """Set callback for TUI avatar state updates."""
        self._avatar_callback = callback

    def _emit_avatar(self, state: str, agent_name: str = "") -> None:
        """Emit avatar state change event."""
        if self._avatar_callback:
            try:
                self._avatar_callback(state, agent_name)
            except Exception:
                pass

    async def execute_agent(self, name: str, task: str, engine: Any) -> AgentOutput:
        """Execute a single agent by name."""
        agent = self._agents.get(name)
        if not agent:
            return AgentOutput(role=name, output=f"Agent '{name}' not found", confidence=0.0)

        self._emit_avatar("working", name)
        self._bus.publish("agent_started", {"agent": name, "task": task[:80]})

        t0 = time.time()
        result = await agent.run(task, self.context, engine)
        duration = (time.time() - t0) * 1000

        self.context.agent_outputs[name] = result

        # Record in profiler
        self._profiler.record(
            role=name,
            duration_ms=duration,
            tokens=result.tokens_used,
        )

        self._bus.publish("agent_completed", {
            "agent": name,
            "duration_ms": round(duration, 1),
            "tokens": result.tokens_used,
            "confidence": result.confidence,
        })
        return result

    async def execute_pipeline(
        self, pipeline: list[str], task: str, engine: Any,
    ) -> list[AgentOutput]:
        """Execute a sequence of agents, piping context between them."""
        results: list[AgentOutput] = []
        self._bus.publish("pipeline_started", {"agents": pipeline, "task": task[:80]})
        for agent_name in pipeline:
            result = await self.execute_agent(agent_name, task, engine)
            results.append(result)
            self.context.output = result.output
        self._bus.publish("pipeline_completed", {"agents": pipeline, "total_results": len(results)})
        return results

    def warm_up_agent_memories(self, pipeline: list[str] | None = None) -> int:
        """Pre-load agent memories from the persistent store.

        Warms up the memory cache for agents in the pipeline
        (or all agents if no pipeline specified).
        """
        targets = pipeline if pipeline else list(self._agents.keys())
        warmed = 0
        for name in targets:
            agent = self._agents.get(name)
            if agent and agent._memory:
                count = self._memory.count(agent.role)
                if count > 0:
                    # Trigger memory access to warm cache
                    agent._memory.get_recent(limit=5)
                    warmed += 1
                    logger.debug(f"Memory warm-up: {name} ({count} memories)")
        logger.info(f"Memory warm-up complete: {warmed}/{len(targets)} agents warmed")
        self._bus.publish("memory_warmup", {"agents_warmed": warmed})
        return warmed

    def get_profiler(self) -> SwarmProfiler:
        """Get the swarm profiler instance."""
        return self._profiler

    async def execute_with_audit(
        self,
        worker_pipeline: list[str],
        task: str,
        engine: Any,
        max_iterations: int = 3,
        min_score: float = 70.0,
    ) -> dict[str, Any]:
        """Execute worker pipeline, then run self-healing audit loop.

        v3.2 improvements:
        1. Static auditors run in parallel (asyncio.gather)
        2. LLM auditors only run if static found issues OR first iteration
        3. Progressive confidence threshold for early exit
        4. Issue severity triage — only CRITICAL/HIGH go to PatchApplier
        5. Patch execution verification with logging
        """
        self.context.task = task
        self._emit_avatar("thinking")

        # Phase 1: Worker execution
        worker_results = await self.execute_pipeline(worker_pipeline, task, engine)
        if worker_results:
            self.context.output = worker_results[-1].output

        # Phase 2: Audit loop
        self._emit_avatar("audit")

        score = 0.0
        iteration = 0
        for iteration in range(max_iterations):
            all_issues: list[dict[str, Any]] = []

            # ── Phase A: Run all static auditors in parallel ──
            active_static = [name for name in STATIC_AUDITORS if name in self._agents]
            static_tasks = [
                self.execute_agent(name, task, engine)
                for name in active_static
            ]
            static_results = await asyncio.gather(
                *static_tasks, return_exceptions=True,
            )

            static_issues: list[dict[str, Any]] = []
            for result, name in zip(static_results, active_static):
                if isinstance(result, AgentOutput):
                    static_issues.extend(result.issues)
                elif isinstance(result, Exception):
                    logger.error(f"Static auditor '{name}' crashed: {result}", exc_info=result)
                    static_issues.append({
                        "type": "auditor_crash",
                        "severity": "CRITICAL",
                        "description": f"The static auditor '{name}' crashed with error: {str(result)}"
                    })
            all_issues.extend(static_issues)

            # ── Phase B: LLM auditors (only if needed) ──
            if static_issues or iteration == 0:
                for auditor_name in LLM_AUDITORS:
                    if auditor_name in self._agents:
                        try:
                            result = await self.execute_agent(
                                auditor_name, task, engine,
                            )
                            all_issues.extend(result.issues)
                        except Exception as e:
                            logger.error(f"LLM auditor '{auditor_name}' crashed: {e}", exc_info=e)
                            all_issues.append({
                                "type": "auditor_crash",
                                "severity": "CRITICAL",
                                "description": f"LLM auditor '{auditor_name}' crashed: {str(e)}"
                            })

            # ── Issue severity triage ──
            critical_issues = [
                i for i in all_issues
                if i.get("severity", "").upper()
                in ("CRITICAL", "HIGH", "MAJOR")
            ]
            info_issues = [
                i for i in all_issues
                if i.get("severity", "").upper()
                in ("LOW", "INFO", "WARNING", "MINOR", "warning")
            ]

            self.context.audit_reports.append({
                "iteration": iteration + 1,
                "issues": critical_issues,  # Only critical to PatchApplier
                "info_issues": info_issues,  # Logged but not patched
                "auditor_count": len(static_tasks) + len(LLM_AUDITORS),
            })

            # Get verifier score
            verifier = self.context.agent_outputs.get("output_verifier")
            score = verifier.metadata.get("score", 50) if verifier else 50

            # ── Progressive confidence threshold ──
            threshold = MIN_SCORES_BY_ITERATION.get(iteration + 1, min_score)
            if score >= threshold and not critical_issues:
                self._emit_avatar("success")
                logger.info(
                    f"Audit PASSED: iteration {iteration + 1}, "
                    f"score {score}, threshold {threshold}",
                )
                break

            # ── Patch only critical/high issues ──
            if critical_issues and "patch_applier" in self._agents:
                patch_result = await self.execute_agent(
                    "patch_applier", task, engine,
                )
                self.context.output = patch_result.output
                self.context.code = patch_result.output

                # Log execution verification results
                exec_verified = patch_result.metadata.get(
                    "execution_verified", False,
                )
                logger.info(
                    f"Patch applied: verified={exec_verified}, "
                    f"confidence={patch_result.confidence:.2f}",
                )
                if not exec_verified:
                    logger.warning(
                        "Patched code failed execution check. "
                        f"Error: {self.context.extra.get('patch_execution_error', 'unknown')}",
                    )

        # Phase 3: Self-reflection
        if "self_reflection" in self._agents:
            await self.execute_agent("self_reflection", task, engine)

        self._emit_avatar("success")

        return {
            "output": self.context.patched_output or self.context.output,
            "worker_results": worker_results,
            "audit_iterations": iteration + 1,
            "final_score": score,
            "issues_found": sum(
                len(r.get("issues", [])) for r in self.context.audit_reports
            ),
        }

    def list_agents(self) -> list[dict[str, Any]]:
        """Return stats for all loaded agents."""
        return [a.get_stats() for a in self._agents.values()]

    def reset(self) -> None:
        """Reset shared context for a new task."""
        self.context = SharedContext()
        self._emit_avatar("idle")
