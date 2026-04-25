"""Swarm Orchestrator — Coordinate agent execution and pipeline management.

Manages the 51-agent swarm: task decomposition, worker pipeline
execution, and the self-healing audit loop. Uses SharedContext
for cross-agent communication.

v3.1: Initializes tool registry and persistent memory for all agents.
Emits avatar state events for TUI integration.
"""

from __future__ import annotations

import logging
from typing import Any

from pravaha.swarm.agents import ALL_AGENTS
from pravaha.swarm.agents.base_agent import AgentMemory, AgentOutput, BaseAgent, SharedContext
from pravaha.swarm.memory.memory_store import MemoryStore
from pravaha.swarm.tools import ToolRegistry

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """Orchestrate the 51-agent swarm execution with self-healing audit.

    v3.1 additions:
    - Initializes ToolRegistry for all agents on construction
    - Attaches persistent MemoryStore to each agent
    - Emits avatar state events for TUI
    """

    def __init__(self, enabled_agents: list[str] | None = None) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._avatar_callback: Any = None

        # Initialize tool registry and memory store
        self._tools = ToolRegistry.default()
        self._memory = MemoryStore()

        for name, cls in ALL_AGENTS.items():
            if enabled_agents is None or name in enabled_agents:
                agent = cls()
                # Attach tools and memory to every agent
                agent.attach_tools(self._tools)
                agent.attach_memory(
                    AgentMemory(self._memory, agent.role)
                )
                self._agents[name] = agent

        self.context = SharedContext()
        logger.info(f"SwarmOrchestrator: {len(self._agents)} agents loaded (v3.1)")

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
        result = await agent.run(task, self.context, engine)
        self.context.agent_outputs[name] = result
        return result

    async def execute_pipeline(
        self, pipeline: list[str], task: str, engine: Any
    ) -> list[AgentOutput]:
        """Execute a sequence of agents, piping context between them."""
        results: list[AgentOutput] = []
        for agent_name in pipeline:
            result = await self.execute_agent(agent_name, task, engine)
            results.append(result)
            self.context.output = result.output
        return results

    async def execute_with_audit(
        self,
        worker_pipeline: list[str],
        task: str,
        engine: Any,
        max_iterations: int = 3,
        min_score: float = 70.0,
    ) -> dict[str, Any]:
        """Execute worker pipeline, then run self-healing audit loop.

        The audit loop:
        1. Run auditor agents on the output
        2. If issues found, run PatchApplier to fix them
        3. Re-audit the patched output
        4. Repeat up to max_iterations
        5. Return final output with audit report
        """
        self.context.task = task
        self._emit_avatar("thinking")

        # Phase 1: Worker execution
        worker_results = await self.execute_pipeline(worker_pipeline, task, engine)
        if worker_results:
            self.context.output = worker_results[-1].output

        # Phase 2: Audit loop
        self._emit_avatar("audit")
        audit_pipeline = [
            "syntax_audit",
            "type_safety",
            "security_audit",
            "logic_flaw",
            "consistency_guard",
            "hallucination_hunter",
            "edge_case_hunter",
            "performance_profiler",
            "output_verifier",
        ]

        score = 0.0
        iteration = 0
        for iteration in range(max_iterations):
            all_issues: list[dict[str, Any]] = []
            audit_results: list[AgentOutput] = []

            for auditor_name in audit_pipeline:
                if auditor_name not in self._agents:
                    continue
                result = await self.execute_agent(auditor_name, task, engine)
                audit_results.append(result)
                all_issues.extend(result.issues)

            self.context.audit_reports.append(
                {
                    "iteration": iteration + 1,
                    "issues": all_issues,
                    "auditor_count": len(audit_results),
                }
            )

            verifier = self.context.agent_outputs.get("output_verifier")
            score = verifier.metadata.get("score", 50) if verifier else 50

            if score >= min_score and not all_issues:
                self._emit_avatar("success")
                logger.info(f"Audit PASSED: iteration {iteration + 1}, score {score}")
                break

            if all_issues and "patch_applier" in self._agents:
                patch_result = await self.execute_agent("patch_applier", task, engine)
                self.context.output = patch_result.output
                self.context.code = patch_result.output

        # Phase 3: Self-reflection
        if "self_reflection" in self._agents:
            await self.execute_agent("self_reflection", task, engine)

        self._emit_avatar("success")

        return {
            "output": self.context.patched_output or self.context.output,
            "worker_results": worker_results,
            "audit_iterations": iteration + 1,
            "final_score": score,
            "issues_found": sum(len(r.get("issues", [])) for r in self.context.audit_reports),
        }

    def list_agents(self) -> list[dict[str, Any]]:
        """Return stats for all loaded agents."""
        return [a.get_stats() for a in self._agents.values()]

    def reset(self) -> None:
        """Reset shared context for a new task."""
        self.context = SharedContext()
        self._emit_avatar("idle")
