"""Swarm Orchestrator — Coordinate agent execution and pipeline management.

Manages the 32-agent swarm: task decomposition, worker pipeline
execution, and the self-healing audit loop. Uses SharedContext
for cross-agent communication.

Phase 5: Core swarm orchestration — the brain of Pravaha's
multi-agent system.
"""

from __future__ import annotations

import logging
from typing import Any

from pravaha.swarm.agents import ALL_AGENTS
from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """Orchestrate the 32-agent swarm execution with self-healing audit."""

    def __init__(self, enabled_agents: list[str] | None = None) -> None:
        self._agents: dict[str, BaseAgent] = {}
        for name, cls in ALL_AGENTS.items():
            if enabled_agents is None or name in enabled_agents:
                self._agents[name] = cls()
        self.context = SharedContext()
        logger.info(f"SwarmOrchestrator: {len(self._agents)} agents loaded")

    async def execute_agent(self, name: str, task: str, engine: Any) -> AgentOutput:
        """Execute a single agent by name."""
        agent = self._agents.get(name)
        if not agent:
            return AgentOutput(role=name, output=f"Agent '{name}' not found", confidence=0.0)
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
            # Update context output for next agent
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

        # Phase 1: Worker execution
        worker_results = await self.execute_pipeline(worker_pipeline, task, engine)
        if worker_results:
            self.context.output = worker_results[-1].output

        # Phase 2: Audit loop
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

            # Store audit report in context
            self.context.audit_reports.append(
                {
                    "iteration": iteration + 1,
                    "issues": all_issues,
                    "auditor_count": len(audit_results),
                }
            )

            # Check verifier score
            verifier = self.context.agent_outputs.get("output_verifier")
            score = verifier.metadata.get("score", 50) if verifier else 50

            if score >= min_score and not all_issues:
                logger.info(f"Audit PASSED: iteration {iteration + 1}, score {score}")
                break

            # Apply patches if issues found
            if all_issues and "patch_applier" in self._agents:
                patch_result = await self.execute_agent("patch_applier", task, engine)
                self.context.output = patch_result.output
                self.context.code = patch_result.output

        # Phase 3: Self-reflection (logged, not shown to user)
        if "self_reflection" in self._agents:
            await self.execute_agent("self_reflection", task, engine)

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
