"""Pipeline DAG — Directed Acyclic Graph for parallel pipeline execution.

Allows agents to run in parallel when their dependencies are satisfied.
Uses asyncio.gather() for concurrent execution of independent steps.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DAGNode:
    """A node in the pipeline DAG."""

    name: str
    agent_role: str
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, done, failed
    result: Any = None
    duration_ms: float = 0.0


class PipelineDAG:
    """DAG-based pipeline execution with parallel steps.

    Agents with no unmet dependencies run concurrently via
    asyncio.gather(). This is a major performance improvement
    over sequential pipeline execution.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, DAGNode] = {}

    def add_node(
        self,
        name: str,
        agent_role: str,
        dependencies: list[str] | None = None,
    ) -> None:
        """Add a node to the DAG."""
        self._nodes[name] = DAGNode(
            name=name,
            agent_role=agent_role,
            dependencies=dependencies or [],
        )

    def get_ready_nodes(self) -> list[DAGNode]:
        """Get nodes whose dependencies are all satisfied."""
        ready = []
        for node in self._nodes.values():
            if node.status != "pending":
                continue
            deps_met = all(
                self._nodes[dep].status == "done"
                for dep in node.dependencies
                if dep in self._nodes
            )
            if deps_met:
                ready.append(node)
        return ready

    def is_complete(self) -> bool:
        """Check if all nodes are done or failed."""
        return all(
            n.status in ("done", "failed") for n in self._nodes.values()
        )

    def has_failed(self) -> bool:
        """Check if any node has failed."""
        return any(n.status == "failed" for n in self._nodes.values())

    async def execute(
        self,
        run_agent: Any,  # async callable(role, context) -> AgentOutput
        context: Any,
        max_parallel: int = 4,
    ) -> dict[str, Any]:
        """Execute the DAG with parallel steps.

        Args:
            run_agent: Async function that takes (agent_role, context)
                       and returns an AgentOutput.
            context: SharedContext passed to each agent.
            max_parallel: Maximum concurrent agent executions.
        """
        t0 = time.time()
        results: dict[str, Any] = {}
        iterations = 0
        max_iterations = len(self._nodes) * 2  # Safety limit

        while not self.is_complete() and iterations < max_iterations:
            iterations += 1
            ready = self.get_ready_nodes()
            if not ready:
                if not self.is_complete():
                    # Deadlock or all remaining have unmet deps
                    logger.warning("DAG execution stalled — unmet dependencies")
                    break
                continue

            # Run ready nodes in parallel (bounded)
            batch = ready[:max_parallel]
            for node in batch:
                node.status = "running"

            tasks = [
                self._run_node(node, run_agent, context)
                for node in batch
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for node in batch:
                if node.result is not None:
                    results[node.name] = node.result

        total_ms = (time.time() - t0) * 1000
        logger.info(
            f"DAG execution complete: {len(results)}/{len(self._nodes)} nodes, "
            f"{total_ms:.0f}ms total, {iterations} iterations"
        )

        return {
            "results": results,
            "total_nodes": len(self._nodes),
            "completed": sum(1 for n in self._nodes.values() if n.status == "done"),
            "failed": sum(1 for n in self._nodes.values() if n.status == "failed"),
            "total_ms": round(total_ms, 1),
        }

    async def _run_node(
        self,
        node: DAGNode,
        run_agent: Any,
        context: Any,
    ) -> None:
        """Execute a single DAG node."""
        t0 = time.time()
        try:
            result = await run_agent(node.agent_role, context)
            node.result = result
            node.status = "done"
            node.duration_ms = (time.time() - t0) * 1000
            logger.debug(
                f"DAG node '{node.name}' completed in {node.duration_ms:.0f}ms"
            )
        except Exception as e:
            node.status = "failed"
            node.duration_ms = (time.time() - t0) * 1000
            logger.error(f"DAG node '{node.name}' failed: {e}")

    def get_execution_order(self) -> list[list[str]]:
        """Get the execution layers (for visualization)."""
        layers: list[list[str]] = []
        done: set[str] = set()

        while len(done) < len(self._nodes):
            layer = []
            for name, node in self._nodes.items():
                if name in done:
                    continue
                if all(dep in done for dep in node.dependencies):
                    layer.append(name)
            if not layer:
                break
            layers.append(layer)
            done.update(layer)

        return layers

    @classmethod
    def from_pipeline_config(
        cls,
        workers: list[str],
        auditors: list[str],
    ) -> PipelineDAG:
        """Build a DAG from a standard pipeline config.

        Workers run in sequence (each depends on previous).
        Auditors run in parallel after all workers complete.
        """
        dag = cls()

        # Workers: sequential chain
        for i, role in enumerate(workers):
            deps = [workers[i - 1]] if i > 0 else []
            dag.add_node(role, agent_role=role, dependencies=deps)

        # Auditors: all depend on last worker, run in parallel
        last_worker = workers[-1] if workers else None
        for role in auditors:
            deps = [last_worker] if last_worker else []
            dag.add_node(
                f"audit_{role}",
                agent_role=role,
                dependencies=deps,
            )

        return dag
