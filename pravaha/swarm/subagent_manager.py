from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pravaha.swarm.agents.base_agent import AgentOutput, SharedContext

logger = logging.getLogger(__name__)

@dataclass
class SubagentHandle:
    """Handle for a spawned subagent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_role: str = ""
    agent_type: str = ""
    task: str = ""
    status: str = "pending"
    result: Any | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    _event: asyncio.Event = field(default_factory=asyncio.Event)
    
    async def wait(self, timeout: float | None = None) -> Any | None:
        """Waits for the subagent to complete."""
        try:
            if timeout is not None:
                await asyncio.wait_for(self._event.wait(), timeout)
            else:
                await self._event.wait()
        except asyncio.TimeoutError:
            pass
        return self.result
        
    def cancel(self) -> None:
        """Cancels the subagent."""
        self.status = "cancelled"
        self.completed_at = time.time()
        self._event.set()


class SubagentManager:
    """Manages spawning and lifecycle of subagents."""
    
    def __init__(self, agent_registry: dict[str, Any], tool_registry: Any, max_concurrent: int = 8):
        """
        Initializes the SubagentManager.
        
        Args:
            agent_registry: Registry mapping agent_type to agent classes.
            tool_registry: Registry of tools to provide to agents.
            max_concurrent: Maximum number of concurrent subagents.
        """
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active: dict[str, SubagentHandle] = {}
        self._completed: list[SubagentHandle] = {}
        self._completed_list: list[SubagentHandle] = []
        self._tasks: dict[str, asyncio.Task] = {}
        
    async def spawn(self, parent_role: str, agent_type: str, task: str, context: Any, engine: Any) -> SubagentHandle:
        """Spawns a single subagent."""
        handle = SubagentHandle(
            parent_role=parent_role,
            agent_type=agent_type,
            task=task
        )
        self._active[handle.id] = handle
        
        agent_cls = self.agent_registry.get(agent_type)
        if not agent_cls:
            handle.status = "failed"
            handle.completed_at = time.time()
            handle._event.set()
            raise ValueError(f"Agent type {agent_type} not found in registry.")
            
        child_context = context.clone() if hasattr(context, "clone") else context
        agent_instance = agent_cls(tools=self.tool_registry, context=child_context)
        
        task_obj = asyncio.create_task(self._run_subagent(handle, agent_instance, child_context, engine))
        self._tasks[handle.id] = task_obj
        
        return handle
        
    async def spawn_batch(self, parent_role: str, tasks: list[tuple[str, str]], context: Any, engine: Any) -> list[SubagentHandle]:
        """Spawns multiple subagents."""
        handles = []
        for agent_type, task in tasks:
            handle = await self.spawn(parent_role, agent_type, task, context, engine)
            handles.append(handle)
        return handles
        
    async def gather_results(self, handles: list[SubagentHandle], timeout: float = 300.0) -> list[Any | None]:
        """Waits for all handles to complete with timeout."""
        results = []
        start_time = time.time()
        for handle in handles:
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                results.append(None)
                continue
            res = await handle.wait(timeout=remaining)
            results.append(res)
        return results
        
    async def _run_subagent(self, handle: SubagentHandle, agent: Any, context: Any, engine: Any) -> None:
        """Runs the subagent."""
        async with self._semaphore:
            if handle.status == "cancelled":
                return
            handle.status = "running"
            try:
                if hasattr(agent, "run") and asyncio.iscoroutinefunction(agent.run):
                    result = await agent.run(handle.task, engine)
                else:
                    result = await agent(handle.task)
                handle.result = result
                handle.status = "completed"
            except Exception as e:
                logger.error(f"Subagent {handle.id} failed: {e}", exc_info=True)
                handle.status = "failed"
            finally:
                handle.completed_at = time.time()
                handle._event.set()
                if handle.id in self._active:
                    self._completed_list.append(self._active.pop(handle.id))
                    
    def get_active(self) -> list[SubagentHandle]:
        """Returns list of active subagents."""
        return list(self._active.values())
        
    def get_stats(self) -> dict:
        """Returns stats about subagents."""
        total_spawned = len(self._active) + len(self._completed_list)
        failed_count = sum(1 for h in self._completed_list if h.status == "failed")
        return {
            "total_spawned": total_spawned,
            "active_count": len(self._active),
            "completed_count": len(self._completed_list) - failed_count,
            "failed_count": failed_count
        }
        
    def cancel_all(self) -> int:
        """Cancels all active subagents."""
        count = 0
        for handle_id, handle in list(self._active.items()):
            handle.cancel()
            task = self._tasks.get(handle_id)
            if task:
                task.cancel()
            self._completed_list.append(self._active.pop(handle_id))
            count += 1
        return count
