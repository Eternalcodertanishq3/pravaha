from __future__ import annotations

import asyncio
import unittest
from pravaha.swarm.subagent_manager import SubagentManager

class MockAgent:
    def __init__(self, tools, context):
        self.tools = tools
        self.context = context
        
    async def run(self, task, engine):
        await asyncio.sleep(0.1)
        return f"result for {task}"

class MockContext:
    def clone(self):
        return MockContext()

class TestSubagentManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = {"mock": MockAgent}
        self.manager = SubagentManager(self.registry, {}, max_concurrent=2)
        self.context = MockContext()
        self.engine = {}
        
    async def test_spawn_and_gather(self):
        handle1 = await self.manager.spawn("parent", "mock", "task1", self.context, self.engine)
        handle2 = await self.manager.spawn("parent", "mock", "task2", self.context, self.engine)
        
        self.assertEqual(len(self.manager.get_active()), 2)
        
        results = await self.manager.gather_results([handle1, handle2])
        self.assertEqual(results, ["result for task1", "result for task2"])
        
        stats = self.manager.get_stats()
        self.assertEqual(stats["completed_count"], 2)
        self.assertEqual(stats["active_count"], 0)

    async def test_batch_spawn(self):
        tasks = [("mock", f"task{i}") for i in range(3)]
        handles = await self.manager.spawn_batch("parent", tasks, self.context, self.engine)
        self.assertEqual(len(handles), 3)
        
        results = await self.manager.gather_results(handles)
        self.assertEqual(len(results), 3)

    async def test_cancel_all(self):
        await self.manager.spawn("parent", "mock", "task1", self.context, self.engine)
        await self.manager.spawn("parent", "mock", "task2", self.context, self.engine)
        
        count = self.manager.cancel_all()
        self.assertEqual(count, 2)
        self.assertEqual(len(self.manager.get_active()), 0)

if __name__ == '__main__':
    unittest.main()
