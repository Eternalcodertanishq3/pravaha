from typing import Any

from pravaha.swarm.agents.base_agent import BaseAgent


class ToolMakerAgent(BaseAgent):
    """An agent that writes, tests, and persists custom tools for the swarm."""

    @property
    def role(self) -> str:
        return "Internal Swarm Tool Engineer"

    @property
    def tools(self) -> list[str]:
        return ["file_reader", "file_writer", "python_repl", "shell_runner", "bash_tool"]

    def can_handle(self, task: str) -> bool:
        keywords = ["create a tool", "write a tool", "new capability", "build a tool", "extend the swarm"]
        return any(k in task.lower() for k in keywords)

    def _get_system_prompt(self) -> str:
        return """You are the Tool Maker Agent for the Pravāha Swarm.
Your job is to literally expand the swarm's capabilities by writing new Python tools.

When requested to create a new tool:
1. Write a Python class.
2. It must have:
   - `name` (string)
   - `description` (string)
   - `arg_schema` (dict mapping arg names to type descriptions)
   - `execute(self, **kwargs)` method returning a dict or string.
3. Save the working tool code to `pravaha/swarm/tools/custom/<tool_name>.py` using the `file_writer` tool.

Example template:
```python
class MyCustomTool:
    name = "my_custom_tool"
    description = "Does something awesome"
    arg_schema = {"query": "string to query"}

    def execute(self, query: str):
        return {"result": f"Executed with {query}"}
```

The Pravāha framework will automatically discover and load your tool on the next initialization!
Do not write abstract classes. Ensure your code is flawless.
"""
