"""Tool Registry — Central registry for all agent tools."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all available tools.

    Tools are registered by name and can be executed asynchronously.
    Each tool must have: name, description, arg_schema, execute(**args).
    """

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name}")

    def get_available(self, names: list[str]) -> list:
        """Get tool instances by name list."""
        return [self._tools[n] for n in names if n in self._tools]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    async def execute(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name with given args.

        Returns JSON string of the result.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return json.dumps({"error": f"Tool '{tool_name}' not available", "success": False})

        try:
            result = tool.execute(**args)
            return json.dumps(result, default=str)
        except TypeError as e:
            return json.dumps({"error": f"Invalid args for {tool_name}: {e}", "success": False})
        except Exception as e:
            return json.dumps({"error": f"Tool error: {e}", "success": False})

    @classmethod
    def default(cls) -> ToolRegistry:
        """Create registry with all standard tools pre-registered."""
        from pravaha.swarm.tools.code_executor import CodeExecutor
        from pravaha.swarm.tools.file_reader import FileReader
        from pravaha.swarm.tools.search_tool import SearchTool
        from pravaha.swarm.tools.shell_runner import ShellRunner
        from pravaha.swarm.tools.web_fetcher import WebFetcher

        registry = cls()
        registry.register(CodeExecutor())
        registry.register(FileReader())
        registry.register(WebFetcher())
        registry.register(SearchTool())
        registry.register(ShellRunner())
        return registry
