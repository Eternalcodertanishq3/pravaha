"""Tool Registry — Central registry for all 12 agent tools.

v3.3: Expanded from 5 to 12 tools:
- CodeExecutor, FileReader, WebFetcher, SearchTool, ShellRunner (original 5)
- PythonRepl, BashTool, JsonTool, HttpClient, FileWriter, GitTool, Calculator (new 7)
"""

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

    def get_tool(self, name: str) -> Any | None:
        """Get a single tool by name."""
        return self._tools.get(name)

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
        """Create registry with all 12 standard tools pre-registered."""
        # Original 5 tools
        # v3.3: 7 new tools
        from pravaha.swarm.tools.bash_tool import BashTool
        from pravaha.swarm.tools.calculator import Calculator
        from pravaha.swarm.tools.code_executor import CodeExecutor
        from pravaha.swarm.tools.file_reader import FileReader
        from pravaha.swarm.tools.file_writer import FileWriter
        from pravaha.swarm.tools.git_tool import GitTool
        from pravaha.swarm.tools.http_client import HttpClient
        from pravaha.swarm.tools.json_tool import JsonTool
        from pravaha.swarm.tools.python_repl import PythonRepl
        from pravaha.swarm.tools.search_tool import SearchTool
        from pravaha.swarm.tools.shell_runner import ShellRunner
        from pravaha.swarm.tools.web_fetcher import WebFetcher

        registry = cls()

        # Original tools
        registry.register(CodeExecutor())
        registry.register(FileReader())
        registry.register(WebFetcher())
        registry.register(SearchTool())
        registry.register(ShellRunner())

        # v3.3 tools
        registry.register(PythonRepl())
        registry.register(BashTool())
        registry.register(JsonTool())
        registry.register(HttpClient())
        registry.register(FileWriter())
        registry.register(GitTool())
        registry.register(Calculator())

        # v4.0: Dynamic Custom Tools Auto-Discovery
        try:
            import importlib
            import inspect
            import pkgutil

            from pravaha.swarm.tools import custom

            for _, module_name, _ in pkgutil.iter_modules(custom.__path__):
                try:
                    module = importlib.import_module(f"pravaha.swarm.tools.custom.{module_name}")
                    for obj_name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            hasattr(obj, "execute")
                            and hasattr(obj, "name")
                            and not obj.__name__.startswith("Base")
                            and obj.__module__ == module.__name__
                        ):
                            tool_instance = obj()
                            registry.register(tool_instance)
                            logger.info(f"Dynamically loaded custom tool: {tool_instance.name}")
                except Exception as e:
                    logger.error(f"Failed to load custom tool {module_name}: {e}")
        except Exception as e:
            logger.warning(f"Could not load custom tools: {e}")

        logger.info(f"ToolRegistry: {len(registry._tools)} tools registered")
        return registry
