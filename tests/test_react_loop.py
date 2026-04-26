"""Tests for the ReAct loop and tool execution."""

from __future__ import annotations

import pytest

from pravaha.swarm.agents.base_agent import BaseAgent, ReActStep, SharedContext, ToolCall


class TestToolCall:
    def test_parse_tool_call(self) -> None:
        result = BaseAgent._parse_tool_call_legacy('execute_python({"code": "print(1)"})')
        assert result is not None
        assert result.tool_name == "execute_python"
        assert result.args == {"code": "print(1)"}

    def test_parse_invalid(self) -> None:
        result = BaseAgent._parse_tool_call_legacy("just some text")
        assert result is None

    def test_parse_malformed_json(self) -> None:
        result = BaseAgent._parse_tool_call_legacy("tool(not json)")
        assert result is not None
        assert result.tool_name == "tool"
        assert "raw" in result.args


class TestReActStep:
    def test_default_values(self) -> None:
        step = ReActStep()
        assert step.thought == ""
        assert step.action is None
        assert step.observation == ""
        assert step.is_final_answer is False

    def test_final_answer(self) -> None:
        step = ReActStep(is_final_answer=True, answer="42")
        assert step.is_final_answer
        assert step.answer == "42"


class TestCodeExecutor:
    def test_simple_execution(self) -> None:
        from pravaha.swarm.tools.code_executor import CodeExecutor
        executor = CodeExecutor()
        result = executor.execute("print('hello')", timeout_s=5)
        assert result["success"] is True
        assert "hello" in result["stdout"]

    def test_timeout_kills(self) -> None:
        from pravaha.swarm.tools.code_executor import CodeExecutor
        executor = CodeExecutor()
        result = executor.execute("while True: pass", timeout_s=2)
        assert result["success"] is False
        assert "KILLED" in result["stderr"] or result["exit_code"] != 0

    def test_syntax_error(self) -> None:
        from pravaha.swarm.tools.code_executor import CodeExecutor
        executor = CodeExecutor()
        result = executor.execute("def (broken", timeout_s=5)
        assert result["success"] is False


class TestFileReader:
    def test_read_existing_file(self) -> None:
        from pravaha.swarm.tools.file_reader import FileReader
        reader = FileReader()
        result = reader.execute("pyproject.toml")
        assert result["success"] is True
        assert "content" in result

    def test_nonexistent_file(self) -> None:
        from pravaha.swarm.tools.file_reader import FileReader
        reader = FileReader()
        result = reader.execute("nonexistent_file_xyz.py")
        assert result["success"] is False

    def test_blocked_extension(self) -> None:
        from pravaha.swarm.tools.file_reader import FileReader
        reader = FileReader()
        result = reader.execute("binary.exe")
        assert result["success"] is False


class TestShellRunner:
    def test_allowed_command(self) -> None:
        from pravaha.swarm.tools.shell_runner import ShellRunner
        runner = ShellRunner()
        result = runner.execute("echo hello", timeout_s=5)
        assert result["success"] is True

    def test_blocked_command(self) -> None:
        from pravaha.swarm.tools.shell_runner import ShellRunner
        runner = ShellRunner()
        result = runner.execute("rm -rf /", timeout_s=5)
        assert result["success"] is False


class TestToolRegistry:
    def test_default_registry(self) -> None:
        from pravaha.swarm.tools import ToolRegistry
        registry = ToolRegistry.default()
        names = registry.list_tools()
        assert "execute_python" in names
        assert "read_file" in names
        assert "web_search" in names

    @pytest.mark.asyncio
    async def test_execute_tool(self) -> None:
        from pravaha.swarm.tools import ToolRegistry
        registry = ToolRegistry.default()
        result = await registry.execute("execute_python", {"code": "print(42)"})
        assert "42" in result
