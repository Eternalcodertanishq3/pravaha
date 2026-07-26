"""Tests for PTYTerminalTool."""

import sys
import pytest
from pravaha.swarm.tools.pty_terminal import PTYTerminalTool

@pytest.fixture
def terminal():
    tool = PTYTerminalTool()
    yield tool
    if tool.process:
        tool.process.kill()

def test_basic_execution(terminal):
    cmd = "echo hello" if sys.platform == "win32" else "echo hello"
    res = terminal.execute(command="execute", cmd_args=cmd)
    assert res["success"] is True
    assert "hello" in res["output"].lower()

def test_ansi_stripping():
    tool = PTYTerminalTool()
    raw = "\x1b[31mError\x1b[0m"
    stripped = tool.ANSI_ESCAPE.sub('', raw)
    assert stripped == "Error"
    if tool.process:
        tool.process.kill()

def test_command_history(terminal):
    terminal.execute(command="execute", cmd_args="echo 1")
    terminal.execute(command="execute", cmd_args="echo 2")
    
    res = terminal.execute(command="get_history")
    assert res["success"] is True
    assert "echo 1" in res["output"]
    assert "echo 2" in res["output"]

def test_interactive_prompt_detection():
    # Since we can't easily mock the subprocess interactively here, we can test the regexes
    tool = PTYTerminalTool()
    assert any(p.search("Do you want to continue? [Y/n]") for p in tool.PROMPT_PATTERNS)
    assert any(p.search("Enter password: ") for p in tool.PROMPT_PATTERNS)
    assert any(p.search("Are you sure? (yes/no)") for p in tool.PROMPT_PATTERNS)
    if tool.process:
        tool.process.kill()
