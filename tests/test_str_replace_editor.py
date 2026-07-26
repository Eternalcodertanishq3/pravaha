"""Tests for StrReplaceEditorTool."""

import os
import pytest
from pathlib import Path
from pravaha.swarm.tools.str_replace_editor import StrReplaceEditorTool

@pytest.fixture
def temp_workspace(tmp_path):
    return tmp_path

def test_create_and_view(temp_workspace):
    tool = StrReplaceEditorTool(workspace_dir=temp_workspace)
    
    # test create
    res = tool.execute(command="create", path="test.txt", content="hello\nworld")
    assert res["success"] is True
    assert (temp_workspace / "test.txt").exists()
    
    # test view
    res = tool.execute(command="view", path="test.txt")
    assert res["success"] is True
    assert "1: hello" in res["output"]
    assert "2: world" in res["output"]

def test_str_replace_unique(temp_workspace):
    tool = StrReplaceEditorTool(workspace_dir=temp_workspace)
    tool.execute(command="create", path="test.txt", content="apple\nbanana\ncherry")
    
    res = tool.execute(command="str_replace", path="test.txt", old_str="banana", new_str="orange")
    assert res["success"] is True
    assert "Line 2" in res["output"] or "line 2" in res["output"]
    
    content = (temp_workspace / "test.txt").read_text()
    assert "orange" in content
    assert "banana" not in content

def test_str_replace_errors(temp_workspace):
    tool = StrReplaceEditorTool(workspace_dir=temp_workspace)
    tool.execute(command="create", path="test.txt", content="apple\napple\ncherry")
    
    # Missing match
    res = tool.execute(command="str_replace", path="test.txt", old_str="banana", new_str="orange")
    assert res["success"] is False
    assert "not found" in res["output"]
    
    # Multiple match
    res = tool.execute(command="str_replace", path="test.txt", old_str="apple", new_str="orange")
    assert res["success"] is False
    assert "Must be unique" in res["output"]

def test_insert(temp_workspace):
    tool = StrReplaceEditorTool(workspace_dir=temp_workspace)
    tool.execute(command="create", path="test.txt", content="line1\nline2")
    
    res = tool.execute(command="insert", path="test.txt", insert_line=1, text="inserted")
    assert res["success"] is True
    
    content = (temp_workspace / "test.txt").read_text()
    assert content.splitlines() == ["line1", "inserted", "line2"]

def test_path_traversal(temp_workspace):
    tool = StrReplaceEditorTool(workspace_dir=temp_workspace)
    res = tool.execute(command="view", path="../outside.txt")
    assert res["success"] is False
    assert "traversal is not allowed" in res["output"]

def test_absolute_path_outside(temp_workspace):
    tool = StrReplaceEditorTool(workspace_dir=temp_workspace)
    res = tool.execute(command="view", path="/tmp/secret.txt")
    if os.name == 'posix':
        assert res["success"] is False
        assert "outside the workspace" in res["output"]
