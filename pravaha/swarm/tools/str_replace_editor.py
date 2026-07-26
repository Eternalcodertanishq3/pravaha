"""StrReplaceEditorTool for token-efficient file editing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class StrReplaceEditorTool:
    """Token-efficient file editing via targeted string replacement."""

    name = "str_replace_editor"
    description = "Token-efficient file editing via targeted string replacement"
    arg_schema = {
        "command": "str",
        "path": "str",
        "old_str": "str (for str_replace)",
        "new_str": "str (for str_replace)",
        "content": "str (for create)",
        "insert_line": "int (for insert)",
        "text": "str (for insert)",
        "start_line": "int (optional for view)",
        "end_line": "int (optional for view)",
    }

    def __init__(self, workspace_dir: str | Path | None = None) -> None:
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else Path.cwd()

    def _validate_path(self, file_path: str) -> Path:
        if ".." in file_path:
            raise ValueError("Path traversal is not allowed.")
        p = Path(file_path)
        if not p.is_absolute():
            p = self.workspace_dir / p
        p = p.resolve()
        if not str(p).startswith(str(self.workspace_dir)):
            raise ValueError("Path is outside the workspace.")
        return p

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool."""
        command = kwargs.get("command")
        path_str = kwargs.get("path")

        if not command or not path_str:
            return {"success": False, "output": "command and path are required."}

        try:
            target_path = self._validate_path(str(path_str))
        except ValueError as e:
            return {"success": False, "output": str(e)}

        if command == "view":
            if not target_path.exists():
                return {"success": False, "output": "File not found."}
            start_line = kwargs.get("start_line")
            end_line = kwargs.get("end_line")
            try:
                lines = target_path.read_text(encoding="utf-8").splitlines()
                start = max(0, int(start_line) - 1) if start_line is not None else 0
                end = int(end_line) if end_line is not None else len(lines)
                end = min(end, start + 300)

                output_lines = []
                for i in range(start, end):
                    if i < len(lines):
                        output_lines.append(f"{i+1}: {lines[i]}")
                return {"success": True, "output": "\n".join(output_lines)}
            except Exception as e:
                return {"success": False, "output": f"Failed to view file: {e}"}

        elif command == "str_replace":
            old_str = kwargs.get("old_str")
            new_str = kwargs.get("new_str", "")
            if not target_path.exists():
                return {"success": False, "output": "File not found."}
            if old_str is None:
                return {"success": False, "output": "old_str is required for str_replace."}

            try:
                content = target_path.read_text(encoding="utf-8")
                count = content.count(old_str)
                if count == 0:
                    return {"success": False, "output": "old_str not found in file."}
                elif count > 1:
                    return {"success": False, "output": f"old_str found {count} times. Must be unique."}

                # find line number
                lines = content.splitlines()
                line_num = -1
                for i, line in enumerate(lines):
                    if old_str in line:
                        line_num = i + 1
                        break

                new_content = content.replace(old_str, str(new_str))
                target_path.write_text(new_content, encoding="utf-8")
                return {"success": True, "output": f"Replaced successfully on line {line_num}."}
            except Exception as e:
                return {"success": False, "output": f"Failed to replace: {e}"}

        elif command == "create":
            if target_path.exists() and not kwargs.get("overwrite", False):
                return {"success": False, "output": "File already exists. Use overwrite=True to overwrite."}
            content = kwargs.get("content", "")
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(str(content), encoding="utf-8")
                return {"success": True, "output": f"File created at {target_path.name}."}
            except Exception as e:
                return {"success": False, "output": f"Failed to create file: {e}"}

        elif command == "insert":
            insert_line = kwargs.get("insert_line")
            text = kwargs.get("text", "")
            if not target_path.exists():
                return {"success": False, "output": "File not found."}
            if insert_line is None:
                return {"success": False, "output": "insert_line is required for insert."}

            try:
                lines = target_path.read_text(encoding="utf-8").splitlines()
                idx = int(insert_line)

                # line 0 means beginning
                lines.insert(idx, str(text))
                target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return {"success": True, "output": f"Inserted at line {idx}."}
            except Exception as e:
                return {"success": False, "output": f"Failed to insert: {e}"}

        return {"success": False, "output": f"Unknown command: {command}"}
