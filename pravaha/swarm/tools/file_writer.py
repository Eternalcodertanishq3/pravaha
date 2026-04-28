"""File Writer Tool — Write files to disk with safety constraints.

Only writes to whitelisted directories and allowed extensions.
Agents can create, append, or overwrite files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class FileWriter:
    """Write content to a file with directory/extension whitelisting."""

    name = "write_file"
    description = "Write content to a file (whitelisted paths and extensions)"
    arg_schema = '{"path": "string", "content": "string", "mode": "w|a"}'

    ALLOWED_DIRS = ["output/", "data/agents/", "data/", "/tmp/pravaha/", "temp/"]
    ALLOWED_EXTENSIONS = {
        ".py", ".txt", ".md", ".json", ".yaml", ".yml",
        ".csv", ".html", ".css", ".js", ".ts", ".toml",
        ".xml", ".sql", ".sh", ".bat", ".log", ".cfg",
    }

    def execute(
        self,
        path: str,
        content: str,
        mode: str = "w",
    ) -> dict[str, Any]:
        """Write content to a file."""
        if not path:
            return {"error": "Path is required", "success": False}

        if mode not in ("w", "a"):
            return {"error": f"Invalid mode '{mode}'. Use 'w' or 'a'.", "success": False}

        # Check extension
        ext = Path(path).suffix.lower()
        if ext and ext not in self.ALLOWED_EXTENSIONS:
            return {
                "error": f"Extension '{ext}' not allowed. "
                         f"Allowed: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}",
                "success": False,
            }

        # Check directory
        norm_path = path.replace("\\", "/")
        allowed = any(norm_path.startswith(d) or f"/{d}" in norm_path for d in self.ALLOWED_DIRS)
        if not allowed:
            return {
                "error": f"Path '{path}' is not in an allowed directory. "
                         f"Allowed prefixes: {', '.join(self.ALLOWED_DIRS)}",
                "success": False,
            }

        try:
            # Create parent directories
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            with open(target, mode, encoding="utf-8") as f:
                f.write(content)

            return {
                "path": str(target.resolve()),
                "bytes_written": len(content.encode("utf-8")),
                "mode": "append" if mode == "a" else "write",
                "success": True,
            }
        except PermissionError:
            return {"error": f"Permission denied: {path}", "success": False}
        except Exception as e:
            return {"error": f"Write failed: {e}", "success": False}
