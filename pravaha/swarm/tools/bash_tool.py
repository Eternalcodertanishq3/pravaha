"""Bash Tool — Execute whitelisted bash/shell commands.

Extends ShellRunner with: pipe operator support (cmd1 | cmd2),
environment variable injection, working directory setting.
"""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
from typing import Any

# Whitelisted commands (safe subset)
ALLOWED_COMMANDS = {
    "ls", "dir", "cat", "head", "tail", "grep", "find", "wc",
    "sort", "uniq", "echo", "pwd", "env", "date", "whoami",
    "python", "python3", "pip", "pip3", "git", "curl", "wget",
    "tree", "file", "which", "type", "diff", "md5sum", "sha256sum",
}

METACHARACTERS = {"|", ";", "&&", "||", "$", "`", ">>", ">", "<"}

# Windows shell builtins that need cmd /c to execute
_WINDOWS_BUILTINS = {"echo", "dir", "date", "type", "set", "cls", "copy", "del", "ren", "mkdir", "rmdir"}

class BashTool:
    """Execute whitelisted bash commands."""

    name = "bash"
    description = "Execute shell commands (whitelisted, no pipe support)"
    arg_schema = '{"command": "string", "cwd": "optional_path", "env": {}}'

    def execute(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int = 15,
    ) -> dict[str, Any]:
        """Execute a shell command string.

        Supports environment variables and working directory.
        """
        # Reject shell metacharacters
        for meta in METACHARACTERS:
            if meta in command:
                return {"error": f"Shell metacharacter '{meta}' is not allowed", "success": False}

        try:
            parts = shlex.split(command)
        except ValueError as e:
            return {"error": f"Command parsing error: {e}", "success": False}

        if not parts:
            return {"error": "Empty command", "success": False}

        base_cmd = parts[0].split("/")[-1]  # Handle absolute paths
        if base_cmd not in ALLOWED_COMMANDS:
            return {
                "error": f"Command '{base_cmd}' not in whitelist. "
                         f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}",
                "success": False,
            }

        # Build environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Resolve working directory
        work_dir = cwd or os.getcwd()
        if not os.path.isdir(work_dir):
            return {"error": f"Working directory not found: {work_dir}", "success": False}

        try:
            # On Windows, shell builtins (echo, dir, etc.) require cmd /c
            run_parts = parts
            if platform.system() == "Windows" and base_cmd in _WINDOWS_BUILTINS:
                run_parts = ["cmd", "/c"] + parts

            result = subprocess.run(
                run_parts,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=work_dir,
                env=run_env,
            )
            return {
                "stdout": result.stdout[:8192],
                "stderr": result.stderr[:4096],
                "exit_code": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout_s}s", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
