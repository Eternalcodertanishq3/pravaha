"""Bash Tool — Execute whitelisted bash/shell commands.

Extends ShellRunner with: pipe operator support (cmd1 | cmd2),
environment variable injection, working directory setting.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

# Whitelisted commands (safe subset)
ALLOWED_COMMANDS = {
    "ls", "dir", "cat", "head", "tail", "grep", "find", "wc",
    "sort", "uniq", "echo", "pwd", "env", "date", "whoami",
    "python", "python3", "pip", "pip3", "git", "curl", "wget",
    "tree", "file", "which", "type", "diff", "md5sum", "sha256sum",
}


class BashTool:
    """Execute whitelisted bash commands with pipeline support."""

    name = "bash"
    description = "Execute shell commands (whitelisted, with pipe support)"
    arg_schema = '{"command": "string", "cwd": "optional_path", "env": {}}'

    def execute(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int = 15,
    ) -> dict[str, Any]:
        """Execute a shell command string.

        Supports pipe operators (cmd1 | cmd2).
        """
        # Validate command against whitelist
        parts = command.strip().split()
        if not parts:
            return {"error": "Empty command", "success": False}

        # Check first command in pipe chain
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
            result = subprocess.run(
                command,
                shell=True,
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
