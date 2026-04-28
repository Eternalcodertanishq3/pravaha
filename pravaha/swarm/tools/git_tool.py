"""Git Tool — Safe git operations for agents.

Supports: status, diff, log, add, commit (no push/pull for safety).
"""

from __future__ import annotations

import subprocess
from typing import Any


class GitTool:
    """Safe git operations: status/diff/log/add/commit."""

    name = "git"
    description = "Safe git operations: status/diff/log/add/commit (no push/pull)"
    arg_schema = '{"command": "status|diff|log|add|commit", "args": "optional extra args"}'

    # Commands that are safe for agents to run
    SAFE_COMMANDS = {"status", "diff", "log", "add", "commit", "branch", "show", "stash"}
    BLOCKED_COMMANDS = {"push", "pull", "fetch", "clone", "remote", "rebase", "reset", "force"}

    def execute(
        self,
        command: str,
        args: str = "",
        cwd: str | None = None,
        timeout_s: int = 15,
    ) -> dict[str, Any]:
        """Execute a git command."""
        command = command.strip().lower()

        if command in self.BLOCKED_COMMANDS:
            return {
                "error": f"Command 'git {command}' is blocked for safety. "
                         f"Safe commands: {', '.join(sorted(self.SAFE_COMMANDS))}",
                "success": False,
            }

        if command not in self.SAFE_COMMANDS:
            return {
                "error": f"Unknown command '{command}'. "
                         f"Allowed: {', '.join(sorted(self.SAFE_COMMANDS))}",
                "success": False,
            }

        # Build full command
        cmd_parts = ["git", command]
        if args:
            cmd_parts.extend(args.split())

        # Special handling for log (limit output)
        if command == "log" and "-n" not in args and "--oneline" not in args:
            cmd_parts.extend(["-n", "20", "--oneline"])

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=cwd,
            )
            return {
                "stdout": result.stdout[:8192],
                "stderr": result.stderr[:4096],
                "exit_code": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"git {command} timed out after {timeout_s}s", "success": False}
        except FileNotFoundError:
            return {"error": "git is not installed or not in PATH", "success": False}
        except Exception as e:
            return {"error": f"git error: {e}", "success": False}
