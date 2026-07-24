"""Shell Runner — Safe shell command execution with whitelist."""

from __future__ import annotations

import logging
import platform
import shlex
import subprocess

logger = logging.getLogger(__name__)

METACHARACTERS = {"|", ";", "&&", "||", "$", "`", ">>", ">", "<"}

# Windows shell builtins that need cmd /c to execute
_WINDOWS_BUILTINS = {"echo", "dir", "date", "type", "set", "cls", "copy", "del", "ren", "mkdir", "rmdir"}

class ShellRunner:
    """Run whitelisted shell commands safely."""

    name = "run_shell"
    description = "Run a whitelisted shell command and return output"
    arg_schema = '{"command": "string", "timeout_s": 5}'

    MAX_OUTPUT_BYTES = 8192

    # Only these commands are allowed
    ALLOWED_COMMANDS = {
        "ls", "dir", "cat", "head", "tail", "wc", "find", "grep",
        "echo", "pwd", "whoami", "date", "uname", "which",
        "pip", "python", "node", "npm", "git", "ruff", "mypy",
        "pytest", "cargo", "rustc",
    }

    def execute(self, command: str, timeout_s: int = 5) -> dict:
        """Execute a whitelisted shell command."""
        timeout_s = min(timeout_s, 10)

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

        base_cmd = parts[0].lower()
        if base_cmd not in self.ALLOWED_COMMANDS:
            return {
                "error": f"Command not allowed: {base_cmd}. "
                f"Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS))}",
                "success": False,
            }

        # Block dangerous flags
        dangerous_flags = {"--rm", "-rf", "--force", "--delete"}
        if any(f in parts for f in dangerous_flags):
            return {"error": "Dangerous flags detected", "success": False}

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
            )
            return {
                "stdout": result.stdout[: self.MAX_OUTPUT_BYTES],
                "stderr": result.stderr[: self.MAX_OUTPUT_BYTES],
                "exit_code": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "error": f"Command timed out ({timeout_s}s)",
                "success": False,
            }
        except Exception as e:
            return {"error": str(e), "success": False}
