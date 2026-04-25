"""Code Executor — Sandboxed subprocess Python execution.

Executes Python code in an isolated subprocess with strict resource
limits. An autonomous coder agent will eventually write `while True`,
so this sandbox ruthlessly kills it after timeout.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeExecutor:
    """Execute Python code in a sandboxed subprocess.

    Returns stdout, stderr, and exit code. This is what makes
    CoderAgent and DebuggerAgent actually verify their outputs
    rather than just claiming they work.

    Security:
    - No shell=True
    - Minimal env (PATH only)
    - Strict timeout (default 5s, ruthless kill)
    - Output capped at MAX_OUTPUT_BYTES
    - On Linux: resource.setrlimit via preexec_fn
    """

    name = "execute_python"
    description = "Execute Python code and return stdout/stderr/exitcode"
    arg_schema = '{"code": "string", "timeout_s": 5}'

    MAX_OUTPUT_BYTES = 8192  # 8KB max output
    DEFAULT_TIMEOUT = 5  # 5-second ruthless kill

    # Max memory for child process (256MB)
    MAX_MEMORY_BYTES = 256 * 1024 * 1024

    def _get_safe_env(self) -> dict[str, str]:
        """Minimal environment for subprocess."""
        env: dict[str, str] = {}
        if platform.system() == "Windows":
            # Windows needs SystemRoot and some PATH entries
            env["PATH"] = os.environ.get("PATH", "")
            env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
            env["TEMP"] = os.environ.get("TEMP", tempfile.gettempdir())
            env["TMP"] = os.environ.get("TMP", tempfile.gettempdir())
        else:
            env["PATH"] = "/usr/bin:/bin:/usr/local/bin"
            env["HOME"] = "/tmp"
        return env

    def _get_preexec_fn(self):
        """Return a preexec function that sets resource limits (Linux only)."""
        if platform.system() != "Linux":
            return None

        def _set_limits():
            import resource

            # CPU time limit (seconds)
            resource.setrlimit(
                resource.RLIMIT_CPU, (self.DEFAULT_TIMEOUT, self.DEFAULT_TIMEOUT)
            )
            # Memory limit
            resource.setrlimit(
                resource.RLIMIT_AS, (self.MAX_MEMORY_BYTES, self.MAX_MEMORY_BYTES)
            )
            # No core dumps
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            # Max file size (1MB)
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024)
            )

        return _set_limits

    def execute(self, code: str, timeout_s: int = 5) -> dict:
        """Execute Python code and return results.

        Args:
            code: Python source code to execute.
            timeout_s: Max seconds before ruthless kill (default 5).

        Returns:
            Dict with stdout, stderr, exit_code, success keys.
        """
        timeout_s = min(timeout_s, 10)  # Hard cap at 10s

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=self._get_safe_env(),
                preexec_fn=self._get_preexec_fn(),
                # No shell=True — ever.
            )
            return {
                "stdout": result.stdout[: self.MAX_OUTPUT_BYTES],
                "stderr": result.stderr[: self.MAX_OUTPUT_BYTES],
                "exit_code": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"KILLED: Execution timed out ({timeout_s}s). "
                "Possible infinite loop detected.",
                "exit_code": -9,
                "success": False,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "success": False,
            }
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
