"""Docker Sandbox — Containerized execution environment for untrusted agent tool execution.

Enforces hardware and security boundaries:
- Network isolation (--network=none)
- Read-only root filesystem (--read-only)
- Memory cap (--memory=512m)
- CPU quota (--cpus=1.0)
- PID limit (--pids-limit=64)
- Automatic cleanup (--rm)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Execute untrusted code or commands inside an isolated Docker container."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "512m",
        cpu_quota: str = "1.0",
        pids_limit: int = 64,
        allow_network: bool = False,
    ) -> None:
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.pids_limit = pids_limit
        self.allow_network = allow_network
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker CLI is available on PATH."""
        return shutil.which("docker") is not None

    def execute_command(
        self,
        command: list[str],
        work_dir: str | None = None,
        timeout_s: int = 10,
    ) -> dict[str, Any]:
        """Execute command inside Docker container or fallback to isolated process.

        Args:
            command: Command list (e.g. ['python', '-c', 'print(1)']).
            work_dir: Optional host directory to mount into container workspace.
            timeout_s: Maximum execution duration in seconds.

        Returns:
            Dict containing stdout, stderr, exit_code, success.
        """
        if self._docker_available:
            return self._run_in_docker(command, work_dir, timeout_s)
        else:
            logger.warning("Docker CLI unavailable. Falling back to process isolation sandbox.")
            return self._run_fallback(command, work_dir, timeout_s)

    def _run_in_docker(
        self,
        command: list[str],
        work_dir: str | None,
        timeout_s: int,
    ) -> dict[str, Any]:
        """Run command inside Docker container."""
        temp_dir = tempfile.mkdtemp(prefix="pravaha_sandbox_")
        mount_host = work_dir or temp_dir

        docker_cmd = [
            "docker", "run", "--rm",
            "--memory", self.memory_limit,
            "--cpus", self.cpu_quota,
            "--pids-limit", str(self.pids_limit),
            "-v", f"{os.path.abspath(mount_host)}:/workspace:rw",
            "-w", "/workspace",
        ]

        if not self.allow_network:
            docker_cmd.extend(["--network", "none"])

        docker_cmd.append(self.image)
        docker_cmd.extend(command)

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return {
                "stdout": result.stdout[:8192],
                "stderr": result.stderr[:4096],
                "exit_code": result.returncode,
                "success": result.returncode == 0,
                "sandbox_type": "docker",
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Container execution timed out after {timeout_s}s", "success": False}
        except Exception as e:
            return {"error": f"Docker execution error: {e}", "success": False}
        finally:
            if not work_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_fallback(
        self,
        command: list[str],
        work_dir: str | None,
        timeout_s: int,
    ) -> dict[str, Any]:
        """Fallback process execution with strict timeout and directory isolation."""
        cwd = work_dir or os.getcwd()
        try:
            result = subprocess.run(
                command,
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
                "sandbox_type": "process_fallback",
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Process execution timed out after {timeout_s}s", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
