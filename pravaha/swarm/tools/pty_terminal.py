"""PTY Terminal Tool for persistent interactive shell sessions."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


class PTYTerminalTool:
    """Persistent interactive terminal with environment state preservation."""

    name = "pty_terminal"
    description = "Persistent interactive terminal with environment state preservation"
    arg_schema = {
        "command": "str (execute, get_history, reset, get_cwd)",
        "cmd_args": "str (shell command for 'execute')",
        "timeout": "float (optional timeout for execute, default 10.0)"
    }

    # Regex to strip ANSI color codes
    ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\([B0]|\x1b\][0-9]*;[^\x07]*\x07')
    
    # Patterns to detect interactive prompts
    PROMPT_PATTERNS = [
        re.compile(r'\[y/n\]', re.IGNORECASE),
        re.compile(r'password:', re.IGNORECASE),
        re.compile(r'\(yes/no\)', re.IGNORECASE)
    ]

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.history: list[str] = []
        self._cwd = os.getcwd()
        self._spawn_shell()

    def _spawn_shell(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.kill()
            
        shell = "cmd.exe" if sys.platform == "win32" else "/bin/bash"
        self.process = subprocess.Popen(
            [shell],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=self._cwd
        )

    def _read_output_with_timeout(self, timeout: float) -> str:
        if not self.process or not self.process.stdout:
            return ""

        output_lines: list[str] = []
        done = threading.Event()
        
        def reader() -> None:
            try:
                # Read until no more output or interactive prompt
                # Note: this is a simple implementation. In a real PTY, we'd read char by char.
                while True:
                    # Non-blocking read or small timeout read is tricky on Windows without win32api
                    # We will do a single readline per loop with a timeout handled by the thread
                    line = self.process.stdout.readline() # type: ignore
                    if not line:
                        break
                    output_lines.append(line)
                    
                    # Check for prompt
                    stripped = self.ANSI_ESCAPE.sub('', line)
                    if any(p.search(stripped) for p in self.PROMPT_PATTERNS):
                        break
            except Exception:
                pass
            finally:
                done.set()

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout)
        
        # We don't forcefully kill the thread, it will die with the process or when output ends.
        raw_output = "".join(output_lines)
        return self.ANSI_ESCAPE.sub('', raw_output)

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute terminal commands."""
        command = kwargs.get("command", "execute")
        
        if command == "reset":
            self._spawn_shell()
            return {"success": True, "output": "Terminal reset."}
            
        elif command == "get_history":
            return {"success": True, "output": "\n".join(self.history[-100:])}
            
        elif command == "get_cwd":
            return {"success": True, "output": self._cwd}
            
        elif command == "execute":
            cmd_args = kwargs.get("cmd_args")
            if not cmd_args:
                return {"success": False, "output": "cmd_args required for execute"}
                
            timeout = float(kwargs.get("timeout", 10.0))
            self.history.append(str(cmd_args))
            if len(self.history) > 100:
                self.history = self.history[-100:]

            if not self.process or self.process.poll() is not None:
                self._spawn_shell()

            try:
                # Keep track of cwd changes (simple heuristic)
                if cmd_args.startswith("cd "):
                    new_dir = cmd_args.split(" ", 1)[1].strip()
                    try:
                        os.chdir(new_dir)
                        self._cwd = os.getcwd()
                    except Exception:
                        pass
                
                self.process.stdin.write(f"{cmd_args}\n") # type: ignore
                self.process.stdin.flush() # type: ignore
                
                # Wait for output
                time.sleep(0.1) # Brief pause to let process write output
                output = self._read_output_with_timeout(timeout)
                
                exit_code = self.process.poll()
                return {
                    "success": True, 
                    "output": output, 
                    "exit_code": exit_code
                }
            except Exception as e:
                return {"success": False, "output": str(e), "exit_code": -1}

        return {"success": False, "output": f"Unknown command: {command}"}
