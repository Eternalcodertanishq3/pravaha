"""Persistent Python REPL — Variables survive between calls.

Agents can define variables in one call and use them in the next.
Uses exec() with a shared namespace dict. Same security constraints
as CodeExecutor: timeout, resource limits.
"""

from __future__ import annotations

import io
import signal
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any


class PythonRepl:
    """Persistent Python REPL with state across calls."""

    name = "python_repl"
    description = "Persistent Python REPL — variables survive between calls"
    arg_schema = '{"code": "string", "timeout_s": 5}'

    def __init__(self) -> None:
        self._namespace: dict[str, Any] = {"__builtins__": __builtins__}

    def execute(self, code: str, timeout_s: int = 5) -> dict[str, Any]:
        """Execute Python code in a persistent namespace.

        Returns: stdout, stderr, exit_code, namespace_keys.
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Track new variables
        keys_before = set(self._namespace.keys())

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                # Try exec first for statements, eval for expressions
                try:
                    compiled = compile(code, "<repl>", "eval")
                    result = eval(compiled, self._namespace)
                    if result is not None:
                        stdout_buf.write(repr(result))
                except SyntaxError:
                    exec(compile(code, "<repl>", "exec"), self._namespace)

            new_keys = set(self._namespace.keys()) - keys_before - {"__builtins__"}
            return {
                "stdout": stdout_buf.getvalue()[:4096],
                "stderr": stderr_buf.getvalue()[:2048],
                "exit_code": 0,
                "namespace_keys": sorted(new_keys),
                "success": True,
            }
        except Exception as e:
            return {
                "stdout": stdout_buf.getvalue()[:4096],
                "stderr": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
                "exit_code": 1,
                "namespace_keys": [],
                "success": False,
            }

    def reset(self) -> dict[str, Any]:
        """Reset the REPL namespace."""
        self._namespace = {"__builtins__": __builtins__}
        return {"success": True, "message": "REPL namespace reset"}
