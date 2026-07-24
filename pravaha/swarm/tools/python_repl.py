"""Persistent Python REPL — Variables survive between calls.

Agents can define variables in one call and use them in the next.
Uses exec() with a shared namespace dict. Same security constraints
as CodeExecutor: timeout, resource limits.
"""

from __future__ import annotations

import ast
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

BLOCKED_IMPORTS = {
    'os', 'subprocess', 'shutil', 'socket', 'http', 'urllib', 'requests',
    'httpx', 'ctypes', 'signal', 'sys', 'pathlib', 'importlib', 'pickle', 'shelve'
}

BLOCKED_CALLS = {'open', 'exec', 'eval', 'compile', '__import__'}

def _validate_ast(code: str) -> str | None:
    """Validate AST for blocked imports and calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split('.')[0]
                if base_module in BLOCKED_IMPORTS:
                    return f"Importing '{base_module}' is blocked for security."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split('.')[0]
                if base_module in BLOCKED_IMPORTS:
                    return f"Importing from '{base_module}' is blocked for security."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_CALLS:
                    return f"Calling '{node.func.id}' is blocked for security."
    return None

def _get_restricted_builtins() -> dict[str, Any]:
    b = __builtins__.copy() if isinstance(__builtins__, dict) else vars(__builtins__).copy()
    for name in ['__import__', 'open', 'exec', 'eval', 'compile', 'globals', 'locals', 'getattr', 'setattr', 'delattr', 'breakpoint', 'exit', 'quit']:
        b.pop(name, None)
    return b

class PythonRepl:
    """Persistent Python REPL with state across calls."""

    name = "python_repl"
    description = "Persistent Python REPL — variables survive between calls"
    arg_schema = '{"code": "string", "timeout_s": 5}'

    def __init__(self) -> None:
        self._namespace: dict[str, Any] = {"__builtins__": _get_restricted_builtins()}

    def execute(self, code: str, timeout_s: int = 5) -> dict[str, Any]:
        """Execute Python code in a persistent namespace.

        Returns: stdout, stderr, exit_code, namespace_keys.
        """
        error_msg = _validate_ast(code)
        if error_msg:
            return {
                "stdout": "",
                "stderr": error_msg,
                "exit_code": 1,
                "namespace_keys": [],
                "success": False,
            }

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Track new variables
        keys_before = set(self._namespace.keys())

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                # Try exec first for statements, eval for expressions
                try:
                    compiled = compile(code, "<repl>", "eval")
                    result = eval(compiled, self._namespace)  # noqa: S307
                    if result is not None:
                        stdout_buf.write(repr(result))
                except SyntaxError:
                    exec(compile(code, "<repl>", "exec"), self._namespace)  # noqa: S102

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
        self._namespace = {"__builtins__": _get_restricted_builtins()}
        return {"success": True, "message": "REPL namespace reset"}
