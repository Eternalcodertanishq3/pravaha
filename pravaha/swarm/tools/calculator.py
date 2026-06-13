"""Calculator Tool — Safe math expression evaluator.

Uses Python's ast module for safe parsing. No eval().
Supports: arithmetic, basic algebra, statistics, constants.
Optionally uses sympy if available for symbolic math.
"""

from __future__ import annotations

import ast
import math
import operator
import statistics
from collections.abc import Callable
from typing import Any

# Safe operators for AST evaluation
SAFE_OPS: dict[type[ast.AST], Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math constants and functions
SAFE_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pow": pow,
    "mean": statistics.mean,
    "median": statistics.median,
    "stdev": statistics.stdev,
}


def _safe_eval(node: ast.AST) -> Any:
    """Safely evaluate an AST node (no exec/eval/import)."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_func = SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        op_func = SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        func = SAFE_NAMES.get(func_name)
        if func is None:
            raise ValueError(f"Unknown function: {func_name}")
        args = [_safe_eval(arg) for arg in node.args]
        return func(*args)
    elif isinstance(node, ast.Name):
        if node.id in SAFE_NAMES:
            return SAFE_NAMES[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    elif isinstance(node, ast.List):
        return [_safe_eval(elt) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_safe_eval(elt) for elt in node.elts)
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class Calculator:
    """Evaluate math expressions safely."""

    name = "calculate"
    description = "Evaluate math expressions safely (arithmetic, trig, stats)"
    arg_schema = '{"expression": "2 * pi * 5 ** 2"}'

    def execute(self, expression: str) -> dict[str, Any]:
        """Evaluate a math expression."""
        if not expression or not expression.strip():
            return {"error": "Empty expression", "success": False}

        expression = expression.strip()

        # Try sympy first if available (for symbolic math)
        sympy_result = self._try_sympy(expression)
        if sympy_result is not None:
            return sympy_result

        # Fall back to safe AST evaluation
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)

            return {
                "result": result,
                "expression": expression,
                "type": type(result).__name__,
                "success": True,
            }
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            return {
                "error": f"Evaluation error: {e}",
                "expression": expression,
                "success": False,
            }
        except SyntaxError as e:
            return {
                "error": f"Syntax error: {e}",
                "expression": expression,
                "success": False,
            }

    @staticmethod
    def _try_sympy(expression: str) -> dict[str, Any] | None:
        """Try to evaluate with sympy for symbolic math."""
        try:
            import sympy
            result = sympy.sympify(expression)
            numeric = float(result.evalf()) if result.is_number else None
            return {
                "result": numeric if numeric is not None else str(result),
                "symbolic": str(result),
                "latex": sympy.latex(result),
                "expression": expression,
                "engine": "sympy",
                "success": True,
            }
        except ImportError:
            return None
        except Exception:
            return None  # Fall back to AST
