"""JSON Tool — Parse, query, validate, and transform JSON data.

Supports dot-path queries and index access for navigating
nested JSON structures.
"""

from __future__ import annotations

import json
from typing import Any


class JsonTool:
    """Parse/query/transform JSON data."""

    name = "json_query"
    description = "Parse/query/transform JSON data with dot-path access"
    arg_schema = '{"data": "json_string", "query": "dot.path or [index]", "action": "query|validate|keys|flatten"}'

    def execute(
        self,
        data: str,
        query: str = "",
        action: str = "query",
    ) -> dict[str, Any]:
        """Execute JSON operation."""
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON: {e}",
                "valid": False,
                "success": False,
            }

        if action == "validate":
            return {
                "valid": True,
                "type": type(parsed).__name__,
                "length": len(parsed) if isinstance(parsed, (list, dict)) else None,
                "success": True,
            }

        if action == "keys":
            if isinstance(parsed, dict):
                return {"keys": list(parsed.keys()), "success": True}
            return {"error": "Data is not a JSON object", "success": False}

        if action == "flatten":
            flat = self._flatten(parsed)
            return {"result": flat, "success": True}

        # Default: query with dot-path
        if not query:
            return {"result": parsed, "success": True}

        result = self._dot_query(parsed, query)
        if result is _SENTINEL:
            return {"error": f"Path '{query}' not found", "success": False}

        return {
            "result": result,
            "type": type(result).__name__,
            "success": True,
        }

    @staticmethod
    def _dot_query(data: Any, path: str) -> Any:
        """Navigate JSON with dot.path and [index] notation."""
        current = data
        # Split on dots but handle array indices
        parts = []
        for part in path.split("."):
            if "[" in part:
                name, rest = part.split("[", 1)
                if name:
                    parts.append(name)
                idx_str = rest.rstrip("]")
                try:
                    parts.append(int(idx_str))
                except ValueError:
                    parts.append(idx_str)
            else:
                parts.append(part)

        for part in parts:
            try:
                if isinstance(part, int):
                    current = current[part]
                elif isinstance(current, dict):
                    current = current[part]
                elif isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                else:
                    return _SENTINEL
            except (KeyError, IndexError, TypeError):
                return _SENTINEL
        return current

    @staticmethod
    def _flatten(data: Any, prefix: str = "", sep: str = ".") -> dict[str, Any]:
        """Flatten nested JSON into dot-path keys."""
        result: dict[str, Any] = {}
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}{sep}{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    result.update(JsonTool._flatten(value, full_key, sep))
                else:
                    result[full_key] = value
        elif isinstance(data, list):
            for i, value in enumerate(data):
                full_key = f"{prefix}[{i}]"
                if isinstance(value, (dict, list)):
                    result.update(JsonTool._flatten(value, full_key, sep))
                else:
                    result[full_key] = value
        else:
            result[prefix] = data
        return result


# Sentinel for missing values
_SENTINEL = object()
