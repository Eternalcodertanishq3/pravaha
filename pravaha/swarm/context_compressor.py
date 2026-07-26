from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 2000
MAX_LINES = 200
SUMMARY_LINES = 50

class ContextCompressor:
    """
    ContextCompressor reduces tool output token usage while preserving critical information.
    """

    def compress(self, content: str, content_type: str = "auto") -> str:
        """
        Compresses content based on its type.

        Args:
            content: The content to compress.
            content_type: The type of content ("auto" to detect automatically).

        Returns:
            The compressed content.
        """
        if not content:
            return content

        if content_type == "auto":
            content_type = self._detect_type(content)

        lines = content.splitlines()
        if len(lines) <= SUMMARY_LINES and content_type != "json_data":
            return content

        if content_type == "build_log":
            return self._compress_build_log(content)
        elif content_type == "source_code":
            return self._compress_source_code(content)
        elif content_type == "stack_trace":
            return self._extract_stack_trace(content)
        elif content_type == "command_output":
            return self._compress_command_output(content)
        elif content_type == "json_data":
            return self._compress_json(content)
        else:
            return self._truncate_with_summary(content)

    def _detect_type(self, content: str) -> str:
        """Detects the type of content."""
        content_stripped = content.strip()

        if "Traceback (most recent call last)" in content or re.search(r'Error:.*(?:line \d+|File ".*")', content):
            return "stack_trace"

        if content_stripped.startswith("{") or content_stripped.startswith("["):
            try:
                json.loads(content_stripped)
                return "json_data"
            except ValueError:
                pass

        lower_content = content.lower()
        build_markers = ["[info]", "[warning]", "[error]", "build", "passed", "failed"]
        marker_count = sum(1 for marker in build_markers if marker in lower_content)
        if marker_count >= 2:
            return "build_log"

        code_markers = ["def ", "class ", "import ", "from "]
        if any(marker in content for marker in code_markers) and "    " in content:
            return "source_code"

        if content_stripped.startswith("$") or content_stripped.startswith(">") or re.search(r'(?m)^[\w@:\-~]+[$#%]\s', content):
            return "command_output"

        return "text"

    def _compress_build_log(self, content: str) -> str:
        """Compresses a build log by keeping ERROR/WARNING lines + first 5 and last 10 lines."""
        lines = content.splitlines()
        if len(lines) <= 15:
            return content

        header = lines[:5]
        footer = lines[-10:]
        middle = lines[5:-10]

        important_lines = []
        stripped_count = 0
        for line in middle:
            lower_line = line.lower()
            if "error" in lower_line or "warning" in lower_line or "failed" in lower_line:
                important_lines.append(line)
            else:
                stripped_count += 1

        result = header
        if important_lines:
            result.append(f"\n... ({stripped_count} lines omitted) ...\n")
            result.extend(important_lines)
        else:
            result.append(f"\n... ({stripped_count} lines omitted, no errors/warnings found) ...\n")

        result.extend(footer)
        return "\n".join(result)

    def _compress_source_code(self, content: str) -> str:
        """Compresses source code by extracting AST if Python and > 200 lines."""
        lines = content.splitlines()
        if len(lines) <= MAX_LINES:
            return content

        try:
            tree = ast.parse(content)
            extracted = []

            class SignatureExtractor(ast.NodeVisitor):
                def visit_ClassDef(self, node):
                    extracted.append(f"Line {node.lineno}: class {node.name}:")
                    self.generic_visit(node)

                def visit_FunctionDef(self, node):
                    extracted.append(f"Line {node.lineno}: def {node.name}(...):")

                def visit_AsyncFunctionDef(self, node):
                    extracted.append(f"Line {node.lineno}: async def {node.name}(...):")

            SignatureExtractor().visit(tree)

            if extracted:
                return "\n".join(extracted)
        except SyntaxError:
            pass

        return "\n".join(lines[:50] + [f"\n... ({len(lines) - 70} lines omitted) ...\n"] + lines[-20:])

    def _extract_stack_trace(self, content: str) -> str:
        """Extracts stack trace block with 3 lines of context."""
        lines = content.splitlines()
        traceback_start = -1

        for i, line in enumerate(lines):
            if "Traceback (most recent call last)" in line or re.search(r'Error:.*(?:line \d+|File ".*")', line):
                traceback_start = i
                break

        if traceback_start == -1:
            return self._truncate_with_summary(content)

        start_idx = max(0, traceback_start - 3)
        end_idx = min(len(lines), traceback_start + 50)

        return "\n".join(lines[start_idx:end_idx])

    def _compress_command_output(self, content: str) -> str:
        """Compresses command output."""
        lines = content.splitlines()
        if len(lines) <= 100:
            return content

        return "\n".join(lines[:20] + [f"\n... ({len(lines) - 50} lines omitted) ...\n"] + lines[-30:])

    def _compress_json(self, content: str) -> str:
        """Compresses JSON by showing structure if >2000 chars."""
        content_stripped = content.strip()
        if len(content_stripped) <= 2000:
            return content

        try:
            data = json.loads(content_stripped)
            return self._summarize_json_structure(data)
        except ValueError:
            return self._truncate_with_summary(content)

    def _summarize_json_structure(self, data: Any, indent: int = 0) -> str:
        """Recursively summarize JSON structure."""
        spacing = " " * indent
        if isinstance(data, dict):
            if not data:
                return "{}"
            result = ["{"]
            for k, v in data.items():
                result.append(f"{spacing}  \"{k}\": {self._summarize_json_structure(v, indent + 2)},")
            result.append(f"{spacing}}}")
            return "\n".join(result)
        elif isinstance(data, list):
            if not data:
                return "[]"
            return f"[... {len(data)} items ...]"
        elif isinstance(data, str):
            if len(data) > 50:
                return f"\"{data[:47]}...\""
            return f"\"{data}\""
        else:
            return str(data)

    def _truncate_with_summary(self, content: str) -> str:
        """Generic fallback truncate."""
        lines = content.splitlines()
        if len(lines) <= SUMMARY_LINES:
            return content

        return "\n".join(lines[:30] + [f"\n... ({len(lines) - 50} lines omitted) ...\n"] + lines[-20:])
