"""Function Calling / Tool Use — OpenAI-compatible tool routing.

Feature 4: Detect when model output is a function call, parse it, and
return structured ToolCall object instead of raw text. Compatible with
the OpenAI tool_calls format.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FunctionDefinition:
    """Definition of a callable function/tool.

    Matches the OpenAI function definition format.

    Attributes:
        name: Function name.
        description: What the function does.
        parameters: JSON Schema for function parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A parsed function/tool call from model output.

    Attributes:
        id: Unique call identifier.
        function_name: Name of the function to call.
        arguments: Parsed argument dict.
        raw_text: Original text that was parsed.
    """

    id: str
    function_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""


class FunctionCallParser:
    """Parse and format function calls from model output.

    Supports multiple function call formats:
    - OpenAI-style: {"name": "...", "arguments": {...}}
    - XML-style: <function>name</function><arguments>...</arguments>
    - Markdown-style: ```tool_call\n{"name": "...", "arguments": {...}}\n```
    """

    # Patterns for detecting function calls in model output
    _JSON_PATTERN = re.compile(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
        re.DOTALL,
    )
    _XML_PATTERN = re.compile(
        r"<function_call>\s*"
        r"<name>([^<]+)</name>\s*"
        r"<arguments>(.*?)</arguments>\s*"
        r"</function_call>",
        re.DOTALL,
    )
    _TOOL_CALL_PATTERN = re.compile(
        r"```tool_call\s*\n(.*?)\n\s*```",
        re.DOTALL,
    )

    def __init__(
        self,
        tools: list[FunctionDefinition] | None = None,
    ) -> None:
        """Initialize the parser.

        Args:
            tools: Available function definitions. Used for validation.
        """
        self.tools = {t.name: t for t in (tools or [])}
        self._call_counter = 0

    def detect(
        self,
        token_ids: list[int],
        text: str,
    ) -> ToolCall | None:
        """Detect if the output text contains a function call.

        Tries multiple formats in order of specificity.

        Args:
            token_ids: Generated token IDs (unused, for future logit analysis).
            text: Generated text to check for function calls.

        Returns:
            ToolCall if detected, None otherwise.
        """
        # Try JSON format first
        call = self._try_json_format(text)
        if call:
            return call

        # Try XML format
        call = self._try_xml_format(text)
        if call:
            return call

        # Try markdown tool_call format
        call = self._try_markdown_format(text)
        if call:
            return call

        return None

    def _try_json_format(self, text: str) -> ToolCall | None:
        """Try to parse OpenAI-style JSON function call."""
        match = self._JSON_PATTERN.search(text)
        if match:
            name = match.group(1)
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = {}

            self._call_counter += 1
            return ToolCall(
                id=f"call_{self._call_counter}",
                function_name=name,
                arguments=args,
                raw_text=match.group(0),
            )
        return None

    def _try_xml_format(self, text: str) -> ToolCall | None:
        """Try to parse XML-style function call."""
        match = self._XML_PATTERN.search(text)
        if match:
            name = match.group(1).strip()
            try:
                args = json.loads(match.group(2).strip())
            except json.JSONDecodeError:
                args = {"raw": match.group(2).strip()}

            self._call_counter += 1
            return ToolCall(
                id=f"call_{self._call_counter}",
                function_name=name,
                arguments=args,
                raw_text=match.group(0),
            )
        return None

    def _try_markdown_format(self, text: str) -> ToolCall | None:
        """Try to parse markdown tool_call block."""
        match = self._TOOL_CALL_PATTERN.search(text)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                name = data.get("name", "unknown")
                args = data.get("arguments", {})

                self._call_counter += 1
                return ToolCall(
                    id=f"call_{self._call_counter}",
                    function_name=name,
                    arguments=args,
                    raw_text=match.group(0),
                )
            except json.JSONDecodeError:
                pass
        return None

    def format_tool_result(
        self,
        result: str,
        call: ToolCall,
    ) -> str:
        """Format a tool execution result for injection back into the conversation.

        Args:
            result: The result text from executing the tool.
            call: The original ToolCall that produced this result.

        Returns:
            Formatted message for the model.
        """
        return f"Tool call result for {call.function_name} (call_id: {call.id}):\n{result}"

    def build_tools_prompt(self) -> str:
        """Build a system prompt section describing available tools.

        Returns:
            Formatted tools description for the system prompt.
        """
        if not self.tools:
            return ""

        lines = ["You have access to the following tools:\n"]
        for name, func in self.tools.items():
            lines.append(f"### {name}")
            lines.append(f"{func.description}")
            if func.parameters:
                lines.append(f"Parameters: {json.dumps(func.parameters, indent=2)}")
            lines.append("")

        lines.append('To call a tool, respond with JSON: {"name": "tool_name", "arguments": {...}}')
        return "\n".join(lines)

    def validate_call(self, call: ToolCall) -> tuple[bool, str]:
        """Validate a tool call against registered function definitions.

        Args:
            call: The tool call to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if call.function_name not in self.tools:
            available = ", ".join(self.tools.keys())
            return False, (f"Unknown function: {call.function_name}. Available: {available}")

        func = self.tools[call.function_name]
        required = func.parameters.get("required", [])
        for param in required:
            if param not in call.arguments:
                return False, f"Missing required parameter: {param}"

        return True, ""
