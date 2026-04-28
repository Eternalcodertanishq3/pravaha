"""Output Validator — Schema-based output validation with retry logic.

Validates agent outputs against expected formats:
- JSON schema compliance
- Required field checking
- Type validation
- Content length constraints
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of output validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    cleaned_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class OutputValidator:
    """Validate agent output against schemas and constraints.

    Provides retry recommendations when validation fails.
    """

    def validate_json(
        self,
        output: str,
        required_fields: list[str] | None = None,
        field_types: dict[str, type] | None = None,
    ) -> ValidationResult:
        """Validate JSON output structure."""
        errors: list[str] = []
        warnings: list[str] = []

        # Try to extract JSON from code blocks
        cleaned = self._extract_json(output)
        if cleaned is None:
            return ValidationResult(
                valid=False,
                errors=["Output is not valid JSON"],
                warnings=[],
                cleaned_output=output,
            )

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                errors=[f"JSON parse error: {e}"],
                warnings=[],
                cleaned_output=cleaned,
            )

        # Check required fields
        if required_fields and isinstance(parsed, dict):
            for field in required_fields:
                if field not in parsed:
                    errors.append(f"Missing required field: '{field}'")
                elif parsed[field] is None:
                    warnings.append(f"Field '{field}' is null")

        # Check field types
        if field_types and isinstance(parsed, dict):
            for field_name, expected_type in field_types.items():
                if field_name in parsed and not isinstance(parsed[field_name], expected_type):
                    actual = type(parsed[field_name]).__name__
                    errors.append(
                        f"Field '{field_name}' expected {expected_type.__name__}, "
                        f"got {actual}"
                    )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_output=cleaned,
        )

    def validate_text(
        self,
        output: str,
        min_length: int = 10,
        max_length: int = 10000,
        must_contain: list[str] | None = None,
        must_not_contain: list[str] | None = None,
    ) -> ValidationResult:
        """Validate text output against constraints."""
        errors: list[str] = []
        warnings: list[str] = []

        if len(output) < min_length:
            errors.append(f"Output too short: {len(output)} < {min_length}")

        if len(output) > max_length:
            warnings.append(f"Output truncated: {len(output)} > {max_length}")

        if must_contain:
            for phrase in must_contain:
                if phrase.lower() not in output.lower():
                    errors.append(f"Missing required content: '{phrase}'")

        if must_not_contain:
            for phrase in must_not_contain:
                if phrase.lower() in output.lower():
                    errors.append(f"Contains forbidden content: '{phrase}'")

        # Check for common LLM issues
        if output.strip().startswith("I cannot") or output.strip().startswith("I'm sorry"):
            warnings.append("Output appears to be a refusal")

        if output.count("```") % 2 != 0:
            warnings.append("Unclosed code block detected")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_output=output[:max_length],
        )

    def validate_code(self, output: str, language: str = "python") -> ValidationResult:
        """Validate code output."""
        errors: list[str] = []
        warnings: list[str] = []

        # Extract code from markdown blocks
        code = self._extract_code(output, language)
        if not code:
            warnings.append("No code block found, using raw output")
            code = output

        if language == "python":
            # Basic syntax check
            try:
                compile(code, "<validation>", "exec")
            except SyntaxError as e:
                errors.append(f"Python syntax error: {e}")

            # Check for common issues
            if "import *" in code:
                warnings.append("Star import detected")
            if "eval(" in code or "exec(" in code:
                warnings.append("eval/exec usage detected")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_output=code,
        )

    def get_retry_prompt(self, validation: ValidationResult, original_task: str) -> str:
        """Generate a retry prompt based on validation errors."""
        error_list = "\n".join(f"- {e}" for e in validation.errors)
        warning_list = "\n".join(f"- {w}" for w in validation.warnings)

        return (
            f"Your previous output had validation errors:\n"
            f"ERRORS:\n{error_list}\n"
            f"{f'WARNINGS:{chr(10)}{warning_list}{chr(10)}' if validation.warnings else ''}"
            f"\nPlease fix these issues and try again.\n"
            f"Original task: {original_task}"
        )

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract JSON from text, handling code blocks."""
        # Try code block first
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if candidate.startswith("{") or candidate.startswith("["):
                return candidate

        # Try raw JSON
        text = text.strip()
        if text.startswith("{") or text.startswith("["):
            return text

        return None

    @staticmethod
    def _extract_code(text: str, language: str = "python") -> str:
        """Extract code from markdown code blocks."""
        pattern = rf"```{language}\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return ""
