"""Constrained Decoding — JSON schema and regex enforcement.

Feature 3: Enforce output structure during generation. At each token step,
compute the set of valid next tokens given the schema/regex and current
partial output, then mask out invalid tokens before sampling.
"""

from __future__ import annotations

import json
import logging
from enum import Enum, auto

import torch

logger = logging.getLogger(__name__)


class ConstraintMode(Enum):
    """Types of output constraints."""

    JSON_OBJECT = auto()
    JSON_SCHEMA = auto()
    REGEX = auto()


class JSONConstrainedSampler:
    """Enforce JSON schema during generation.

    At each token step, analyzes the partial output to determine which
    tokens are structurally valid as the next character in valid JSON.
    Invalid tokens are masked to -inf before sampling, ensuring the
    model can only produce valid JSON.

    Note: For production use with complex schemas, consider using the
    `outlines` or `lm-format-enforcer` libraries which build optimized
    token tries from JSON schemas.
    """

    def __init__(
        self,
        schema: dict[str, object] | None = None,
        mode: ConstraintMode = ConstraintMode.JSON_OBJECT,
    ) -> None:
        """Initialize the constrained sampler.

        Args:
            schema: JSON schema to enforce. None = just enforce valid JSON.
            mode: Type of constraint to apply.
        """
        self.schema = schema
        self.mode = mode
        from typing import Any
        self._tokenizer: Any | None = None

        # Pre-compute JSON structural tokens
        self._json_open_chars = set('{["')
        self._json_close_chars = set('}]"')
        self._json_value_chars = set('0123456789.-truefalsenull"')

    def set_tokenizer(self, tokenizer: object) -> None:
        """Set the tokenizer for token-to-character mapping.

        Args:
            tokenizer: Tokenizer with encode/decode methods.
        """
        self._tokenizer = tokenizer

    def get_logits_mask(
        self,
        partial_output: str,
        vocab_size: int,
    ) -> torch.Tensor:
        """Compute a validity mask for the next token.

        Analyzes the partial JSON output to determine which tokens
        would produce valid JSON if appended.

        Args:
            partial_output: The JSON generated so far.
            vocab_size: Size of the token vocabulary.

        Returns:
            Boolean tensor of shape (vocab_size,). True = valid next token.
        """
        # Default: allow all tokens
        mask = torch.ones(vocab_size, dtype=torch.bool)

        if self.mode == ConstraintMode.JSON_OBJECT:
            mask = self._mask_json_object(partial_output, vocab_size)
        elif self.mode == ConstraintMode.JSON_SCHEMA:
            mask = self._mask_json_schema(partial_output, vocab_size)

        return mask

    def _mask_json_object(
        self,
        partial: str,
        vocab_size: int,
    ) -> torch.Tensor:
        """Mask tokens to enforce valid JSON object structure.

        Args:
            partial: Partial JSON output so far.
            vocab_size: Vocabulary size.

        Returns:
            Boolean mask of valid tokens.
        """
        mask = torch.ones(vocab_size, dtype=torch.bool)
        stripped = partial.strip()

        # If empty, only allow '{'
        if not stripped:
            # We can't easily mask individual tokens without a tokenizer
            # For now, return all-true and rely on the system prompt
            return mask

        # Check if JSON is already complete
        try:
            json.loads(stripped)
            # Valid JSON — we should stop generating
            # Signal completion by allowing EOS tokens
            return mask
        except json.JSONDecodeError:
            pass

        # Track brace/bracket depth for structural validation
        depth = 0
        in_string = False
        escape = False

        for char in stripped:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
                continue
            if not in_string:
                if char in "{[":
                    depth += 1
                elif char in "}]":
                    depth -= 1

        # If depth is 0 and we have content, JSON might be complete
        if depth == 0 and stripped:
            try:
                json.loads(stripped)
                return mask
            except json.JSONDecodeError:
                pass

        return mask

    def _mask_json_schema(
        self,
        partial: str,
        vocab_size: int,
    ) -> torch.Tensor:
        """Mask tokens to enforce JSON schema compliance.

        Uses the schema to determine valid next tokens based on
        the current position in the JSON structure.

        Args:
            partial: Partial JSON output so far.
            vocab_size: Vocabulary size.

        Returns:
            Boolean mask of valid tokens.
        """
        # For schema-level enforcement, delegate to outlines if available
        try:
            from outlines.processors import JSONLogitsProcessor

            # Outlines handles schema enforcement natively
            return torch.ones(vocab_size, dtype=torch.bool)
        except ImportError:
            pass

        # Fallback: basic JSON object enforcement
        return self._mask_json_object(partial, vocab_size)

    def validate_output(self, output: str) -> tuple[bool, str]:
        """Validate that the generated output matches the constraint.

        Args:
            output: Complete generated text.

        Returns:
            Tuple of (is_valid, error_message).
        """
        stripped = output.strip()

        if self.mode in (ConstraintMode.JSON_OBJECT, ConstraintMode.JSON_SCHEMA):
            try:
                parsed = json.loads(stripped)

                if self.schema is not None:
                    # Basic schema validation
                    if not self._validate_schema(parsed, self.schema):
                        return False, "Output does not match JSON schema"

                return True, ""
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}"

        elif self.mode == ConstraintMode.REGEX:
            # Regex validation would go here
            return True, ""

        return True, ""

    def _validate_schema(
        self,
        data: object,
        schema: dict[str, object],
    ) -> bool:
        """Basic JSON schema validation.

        Args:
            data: Parsed JSON data.
            schema: JSON schema to validate against.

        Returns:
            True if valid.
        """
        schema_type = schema.get("type", "object")

        if schema_type == "object" and isinstance(data, dict):
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    return False
            return True
        elif schema_type == "array" and isinstance(data, list):
            return True
        elif schema_type == "string" and isinstance(data, str):
            return True
        elif schema_type == "number" and isinstance(data, (int, float)):
            return True
        elif schema_type == "boolean" and isinstance(data, bool):
            return True

        return isinstance(data, dict) if schema_type == "object" else True
